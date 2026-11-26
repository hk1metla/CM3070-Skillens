"""Reproduce Skillens rankings and audit the evaluation denominator.

This companion does not edit repository source files. It uses its original split helper,
ingestion code, recommender classes and metric implementations. It replaces only
repeated full-frame user scans with pre-grouped data. Run with PYTHONHASHSEED=0.
No learner-level records are written to the public output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from src.data.make_splits import _split_user_history
    from src.data import ingest
    from src.models.tfidf import TfidfRecommender
    from src.models.itemknn import ItemKNNRecommender
    from src.models.hybrid import HybridRecommender
    from src.models.popularity import PopularityRecommender
    from src.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
    from src.eval.statistical import bootstrap_confidence_interval

    work = tempfile.TemporaryDirectory(prefix='skillens-oulad-')
    raw = Path(work.name)
    with zipfile.ZipFile(args.data_zip) as z:
        for name in ['vle.csv', 'courses.csv', 'studentInfo.csv']:
            (raw / name).write_bytes(z.read(name))
        vle = pd.read_csv(raw / 'vle.csv')
        courses = pd.read_csv(raw / 'courses.csv')
        ids = ('oulad_' + courses.code_module + '_' + courses.code_presentation).tolist()
        site_map = dict(zip(vle.id_site, 'oulad_' + vle.code_module + '_' + vle.code_presentation))
        parts = []
        row_count = 0
        click_sum = 0
        with z.open('studentVle.csv') as f:
            for chunk in pd.read_csv(f, chunksize=200_000,
                    usecols=['id_student', 'id_site', 'date', 'sum_click'],
                    dtype={'id_student':'int32','id_site':'int32','date':'int16','sum_click':'int32'}):
                row_count += len(chunk)
                click_sum += int(chunk.sum_click.sum())
                mapped = chunk.id_site.map(site_map)
                keep = mapped.notna()
                parts.append(pd.DataFrame({'user_id':chunk.loc[keep, 'id_student'],
                    'item_id':mapped[keep], 'day':chunk.loc[keep, 'date']}))
    events = pd.concat(parts, ignore_index=True)
    del parts
    date_map = {int(d):(pd.Timestamp('2013-01-01') + pd.Timedelta(days=int(d))).strftime('%Y-%m-%dT%H:%M:%S')
                for d in events.day.unique()}
    events['timestamp'] = events.day.map(date_map)
    events = events.drop(columns='day').sort_values(['user_id','timestamp'])
    print('Loaded', len(events), 'mapped rows;', events.user_id.nunique(), 'users', flush=True)

    train_parts = []
    user_records = {}
    total_sizes = np.zeros(3, dtype=np.int64)
    test_users = 0
    violations = 0
    tied_boundaries = 0
    for uid, group in events.groupby('user_id', sort=True):
        train, val, test = _split_user_history(group, 0.6, 0.2)
        total_sizes += [len(train),len(val),len(test)]
        if len(test): test_users += 1
        for a,b in [(train,val),(val,test)]:
            if len(a) and len(b):
                violations += int(a.timestamp.max() > b.timestamp.min())
                tied_boundaries += int(a.timestamp.max() == b.timestamp.min())
        train_parts.append(train[['user_id','item_id']])
        tr_items = train.item_id.tolist()
        te_items = test.item_id.tolist()
        relevant = set(te_items) - set(tr_items)
        user_records[str(uid)] = {'train_first3':tr_items[:3],
            'train_unique':sorted(set(tr_items)), 'test_unique':sorted(set(te_items)),
            'val_unique':sorted(set(val.item_id.tolist())),
            'relevant':sorted(relevant),'n_train':len(train),'n_test':len(test)}
    train = pd.concat(train_parts, ignore_index=True)
    train['user_id'] = train.user_id.astype(str)
    del events, train_parts

    ingest.OULAD_DIR = str(raw)
    items = ingest._ingest_oulad()
    assert items.item_id.tolist() == ids
    lookup = items.set_index('item_id')
    models = {'popularity':PopularityRecommender(),'tfidf':TfidfRecommender(),
        'itemknn':ItemKNNRecommender(k=50),'hybrid':HybridRecommender(w_content=.6,w_cf=.4)}
    models['popularity'].fit(train)
    models['tfidf'].fit(items)
    models['itemknn'].fit(train)
    models['hybrid'].fit(items,train)
    eligible = {u:r for u,r in user_records.items() if r['n_test'] and r['relevant']}
    print('Eligible novel-item users:',len(eligible),'of',test_users,'test users',flush=True)
    summary = {'raw_rows':row_count,'raw_sum_click':click_sum,'mapped_rows':int(sum(total_sizes)),
        'users':len(user_records),'items':len(items),'test_users':test_users,
        'eligible_users':len(eligible),'eligible_fraction_of_test':len(eligible)/test_users,
        'split_sizes':dict(zip(['train','validation','test'],map(int,total_sizes))),
        'relative_day_boundary_violations':violations,'tied_user_split_boundaries':tied_boundaries,
        'n_zero_train_test_users':sum(r['n_test']>0 and r['n_train']==0 for r in user_records.values()),
        'n_relevant_distribution':pd.Series([len(r['relevant']) for r in eligible.values()]).value_counts().sort_index().to_dict(),
        'tfidf_vocabulary_size':len(models['tfidf'].vectorizer.vocabulary_),
        'train_unique_pairs':len(train.drop_duplicates()),
        'train_matrix_sparsity':1-len(train.drop_duplicates())/(len(user_records)*len(items))}

    # Historical calls preserve the original candidate handling. Controlled calls
    # use a common unseen-item catalogue, score every item, and break ties by ID.
    metrics = []
    per_user = {}
    rec_cache = {}
    def record(protocol,name,uid,recs,rel):
        vals = {'precision':precision_at_k(recs,rel,10),
                'recall':recall_at_k(recs,rel,10),'ndcg':ndcg_at_k(recs,rel,10)}
        per_user.setdefault((protocol,name),{})[uid] = vals
        rec_cache[(protocol,name,uid)] = recs

    all_ids = items.item_id.tolist()
    pop_scores = train.item_id.value_counts().to_dict()
    content = models['tfidf']
    cf = models['itemknn']
    from sklearn.metrics.pairwise import cosine_similarity
    for uid,r in sorted(eligible.items()):
        goal = ' '.join(lookup.loc[i,'title'] for i in r['train_first3'])
        hist = pd.DataFrame({'user_id':[uid]*len(r['train_unique']),'item_id':r['train_unique']})
        rel = r['relevant']
        for name,model in models.items():
            if name=='popularity': recs=model.recommend(k=10)
            elif name=='tfidf': recs=model.recommend(goal,k=10)
            elif name=='itemknn': recs=model.recommend(uid,hist,k=10)
            else: recs=model.recommend(goal=goal,user_id=uid,user_interactions=hist,k=10)
            record('repository',name,uid,recs.item_id.tolist(),rel)

        candidates = sorted(set(all_ids)-set(r['train_unique']))
        text_scores = cosine_similarity(content.vectorizer.transform([goal]),content.item_matrix).ravel()
        text_scores = dict(zip(all_ids,map(float,text_scores)))
        cf_indices = [cf.item_to_idx[i] for i in r['train_unique']]
        cf_vector = cf.item_similarity_matrix[cf_indices].mean(axis=0)
        cf_scores = dict(zip(cf.item_ids,map(float,cf_vector)))
        def normalize(scores):
            v=np.array([scores.get(i,0.) for i in candidates]);span=v.max()-v.min()
            return dict(zip(candidates,((v-v.min())/span if span else np.zeros_like(v))))
        tn,cn = normalize(text_scores),normalize(cf_scores)
        controlled={'popularity':pop_scores,'tfidf':text_scores,'itemknn':cf_scores,
                    'hybrid':{i:.6*tn[i]+.4*cn[i] for i in candidates}}
        for name,scores in controlled.items():
            ranked=sorted(candidates,key=lambda i:(-scores.get(i,0.),i))[:10]
            record('common_unseen',name,uid,ranked,rel)
    for (protocol,name),keyed in per_user.items():
        row={'protocol':protocol,'model':name,'n_users':len(keyed)}
        for metric in ['precision','recall','ndcg']:
            values=[keyed[u][metric] for u in sorted(keyed)]
            mean,lo,hi=bootstrap_confidence_interval(values,n_bootstrap=10000,random_seed=42)
            row.update({metric:mean,metric+'_ci_lower':lo,metric+'_ci_upper':hi})
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(out/'ranking_results.csv',index=False)
    # Paired nDCG tests: all six pairs in both policies form one 12-test family.
    sig=[]
    for protocol in ['repository','common_unseen']:
        for a,b in itertools.combinations(models,2):
            aa=per_user[(protocol,a)];bb=per_user[(protocol,b)]
            common=sorted(set(aa)&set(bb))
            delta=np.array([aa[u]['ndcg']-bb[u]['ndcg'] for u in common])
            p=1. if np.all(delta==0) else float(wilcoxon(delta,zero_method='wilcox',correction=False,alternative='two-sided',method='asymptotic').pvalue)
            sig.append({'protocol':protocol,'model_a':a,'model_b':b,'n_paired':len(common),
                'mean_a_minus_b':float(delta.mean()),'wilcoxon_p_raw':p,
                'wilcoxon_p_bonferroni_6':min(1.,6*p),'wilcoxon_p_bonferroni_12':min(1.,12*p),
                'wins_a':int((delta>0).sum()),'wins_b':int((delta<0).sum()),'ties':int((delta==0).sum()),
                'wilcoxon_zero_method':'wilcox','wilcoxon_method':'asymptotic','continuity_correction':False})
    pd.DataFrame(sig).to_csv(out/'paired_tests.csv',index=False)
    # Aggregate paired-score frequencies permit independent p-value checks
    # without writing student identifiers or demographic links.
    paired_counts=[]
    for protocol in ['repository','common_unseen']:
        for a,b in itertools.combinations(models,2):
            aa=per_user[(protocol,a)];bb=per_user[(protocol,b)]
            pairs=pd.DataFrame([(aa[u]['ndcg'],bb[u]['ndcg']) for u in sorted(eligible)],columns=['ndcg_a','ndcg_b'])
            for (va,vb),frequency in pairs.groupby(['ndcg_a','ndcg_b']).size().items():
                paired_counts.append({'protocol':protocol,'model_a':a,'model_b':b,'ndcg_a':va,'ndcg_b':vb,'count':int(frequency)})
    pd.DataFrame(paired_counts).to_csv(out/'paired_rank_counts.csv',index=False)

    # Complete group census among eligible users, no pre-eligibility sampling.
    demo=pd.read_csv(raw/'studentInfo.csv').drop_duplicates('id_student',keep='first')
    demo['user_id']=demo.id_student.astype(str)
    demo=demo[demo.user_id.isin(user_records)]
    groups=[]
    for category in ['gender','age_band','highest_education','disability']:
        for group,dg in demo.groupby(category,dropna=False,sort=True):
            test_ids=[u for u in dg.user_id if user_records[u]['n_test']]
            eligible_ids=[u for u in test_ids if u in eligible]
            row={'category':category,'group':str(group),'n_test_users':len(test_ids),
                 'n_evaluable':len(eligible_ids),'eligibility_rate':len(eligible_ids)/len(test_ids) if test_ids else None}
            for protocol in ['repository','common_unseen']:
                vals=[per_user[(protocol,'hybrid')][u]['ndcg'] for u in eligible_ids]
                if len(vals) >= 30:
                    mean,lo,hi=bootstrap_confidence_interval(vals,n_bootstrap=10000,random_seed=42)
                    row.update({protocol+'_ndcg':mean,protocol+'_ci_lower':lo,protocol+'_ci_upper':hi})
                else: row.update({protocol+'_ndcg':None,protocol+'_ci_lower':None,protocol+'_ci_upper':None})
            groups.append(row)
    pd.DataFrame(groups).to_csv(out/'group_results.csv',index=False)

    # Repository API invariant checks, including the history type failure.
    checks={}
    checks['split_total_equals_mapped_rows']=sum(total_sizes)==row_count
    checks['per_user_relative_day_nondecreasing']=violations==0
    checks['all_metrics_bounded']=all(0<=r[m]<=1 for r in metrics for m in ['precision','recall','ndcg'])
    checks['common_unseen_excludes_history']=all(not(set(recs)&set(user_records[uid]['train_unique']))
        for (protocol,name,uid),recs in rec_cache.items() if protocol=='common_unseen')
    checks['aligned_model_user_sets']=all(set(per_user[(p,n)])==set(eligible) for p in ['repository','common_unseen'] for n in models)
    probe_uid=sorted(eligible)[0];pr=eligible[probe_uid]
    probe=pd.DataFrame({'user_id':[int(probe_uid)]*len(pr['train_unique']),'item_id':pr['train_unique']})
    probe_recs=cf.recommend(str(probe_uid),probe,k=10)
    checks['integer_history_string_query_returns_empty']=len(probe_recs)==0
    checks['knn_parameter_unused_for_neighbor_truncation']=not any(
        'self.k' in line for line in (repo/'src/models/itemknn.py').read_text().splitlines()
        if 'self.k = ' not in line)
    (out/'verification_checks.json').write_text(json.dumps({k:bool(v) for k,v in checks.items()},indent=2)+'\n')
    (out/'data_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    work.cleanup()
    manifest={'repository':'https://github.com/hk1metla/CM3070-Skillens',
        'git_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'dataset_zip_sha256':digest(args.data_zip),'script_sha256':digest(__file__),
        'python':sys.version.split()[0],'pythonhashseed':os.environ.get('PYTHONHASHSEED'),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        'bootstrap_resamples':10000,'bootstrap_seed':42,'k':10,
        'weights':{'content':.6,'cf':.4},'weight_selection':'repository defaults; no new selection',
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'wilcoxon_method':'asymptotic','wilcoxon_zero_method':'wilcox',
        'continuity_correction':False,'n_pairwise_tests_in_family':12,'notes':[
            'Uses original item text, including author-assigned subject labels. These labels are unverified.',
            'Uses repository relative-day split; does not establish real calendar chronology.',
            'common_unseen is a separate sensitivity analysis; it does not reproduce the original candidate handling.',
            'No raw or per-learner records are included in the public audit outputs.']}
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(pd.DataFrame(sig).to_string(index=False),flush=True)
    print(pd.DataFrame(groups).to_string(index=False),flush=True)
    print('Checks',checks,flush=True)


if __name__=='__main__':
    main()
