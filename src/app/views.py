import streamlit as st
from src.app.shared import (
    get_model_by_name,
    get_selected_model,
    is_oulad_item,
    load_interactions,
    load_items,
    load_tfidf_model,
    log_click,
    log_feedback,
)
from src.explain.template import build_explanation


# ── Shared: push all page content below the fixed nav bar ────────────────────
_PAGE_OFFSET_CSS = """
<style>
section.main > div:first-child { padding-top: 4.5rem !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────

def render_home() -> None:
    st.markdown(_PAGE_OFFSET_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="snap-section">
            <div class="app-hero">
                <div class="hero-glow"></div>
                <div class="hero-gradient"></div>
                <div class="hero-inner">
                    <div class="hero-grid">
                        <div>
                            <span class="badge">Goal-driven</span>
                            <span class="badge">Explainable</span>
                            <span class="badge">Personalized</span>
                            <h1>One app for every learning goal.</h1>
                            <p>Skillens turns intent into a focused learning path with transparent, explainable recommendations.</p>
                            <div class="hero-actions">
                                <a class="pill">No hidden models</a>
                                <a class="pill">Fast results</a>
                                <a class="pill">Clear reasoning</a>
                            </div>
                            <div style="margin-top: 20px;">
                                <div class="section-title">Start with a goal</div>
                                <div class="section-subtitle">
                                    Skillens is built for learners who want clarity. Share your goal and get a shortlist of courses
                                    that match the skills you want to build.
                                </div>
                                <div style="margin-top: 16px;">
                                    <a class="nav-login" href="?page=explore" target="_self">Get started</a>
                                </div>
                            </div>
                        </div>
                        <div class="stack-card">
                            <div class="section-title">Your learning summary</div>
                            <p class="rec-meta">Goal \u2192 Ranked shortlist \u2192 Next steps</p>
                            <div style="margin-top: 16px;">
                                <div class="pill">Build ML foundations</div>
                                <div class="pill">Practical projects</div>
                                <div class="pill">Career-ready skills</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="content-contrast">', unsafe_allow_html=True)

    # WHY SKILLENS
    st.markdown('<div class="snap-section"><div class="section-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-eyebrow">WHY SKILLENS</div>
            <div class="section-title-lg">What makes Skillens feel different</div>
            <div class="section-subtitle">A research prototype that removes decision fatigue and keeps you focused on your goal.</div>
            <div class="section-divider"></div>
        </div>
        <div class="feature-grid">
            <div class="feature-card-pro">
                <div class="feature-icon">\U0001f3af</div>
                <strong>Goal-first design</strong>
                <p>We center the experience around what you want to achieve.</p>
            </div>
            <div class="feature-card-pro">
                <div class="feature-icon">\U0001f50d</div>
                <strong>Clear reasoning</strong>
                <p>Every pick includes a short explanation you can trust.</p>
            </div>
            <div class="feature-card-pro">
                <div class="feature-icon">\u26a1</div>
                <strong>Transparent models</strong>
                <p>Switch between TF-IDF, ItemKNN, Hybrid, and Semantic recommenders.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    # FLOW
    st.markdown('<div class="snap-section"><div class="section-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-eyebrow">FLOW</div>
            <div class="section-title-lg">Unify your learning journey</div>
            <div class="section-subtitle">Three concise steps that keep you moving from goal to outcome.</div>
            <div class="section-divider"></div>
        </div>
        <div class="section-shell">
            <div class="feature-card"><strong>Pick a goal</strong><p>Start with a clear outcome and the system handles the rest.</p></div>
            <br />
            <div class="feature-card"><strong>Get curated paths</strong><p>See a ranked shortlist of courses that fit your intent.</p></div>
            <br />
            <div class="feature-card"><strong>Act with confidence</strong><p>Each recommendation includes a short explanation.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    # IMPACT — FIX Issue 3: stats are factually grounded; removed unsupported marketing copy
    st.markdown('<div class="snap-section"><div class="section-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-eyebrow">RESULTS</div>
            <div class="section-title-lg">Evaluation results</div>
            <div class="section-subtitle">OULAD contains 26,074 learners. Only 43 had a test module-presentation absent from their training history. Each had one eligible item.</div>
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # FIX Issue 3: replaced "600+ Curated courses / Fast / Transparent" with
    # actual evidenced numbers from the report (Table 10).
    stats_content = [
        ("0.703", "Hybrid NDCG@10"),
        ("43", "Evaluated learners"),
        ("0.574", "ItemKNN NDCG@10"),
    ]
    for col, (value, label) in zip(st.columns(3), stats_content):
        with col:
            st.markdown(
                f'<div class="feature-card"><strong style="font-size:24px">{value}</strong><p>{label}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.markdown("### Explore in three steps")
    steps = [
        ("01", "Define your goal", "Tell us what you want to achieve."),
        ("02", "Get matched", "We rank courses that fit your intent."),
        ("03", "Start learning", "Pick a course and keep moving."),
    ]
    for col, (num, title, desc) in zip(st.columns(3), steps):
        with col:
            st.markdown(
                f'<div class="feature-card"><strong>{num}</strong><p><strong>{title}</strong></p><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div></div>", unsafe_allow_html=True)

    # FIX Issue 3: REMOVED named testimonials (Amir K., Sofia R., Jordan L.)
    # Those were fictional quotes inconsistent with academic credibility.
    # Replaced with factual project context.
    st.markdown('<div class="snap-section"><div class="section-container">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header">
            <div class="section-eyebrow">ABOUT</div>
            <div class="section-title-lg">About this project</div>
            <div class="section-subtitle">CM3070 Final Year Project, University of London. BSc Computer Science (ML &amp; AI).</div>
            <div class="section-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    about_items = [
        ("Dataset", "Open University Learning Analytics Dataset (OULAD). 10.6M VLE interaction events."),
        ("Models", "TF-IDF content-based, ItemKNN collaborative, Hybrid fusion, Sentence-BERT semantic."),
        ("Evaluation", "Per-user temporal splits. Leakage-safe protocol. NDCG@10, Precision@10, Recall@10."),
    ]
    for col, (title, desc) in zip(st.columns(3), about_items):
        with col:
            st.markdown(
                f'<div class="feature-card"><strong>{title}</strong><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="site-footer">
            <div class="footer-inner">
                <span>\u00a9 2026 Skillens \u2014 CM3070 Final Year Project.</span>
                <div class="footer-links">
                    <a href="?page=home" target="_self">Home</a>
                    <a href="?page=explore" target="_self">Explore</a>
                    <a href="?page=login" target="_self">Guest session</a>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE / RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_recommendations() -> None:
    st.markdown(_PAGE_OFFSET_CSS, unsafe_allow_html=True)

    # Form card styling
    st.markdown(
        """
        <style>
        div[data-testid="stForm"] {
            background: #f8fafc;
            border-radius: 22px;
            padding: 32px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 20px 40px rgba(15,23,42,0.12);
        }
        div[data-testid="stForm"] label,
        div[data-testid="stForm"] p,
        div[data-testid="stForm"] .stMarkdown { color: #0f172a !important; }
        div[data-testid="stForm"] input,
        div[data-testid="stForm"] textarea { background: #eef2f7 !important; color: #0f172a !important; }
        div[data-testid="stForm"] input::placeholder { color: #64748b !important; opacity: 1; }
        div[data-testid="stForm"] .stRadio label { color: #0f172a !important; font-size: 0.88rem !important; }
        div[data-testid="stForm"] .stFormSubmitButton > button,
        div[data-testid="stForm"] .stButton > button {
            background: #0f172a !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-hero app-hero--compact enterprise-hero">
            <div class="hero-glow"></div>
            <div class="hero-gradient"></div>
            <div class="hero-inner">
                <div class="hero-grid">
                    <div>
                        <span class="badge">Personalized</span>
                        <span class="badge">Fast</span>
                        <span class="badge">Transparent</span>
                        <h1>Find the right course</h1>
                        <p>Share your learning goal and we will curate a ranked shortlist.</p>
                    </div>
                    <div class="stack-card stack-card--enterprise">
                        <div class="section-title">Your goal input</div>
                        <p class="rec-meta">We transform goals into ranked matches.</p>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    items = load_items()

    st.markdown("### Enter Your Course")

    # ── FR1: Free-text goal input ─────────────────────────────────────────────
    goal_text = st.text_input(
        "Your learning goal",
        placeholder="e.g. I want to learn machine learning from scratch",
        help="Type a goal, or leave blank and use the dropdown / quick goals below.",
    )

    # ── Course dropdown (fallback) ────────────────────────────────────────────
    if "course_titles" not in st.session_state:
        st.session_state["course_titles"] = items["title"].dropna().unique().tolist()

    selected_title = st.selectbox(
        "Search courses",
        options=[""] + st.session_state["course_titles"],
        index=0,
        format_func=lambda x: x if x else "Enter your course",
    )

    # ── Quick-goal presets ────────────────────────────────────────────────────
    presets = [
        "Machine learning foundations",
        "Business analytics and dashboards",
        "Front-end web development",
        "Cloud support essentials",
    ]
    preset_choice = st.radio("Quick goals", presets, horizontal=True)

    # ── Model selector (FR2: visible on-page control) ─────────────────────────
    model_key = st.radio(
        "Recommender model",
        options=["hybrid", "tfidf", "itemknn", "semantic"],
        format_func=lambda x: {
            "hybrid":   "Hybrid (TF-IDF + ItemKNN)  \u2605",
            "tfidf":    "TF-IDF (content-based)",
            "itemknn":  "ItemKNN (collaborative)",
            "semantic": "Semantic (Sentence-BERT)",
        }[x],
        index=["hybrid", "tfidf", "itemknn", "semantic"].index(get_selected_model()),
        horizontal=True,
        help="Hybrid combines text and collaborative signals. "
             "Other models are available for comparison.",
    )

    # FIX Issue 2: show a clear note when ItemKNN is selected in the UI so the
    # marker understands this is centrality-based cold-start, not a history-based
    # personalised ranking (report §3.4.3 / §4.4.3 describes the history-based
    # version used in offline evaluation).
    if model_key == "itemknn":
        st.info(
            "**ItemKNN — UI demo mode.** In the offline evaluation pipeline, ItemKNN "
            "generates personalised recommendations from a learner's OULAD interaction history "
            "(report §4.4.3). In this UI, no stored user history is available, so recommendations "
            "are ranked by collaborative centrality: items most widely co-engaged across the "
            "OULAD learner population appear first. The offline evaluation results (NDCG@10 = 0.574) "
            "reflect the history-based mode for 43 eligible learners.",
            icon="\u2139\ufe0f",
        )

    # ── Top-N slider ──────────────────────────────────────────────────────────
    k = st.slider("How many results would you like?", min_value=3, max_value=12, value=6)

    # ── Load selected model ───────────────────────────────────────────────────
    if model_key in ("hybrid", "itemknn"):
        interactions = load_interactions()
        model = get_model_by_name(model_key, items, interactions)
    else:
        model = get_model_by_name(model_key, items)

    # ── Resolve final goal: free-text > dropdown > preset ────────────────────
    goal = goal_text.strip() or selected_title.strip() or preset_choice

    # ── Recommendations ───────────────────────────────────────────────────────
    if goal:
        with st.spinner("Finding the best courses for you..."):
            recs = model.recommend(goal, k=k)
            results = recs.merge(items, on="item_id", how="left")

        returned_item_ids = results["item_id"].tolist()

        st.markdown("### Top matches for you")
        for _, row in results.iterrows():
            score = float(row["score"])
            explanation = build_explanation(row["title"], similarity_score=score)
            item_id = row["item_id"]
            is_oulad = is_oulad_item(item_id)

            if is_oulad:
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div class="rec-card-header">
                            <h3>{row["title"]}</h3>
                            <span class="badge" style="background:#6366f1;color:white;padding:4px 8px;border-radius:4px;font-size:11px;">OULAD</span>
                        </div>
                        <p class="rec-card-description">{row["description"]}</p>
                        <div class="rec-card-footer">
                            <div class="rec-meta">\U0001f3db\ufe0f {row["institution"]}</div>
                            <div class="rec-meta rec-score">Match: {score:.2f}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("\U0001f4da View Course Details", expanded=False):
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"*{row['description']}*")
                    st.markdown("---")
                    st.markdown("**About this course:**")
                    st.info(
                        "This is an Open University module from the OULAD dataset. "
                        "OULAD is anonymized for privacy protection and does not include URLs to original content. "
                        "In a real deployment, this would link to the internal Learning Management System."
                    )
                    st.markdown("**Why this recommendation?**")
                    st.write(explanation)
            else:
                course_url = str(row.get("course_url", "#"))
                if course_url in ("nan", "", "#"):
                    course_url = "#"

                st.markdown(
                    f"""
                    <a href="{course_url}" target="_blank" class="rec-card-link">
                        <div class="rec-card rec-card--clickable">
                            <div class="rec-card-header">
                                <h3>{row["title"]}</h3>
                                <div class="rec-card-arrow">\u2192</div>
                            </div>
                            <p class="rec-card-description">{row["description"]}</p>
                            <div class="rec-card-footer">
                                <div class="rec-meta">\U0001f3db\ufe0f {row["institution"]}</div>
                                <div class="rec-meta rec-score">Match: {score:.2f}</div>
                            </div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("\U0001f4d6 Open & Track", key=f"open_{item_id}"):
                    log_click(goal, item_id, model_key, returned_item_ids)
                    if course_url and course_url != "#":
                        st.markdown(
                            f'<meta http-equiv="refresh" content="0;url={course_url}">',
                            unsafe_allow_html=True,
                        )

                with st.expander("Why this recommendation?"):
                    st.write(explanation)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("\U0001f44d Helpful", key=f"up_{row['item_id']}"):
                    log_feedback(
                        goal, row["item_id"], "up",
                        model_used=model_key,
                        returned_item_ids=returned_item_ids,
                    )
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("\U0001f44e Not for me", key=f"down_{row['item_id']}"):
                    log_feedback(
                        goal, row["item_id"], "down",
                        model_used=model_key,
                        returned_item_ids=returned_item_ids,
                    )
                    st.info("We'll improve our recommendations.")

    # ── Popular courses ───────────────────────────────────────────────────────
    st.markdown("### Popular courses")
    if "popular_courses" not in st.session_state:
        try:
            from src.models.popularity import PopularityRecommender
            train_interactions = load_interactions()
            if not train_interactions.empty:
                popularity = PopularityRecommender()
                popularity.fit(train_interactions)
                top_items = popularity.recommend(k=3)["item_id"].tolist()
                st.session_state["popular_courses"] = items[items["item_id"].isin(top_items)]
            else:
                st.session_state["popular_courses"] = items.head(3)
        except Exception:
            st.session_state["popular_courses"] = items.head(3)

    popular_df = st.session_state["popular_courses"]
    for col, (_, row) in zip(st.columns(3), popular_df.iterrows()):
        with col:
            st.markdown(
                f"""
                <div class="rec-card rec-card--enterprise">
                    <h3>{row["title"]}</h3>
                    <p>{row["description"]}</p>
                    <div class="rec-meta">Institution: {row["institution"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────

def render_login() -> None:
    st.markdown(_PAGE_OFFSET_CSS, unsafe_allow_html=True)
    st.header("Guest session")
    st.write("Explore the catalogue directly, or choose a display name for this browser session.")
    name = st.session_state.get("active_user")
    if name:
        st.write("Current display name:", name)
        if st.button("End guest session"):
            st.session_state.pop("active_user", None)
            st.rerun()
    else:
        with st.form("guest_session"):
            alias = st.text_input("Display name", max_chars=40, placeholder="Guest")
            if st.form_submit_button("Start guest session"):
                st.session_state["active_user"] = alias.strip() or "Guest"
                st.rerun()
    st.markdown("[Explore recommendations](?page=explore)")


def render_signup() -> None:
    render_login()
