import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ITEMS_PATH = os.path.join(BASE_DIR, "data", "processed", "items.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "interactions.csv")
OULAD_DIR = os.path.join(BASE_DIR, "data", "raw", "oulad")


def _build_oulad_interactions() -> pd.DataFrame:
    """
    Build interactions from OULAD studentVle.csv.
    Maps VLE activities to module-level items (since we now use modules as items).
    """
    student_vle_path = os.path.join(OULAD_DIR, "studentVle.csv")
    vle_path = os.path.join(OULAD_DIR, "vle.csv")
    
    if not os.path.exists(student_vle_path) or not os.path.exists(vle_path):
        return pd.DataFrame()

    # Load VLE mapping to get module codes for each site
    vle = pd.read_csv(vle_path)
    vle_mapping = vle[["id_site", "code_module", "code_presentation"]].copy()
    vle_mapping["id_site"] = vle_mapping["id_site"].astype(str)
    
    # Create module-level item_id (matches ingest.py format)
    vle_mapping["item_id"] = (
        "oulad_" + vle_mapping["code_module"].astype(str) + "_" + 
        vle_mapping["code_presentation"].astype(str)
    )
    site_to_item = dict(zip(vle_mapping["id_site"], vle_mapping["item_id"]))

    base_date = datetime(2013, 1, 1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    header_written = False
    for chunk in pd.read_csv(student_vle_path, chunksize=200_000):
        chunk = chunk.dropna(subset=["date", "id_student", "id_site"])
        chunk["id_site"] = chunk["id_site"].astype(str)
        
        # Map VLE site to module-level item_id
        chunk["item_id"] = chunk["id_site"].map(site_to_item)
        chunk = chunk.dropna(subset=["item_id"])  # Drop if site not in mapping
        
        chunk["timestamp"] = (
            base_date + pd.to_timedelta(chunk["date"].astype(int), unit="D")
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
        chunk = chunk.rename(columns={"id_student": "user_id"})
        chunk["user_id"] = chunk["user_id"].astype(str)

        out = chunk[["user_id", "item_id", "timestamp"]].copy()
        out["event_type"] = "click"
        out.to_csv(
            OUTPUT_PATH,
            mode="a",
            index=False,
            header=not header_written,
        )
        header_written = True

    if os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    return pd.DataFrame()


def build_interactions(seed: int = 42, data_mode: str = "oulad") -> pd.DataFrame:
    if data_mode not in ("oulad", "synthetic"):
        raise ValueError("data_mode must be oulad or synthetic")
    if data_mode == "oulad":
        for filename in ("courses.csv", "vle.csv", "studentVle.csv"):
            if not os.path.isfile(os.path.join(OULAD_DIR, filename)):
                raise FileNotFoundError(f"Missing OULAD input {filename} in {OULAD_DIR}. Use --data-mode synthetic for a fixture.")
        interactions = _build_oulad_interactions()
        if not interactions.empty:
            return interactions
        raise ValueError("No mapped OULAD activity rows were found.")

    items = pd.read_csv(ITEMS_PATH)

    # No real interaction data was provided, so we generate a small, deterministic
    # implicit dataset to make the pipeline fully runnable for evaluation/demo.
    rng = np.random.default_rng(seed)
    user_pool = [f"user_{i:04d}" for i in range(1, 251)]

    rows = []
    start_date = datetime(2024, 1, 1)

    for _, item in items.iterrows():
        # Base popularity proxy. Keeps counts reasonable and deterministic.
        base = max(5, min(30, int(rng.poisson(12))))
        for _ in range(base):
            user_id = rng.choice(user_pool)
            day_offset = int(rng.integers(0, 180))
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item["item_id"],
                    "timestamp": (start_date + timedelta(days=day_offset)).isoformat(),
                    "event_type": "view",
                }
            )

    interactions = pd.DataFrame(rows)
    interactions.to_csv(OUTPUT_PATH, index=False)
    return interactions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mode", choices=["oulad", "synthetic"], default="oulad")
    build_interactions(data_mode=parser.parse_args().data_mode)

