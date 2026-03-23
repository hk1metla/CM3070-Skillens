import os
import sys
from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st

from src.models.tfidf import TfidfRecommender


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "items.csv")
DEPLOY_ITEMS_PATH = os.path.join(BASE_DIR, "data", "deploy", "items.csv")

# FIX (Issue 1): Path changed from results/logs/feedback.csv to data/feedback/feedback.csv
# to match report §3.6: "Feedback is logged to a CSV file under data/feedback/
# to keep evaluation runs separate from interactive usage."
LOG_PATH = os.path.join(BASE_DIR, "data", "feedback", "feedback.csv")

MODEL_PATH = os.path.join(BASE_DIR, "results", "logs", "selected_model.txt")


def is_oulad_item(item_id: str) -> bool:
    """Return True if the item is from the OULAD dataset (has 'oulad_' prefix)."""
    return str(item_id).startswith("oulad_")


def _resolve_items_csv_path() -> str:
    """Prefer local pipeline output; fall back to bundled demo data."""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    if os.path.exists(DEPLOY_ITEMS_PATH):
        return DEPLOY_ITEMS_PATH
    raise FileNotFoundError(
        f"items.csv not found. Expected {DATA_PATH} (after --prepare-data) "
        f"or {DEPLOY_ITEMS_PATH}."
    )


def load_items() -> pd.DataFrame:
    """Load items CSV with caching to avoid repeated disk reads."""
    return pd.read_csv(_resolve_items_csv_path())


@st.cache_resource(show_spinner=False)
def load_tfidf_model(items: pd.DataFrame) -> TfidfRecommender:
    model = TfidfRecommender()
    model.fit(items)
    return model


@st.cache_resource(show_spinner=False)
def load_hybrid_model(items: pd.DataFrame, interactions: pd.DataFrame):
    """Load hybrid TF-IDF + ItemKNN model (w_content=0.6, w_cf=0.4)."""
    from src.models.hybrid import HybridRecommender
    model = HybridRecommender(w_content=0.6, w_cf=0.4)
    model.fit(items, interactions)
    return model


@st.cache_resource(show_spinner=False)
def load_itemknn_model(items: pd.DataFrame, interactions: pd.DataFrame):
    """Load ItemKNN collaborative filtering model."""
    from src.models.itemknn import ItemKNNRecommender
    model = ItemKNNRecommender()
    model.fit(interactions)
    return model


@st.cache_resource(show_spinner=False)
def load_semantic_model(items: pd.DataFrame):
    """Load Sentence-BERT semantic embedding model."""
    from src.models.semantic import SemanticRecommender
    model = SemanticRecommender()
    model.fit(items)
    return model


@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    """Load interaction data (train split) with caching."""
    for path in (
        os.path.join(BASE_DIR, "data", "processed", "train.csv"),
        os.path.join(BASE_DIR, "data", "deploy", "train.csv"),
    ):
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def get_model_by_name(
    model_name: str,
    items: pd.DataFrame,
    interactions: Optional[pd.DataFrame] = None,
):
    """
    Route model_name to the appropriate cached loader.

    Supported values:
        "tfidf"    — TF-IDF content-based recommender
        "hybrid"   — Hybrid (TF-IDF + ItemKNN, w_content=0.6)
        "itemknn"  — ItemKNN collaborative filtering
        "semantic" — Sentence-BERT semantic recommender
    """
    if model_name == "tfidf":
        return load_tfidf_model(items)

    elif model_name == "hybrid":
        if interactions is None:
            interactions = load_interactions()
        return load_hybrid_model(items, interactions)

    elif model_name == "itemknn":
        if interactions is None:
            interactions = load_interactions()
        return load_itemknn_model(items, interactions)

    elif model_name == "semantic":
        return load_semantic_model(items)

    else:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Valid options: 'tfidf', 'hybrid', 'itemknn', 'semantic'."
        )


def log_feedback(
    goal: str,
    item_id: str,
    feedback: str,
    model_used: str = "tfidf",
    returned_item_ids: Optional[List[str]] = None,
    event_type: str = "feedback",
    time_spent: Optional[float] = None,
) -> None:
    """Log user feedback to data/feedback/feedback.csv (report §3.6)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "goal_text": goal,
                "item_id": item_id,
                "feedback": feedback,
                "model_used": model_used,
                "returned_item_ids": (
                    ",".join(returned_item_ids) if returned_item_ids else ""
                ),
                "event_type": event_type,
                "time_spent": time_spent if time_spent is not None else "",
            }
        ]
    )
    if os.path.exists(LOG_PATH):
        record.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        record.to_csv(LOG_PATH, index=False)


def log_click(
    goal: str,
    item_id: str,
    model_used: str,
    returned_item_ids: List[str],
) -> None:
    """Log when user clicks on a recommendation."""
    log_feedback(
        goal=goal,
        item_id=item_id,
        feedback="click",
        model_used=model_used,
        returned_item_ids=returned_item_ids,
        event_type="click",
    )


def log_completion(
    goal: str,
    item_id: str,
    model_used: str,
    time_spent: float,
) -> None:
    """Log when user completes an item."""
    log_feedback(
        goal=goal,
        item_id=item_id,
        feedback="completion",
        model_used=model_used,
        returned_item_ids=None,
        event_type="completion",
        time_spent=time_spent,
    )


def get_active_user() -> Optional[str]:
    return st.session_state.get("active_user")


def set_active_user(email: str) -> None:
    st.session_state["active_user"] = email


def clear_active_user() -> None:
    st.session_state.pop("active_user", None)


def get_selected_model(default: str = "tfidf") -> str:
    """Get the globally selected model, persisted across pages."""
    if "selected_model" in st.session_state:
        return st.session_state["selected_model"]
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as fh:
                value = fh.read().strip()
            if value:
                st.session_state["selected_model"] = value
                return value
        except OSError:
            pass
    st.session_state["selected_model"] = default
    return default


def set_selected_model(model_key: str) -> None:
    """Persist the globally selected model to disk and session state."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        with open(MODEL_PATH, "w", encoding="utf-8") as fh:
            fh.write(model_key)
    except OSError:
        pass
    st.session_state["selected_model"] = model_key
