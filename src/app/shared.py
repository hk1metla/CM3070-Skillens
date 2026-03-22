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
# Bundled subset for Streamlit Cloud / clones without running --prepare-data (see data/deploy/README.md)
DEPLOY_ITEMS_PATH = os.path.join(BASE_DIR, "data", "deploy", "items.csv")
LOG_PATH = os.path.join(BASE_DIR, "results", "logs", "feedback.csv")
MODEL_PATH = os.path.join(BASE_DIR, "results", "logs", "selected_model.txt")


def is_oulad_item(item_id: str) -> bool:
    """
    Check if an item is from OULAD dataset.
    
    Per dual-dataset strategy: OULAD items don't have external URLs
    and should show internal detail views instead of external links.
    
    Args:
        item_id: Item identifier (should start with "oulad_" for OULAD items)
        
    Returns:
        True if item is from OULAD dataset, False otherwise
    """
    return str(item_id).startswith("oulad_")


def _resolve_items_csv_path() -> str:
    """Prefer local pipeline output; fall back to repo-bundled demo data (e.g. Streamlit Cloud)."""
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    if os.path.exists(DEPLOY_ITEMS_PATH):
        return DEPLOY_ITEMS_PATH
    raise FileNotFoundError(
        f"items.csv not found. Expected {DATA_PATH} (after --prepare-data) or {DEPLOY_ITEMS_PATH}."
    )


def load_items() -> pd.DataFrame:
    """Load items with caching to avoid repeated disk reads."""
    return pd.read_csv(_resolve_items_csv_path())


@st.cache_resource(show_spinner=False)
def load_tfidf_model(items: pd.DataFrame) -> TfidfRecommender:
    model = TfidfRecommender()
    model.fit(items)
    return model


@st.cache_resource(show_spinner=False)
def load_hybrid_model(items: pd.DataFrame, interactions: pd.DataFrame):
    """Load hybrid model combining content and collaborative filtering."""
    from src.models.hybrid import HybridRecommender
    
    model = HybridRecommender(w_content=0.6, w_cf=0.4)
    model.fit(items, interactions)
    return model


@st.cache_resource(show_spinner=False)
def load_semantic_model(items: pd.DataFrame):
    """Load semantic embedding model using Sentence-BERT."""
    from src.models.semantic import SemanticRecommender
    
    model = SemanticRecommender()
    model.fit(items)
    return model


@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    """Load interaction data for hybrid model with caching."""
    for interactions_path in (
        os.path.join(BASE_DIR, "data", "processed", "train.csv"),
        os.path.join(BASE_DIR, "data", "deploy", "train.csv"),
    ):
        if os.path.exists(interactions_path):
            return pd.read_csv(interactions_path)
    return pd.DataFrame()


def get_model_by_name(model_name: str, items: pd.DataFrame, interactions: Optional[pd.DataFrame] = None):
    """
    Get model by name.
    
    Models are cached via @st.cache_resource decorators on individual load functions.
    This function just routes to the appropriate loader.
    
    Args:
        model_name: Name of the model ("tfidf", "hybrid", "semantic")
        items: Items DataFrame (cached via load_items())
        interactions: Interactions DataFrame (cached via load_interactions(), required for hybrid)
        
    Returns:
        Model instance (cached)
    """
    if model_name == "tfidf":
        return load_tfidf_model(items)
    elif model_name == "hybrid":
        if interactions is None:
            interactions = load_interactions()
        return load_hybrid_model(items, interactions)
    elif model_name == "semantic":
        return load_semantic_model(items)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def log_feedback(
    goal: str,
    item_id: str,
    feedback: str,
    model_used: str = "tfidf",
    returned_item_ids: Optional[List[str]] = None,
    event_type: str = "feedback",
    time_spent: Optional[float] = None,
) -> None:
    """
    Log user feedback with complete context for evaluation.
    
    Args:
        goal: User's goal text
        item_id: Item ID that received feedback
        feedback: "up", "down", "click", or "completion"
        model_used: Which model generated the recommendation
        returned_item_ids: All K items that were shown to the user
        event_type: Type of event ("feedback", "click", "completion")
        time_spent: Time spent on item in seconds (optional)
    """
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


def log_click(goal: str, item_id: str, model_used: str, returned_item_ids: List[str]) -> None:
    """Log when user clicks on a recommendation."""
    log_feedback(
        goal=goal,
        item_id=item_id,
        feedback="click",
        model_used=model_used,
        returned_item_ids=returned_item_ids,
        event_type="click",
    )


def log_completion(goal: str, item_id: str, model_used: str, time_spent: float) -> None:
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
    """Get the globally selected model, persisted across pages/sessions."""
    if "selected_model" in st.session_state:
        return st.session_state["selected_model"]

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                st.session_state["selected_model"] = value
                return value
        except OSError:
            pass

    st.session_state["selected_model"] = default
    return default


def set_selected_model(model_key: str) -> None:
    """Persist and set the globally selected model."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    try:
        with open(MODEL_PATH, "w", encoding="utf-8") as handle:
            handle.write(model_key)
    except OSError:
        pass
    st.session_state["selected_model"] = model_key

