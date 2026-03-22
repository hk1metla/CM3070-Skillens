"""
Control Room / Analytics Dashboard.

Hidden admin page for viewing results, metrics, and analysis.
Accessible via ?page=control (not shown in navbar).
"""

import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_FINAL_DIR = os.path.join(BASE_DIR, "results", "final")
DEPLOY_DIR = os.path.join(BASE_DIR, "data", "deploy")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_PATH = os.path.join(RESULTS_DIR, "logs", "feedback.csv")


def _resolve_results_csv(filename: str) -> Optional[str]:
    """Pipeline writes to results/final/; legacy path was results/; Cloud bundles data/deploy/."""
    for directory in (RESULTS_FINAL_DIR, RESULTS_DIR, DEPLOY_DIR):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return None


def load_evaluation_results() -> pd.DataFrame:
    """Load comprehensive evaluation results."""
    results_path = _resolve_results_csv("comprehensive_metrics.csv")
    if results_path:
        return pd.read_csv(results_path)
    return pd.DataFrame()


def load_feedback_logs() -> pd.DataFrame:
    """Load user feedback logs."""
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def load_ablation_results() -> pd.DataFrame:
    """Load ablation study results."""
    ablation_path = _resolve_results_csv("ablation_study.csv")
    if ablation_path:
        return pd.read_csv(ablation_path)
    return pd.DataFrame()


def render_control_nav(active_tab: str = "performance") -> None:
    """Render navigation bar for control room matching main site design."""
    from src.app.ui import inject_css
    
    inject_css()  # Use same CSS as main site
    
    # Map tab names
    tab_map = {
        "performance": "Performance",
        "metrics": "Metrics", 
        "feedback": "Feedback",
        "ablation": "Ablation",
        "status": "Status",
    }
    
    # Build navigation links
    nav_items = []
    for tab_key, tab_label in tab_map.items():
        active_class = "active" if active_tab == tab_key else ""
        nav_items.append(
            f'<a href="?page=control&tab={tab_key}" class="{active_class}" target="_self">{tab_label}</a>'
        )
    
    st.markdown(
        f"""
        <div class="nav-shell">
            <div class="nav-logo">Control Room</div>
            <div class="nav-links">
                {''.join(nav_items)}
            </div>
            <a href="?page=home" class="nav-login" target="_self">Back to Site</a>
        </div>
        <div style="height: 80px;"></div>
        """,
        unsafe_allow_html=True,
    )


def render_control_room() -> None:
    """Render the control room dashboard."""
    
    # Get active tab from query params (with error handling for browser extensions)
    try:
        active_tab = st.query_params.get("tab", "performance")
        if isinstance(active_tab, list):
            active_tab = active_tab[0] if active_tab else "performance"
    except (AttributeError, TypeError, KeyError):
        active_tab = "performance"
    if isinstance(active_tab, list):
        active_tab = active_tab[0] if active_tab else "performance"
    
    # Render navigation bar (matching main site design)
    render_control_nav(active_tab)
    
    st.markdown(
        """
        <style>
        .control-room-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4ade80;
        }
        .model-control-panel {
            background: rgba(99, 102, 241, 0.1);
            padding: 24px;
            border-radius: 12px;
            border: 2px solid rgba(99, 102, 241, 0.3);
            margin: 20px 0;
        }
        .model-control-panel h2 {
            margin-top: 0;
            color: #818cf8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header
    st.markdown(
        """
        <div class="control-room-header">
            <h1>Skillens Control Room</h1>
            <p>Analytics Dashboard & Model Performance Monitor</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Route to appropriate section based on query params
    if active_tab == "metrics":
        render_evaluation_metrics()
    elif active_tab == "feedback":
        render_user_feedback()
    elif active_tab == "ablation":
        render_ablation_study()
    elif active_tab == "status":
        render_system_status()
    else:  # default to performance
        render_model_performance()


def render_model_performance() -> None:
    """Render model performance comparison."""
    st.header("Model Performance Comparison")
    
    results = load_evaluation_results()
    
    if results.empty:
        st.warning("No evaluation results found.")
        st.markdown(
            "The Control Room reads **`comprehensive_metrics.csv`** from `results/final/`, `results/`, or **`data/deploy/`** (bundled for Streamlit Cloud)."
        )
        st.markdown("**Locally**, run the full pipeline:")
        st.code(
            "python -m src.eval.pipeline --config configs/experiment.yaml --out results/final",
            language="bash",
        )
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_ndcg = results.loc[results["ndcg_mean"].idxmax()]
        st.metric(
            "Best NDCG",
            f"{best_ndcg['ndcg_mean']:.4f}",
            f"Model: {best_ndcg['model']}"
        )
    
    with col2:
        best_precision = results.loc[results["precision_mean"].idxmax()]
        st.metric(
            "Best Precision",
            f"{best_precision['precision_mean']:.4f}",
            f"Model: {best_precision['model']}"
        )
    
    with col3:
        best_diversity = results.loc[results["diversity_mean"].idxmax()]
        st.metric(
            "Best Diversity",
            f"{best_diversity['diversity_mean']:.4f}",
            f"Model: {best_diversity['model']}"
        )
    
    with col4:
        best_coverage = results.loc[results["catalog_coverage"].idxmax()]
        st.metric(
            "Best Coverage",
            f"{best_coverage['catalog_coverage']:.4f}",
            f"Model: {best_coverage['model']}"
        )
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # NDCG comparison
        fig_ndcg = px.bar(
            results,
            x="model",
            y="ndcg_mean",
            title="NDCG@K Comparison",
            color="ndcg_mean",
            color_continuous_scale="Viridis",
        )
        fig_ndcg.update_layout(showlegend=False)
        st.plotly_chart(fig_ndcg, use_container_width=True)
    
    with col2:
        # Precision comparison
        fig_precision = px.bar(
            results,
            x="model",
            y="precision_mean",
            title="Precision@K Comparison",
            color="precision_mean",
            color_continuous_scale="Plasma",
        )
        fig_precision.update_layout(showlegend=False)
        st.plotly_chart(fig_precision, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Metrics Table")
    display_cols = [
        "model",
        "precision_mean",
        "recall_mean",
        "ndcg_mean",
        "diversity_mean",
        "catalog_coverage",
        "gini_coefficient",
    ]
    available_cols = [col for col in display_cols if col in results.columns]
    st.dataframe(results[available_cols], use_container_width=True)


def render_evaluation_metrics() -> None:
    """Render detailed evaluation metrics."""
    st.header("Evaluation Metrics Analysis")
    
    results = load_evaluation_results()
    
    if results.empty:
        st.warning("No evaluation results found.")
        st.caption(
            "Add `data/deploy/comprehensive_metrics.csv` to the repo (from local `results/final/`) or run the pipeline locally."
        )
        return
    
    # Diversity vs Accuracy trade-off
    st.subheader("Diversity vs Accuracy Trade-off")
    
    fig_tradeoff = px.scatter(
        results,
        x="ndcg_mean",
        y="diversity_mean",
        size="catalog_coverage",
        color="model",
        hover_data=["precision_mean", "recall_mean"],
        title="Diversity vs Accuracy Trade-off",
        labels={
            "ndcg_mean": "Accuracy (NDCG)",
            "diversity_mean": "Diversity",
            "catalog_coverage": "Coverage",
        },
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)
    
    # Coverage analysis
    st.subheader("Catalog Coverage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_coverage = px.bar(
            results,
            x="model",
            y="catalog_coverage",
            title="Catalog Coverage by Model",
            color="catalog_coverage",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_coverage, use_container_width=True)
    
    with col2:
        fig_gini = px.bar(
            results,
            x="model",
            y="gini_coefficient",
            title="Inequality (Gini Coefficient) - Lower is Better",
            color="gini_coefficient",
            color_continuous_scale="Reds_r",
        )
        st.plotly_chart(fig_gini, use_container_width=True)
    
    # Metric distributions
    st.subheader("Metric Distributions")
    
    metric_choice = st.selectbox(
        "Select metric to view distribution",
        ["ndcg_mean", "precision_mean", "recall_mean", "diversity_mean"],
    )
    
    fig_dist = px.histogram(
        results,
        x=metric_choice,
        nbins=20,
        title=f"Distribution of {metric_choice.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig_dist, use_container_width=True)


def render_user_feedback() -> None:
    """Render user feedback analytics."""
    st.header("User Feedback Analytics")
    
    feedback_df = load_feedback_logs()
    
    if feedback_df.empty:
        st.info("No feedback data yet. Feedback will appear here as users interact with recommendations.")
        return
    
    # Convert timestamp if needed
    if "timestamp" in feedback_df.columns:
        feedback_df["timestamp"] = pd.to_datetime(feedback_df["timestamp"], errors="coerce")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_feedback = len(feedback_df)
        st.metric("Total Feedback", total_feedback)
    
    with col2:
        if "feedback" in feedback_df.columns:
            positive = len(feedback_df[feedback_df["feedback"] == "up"])
            st.metric("Positive", positive, f"{positive/total_feedback*100:.1f}%")
    
    with col3:
        if "feedback" in feedback_df.columns:
            negative = len(feedback_df[feedback_df["feedback"] == "down"])
            st.metric("Negative", negative, f"{negative/total_feedback*100:.1f}%")
    
    with col4:
        if "event_type" in feedback_df.columns:
            clicks = len(feedback_df[feedback_df["event_type"] == "click"])
            st.metric("Clicks", clicks)
    
    # Feedback over time
    if "timestamp" in feedback_df.columns and not feedback_df["timestamp"].isna().all():
        st.subheader("Feedback Over Time")
        
        feedback_df["date"] = feedback_df["timestamp"].dt.date
        daily_feedback = feedback_df.groupby("date").size().reset_index(name="count")
        
        fig_timeline = px.line(
            daily_feedback,
            x="date",
            y="count",
            title="Daily Feedback Volume",
            markers=True,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Feedback by model
    if "model_used" in feedback_df.columns:
        st.subheader("Feedback by Model")
        
        model_feedback = feedback_df.groupby("model_used").agg({
            "feedback": lambda x: (x == "up").sum() if "feedback" in feedback_df.columns else 0,
        }).reset_index()
        
        fig_model = px.bar(
            model_feedback,
            x="model_used",
            y="feedback",
            title="Positive Feedback by Model",
        )
        st.plotly_chart(fig_model, use_container_width=True)
    
    # Event type breakdown
    if "event_type" in feedback_df.columns:
        st.subheader("Event Type Breakdown")
        
        event_counts = feedback_df["event_type"].value_counts().reset_index()
        event_counts.columns = ["event_type", "count"]
        
        fig_events = px.pie(
            event_counts,
            values="count",
            names="event_type",
            title="Event Type Distribution",
        )
        st.plotly_chart(fig_events, use_container_width=True)
    
    # Recent feedback table
    st.subheader("Recent Feedback")
    display_cols = ["timestamp", "goal_text", "item_id", "feedback", "model_used", "event_type"]
    available_cols = [col for col in display_cols if col in feedback_df.columns]
    
    if available_cols:
        recent_feedback = feedback_df[available_cols].tail(50).sort_values(
            "timestamp" if "timestamp" in available_cols else available_cols[0],
            ascending=False,
        )
        st.dataframe(recent_feedback, use_container_width=True)


def render_ablation_study() -> None:
    """Render ablation study results."""
    st.header("Ablation Study: Component Contributions")
    
    ablation_df = load_ablation_results()
    
    if ablation_df.empty:
        st.warning("No ablation study results found.")
        st.markdown("Expected file: **`ablation_study.csv`** in `results/final/`, `results/`, or **`data/deploy/`**.")
        st.markdown("**Locally**, run:")
        st.code(
            "python -m src.eval.pipeline --config configs/experiment.yaml --out results/final",
            language="bash",
        )
        return
    
    # Improvement visualization
    st.subheader("Component Contribution Analysis")
    
    if "ndcg_improvement" in ablation_df.columns:
        fig_improvement = px.bar(
            ablation_df,
            x="configuration",
            y="ndcg_improvement",
            title="Improvement Over Baseline (%)",
            color="ndcg_improvement",
            color_continuous_scale="Greens",
        )
        fig_improvement.update_xaxes(tickangle=45)
        st.plotly_chart(fig_improvement, use_container_width=True)
    
    # NDCG comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ndcg = px.bar(
            ablation_df,
            x="configuration",
            y="ndcg",
            title="NDCG by Configuration",
            color="ndcg",
            color_continuous_scale="Blues",
        )
        fig_ndcg.update_xaxes(tickangle=45)
        st.plotly_chart(fig_ndcg, use_container_width=True)
    
    with col2:
        if "precision" in ablation_df.columns and "recall" in ablation_df.columns:
            fig_pr = go.Figure()
            fig_pr.add_trace(
                go.Bar(name="Precision", x=ablation_df["configuration"], y=ablation_df["precision"])
            )
            fig_pr.add_trace(
                go.Bar(name="Recall", x=ablation_df["configuration"], y=ablation_df["recall"])
            )
            fig_pr.update_layout(
                title="Precision & Recall by Configuration",
                xaxis_tickangle=45,
                barmode="group",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    
    # Detailed table
    st.subheader("Ablation Study Results")
    st.dataframe(ablation_df, use_container_width=True)


def render_system_status() -> None:
    """Render system status and health checks."""
    st.header("System Status & Model Control")
    
    # Data statistics (First)
    st.subheader("Data Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        items_path = os.path.join(DATA_DIR, "items.csv")
        if os.path.exists(items_path):
            items_df = pd.read_csv(items_path)
            st.metric("Total Items", len(items_df))
    
    with col2:
        interactions_path = os.path.join(DATA_DIR, "interactions.csv")
        if os.path.exists(interactions_path):
            interactions_df = pd.read_csv(interactions_path)
            st.metric("Total Interactions", len(interactions_df))
            if "user_id" in interactions_df.columns:
                st.metric("Unique Users", interactions_df["user_id"].nunique())
    
    with col3:
        if os.path.exists(LOG_PATH):
            feedback_df = pd.read_csv(LOG_PATH)
            st.metric("Total Feedback", len(feedback_df))
    
    # Model Control (Second) - More prominent section
    st.markdown("---")
    st.markdown(
        """
        <div style="background: rgba(99, 102, 241, 0.15); padding: 28px; border-radius: 16px; border: 2px solid rgba(99, 102, 241, 0.4); margin: 24px 0; box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);">
            <h2 style="margin-top: 0; margin-bottom: 8px; color: #818cf8; font-size: 24px;">Model Control</h2>
            <p style="color: #cbd5e1; margin-bottom: 0; font-size: 14px;">Change the model used in the UI for all users</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    model_options = {
        "TF-IDF (Fast & Accurate)": "tfidf",
        "Hybrid (Best Quality)": "hybrid",
        "Semantic (Best Matching)": "semantic",
    }
    
    from src.app.shared import get_selected_model, set_selected_model

    current_model = get_selected_model()
    current_model_name = [k for k, v in model_options.items() if v == current_model][0] if current_model in model_options.values() else "TF-IDF (Fast & Accurate)"
    
    # Better layout with proper spacing and alignment
    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_model_name = st.selectbox(
            "Select model for UI",
            options=list(model_options.keys()),
            index=list(model_options.values()).index(current_model) if current_model in model_options.values() else 0,
            help="This changes the model used in the Explore page for all users.",
        )
    
    with col2:
        # Align button properly with selectbox (no extra margin needed)
        if st.button("Apply Model", type="primary", use_container_width=True):
            new_model_key = model_options[selected_model_name]
            set_selected_model(new_model_key)
            st.success(f"Model changed to {selected_model_name}! Users will now see recommendations from this model.")
            st.rerun()
    
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    st.info(f"**Current Model**: {current_model_name} ({current_model})")
    
    # File existence checks (Third)
    st.subheader("Data & Results Files")
    
    _cm = _resolve_results_csv("comprehensive_metrics.csv")
    _ab = _resolve_results_csv("ablation_study.csv")
    files_to_check = {
        "Items Data": os.path.join(DATA_DIR, "items.csv"),
        "Interactions": os.path.join(DATA_DIR, "interactions.csv"),
        "Train Split": os.path.join(DATA_DIR, "train.csv"),
        "Test Split": os.path.join(DATA_DIR, "test.csv"),
        "Evaluation Results": _cm or os.path.join(RESULTS_FINAL_DIR, "comprehensive_metrics.csv"),
        "Ablation Study": _ab or os.path.join(RESULTS_FINAL_DIR, "ablation_study.csv"),
        "Feedback Logs": LOG_PATH,
    }
    
    status_df = []
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status_df.append({
            "File": name,
            "Status": "Exists" if exists else "Missing",
            "Size (KB)": f"{size / 1024:.2f}" if exists else "0",
            "Path": path,
        })
    
    st.dataframe(pd.DataFrame(status_df), use_container_width=True)
