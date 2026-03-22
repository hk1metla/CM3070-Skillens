"""
Generate cold-start performance plot from canonical CSV data.
Shows NDCG@10 by user history length — must match Table~\\ref{tab:coldstart} (history_truncation.csv).

Usage:
  python generate_coldstart_plot.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "final", "history_truncation.csv")
OUT_DIR = os.path.join(BASE_DIR, "results", "final", "plots")
# Primary output; copy to 28.png for reports that include figures by number
OUTPUT_PRIMARY = os.path.join(OUT_DIR, "coldstart_performance.png")
OUTPUT_REPORT = os.path.join(OUT_DIR, "28.png")


def generate_coldstart_plot():
    """Generate line chart for cold-start performance from canonical CSV."""
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df["n_users"] > 0]

    hybrid_df = df[df["model"] == "hybrid"].sort_values("bin")
    tfidf_df = df[df["model"] == "tfidf"].sort_values("bin")

    history_labels = hybrid_df["bin"].values
    hybrid_ndcg = hybrid_df["ndcg"].values.astype(float)
    content_ndcg = tfidf_df["ndcg"].values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(history_labels))

    # Table uses 3 d.p.; hybrid and content-only are identical in this pipeline
    identical = np.allclose(hybrid_ndcg, content_ndcg, rtol=1e-4)
    if identical:
        ax.plot(
            x,
            hybrid_ndcg,
            marker="o",
            linewidth=2.5,
            markersize=11,
            label="Hybrid / Content-Only (identical)",
            color="#2c3e50",
        )
        for i, v in enumerate(hybrid_ndcg):
            ax.text(
                i,
                v + 0.006,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#2c3e50",
            )
    else:
        ax.plot(
            x,
            hybrid_ndcg,
            marker="o",
            linewidth=2,
            markersize=10,
            label="Hybrid",
            color="#2ecc71",
        )
        ax.plot(
            x,
            content_ndcg,
            marker="s",
            linewidth=2,
            markersize=10,
            label="Content-Only",
            color="#3498db",
        )
        for i, (hyb, cont) in enumerate(zip(hybrid_ndcg, content_ndcg)):
            ax.text(
                i,
                hyb + 0.01,
                f"{hyb:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#2ecc71",
            )
            offset = -0.015 if hyb != cont else 0.01
            va = "top" if hyb != cont else "bottom"
            ax.text(
                i,
                cont + offset,
                f"{cont:.3f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
                color="#3498db",
            )

    ax.set_xlabel("User interaction history length", fontsize=12, fontweight="bold")
    ax.set_ylabel("NDCG@10", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cold-start performance: NDCG@10 by user history length",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(history_labels)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")

    # Keep scale consistent with tabulated values (~0.35–0.50), not a compressed band
    y_min = max(0.0, float(np.min(np.concatenate([hybrid_ndcg, content_ndcg]))) - 0.03)
    y_max = float(np.max(np.concatenate([hybrid_ndcg, content_ndcg]))) + 0.03
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PRIMARY, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_REPORT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved cold-start plot to {OUTPUT_PRIMARY}")
    print(f"✓ Saved report copy to {OUTPUT_REPORT}")


if __name__ == "__main__":
    generate_coldstart_plot()
