"""
ItemKNN Collaborative Filtering Recommender.

Uses item-item similarity based on user interaction patterns.
This is a key component for the hybrid recommender system.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNNRecommender:
    """
    Item-based Collaborative Filtering using K-Nearest Neighbors.

    For each item, finds similar items based on user interaction patterns,
    then recommends items similar to those the user has interacted with.

    recommend() supports two call patterns so the same class works in both
    the Streamlit UI and the offline evaluation pipeline:

      Pattern 1 — eval pipeline / HybridRecommender (original, unchanged):
            model.recommend(user_id, interactions_df, k=10, exclude_items=None)
            Returns personalised recs from the user's interaction history.

      Pattern 2 — Streamlit UI (text-goal, matches TF-IDF / Hybrid interface):
            model.recommend(goal_text, k=10)
            Returns items ranked by collaborative centrality (cold-start).
    """

    def __init__(self, k: int = 50, min_interactions: int = 2):
        self.k = k
        self.min_interactions = min_interactions
        self.item_similarity_matrix = None
        self.item_ids = None
        self.user_item_matrix = None
        self.item_to_idx = None
        self.idx_to_item = None

    # ── FIT ──────────────────────────────────────────────────────────────────

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Fit the ItemKNN model on interaction data.

        Args:
            interactions: DataFrame with columns ['user_id', 'item_id', 'timestamp']
        """
        users = interactions["user_id"].unique()
        items = interactions["item_id"].unique()

        self.item_ids = sorted(items)
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        user_to_idx = {user: idx for idx, user in enumerate(users)}

        # Aggregate interactions and apply log-transform (Hu et al., 2008)
        counts = (
            interactions
            .groupby(["user_id", "item_id"])
            .size()
            .reset_index(name="click_count")
        )

        rows, cols, data = [], [], []
        for _, row in counts.iterrows():
            rows.append(user_to_idx[row["user_id"]])
            cols.append(self.item_to_idx[row["item_id"]])
            data.append(float(np.log1p(row["click_count"])))

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(self.item_ids))
        )

        item_user_matrix = self.user_item_matrix.T
        item_counts = np.array(item_user_matrix.sum(axis=1)).flatten()

        if (item_counts >= self.min_interactions).sum() == 0:
            raise ValueError("No items meet minimum interaction threshold")

        # Compute item-item cosine similarity; zero diagonal (no self-similarity)
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        np.fill_diagonal(self.item_similarity_matrix, 0)

    # ── UNIFIED recommend() — BACKWARDS COMPATIBLE ───────────────────────────

    def recommend(
        self,
        user_id_or_query,
        interactions: Optional[pd.DataFrame] = None,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate recommendations.

        Pattern 1 (eval pipeline / HybridRecommender):
            model.recommend(user_id, interactions_df, k=10, exclude_items=None)

        Pattern 2 (Streamlit UI text-goal):
            model.recommend(goal_text, k=10)
        """
        if self.item_similarity_matrix is None:
            raise ValueError("Model is not fitted. Call fit() before recommend().")

        if isinstance(interactions, pd.DataFrame):
            # Pattern 1 — personalised, history-based (eval pipeline)
            return self._recommend_for_user(
                user_id=user_id_or_query,
                interactions=interactions,
                k=k,
                exclude_items=exclude_items,
            )
        else:
            # Pattern 2 — UI text-goal, cold-start centrality ranking
            effective_k = interactions if isinstance(interactions, int) else k
            return self._recommend_by_centrality(k=effective_k)

    # ── PRIVATE: centrality ranking for UI cold-start ─────────────────────────

    def _recommend_by_centrality(self, k: int = 10) -> pd.DataFrame:
        """
        Rank items by their aggregate cosine similarity to all other items.
        Items widely co-engaged across many users rank first.
        Used when no user interaction history is available (UI cold-start).
        """
        centrality = self.item_similarity_matrix.sum(axis=1)
        max_score = centrality.max()
        if max_score > 0:
            centrality = centrality / max_score

        ranked = sorted(
            (
                (self.idx_to_item[idx], float(score))
                for idx, score in enumerate(centrality)
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return pd.DataFrame(
            {
                "item_id": [item for item, _ in ranked],
                "score": [score for _, score in ranked],
            }
        )

    # ── PRIVATE: user-history ranking (eval pipeline / HybridRecommender) ────

    def _recommend_for_user(
        self,
        user_id: str,
        interactions: pd.DataFrame,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Personalised recommendations from a user's interaction history.
        Called by the offline evaluation pipeline and HybridRecommender.
        Behaviour is identical to the original implementation.
        """
        user_items = set(interactions[interactions["user_id"] == user_id]["item_id"])

        if not user_items:
            return pd.DataFrame({"item_id": [], "score": []})

        user_item_indices = [
            self.item_to_idx[item] for item in user_items if item in self.item_to_idx
        ]

        if not user_item_indices:
            return pd.DataFrame({"item_id": [], "score": []})

        item_scores = np.zeros(len(self.item_ids))
        for idx in user_item_indices:
            item_scores += self.item_similarity_matrix[idx, :]
        item_scores /= len(user_item_indices)

        item_scores_dict = {
            self.idx_to_item[idx]: float(score)
            for idx, score in enumerate(item_scores)
        }

        for item in user_items:
            item_scores_dict.pop(item, None)

        if exclude_items:
            for item in exclude_items:
                item_scores_dict.pop(item, None)

        sorted_items = sorted(
            item_scores_dict.items(), key=lambda x: x[1], reverse=True
        )[:k]

        return pd.DataFrame(
            {
                "item_id": [item for item, _ in sorted_items],
                "score": [score for _, score in sorted_items],
            }
        )

    # ── recommend_for_new_user (unchanged, kept for compatibility) ────────────

    def recommend_for_new_user(
        self, goal_items: List[str], k: int = 10
    ) -> pd.DataFrame:
        """
        Recommend for a new user based on seed item IDs (cold-start).

        Args:
            goal_items: Item IDs matching the user's stated goal.
            k:          Number of recommendations to return.
        """
        if self.item_similarity_matrix is None:
            raise ValueError("Model is not fitted.")

        if not goal_items:
            return pd.DataFrame({"item_id": [], "score": []})

        goal_indices = [
            self.item_to_idx[item]
            for item in goal_items
            if item in self.item_to_idx
        ]

        if not goal_indices:
            return pd.DataFrame({"item_id": [], "score": []})

        item_scores = np.zeros(len(self.item_ids))
        for idx in goal_indices:
            item_scores += self.item_similarity_matrix[idx, :]
        item_scores /= len(goal_indices)

        item_scores_dict = {
            self.idx_to_item[idx]: float(score)
            for idx, score in enumerate(item_scores)
        }

        for item in goal_items:
            item_scores_dict.pop(item, None)

        sorted_items = sorted(
            item_scores_dict.items(), key=lambda x: x[1], reverse=True
        )[:k]

        return pd.DataFrame(
            {
                "item_id": [item for item, _ in sorted_items],
                "score": [score for _, score in sorted_items],
            }
        )
