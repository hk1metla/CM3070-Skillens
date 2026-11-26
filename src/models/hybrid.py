"""
Hybrid Recommender System.

Combines Content-Based (TF-IDF) and Collaborative Filtering (ItemKNN) models
using weighted score fusion with min-max normalisation.
"""

from typing import List, Optional

import pandas as pd

from src.models.itemknn import ItemKNNRecommender
from src.models.tfidf import TfidfRecommender


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    
    Score fusion formula:
    score = w_content * content_score + w_cf * cf_score
    
    The reported baseline fixes the content and collaborative weights at 0.6 and 0.4.
    """

    def __init__(
        self,
        w_content: float = 0.6,
        w_cf: float = 0.4,
        k_neighbors: int = 50,
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            w_content: Weight for content-based (TF-IDF) scores
            w_cf: Weight for collaborative filtering (ItemKNN) scores
            k_neighbors: Number of neighbors for ItemKNN
        """
        self.w_content = w_content
        self.w_cf = w_cf
        self.k_neighbors = k_neighbors
        
        self.content_model: Optional[TfidfRecommender] = None
        self.cf_model: Optional[ItemKNNRecommender] = None
        self.items: Optional[pd.DataFrame] = None

    def fit(
        self,
        items: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> None:
        """
        Fit both content and collaborative filtering models.
        
        Args:
            items: Item metadata DataFrame
            interactions: User-item interactions DataFrame
        """
        self.items = items
        
        # Fit content-based model
        self.content_model = TfidfRecommender()
        self.content_model.fit(items)
        
        # Fit collaborative filtering model
        self.cf_model = ItemKNNRecommender(k=self.k_neighbors)
        self.cf_model.fit(interactions)

    def recommend(
        self,
        goal: str,
        user_id: Optional[str] = None,
        user_interactions: Optional[pd.DataFrame] = None,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate hybrid recommendations.
        
        Args:
            goal: User's goal text (for content-based)
            user_id: User ID (for collaborative filtering)
            user_interactions: User's historical interactions (for CF)
            k: Number of recommendations
            exclude_items: Items to exclude
            
        Returns:
            DataFrame with columns ['item_id', 'score', 'content_score', 'cf_score']
        """
        if self.content_model is None or self.cf_model is None:
            raise ValueError("Model is not fitted.")
        
        # Get content-based recommendations
        content_recs = self.content_model.recommend(goal, k=k * 2)  # Get more for fusion
        content_scores = dict(zip(content_recs["item_id"], content_recs["score"]))
        
        # Get collaborative filtering recommendations
        if user_id and user_interactions is not None:
            # Warm user: use CF
            cf_recs = self.cf_model.recommend(
                user_id, user_interactions, k=k * 2, exclude_items=exclude_items
            )
        else:
            # Cold-start: use goal items to seed CF
            # Find items matching goal from content model
            goal_items = content_recs["item_id"].head(5).tolist()
            cf_recs = self.cf_model.recommend_for_new_user(goal_items, k=k * 2)
        
        cf_scores = dict(zip(cf_recs["item_id"], cf_recs["score"]))
        
        # Normalize scores to [0, 1] range for fair fusion
        if content_scores:
            max_content = max(content_scores.values())
            min_content = min(content_scores.values())
            content_range = max_content - min_content if max_content != min_content else 1.0
            content_scores = {
                item: (score - min_content) / content_range
                for item, score in content_scores.items()
            }
        
        if cf_scores:
            max_cf = max(cf_scores.values())
            min_cf = min(cf_scores.values())
            cf_range = max_cf - min_cf if max_cf != min_cf else 1.0
            cf_scores = {
                item: (score - min_cf) / cf_range
                for item, score in cf_scores.items()
            }
        
        # Combine all candidate items
        all_items = set(content_scores.keys()) | set(cf_scores.keys())
        
        if exclude_items:
            all_items = all_items - set(exclude_items)
        
        # Compute hybrid scores
        hybrid_scores = []
        for item in all_items:
            content_score = content_scores.get(item, 0.0)
            cf_score = cf_scores.get(item, 0.0)
            
            # Weighted fusion
            hybrid_score = (
                self.w_content * content_score + self.w_cf * cf_score
            )
            
            hybrid_scores.append({
                "item_id": item,
                "score": hybrid_score,
                "content_score": content_score,
                "cf_score": cf_score,
            })
        
        # Sort by hybrid score and return top k
        result_df = pd.DataFrame(hybrid_scores)
        result_df = result_df.sort_values("score", ascending=False).head(k)
        
        return result_df[["item_id", "score"]].reset_index(drop=True)
