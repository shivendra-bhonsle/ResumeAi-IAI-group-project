"""
Ranking Module for ResumeAI
Person 3: Semantic Similarity + Final Ranking

This module provides:
1. Semantic similarity scoring using sentence transformers
2. Ranking engine that combines all scoring components
3. Integration with Person 2's scoring modules
4. Output formatting for Person 4's UI
"""

from src.ranking.semantic_similarity import SemanticSimilarityScorer, compute_semantic_similarity
from src.ranking.ranking_engine import RankingEngine, rank_resumes

__all__ = [
    'SemanticSimilarityScorer',
    'compute_semantic_similarity',
    'RankingEngine',
    'rank_resumes',
]