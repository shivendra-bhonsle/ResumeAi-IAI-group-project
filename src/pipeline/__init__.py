"""
Pipeline Module for ResumeAI

Main orchestration for end-to-end resume ranking.
"""

from src.pipeline.main_pipeline import ResumePipeline, rank_candidates

__all__ = ["ResumePipeline", "rank_candidates"]
