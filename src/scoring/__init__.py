"""
Scoring Modules for ResumeAI

This package contains all scoring modules for candidate evaluation:
- Skills Matching
- Experience Scoring
- Education Scoring
- Location Scoring

Each scorer returns a normalized score (0-1) for integration with the ranking engine.
"""

from src.scoring.skills_matcher import SkillsScorer, score_skills
from src.scoring.experience_scorer import ExperienceScorer, score_experience
from src.scoring.education_scorer import EducationScorer, score_education
from src.scoring.location_scorer import LocationScorer, score_location

__all__ = [
    "SkillsScorer",
    "ExperienceScorer",
    "EducationScorer",
    "LocationScorer",
    "score_skills",
    "score_experience",
    "score_education",
    "score_location",
]
