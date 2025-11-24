"""
Skills Matching Module for ResumeAI
Scores candidate skills against job requirements using exact and fuzzy matching.

Team Integration:
- Uses Resume.skills.get_all_skills_flat()
- Uses JobDescription.required_skills
- Returns normalized score (0-1) for ranking engine
"""

from typing import List, Dict, Any
from rapidfuzz import fuzz


class SkillsScorer:
    """
    Score candidate skills against job requirements.

    Features:
    - Exact matching for skill overlap
    - Fuzzy matching for skill variants (e.g., "node.js" vs "nodejs")
    - Bonus points for nice-to-have skills
    - Detailed breakdown of matches
    """

    # Skill synonyms for normalization
    SKILL_SYNONYMS = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "nodejs": "node.js",
        "reactjs": "react",
        "k8s": "kubernetes",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
    }

    def __init__(self, fuzzy_threshold: int = 85):
        """
        Initialize skills scorer.

        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
                           Default: 85 (reduced from 90 for better matching)
        """
        self.fuzzy_threshold = fuzzy_threshold

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize skill name (lowercase, trim, apply synonyms).

        Args:
            skill: Raw skill name

        Returns:
            Normalized skill name
        """
        if not skill:
            return ""

        skill = skill.lower().strip()
        skill = self.SKILL_SYNONYMS.get(skill, skill)
        return " ".join(skill.split())

    def score(self, resume, job) -> float:
        """
        Score resume against job requirements.

        Args:
            resume: Resume object
            job: JobDescription object

        Returns:
            float: Score between 0 and 1
        """
        # Get skills from resume and job
        candidate_skills = [self.normalize_skill(s) for s in resume.skills.get_all_skills_flat()]
        required_skills = [self.normalize_skill(s) for s in job.required_skills.must_have]
        nice_to_have = [self.normalize_skill(s) for s in job.required_skills.nice_to_have]

        # Calculate detailed scores
        result = self._score_skills_detailed(required_skills, nice_to_have, candidate_skills)

        # Return normalized score (0-1)
        return result["score"] / 100.0

    def score_detailed(self, resume, job) -> Dict[str, Any]:
        """
        Get detailed scoring breakdown.

        Args:
            resume: Resume object
            job: JobDescription object

        Returns:
            Dict with detailed scoring information
        """
        candidate_skills = [self.normalize_skill(s) for s in resume.skills.get_all_skills_flat()]
        required_skills = [self.normalize_skill(s) for s in job.required_skills.must_have]
        nice_to_have = [self.normalize_skill(s) for s in job.required_skills.nice_to_have]

        result = self._score_skills_detailed(required_skills, nice_to_have, candidate_skills)
        result["normalized_score"] = result["score"] / 100.0

        return result

    def _score_skills_detailed(
        self,
        required: List[str],
        nice_to_have: List[str],
        candidate: List[str]
    ) -> Dict[str, Any]:
        """
        Internal method for detailed skills scoring.

        Args:
            required: List of required skills
            nice_to_have: List of nice-to-have skills
            candidate: List of candidate skills

        Returns:
            Dict with scoring breakdown
        """
        required_set = set(required)
        candidate_set = set(candidate)

        # Exact matches
        exact_matches = required_set & candidate_set

        # Fuzzy matching for remaining required skills
        remaining = required_set - exact_matches
        fuzzy_matches = set()
        fuzzy_pairs = []

        for req_skill in remaining:
            best_match = None
            best_score = 0

            for cand_skill in candidate_set:
                similarity = fuzz.token_sort_ratio(req_skill, cand_skill)
                if similarity >= self.fuzzy_threshold and similarity > best_score:
                    best_match = cand_skill
                    best_score = similarity

            if best_match:
                fuzzy_matches.add(req_skill)
                fuzzy_pairs.append({
                    "required": req_skill,
                    "matched": best_match,
                    "similarity": best_score
                })

        # Total matched required skills
        matched_total = len(exact_matches | fuzzy_matches)
        coverage = matched_total / len(required_set) if required_set else 0

        # Base score from coverage
        base_score = coverage * 100

        # Bonus from nice-to-have skills (up to 10 points)
        nice_matches = set(nice_to_have) & candidate_set
        bonus = min(len(nice_matches) * 2, 10)

        # Final score (capped at 100)
        final_score = min(base_score + bonus, 100)

        return {
            "score": final_score,
            "base_score": base_score,
            "bonus": bonus,
            "matched_required": matched_total,
            "total_required": len(required_set),
            "coverage": coverage,
            "exact_matches": list(exact_matches),
            "fuzzy_matches": fuzzy_pairs,
            "nice_matches": list(nice_matches),
            "nice_to_have_count": len(nice_matches),
        }


# Convenience function for quick scoring
def score_skills(resume, job) -> float:
    """
    Quick helper to score skills.

    Args:
        resume: Resume object
        job: JobDescription object

    Returns:
        float: Score between 0 and 1
    """
    scorer = SkillsScorer()
    return scorer.score(resume, job)
