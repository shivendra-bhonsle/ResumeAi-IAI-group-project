"""
Experience Scoring Module for ResumeAI
Scores candidate experience against job requirements.

Team Integration:
- Uses Resume.calculate_total_experience()
- Uses JobDescription.required_experience
- Returns normalized score (0-1) for ranking engine
"""

from typing import Dict, Any


class ExperienceScorer:
    """
    Score candidate years of experience against job requirements.

    Scoring Logic:
    - Meets or exceeds requirement: 1.0
    - Double or more (overqualified): 0.9
    - Below requirement: Proportional (years/required)
    - No requirement specified: 0.0
    """

    def __init__(self, overqualified_penalty: bool = True):
        """
        Initialize experience scorer.

        Args:
            overqualified_penalty: If True, slightly penalize being >2x overqualified
                                  (some jobs prefer not too senior candidates)
        """
        self.overqualified_penalty = overqualified_penalty

    def score(self, resume, job) -> float:
        """
        Score resume experience against job requirements.

        Args:
            resume: Resume object
            job: JobDescription object

        Returns:
            float: Score between 0 and 1
        """
        required_years = job.required_experience.get_target_years()
        candidate_years = resume.calculate_total_experience()

        result = self._score_experience(required_years, candidate_years)
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
        required_years = job.required_experience.get_target_years()
        candidate_years = resume.calculate_total_experience()

        result = self._score_experience(required_years, candidate_years)
        result["normalized_score"] = result["score"] / 100.0
        result["required_years"] = required_years
        result["candidate_years"] = candidate_years

        return result

    def _score_experience(self, required_years: float, candidate_years: float) -> Dict[str, Any]:
        """
        Internal method for experience scoring.

        Args:
            required_years: Years of experience required
            candidate_years: Years of experience candidate has

        Returns:
            Dict with score and status
        """
        if required_years == 0 or required_years == None:
            return {
                "score": 0,
                "status": "no_requirement",
                "message": "No experience requirement specified"
            }

        # Candidate significantly overqualified (2x or more)
        if candidate_years >= required_years * 2:
            if self.overqualified_penalty:
                return {
                    "score": 90,
                    "status": "overqualified",
                    "message": f"Candidate has {candidate_years:.1f} years vs {required_years:.1f} required (overqualified)"
                }
            else:
                return {
                    "score": 100,
                    "status": "overqualified",
                    "message": f"Candidate exceeds requirement ({candidate_years:.1f} vs {required_years:.1f} years)"
                }

        # Candidate meets or slightly exceeds requirement
        if candidate_years >= required_years:
            return {
                "score": 100,
                "status": "meets",
                "message": f"Candidate meets requirement ({candidate_years:.1f} vs {required_years:.1f} years)"
            }

        # Candidate below requirement - proportional scoring
        ratio = candidate_years / required_years
        score = ratio * 100

        # Very close to requirement (90%+) - give small bonus
        if ratio >= 0.9:
            score = min(score + 5, 100)
            status = "nearly_meets"
        else:
            status = "below"

        return {
            "score": score,
            "status": status,
            "message": f"Candidate has {candidate_years:.1f} years vs {required_years:.1f} required ({ratio:.1%} of requirement)"
        }


# Convenience function for quick scoring
def score_experience(resume, job) -> float:
    """
    Quick helper to score experience.

    Args:
        resume: Resume object
        job: JobDescription object

    Returns:
        float: Score between 0 and 1
    """
    scorer = ExperienceScorer()
    return scorer.score(resume, job)
