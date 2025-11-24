"""
Location Scoring Module for ResumeAI
Scores candidate location against job requirements.

Team Integration:
- Uses Resume.personal_info.location
- Uses JobDescription.location_requirement
- Returns normalized score (0-1) for ranking engine
"""

from typing import Dict, Any


class LocationScorer:
    """
    Score candidate location against job requirements.

    Scoring Logic:
    - Remote work and candidate is remote-friendly: 100
    - Same city: 100
    - Same country: 50
    - Different location (relocation needed): 0
    - Unknown candidate location: Neutral score (50) if remote allowed, else 0
    """

    def __init__(self, unknown_location_penalty: bool = False):
        """
        Initialize location scorer.

        Args:
            unknown_location_penalty: If False, give benefit of doubt for unknown locations
                                     If True, penalize unknown locations
        """
        self.unknown_location_penalty = unknown_location_penalty

    def score(self, resume, job) -> float:
        """
        Score resume location against job requirements.

        Args:
            resume: Resume object
            job: JobDescription object

        Returns:
            float: Score between 0 and 1
        """
        job_location = job.location_requirement.location
        remote_allowed = job.location_requirement.remote_allowed
        candidate_city = resume.personal_info.location.city
        candidate_country = resume.personal_info.location.country

        result = self._score_location(
            job_location,
            candidate_city,
            candidate_country,
            remote_allowed
        )

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
        job_location = job.location_requirement.location
        remote_allowed = job.location_requirement.remote_allowed
        candidate_city = resume.personal_info.location.city
        candidate_country = resume.personal_info.location.country

        result = self._score_location(
            job_location,
            candidate_city,
            candidate_country,
            remote_allowed
        )

        result["normalized_score"] = result["score"] / 100.0
        result["job_location"] = job_location
        result["candidate_city"] = candidate_city
        result["candidate_country"] = candidate_country
        result["remote_allowed"] = remote_allowed

        return result

    def _score_location(
        self,
        job_location: str,
        candidate_city: str,
        candidate_country: str,
        remote_allowed: bool
    ) -> Dict[str, Any]:
        """
        Internal method for location scoring.

        Args:
            job_location: Job location string
            candidate_city: Candidate's city
            candidate_country: Candidate's country
            remote_allowed: Whether remote work is allowed

        Returns:
            Dict with score and status
        """
        # Normalize for comparison
        job_loc_lower = job_location.lower() if job_location else ""
        city_lower = candidate_city.lower() if candidate_city else ""
        country_lower = candidate_country.lower() if candidate_country else ""

        # Handle unknown candidate location
        if city_lower in ["unknown", ""] and country_lower in ["unknown", ""]:
            if remote_allowed:
                return {
                    "score": 75,  # Give benefit of doubt if remote allowed
                    "status": "unknown_remote_ok",
                    "message": "Candidate location unknown, but remote work is allowed"
                }
            elif not self.unknown_location_penalty:
                return {
                    "score": 50,  # Neutral score
                    "status": "unknown_neutral",
                    "message": "Candidate location unknown (neutral scoring)"
                }
            else:
                return {
                    "score": 0,
                    "status": "unknown_not_remote",
                    "message": "Candidate location unknown and remote not allowed"
                }

        # Remote work scenario
        if remote_allowed:
            # If candidate explicitly prefers remote or is in a "remote" location
            if city_lower == "remote" or "remote" in city_lower:
                return {
                    "score": 100,
                    "status": "remote",
                    "message": "Remote work allowed and candidate is remote-friendly"
                }

            # Even if not explicitly remote, remote is allowed so give high score
            return {
                "score": 90,
                "status": "remote_allowed",
                "message": "Remote work allowed"
            }

        # City match (exact or substring)
        if city_lower and city_lower in job_loc_lower:
            return {
                "score": 100,
                "status": "same_city",
                "message": f"Candidate in same city ({candidate_city})"
            }

        # Country match (exact or substring)
        if country_lower and country_lower in job_loc_lower:
            return {
                "score": 50,
                "status": "same_country",
                "message": f"Candidate in same country ({candidate_country}), relocation within country may be needed"
            }

        # No match - relocation required
        return {
            "score": 0,
            "status": "relocation_needed",
            "message": f"Candidate location ({candidate_city}, {candidate_country}) differs from job location ({job_location})"
        }


# Convenience function for quick scoring
def score_location(resume, job) -> float:
    """
    Quick helper to score location.

    Args:
        resume: Resume object
        job: JobDescription object

    Returns:
        float: Score between 0 and 1
    """
    scorer = LocationScorer()
    return scorer.score(resume, job)
