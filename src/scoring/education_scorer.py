"""
Education Scoring Module for ResumeAI
Scores candidate education against job requirements.

Team Integration:
- Uses Resume.get_highest_education_level()
- Uses JobDescription.education_requirement
- Returns normalized score (0-1) for ranking engine
"""

from typing import Dict, Any


class EducationScorer:
    """
    Score candidate education against job requirements.

    Education Levels (numeric scale):
    0 = Unknown
    1 = High School / Diploma
    2 = Associate's Degree
    3 = Bachelor's Degree (B.Tech, B.E, B.S)
    4 = Master's Degree (M.Tech, M.E, M.S, MBA)
    5 = Doctorate (PhD, Doctor)

    Scoring Logic:
    - Meets or exceeds requirement: 100
    - One level below: 75
    - Two or more levels below: 50
    """

    # Education level mapping (for parsing)
    EDU_LEVEL_MAP = {
        "high school": 1,
        "high_school": 1,
        "diploma": 1,
        "associate": 2,
        "associate's": 2,
        "bachelor": 3,
        "bachelor's": 3,
        "b.tech": 3,
        "b.e": 3,
        "b.s": 3,
        "b.a": 3,
        "master": 4,
        "master's": 4,
        "m.tech": 4,
        "m.e": 4,
        "m.s": 4,
        "m.a": 4,
        "mba": 4,
        "phd": 5,
        "ph.d": 5,
        "doctorate": 5,
        "doctor": 5,
    }

    def __init__(self):
        """Initialize education scorer."""
        pass

    def score(self, resume, job) -> float:
        """
        Score resume education against job requirements.

        Args:
            resume: Resume object
            job: JobDescription object

        Returns:
            float: Score between 0 and 1
        """
        # Get education levels
        job_level = self._parse_edu_level(job.education_requirement.min_level)
        candidate_level = resume.get_highest_education_level()

        # If candidate level is 0 (unknown), try custom parsing
        if candidate_level == 0 and resume.education:
            candidate_level = max(
                self._parse_edu_level(edu.degree.level)
                for edu in resume.education
            )

        result = self._score_education(job_level, candidate_level)
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
        job_level = self._parse_edu_level(job.education_requirement.min_level)
        candidate_level = resume.get_highest_education_level()

        # Fallback to custom parsing if needed
        if candidate_level == 0 and resume.education:
            candidate_level = max(
                self._parse_edu_level(edu.degree.level)
                for edu in resume.education
            )

        result = self._score_education(job_level, candidate_level)
        result["normalized_score"] = result["score"] / 100.0
        result["job_level"] = job_level
        result["candidate_level"] = candidate_level
        result["job_level_name"] = job.education_requirement.min_level
        result["candidate_education"] = [
            {
                "degree": edu.degree.level,
                "field": edu.degree.field,
                "institution": edu.institution.name
            }
            for edu in resume.education
        ] if resume.education else []

        return result

    def _parse_edu_level(self, level_str) -> int:
        """
        Parse education level string to numeric value.

        Args:
            level_str: Education level as string or int

        Returns:
            int: Numeric level (0-5)
        """
        if isinstance(level_str, int):
            return level_str

        if not level_str or level_str == "unknown":
            return 0

        level_lower = str(level_str).lower().strip()

        # Try exact match first
        if level_lower in self.EDU_LEVEL_MAP:
            return self.EDU_LEVEL_MAP[level_lower]

        # Try substring match
        for key, value in self.EDU_LEVEL_MAP.items():
            if key in level_lower:
                return value

        return 0

    def _score_education(self, job_level: int, candidate_level: int) -> Dict[str, Any]:
        """
        Internal method for education scoring.

        Args:
            job_level: Required education level (0-5)
            candidate_level: Candidate's education level (0-5)

        Returns:
            Dict with score and status
        """
        if job_level == 0:
            return {
                "score": 100,
                "status": "no_requirement",
                "message": "No education requirement specified"
            }

        if candidate_level == 0:
            return {
                "score": 50,
                "status": "unknown",
                "message": "Candidate education level unknown"
            }

        # Meets or exceeds requirement
        if candidate_level >= job_level:
            if candidate_level > job_level:
                return {
                    "score": 100,
                    "status": "exceeds",
                    "message": f"Candidate exceeds requirement (level {candidate_level} vs {job_level})"
                }
            else:
                return {
                    "score": 100,
                    "status": "meets",
                    "message": f"Candidate meets requirement (level {candidate_level})"
                }

        # One level below
        if candidate_level == job_level - 1:
            return {
                "score": 75,
                "status": "slightly_below",
                "message": f"Candidate one level below requirement (level {candidate_level} vs {job_level})"
            }

        # Two or more levels below
        return {
            "score": 50,
            "status": "significant_gap",
            "message": f"Candidate below requirement (level {candidate_level} vs {job_level})"
        }


# Convenience function for quick scoring
def score_education(resume, job) -> float:
    """
    Quick helper to score education.

    Args:
        resume: Resume object
        job: JobDescription object

    Returns:
        float: Score between 0 and 1
    """
    scorer = EducationScorer()
    return scorer.score(resume, job)
