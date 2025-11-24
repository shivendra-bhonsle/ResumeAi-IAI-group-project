"""
Job Description data schema with ML-friendly methods.

This module defines the complete structure for parsed job description data,
optimized for integration with ML scoring modules.

Team Integration:
- Person 2 (Scoring): Use extract_features() for all scoring requirements
- Person 3 (Ranking): Use get_text_for_embedding() for semantic similarity
- Person 4 (UI): Use to_dict() for API responses
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
from src.models.base_schema import (
    MLReadyBaseModel,
    EducationLevel,
    Location,
    normalize_text,
    safe_parse_int,
)


# ==========================================
# Required Skills
# ==========================================

class RequiredSkills(MLReadyBaseModel):
    """Skills required for the job"""
    must_have: List[str] = Field(default_factory=list, description="Required/must-have skills")
    nice_to_have: List[str] = Field(default_factory=list, description="Optional/nice-to-have skills")

    def get_all_required_skills(self) -> List[str]:
        """
        Get all must-have skills (normalized).

        Returns:
            List[str]: Normalized, deduplicated, sorted list of required skills

        Team Usage:
            Person 2 (Skills Matcher): Primary input for skills matching
        """
        skills = [normalize_text(skill) for skill in self.must_have if skill]
        return sorted(list(set(skills)))

    def get_all_nice_to_have_skills(self) -> List[str]:
        """Get all nice-to-have skills (normalized)"""
        skills = [normalize_text(skill) for skill in self.nice_to_have if skill]
        return sorted(list(set(skills)))

    def get_all_skills_combined(self) -> List[str]:
        """Get both required and nice-to-have skills"""
        all_skills = set(self.get_all_required_skills())
        all_skills.update(self.get_all_nice_to_have_skills())
        return sorted(list(all_skills))

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}required_skills": ",".join(self.get_all_required_skills()),
            f"{prefix}num_required_skills": len(self.get_all_required_skills()),
            f"{prefix}nice_to_have_skills": ",".join(self.get_all_nice_to_have_skills()),
            f"{prefix}num_nice_to_have": len(self.get_all_nice_to_have_skills()),
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "required_skills": self.get_all_required_skills(),
            "nice_to_have_skills": self.get_all_nice_to_have_skills(),
            "num_required_skills": len(self.get_all_required_skills()),
            "num_nice_to_have": len(self.get_all_nice_to_have_skills()),
        }


# ==========================================
# Experience Requirements
# ==========================================

class ExperienceRequirement(MLReadyBaseModel):
    """Experience requirements for the job"""
    min_years: Optional[float] = Field(default=None, description="Minimum years of experience")
    max_years: Optional[float] = Field(default=None, description="Maximum years (if specified)")
    preferred_years: Optional[float] = Field(default=None, description="Preferred years")

    @classmethod
    def from_string(cls, exp_str: str) -> "ExperienceRequirement":
        """
        Parse experience requirement from various string formats.

        Examples:
            "5+ years" → min_years=5
            "3-5 years" → min_years=3, max_years=5
            "5 to 7 years" → min_years=5, max_years=7
            "At least 5 years" → min_years=5
        """
        import re

        exp_str = exp_str.lower()

        # Pattern: "3-5 years" or "3 to 5 years"
        range_match = re.search(r'(\d+)\s*[-to]+\s*(\d+)', exp_str)
        if range_match:
            min_y = float(range_match.group(1))
            max_y = float(range_match.group(2))
            return cls(min_years=min_y, max_years=max_y, preferred_years=(min_y + max_y) / 2)

        # Pattern: "5+ years" or "5 or more"
        plus_match = re.search(r'(\d+)\+', exp_str)
        if plus_match:
            return cls(min_years=float(plus_match.group(1)))

        # Pattern: "at least 5"
        least_match = re.search(r'at least\s*(\d+)', exp_str)
        if least_match:
            return cls(min_years=float(least_match.group(1)))

        # Pattern: just a number "5 years"
        num_match = re.search(r'(\d+)', exp_str)
        if num_match:
            return cls(min_years=float(num_match.group(1)))

        return cls(min_years=0.0)

    def get_target_years(self) -> float:
        """
        Get single target value for comparison.

        Returns:
            float: Preferred years, or average of min/max, or min_years

        Team Usage:
            Person 2 (Experience Scorer): Primary input for experience comparison
        """
        if self.preferred_years:
            return self.preferred_years

        if self.max_years:
            return (self.min_years + self.max_years) / 2

        return self.min_years

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}min_years": self.min_years,
            f"{prefix}max_years": self.max_years,
            f"{prefix}target_years": self.get_target_years(),
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "min_years": self.min_years,
            "target_years": self.get_target_years(),
        }


# ==========================================
# Education Requirements
# ==========================================

class EducationRequirement(MLReadyBaseModel):
    """Education requirements for the job"""
    min_level: EducationLevel = Field(default=EducationLevel.UNKNOWN, description="Minimum education level")
    preferred_level: Optional[EducationLevel] = Field(default=None, description="Preferred education level")
    field: Optional[str] = Field(default=None, description="Required field of study")

    def get_min_level_numeric(self) -> int:
        """
        Get minimum education level as numeric value.

        Returns:
            int: Numeric level (0-5)

        Team Usage:
            Person 2 (Education Scorer): Primary input for education matching
        """
        if isinstance(self.min_level, EducationLevel):
            return self.min_level.to_numeric()
        return 0

    def get_preferred_level_numeric(self) -> int:
        """Get preferred education level as numeric"""
        if self.preferred_level and isinstance(self.preferred_level, EducationLevel):
            return self.preferred_level.to_numeric()
        return self.get_min_level_numeric()

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}min_education": self.min_level.value if isinstance(self.min_level, EducationLevel) else self.min_level,
            f"{prefix}min_education_numeric": self.get_min_level_numeric(),
            f"{prefix}required_field": self.field,
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "min_level_numeric": self.get_min_level_numeric(),
            "preferred_level_numeric": self.get_preferred_level_numeric(),
            "min_level": self.min_level.value if isinstance(self.min_level, EducationLevel) else self.min_level,
            "required_field": self.field,
        }


# ==========================================
# Location Requirements
# ==========================================

class LocationRequirement(MLReadyBaseModel):
    """Location requirements for the job"""
    location: Optional[str] = Field(default="Unknown", description="Job location (city/country)")
    remote_allowed: bool = Field(default=False, description="Whether remote work is allowed")
    hybrid: bool = Field(default=False, description="Whether hybrid work is offered")

    def model_post_init(self, __context):
        """Handle None values after initialization"""
        if self.location is None:
            self.location = "Unknown"

    def get_location_city(self) -> str:
        """Extract city from location string"""
        if not self.location or self.location == "Unknown":
            return "Unknown"

        # Handle formats like "Pune, India" or "New York, NY"
        parts = self.location.split(',')
        return parts[0].strip() if parts else self.location

    def get_location_country(self) -> str:
        """Extract country from location string"""
        if not self.location or self.location == "Unknown":
            return "Unknown"

        parts = self.location.split(',')
        return parts[-1].strip() if len(parts) > 1 else "Unknown"

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}location": self.location,
            f"{prefix}city": self.get_location_city(),
            f"{prefix}country": self.get_location_country(),
            f"{prefix}remote_allowed": self.remote_allowed,
            f"{prefix}hybrid": self.hybrid,
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "city": self.get_location_city(),
            "country": self.get_location_country(),
            "remote_allowed": self.remote_allowed,
            "hybrid": self.hybrid,
        }


# ==========================================
# Complete Job Description Model
# ==========================================

class JobDescription(MLReadyBaseModel):
    """
    Complete job description data model with ML-ready methods.

    This is the main class that parsers will return and other team members will use.

    Team Integration:
        Person 2 (Scoring Modules):
            - job.extract_features() → All requirements for scoring
            - job.required_skills.get_all_required_skills() → Skills list
            - job.required_experience.get_target_years() → Experience years
            - job.education_requirement.get_min_level_numeric() → Education level

        Person 3 (Ranking Engine):
            - job.get_text_for_embedding() → Text for semantic similarity
            - job.validate_for_ml() → Check data completeness

        Person 4 (UI):
            - job.dict() → Clean JSON for API
            - job.format_for_display() → Human-readable summary
    """
    job_id: Optional[str] = Field(default=None, description="Unique job identifier")
    title: str = Field(default="Unknown", description="Job title")
    company: Optional[str] = Field(default="Unknown", description="Company name")
    role: Optional[str] = Field(default=None, description="Role/position type")

    def model_post_init(self, __context):
        """Handle None values after initialization"""
        if self.company is None:
            self.company = "Unknown"
    description: str = Field(default="", description="Full job description")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")

    required_skills: RequiredSkills = Field(default_factory=RequiredSkills, description="Required skills")
    required_experience: ExperienceRequirement = Field(default_factory=ExperienceRequirement, description="Experience requirements")
    education_requirement: EducationRequirement = Field(default_factory=EducationRequirement, description="Education requirements")
    location_requirement: LocationRequirement = Field(default_factory=LocationRequirement, description="Location requirements")

    # Optional fields
    salary_range: Optional[str] = Field(default=None, description="Salary range")
    benefits: Optional[List[str]] = Field(default_factory=list, description="Benefits offered")
    company_name: Optional[str] = Field(default=None, description="Company name")
    company_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Company information")

    # ==========================================
    # ML-FRIENDLY METHODS
    # ==========================================

    def get_text_for_embedding(self) -> str:
        """
        Combined text for semantic similarity analysis.

        Returns:
            str: Description + responsibilities combined

        Team Usage:
            Person 3 (Semantic Scorer): Input for sentence transformer embedding
        """
        texts = []

        if self.description:
            texts.append(self.description)

        if self.responsibilities:
            texts.extend(self.responsibilities)

        return " ".join(texts)

    def extract_features(self) -> Dict[str, Any]:
        """
        Extract ALL requirements for ML scoring modules.

        Returns:
            Dict containing all job requirements in ML-friendly format

        Team Usage:
            Person 2: PRIMARY method - contains all inputs for all scoring modules:
                - Skills Matcher: features["required_skills"]
                - Experience Scorer: features["required_years"]
                - Education Scorer: features["required_education_level"]
                - Location Scorer: features["location"], features["remote_allowed"]

            Person 3: Use features["description_text"] for semantic similarity
        """
        return {
            # Skills requirements
            "required_skills": self.required_skills.get_all_required_skills(),
            "nice_to_have_skills": self.required_skills.get_all_nice_to_have_skills(),
            "all_skills": self.required_skills.get_all_skills_combined(),
            "num_required_skills": len(self.required_skills.get_all_required_skills()),

            # Experience requirements
            "required_years": self.required_experience.get_target_years(),
            "min_years": self.required_experience.min_years,
            "max_years": self.required_experience.max_years,

            # Education requirements
            "required_education_level": self.education_requirement.get_min_level_numeric(),
            "preferred_education_level": self.education_requirement.get_preferred_level_numeric(),
            "required_field": self.education_requirement.field,

            # Location requirements
            "location": self.location_requirement.location,
            "location_city": self.location_requirement.get_location_city(),
            "location_country": self.location_requirement.get_location_country(),
            "remote_allowed": self.location_requirement.remote_allowed,
            "hybrid": self.location_requirement.hybrid,

            # Text for semantic analysis
            "description_text": self.get_text_for_embedding(),
            "job_title": self.title,

            # Metadata
            "has_salary_info": bool(self.salary_range),
            "num_responsibilities": len(self.responsibilities),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary for storage/display.

        Returns:
            Dict with flat structure

        Team Usage:
            Person 3: Easy DataFrame conversion
            Person 4: Table display in UI
        """
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company_name,
            "required_years": self.required_experience.get_target_years(),
            "required_education": self.education_requirement.min_level.value if isinstance(self.education_requirement.min_level, EducationLevel) else self.education_requirement.min_level,
            "required_skills": ",".join(self.required_skills.get_all_required_skills()),
            "num_required_skills": len(self.required_skills.get_all_required_skills()),
            "location": self.location_requirement.location,
            "remote": self.location_requirement.remote_allowed,
            "salary_range": self.salary_range,
        }

    def validate_for_ml(self) -> List[str]:
        """
        Validate if job description has sufficient data for ML scoring.

        Returns:
            List of validation issues. Empty list = ready for scoring.

        Team Usage:
            Person 2 & 3: Check data quality before scoring
            Person 4: Show validation warnings in UI
        """
        issues = []

        # Check skills
        if not self.required_skills.get_all_required_skills():
            issues.append("no_required_skills")

        # Check experience
        if self.required_experience.min_years == 0:
            issues.append("experience_requirement_not_specified")

        # Check education
        if self.education_requirement.get_min_level_numeric() == 0:
            issues.append("education_requirement_not_specified")

        # Check description
        if not self.description or len(self.description) < 50:
            issues.append("insufficient_job_description")

        # Check responsibilities
        if not self.responsibilities:
            issues.append("no_responsibilities_listed")

        return issues

    def format_for_display(self) -> Dict[str, Any]:
        """
        Format job description for UI display.

        Returns:
            Dict with human-readable format

        Team Usage:
            Person 4: Display job summary in UI
        """
        return {
            "title": self.title,
            "company": self.company_name or "Unknown Company",
            "location": self.location_requirement.location,
            "remote_friendly": "Yes" if self.location_requirement.remote_allowed else "No",
            "experience_required": f"{self.required_experience.get_target_years():.0f}+ years",
            "education_required": self.education_requirement.min_level.value if isinstance(self.education_requirement.min_level, EducationLevel) else "Not specified",
            "key_skills": self.required_skills.get_all_required_skills()[:10],  # Top 10
            "num_responsibilities": len(self.responsibilities),
            "salary": self.salary_range or "Not disclosed",
        }

    def get_requirements_summary(self) -> str:
        """
        Get a concise summary of requirements.

        Returns:
            str: Human-readable summary

        Team Usage:
            Person 4: Quick summary for UI display
        """
        parts = []

        # Experience
        if self.required_experience.min_years > 0:
            parts.append(f"{self.required_experience.get_target_years():.0f}+ years experience")

        # Education
        if self.education_requirement.get_min_level_numeric() > 0:
            edu_level = self.education_requirement.min_level.value.replace('_', ' ').title()
            parts.append(f"{edu_level} degree")

        # Skills count
        num_skills = len(self.required_skills.get_all_required_skills())
        if num_skills > 0:
            parts.append(f"{num_skills} required skills")

        # Location
        if self.location_requirement.remote_allowed:
            parts.append("Remote OK")
        else:
            parts.append(f"Location: {self.location_requirement.get_location_city()}")

        return " | ".join(parts) if parts else "Requirements not specified"
