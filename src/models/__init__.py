"""
Data models for Resume and Job Description parsing.

This module provides ML-ready data models with:
- Automatic validation via Pydantic
- Feature extraction for ML models
- Flat dictionary conversion for pandas
- Text extraction for embeddings
- Data quality validation

Usage:
    from src.models import Resume, JobDescription
    from src.models import EducationLevel, ExperienceLevel

Team Integration:
    - Import Resume and JobDescription for all data operations
    - Use extract_features() for ML scoring
    - Use to_flat_dict() for DataFrame conversion
    - Use get_text_for_embedding() for semantic similarity
"""

# Base models and utilities
from src.models.base_schema import (
    MLReadyBaseModel,
    EducationLevel,
    ExperienceLevel,
    EmploymentType,
    RemotePreference,
    Location,
    DateRange,
    normalize_text,
    safe_parse_int,
    safe_parse_float,
)

# Resume models
from src.models.resume_schema import (
    Resume,
    PersonalInfo,
    Skills,
    TechnicalSkills,
    Skill,
    Experience,
    Education,
    Degree,
    Institution,
    Project,
)

# Job description models
from src.models.job_schema import (
    JobDescription,
    RequiredSkills,
    ExperienceRequirement,
    EducationRequirement,
    LocationRequirement,
)

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Main models
    "Resume",
    "JobDescription",

    # Enums
    "EducationLevel",
    "ExperienceLevel",
    "EmploymentType",
    "RemotePreference",

    # Common models
    "Location",
    "DateRange",

    # Resume components
    "PersonalInfo",
    "Skills",
    "TechnicalSkills",
    "Skill",
    "Experience",
    "Education",
    "Degree",
    "Institution",
    "Project",

    # Job description components
    "RequiredSkills",
    "ExperienceRequirement",
    "EducationRequirement",
    "LocationRequirement",

    # Base
    "MLReadyBaseModel",

    # Utilities
    "normalize_text",
    "safe_parse_int",
    "safe_parse_float",
]
