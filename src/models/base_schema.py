"""
Base schema classes and utilities for ML-ready data models.

This module provides:
- MLReadyBaseModel: Base class with ML-friendly methods
- Enums for categorical data (EducationLevel, ExperienceLevel)
- Common validators and utilities
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime
from dateutil import parser as date_parser


# ==========================================
# Enums for Categorical Data
# ==========================================

class EducationLevel(str, Enum):
    """Education levels with numeric mapping for ML models"""
    HIGH_SCHOOL = "high_school"
    DIPLOMA = "diploma"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    UNKNOWN = "unknown"

    def to_numeric(self) -> int:
        """
        Convert education level to numeric value for ML models.

        Returns:
            int: Numeric representation (0-6)
        """
        mapping = {
            "unknown": 0,
            "high_school": 1,
            "diploma": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "doctorate": 5,
        }
        return mapping.get(self.value, 0)

    @classmethod
    def from_string(cls, level_str: str) -> "EducationLevel":
        """
        Parse education level from various string formats.

        Examples:
            "Bachelor's" → BACHELOR
            "M.Tech" → MASTER
            "PhD" → DOCTORATE
        """
        level_str = level_str.lower().strip()

        # High school variations
        if any(x in level_str for x in ["high school", "secondary", "diploma"]):
            return cls.HIGH_SCHOOL

        # Associate variations
        if "associate" in level_str:
            return cls.ASSOCIATE

        # Bachelor variations
        if any(x in level_str for x in ["bachelor", "b.e", "b.tech", "b.s", "ba", "bsc"]):
            return cls.BACHELOR

        # Master variations
        if any(x in level_str for x in ["master", "m.e", "m.tech", "ms", "mba", "ma", "msc"]):
            return cls.MASTER

        # Doctorate variations
        if any(x in level_str for x in ["phd", "doctorate", "doctoral"]):
            return cls.DOCTORATE

        return cls.UNKNOWN


class ExperienceLevel(str, Enum):
    """Experience level categories"""
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

    def to_numeric(self) -> int:
        """Convert to numeric value (0-6)"""
        mapping = {
            "unknown": 0,
            "intern": 1,
            "entry": 2,
            "mid": 3,
            "senior": 4,
            "lead": 5,
            "executive": 6,
        }
        return mapping.get(self.value, 0)


class EmploymentType(str, Enum):
    """Employment type categories"""
    FULL_TIME = "full-time"
    PART_TIME = "part-time"
    CONTRACT = "contract"
    INTERN = "intern"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class RemotePreference(str, Enum):
    """Remote work preferences"""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


# ==========================================
# Base Model with ML Utilities
# ==========================================

class MLReadyBaseModel(BaseModel):
    """
    Base class for all data models with ML-friendly utilities.

    All schemas inherit from this to get:
    - Flat dictionary conversion for pandas
    - Feature extraction for ML models
    - Validation for ML readiness
    - Completeness scoring
    """

    class Config:
        """Pydantic configuration"""
        # Allow extra fields for forward compatibility
        extra = "allow"
        # Use enum values instead of enum objects
        use_enum_values = True
        # Allow arbitrary types
        arbitrary_types_allowed = True

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        """
        Convert nested structure to flat dictionary for pandas DataFrame.

        Args:
            prefix: Prefix for field names (used in recursion)

        Returns:
            Dict with flat structure

        Note: Override in subclasses for custom flattening
        """
        raise NotImplementedError("Subclasses must implement to_flat_dict()")

    def extract_features(self) -> Dict[str, Any]:
        """
        Extract ML-ready features from the model.

        Returns:
            Dict containing:
                - Numerical features (int, float)
                - Categorical features (str, List[str])
                - Text features (str) for embeddings

        Note: Override in subclasses
        """
        raise NotImplementedError("Subclasses must implement extract_features()")

    def get_missing_fields(self, critical_fields: Optional[List[str]] = None) -> List[str]:
        """
        Check which fields are missing or have placeholder values.

        Args:
            critical_fields: List of field names to check. If None, checks all fields.

        Returns:
            List of field names that are missing/invalid
        """
        missing = []
        data = self.dict()

        fields_to_check = critical_fields if critical_fields else data.keys()

        for field_name in fields_to_check:
            if field_name not in data:
                missing.append(field_name)
                continue

            field_value = data[field_name]

            # Check for placeholder values
            if field_value in ["Unknown", "unknown", None, "", []]:
                missing.append(field_name)
            elif isinstance(field_value, str) and field_value.strip() == "":
                missing.append(field_name)

        return missing

    def completeness_score(self, critical_fields: Optional[List[str]] = None) -> float:
        """
        Calculate completeness score (0-1) indicating data quality.

        Args:
            critical_fields: List of critical field names. If None, uses all fields.

        Returns:
            float: Score between 0 (no data) and 1 (complete data)
        """
        data = self.dict()
        fields_to_check = critical_fields if critical_fields else data.keys()

        if not fields_to_check:
            return 0.0

        total_fields = len(fields_to_check)
        missing_fields = len(self.get_missing_fields(critical_fields))

        return (total_fields - missing_fields) / total_fields

    def validate_for_ml(self) -> List[str]:
        """
        Validate if the model has sufficient data for ML scoring.

        Returns:
            List of validation errors/warnings. Empty list means valid.

        Note: Override in subclasses with specific validation logic
        """
        return []


# ==========================================
# Common Field Models
# ==========================================

class DateRange(MLReadyBaseModel):
    """Date range for experience/education"""
    start: str = Field(default="Unknown", description="Start date")
    end: str = Field(default="Present", description="End date (or 'Present')")
    duration: Optional[str] = Field(default=None, description="Duration string (e.g., '2 years 3 months')")

    def calculate_duration_years(self) -> float:
        """
        Calculate duration in years.

        Returns:
            float: Duration in years (fractional)
        """
        if self.start in ["Unknown", "unknown", None] or self.end in ["Unknown", "unknown", None]:
            return 0.0

        try:
            start_date = self._parse_date(self.start)

            # Handle "Present" as current date
            if self.end.lower() in ["present", "current", "now"]:
                end_date = datetime.now()
            else:
                end_date = self._parse_date(self.end)

            # Calculate difference in years
            delta = end_date - start_date
            years = delta.days / 365.25  # Account for leap years

            return max(0.0, years)  # Ensure non-negative

        except Exception:
            return 0.0

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        """Parse various date formats"""
        try:
            return date_parser.parse(date_str)
        except Exception:
            # If parsing fails, return epoch
            return datetime(1970, 1, 1)

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}start": self.start,
            f"{prefix}end": self.end,
            f"{prefix}duration_years": self.calculate_duration_years(),
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "duration_years": self.calculate_duration_years(),
        }

    def validate_for_ml(self) -> List[str]:
        issues = []
        if self.calculate_duration_years() == 0.0:
            issues.append("Cannot calculate duration from dates")
        return issues


class Location(MLReadyBaseModel):
    """Location information"""
    city: str = Field(default="Unknown", description="City name")
    country: str = Field(default="Unknown", description="Country name")
    remote_preference: RemotePreference = Field(default=RemotePreference.UNKNOWN, description="Remote work preference")

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}city": self.city,
            f"{prefix}country": self.country,
            f"{prefix}remote_preference": self.remote_preference.value if isinstance(self.remote_preference, RemotePreference) else self.remote_preference,
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "country": self.country,
            "remote_preference": self.remote_preference.value if isinstance(self.remote_preference, RemotePreference) else self.remote_preference,
        }

    def validate_for_ml(self) -> List[str]:
        return []


# ==========================================
# Helper Functions
# ==========================================

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Args:
        text: Input text

    Returns:
        Normalized text (lowercase, stripped, deduplicated spaces)
    """
    if not text or text == "Unknown":
        return ""

    # Lowercase and strip
    text = text.lower().strip()

    # Replace multiple spaces with single space
    import re
    text = re.sub(r'\s+', ' ', text)

    return text


def safe_parse_int(value: Any, default: int = 0) -> int:
    """Safely parse integer with fallback"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_parse_float(value: Any, default: float = 0.0) -> float:
    """Safely parse float with fallback"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
