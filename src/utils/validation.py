"""
Validation utilities for parsed data.

This module provides validation functions for:
- Email addresses
- Phone numbers
- Dates and date ranges
- Skills
- Experience data
- Education data
"""

import re
from typing import Optional, List
from datetime import datetime


# ==========================================
# Email Validation
# ==========================================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address string

    Returns:
        bool: True if valid email format

    Example:
        >>> is_valid_email("john@example.com")
        True
        >>> is_valid_email("invalid-email")
        False
    """
    if not email or email == "Unknown":
        return False

    # RFC 5322 simplified pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


# ==========================================
# Phone Validation
# ==========================================

def is_valid_phone(phone: str) -> bool:
    """
    Validate phone number format (basic check).

    Args:
        phone: Phone number string

    Returns:
        bool: True if appears to be a valid phone number

    Example:
        >>> is_valid_phone("+1-555-0123")
        True
        >>> is_valid_phone("abc")
        False
    """
    if not phone or phone == "Unknown":
        return False

    # Remove common separators
    digits = re.sub(r'[\s\-\(\)\+\.]', '', phone)

    # Check if it's mostly digits
    # Valid phone should have at least 7 digits
    return len(digits) >= 7 and digits.isdigit()


# ==========================================
# Date Validation
# ==========================================

def is_valid_date(date_str: str) -> bool:
    """
    Check if string represents a valid date.

    Args:
        date_str: Date string

    Returns:
        bool: True if valid date format

    Example:
        >>> is_valid_date("2020-01-15")
        True
        >>> is_valid_date("Present")
        True
        >>> is_valid_date("Invalid")
        False
    """
    if not date_str or date_str == "Unknown":
        return False

    # Special cases
    if date_str.lower() in ["present", "current", "now"]:
        return True

    # Try parsing common formats
    date_formats = [
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]

    for fmt in date_formats:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue

    return False


def is_valid_date_range(start: str, end: str) -> bool:
    """
    Validate date range (start should be before end).

    Args:
        start: Start date
        end: End date

    Returns:
        bool: True if valid range

    Example:
        >>> is_valid_date_range("2020-01", "2022-01")
        True
        >>> is_valid_date_range("2022-01", "2020-01")
        False
    """
    if not is_valid_date(start):
        return False

    if end.lower() in ["present", "current", "now"]:
        return True

    if not is_valid_date(end):
        return False

    # Both are valid dates, check order
    try:
        # Simple year comparison
        start_year = int(re.search(r'\d{4}', start).group())
        end_year = int(re.search(r'\d{4}', end).group())
        return start_year <= end_year
    except (AttributeError, ValueError):
        return True  # Can't determine, assume valid


# ==========================================
# Skills Validation
# ==========================================

def is_valid_skill_name(skill: str) -> bool:
    """
    Validate skill name.

    Args:
        skill: Skill name

    Returns:
        bool: True if valid skill name

    Example:
        >>> is_valid_skill_name("Python")
        True
        >>> is_valid_skill_name("")
        False
    """
    if not skill or skill == "Unknown":
        return False

    # Skill should be at least 2 characters
    # Can contain letters, numbers, spaces, hyphens, dots
    pattern = r'^[a-zA-Z0-9\s\-\.+#]{2,}$'
    return bool(re.match(pattern, skill.strip()))


def validate_skills_list(skills: List[str]) -> List[str]:
    """
    Validate and filter skills list.

    Args:
        skills: List of skill names

    Returns:
        List[str]: Valid skills only

    Example:
        >>> validate_skills_list(["Python", "", "Invalid!", "Java"])
        ["Python", "Java"]
    """
    return [s for s in skills if is_valid_skill_name(s)]


# ==========================================
# Experience Validation
# ==========================================

def validate_experience_years(years: float) -> bool:
    """
    Validate experience years value.

    Args:
        years: Years of experience

    Returns:
        bool: True if valid

    Example:
        >>> validate_experience_years(5.5)
        True
        >>> validate_experience_years(-1)
        False
        >>> validate_experience_years(100)
        False
    """
    # Should be positive and reasonable (< 60 years)
    return 0 <= years <= 60


# ==========================================
# Education Validation
# ==========================================

def is_valid_education_level(level: str) -> bool:
    """
    Validate education level.

    Args:
        level: Education level string

    Returns:
        bool: True if valid education level

    Example:
        >>> is_valid_education_level("bachelor")
        True
        >>> is_valid_education_level("invalid")
        False
    """
    valid_levels = {
        "high_school", "diploma", "associate",
        "bachelor", "master", "doctorate",
        "unknown"
    }

    return level.lower() in valid_levels


# ==========================================
# Completeness Validation
# ==========================================

def check_data_completeness(data: dict, required_fields: List[str]) -> dict:
    """
    Check if required fields are present and not "Unknown".

    Args:
        data: Dictionary to check
        required_fields: List of required field names

    Returns:
        dict: {
            "is_complete": bool,
            "missing_fields": List[str],
            "unknown_fields": List[str]
        }

    Example:
        >>> data = {"name": "John", "email": "Unknown", "phone": "555-0123"}
        >>> check_data_completeness(data, ["name", "email", "skills"])
        {
            "is_complete": False,
            "missing_fields": ["skills"],
            "unknown_fields": ["email"]
        }
    """
    missing = []
    unknown = []

    for field in required_fields:
        if field not in data:
            missing.append(field)
        elif data[field] in ["Unknown", "unknown", None, "", []]:
            unknown.append(field)

    return {
        "is_complete": len(missing) == 0 and len(unknown) == 0,
        "missing_fields": missing,
        "unknown_fields": unknown,
        "completeness_score": (
            (len(required_fields) - len(missing) - len(unknown)) / len(required_fields)
            if required_fields else 1.0
        )
    }


# ==========================================
# JSON Schema Validation
# ==========================================

def validate_json_structure(data: dict, required_keys: List[str]) -> bool:
    """
    Check if JSON has required keys.

    Args:
        data: JSON dictionary
        required_keys: List of required key names

    Returns:
        bool: True if all required keys present

    Example:
        >>> data = {"name": "John", "email": "john@example.com"}
        >>> validate_json_structure(data, ["name", "email"])
        True
        >>> validate_json_structure(data, ["name", "phone"])
        False
    """
    return all(key in data for key in required_keys)


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    print("=== Validation Utilities Test ===\n")

    # Test email validation
    print("Email Validation:")
    emails = ["john@example.com", "invalid-email", "test@domain.co.uk", ""]
    for email in emails:
        print(f"  {email:<25} → {is_valid_email(email)}")

    # Test phone validation
    print("\nPhone Validation:")
    phones = ["+1-555-0123", "555.0123", "1234567890", "abc", ""]
    for phone in phones:
        print(f"  {phone:<25} → {is_valid_phone(phone)}")

    # Test date validation
    print("\nDate Validation:")
    dates = ["2020-01-15", "2020-01", "2020", "Present", "Invalid", ""]
    for date in dates:
        print(f"  {date:<25} → {is_valid_date(date)}")

    # Test skill validation
    print("\nSkill Validation:")
    skills = ["Python", "C++", "Node.js", "", "X", "Invalid!@#"]
    for skill in skills:
        print(f"  {skill:<25} → {is_valid_skill_name(skill)}")

    # Test experience validation
    print("\nExperience Validation:")
    years_list = [5.5, 0, -1, 45, 100]
    for years in years_list:
        print(f"  {years:<25} → {validate_experience_years(years)}")

    # Test completeness check
    print("\nCompleteness Check:")
    data = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "Unknown",
        "skills": []
    }
    required = ["name", "email", "phone", "skills", "experience"]
    result = check_data_completeness(data, required)
    print(f"  Data: {data}")
    print(f"  Required: {required}")
    print(f"  Result: {result}")
