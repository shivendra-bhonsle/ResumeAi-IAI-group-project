"""
Text preprocessing and normalization utilities.

This module provides utilities for cleaning and normalizing text
from resumes and job descriptions before LLM parsing.
"""

import re
from typing import List, Optional
from datetime import datetime
import logging

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

logger = logging.getLogger(__name__)


# ==========================================
# Text Cleaning
# ==========================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text for LLM processing.

    Operations:
        - Remove extra whitespace
        - Fix common encoding issues
        - Normalize line breaks
        - Remove special characters that break parsing

    Args:
        text: Raw text to clean

    Returns:
        str: Cleaned text

    Example:
        >>> clean_text("Hello    World\\n\\n\\n")
        "Hello World\\n"
    """
    if not text:
        return ""

    # Fix common encoding issues
    text = text.replace('\u2019', "'")  # Smart apostrophe
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')  # Smart quotes
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2022', '•')  # Bullet
    text = text.replace('\xa0', ' ')    # Non-breaking space

    # Normalize line breaks (convert multiple to double)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Replace multiple spaces with single space (but preserve single newlines)
    lines = text.split('\n')
    lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    text = '\n'.join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace while preserving structure.

    Args:
        text: Input text

    Returns:
        str: Text with normalized whitespace
    """
    if not text:
        return ""

    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


def truncate_text(text: str, max_length: int = 10000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Useful for limiting token count in LLM requests.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


# ==========================================
# Date Parsing and Normalization
# ==========================================

def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize various date formats to ISO format (YYYY-MM-DD).

    Handles formats like:
        - "Jan 2020" → "2020-01-01"
        - "January 2020" → "2020-01-01"
        - "2020-01" → "2020-01-01"
        - "2020" → "2020-01-01"
        - "Present" → "Present"

    Args:
        date_str: Date string in various formats

    Returns:
        Optional[str]: Normalized date or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()

    # Handle special cases
    if date_str.lower() in ["present", "current", "now", "ongoing"]:
        return "Present"

    if date_str.lower() in ["unknown", "n/a", "na", ""]:
        return None

    # Try parsing with dateutil
    if date_parser:
        try:
            parsed_date = date_parser.parse(date_str, fuzzy=True)
            return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug(f"Could not parse date '{date_str}': {e}")

    # Fallback: Try to extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return f"{year_match.group(0)}-01-01"

    return None


def parse_date_range(date_range_str: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse date range from various formats.

    Examples:
        "Jan 2020 - Dec 2022" → ("2020-01-01", "2022-12-01")
        "2020 - Present" → ("2020-01-01", "Present")
        "2020-01 to 2022-06" → ("2020-01-01", "2022-06-01")

    Args:
        date_range_str: Date range string

    Returns:
        tuple: (start_date, end_date) or (None, None)
    """
    if not date_range_str:
        return None, None

    # Split on common separators
    separators = [' - ', ' – ', ' to ', ' - ', '-', '–']

    start_str, end_str = None, None

    for sep in separators:
        if sep in date_range_str:
            parts = date_range_str.split(sep, 1)
            if len(parts) == 2:
                start_str, end_str = parts
                break

    if not start_str:
        # No separator found, treat as single date
        start_str = date_range_str
        end_str = "Present"

    # Normalize both dates
    start_date = normalize_date(start_str)
    end_date = normalize_date(end_str)

    return start_date, end_date


def calculate_duration(start_date: str, end_date: str) -> float:
    """
    Calculate duration in years between two dates.

    Args:
        start_date: Start date (ISO format or "Present")
        end_date: End date (ISO format or "Present")

    Returns:
        float: Duration in years (fractional)
    """
    if not start_date or start_date == "Unknown":
        return 0.0

    try:
        start = datetime.fromisoformat(start_date.replace("Present", str(datetime.now().date())))

        if end_date and end_date != "Unknown":
            if end_date.lower() in ["present", "current"]:
                end = datetime.now()
            else:
                end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        delta = end - start
        years = delta.days / 365.25

        return max(0.0, years)

    except Exception as e:
        logger.debug(f"Could not calculate duration: {e}")
        return 0.0


# ==========================================
# Skill Normalization
# ==========================================

def normalize_skill(skill: str) -> str:
    """
    Normalize skill name for consistent matching.

    Operations:
        - Lowercase
        - Remove special characters
        - Trim whitespace
        - Handle common variations

    Args:
        skill: Skill name

    Returns:
        str: Normalized skill name

    Example:
        >>> normalize_skill("  Python 3.x  ")
        "python"
        >>> normalize_skill("Node.js")
        "nodejs"
    """
    if not skill:
        return ""

    # Lowercase
    skill = skill.lower().strip()

    # Remove version numbers
    skill = re.sub(r'\d+(\.\d+)*\+?', '', skill)

    # Remove special characters except spaces and hyphens
    skill = re.sub(r'[^\w\s-]', '', skill)

    # Replace spaces and hyphens with nothing (e.g., "node.js" → "nodejs")
    skill = re.sub(r'[\s-]', '', skill)

    # Trim again
    skill = skill.strip()

    return skill


def normalize_skills_list(skills: List[str]) -> List[str]:
    """
    Normalize list of skills and remove duplicates.

    Args:
        skills: List of skill names

    Returns:
        List[str]: Normalized, deduplicated, sorted list

    Example:
        >>> normalize_skills_list(["Python 3", "  python  ", "Java"])
        ["java", "python"]
    """
    if not skills:
        return []

    # Normalize each skill
    normalized = [normalize_skill(skill) for skill in skills]

    # Remove empty strings
    normalized = [s for s in normalized if s]

    # Remove duplicates and sort
    return sorted(list(set(normalized)))


def apply_skill_synonyms(skills: List[str], synonym_dict: dict) -> List[str]:
    """
    Apply skill synonyms for better matching.

    Example:
        synonym_dict = {"javascript": ["js", "javascript", "ecmascript"]}
        apply_skill_synonyms(["js"], synonym_dict) → ["javascript"]

    Args:
        skills: List of skills
        synonym_dict: Dict mapping canonical skill to list of synonyms

    Returns:
        List[str]: Skills with synonyms applied
    """
    canonical_skills = set()

    for skill in skills:
        skill_normalized = normalize_skill(skill)

        # Find canonical name
        canonical = skill_normalized
        for canonical_name, synonyms in synonym_dict.items():
            if skill_normalized in [normalize_skill(s) for s in synonyms]:
                canonical = canonical_name
                break

        canonical_skills.add(canonical)

    return sorted(list(canonical_skills))


# ==========================================
# Experience Parsing
# ==========================================

def parse_experience_years(exp_str: str) -> float:
    """
    Parse experience years from various string formats.

    Examples:
        "5 years" → 5.0
        "5+ years" → 5.0
        "3-5 years" → 4.0 (average)
        "6 months" → 0.5

    Args:
        exp_str: Experience string

    Returns:
        float: Years of experience
    """
    if not exp_str:
        return 0.0

    exp_str = exp_str.lower().strip()

    # Handle range (e.g., "3-5 years")
    range_match = re.search(r'(\d+)\s*[-to]+\s*(\d+)', exp_str)
    if range_match:
        min_years = float(range_match.group(1))
        max_years = float(range_match.group(2))
        return (min_years + max_years) / 2

    # Handle "+ years" (e.g., "5+ years")
    plus_match = re.search(r'(\d+)\+', exp_str)
    if plus_match:
        return float(plus_match.group(1))

    # Handle "X months"
    months_match = re.search(r'(\d+)\s*months?', exp_str)
    if months_match:
        return float(months_match.group(1)) / 12

    # Handle "X years"
    years_match = re.search(r'(\d+)\s*years?', exp_str)
    if years_match:
        return float(years_match.group(1))

    # Try to extract any number
    number_match = re.search(r'\d+', exp_str)
    if number_match:
        return float(number_match.group(0))

    return 0.0


# ==========================================
# Email and Phone Validation
# ==========================================

def is_valid_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address

    Returns:
        bool: True if valid format
    """
    if not email:
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def extract_email(text: str) -> Optional[str]:
    """
    Extract email address from text.

    Args:
        text: Text containing email

    Returns:
        Optional[str]: First email found or None
    """
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return None


def normalize_phone(phone: str) -> str:
    """
    Normalize phone number format.

    Args:
        phone: Phone number in any format

    Returns:
        str: Normalized phone number
    """
    if not phone:
        return ""

    # Remove all non-digit characters except +
    phone = re.sub(r'[^\d+]', '', phone)

    return phone


# ==========================================
# URL Extraction
# ==========================================

def extract_url(text: str, domain: Optional[str] = None) -> Optional[str]:
    """
    Extract URL from text, optionally filtered by domain.

    Args:
        text: Text containing URL
        domain: Optional domain filter (e.g., "linkedin.com")

    Returns:
        Optional[str]: First URL found or None
    """
    pattern = r'https?://[^\s]+'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    if domain:
        for url in matches:
            if domain in url:
                return url
        return None

    return matches[0]


def extract_linkedin_url(text: str) -> Optional[str]:
    """Extract LinkedIn profile URL"""
    return extract_url(text, domain="linkedin.com")


def extract_github_url(text: str) -> Optional[str]:
    """Extract GitHub profile URL"""
    return extract_url(text, domain="github.com")


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    # Test text cleaning
    print("Testing text cleaning:")
    dirty_text = "Hello    World\n\n\n\nTest"
    print(f"Original: {repr(dirty_text)}")
    print(f"Cleaned: {repr(clean_text(dirty_text))}")

    # Test date parsing
    print("\nTesting date parsing:")
    dates = ["Jan 2020", "2020-01-15", "Present", "Unknown"]
    for date in dates:
        print(f"{date} → {normalize_date(date)}")

    # Test skill normalization
    print("\nTesting skill normalization:")
    skills = ["  Python 3.x  ", "Node.js", "JAVA", "python"]
    print(f"Original: {skills}")
    print(f"Normalized: {normalize_skills_list(skills)}")

    # Test experience parsing
    print("\nTesting experience parsing:")
    exp_strings = ["5 years", "3-5 years", "5+ years", "6 months"]
    for exp in exp_strings:
        print(f"{exp} → {parse_experience_years(exp)} years")
