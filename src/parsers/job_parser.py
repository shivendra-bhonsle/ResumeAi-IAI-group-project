"""
Job description parser using Gemini API.

This module provides the JobParser class that parses job descriptions
into structured JobDescription objects ready for ML scoring.

Usage:
    parser = JobParser()
    job = parser.parse("Senior Python Developer needed. 5+ years exp...")
    features = job.extract_features()  # For ML scoring
"""

from typing import Optional, Type
import logging

from pydantic import BaseModel

from src.parsers.base_parser import BaseParser, ParsingError
from src.parsers.prompt_templates import get_job_prompt
from src.parsers.gemini_client import GeminiClient
from src.models import JobDescription

logger = logging.getLogger(__name__)


class JobParser(BaseParser):
    """
    Parser for job descriptions.

    Features:
        - Parses plain text job descriptions
        - Returns ML-ready JobDescription objects
        - Extracts requirements (skills, experience, education)
        - Validates and normalizes data

    Example:
        # Parse job description
        parser = JobParser()

        job_text = '''
        Senior Python Developer
        5+ years of experience required.
        Skills: Python, Django, PostgreSQL, AWS
        Education: Bachelor's degree in CS
        Location: San Francisco (Remote OK)
        '''

        job = parser.parse(job_text)

        # Access ML features
        features = job.extract_features()
        required_skills = features["required_skills"]
        required_years = features["required_years"]

        # For semantic similarity
        text = job.get_text_for_embedding()
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        max_text_length: int = 10000,
        temperature: float = 0.0,
    ):
        """
        Initialize job parser.

        Args:
            gemini_client: Gemini API client (creates default if not provided)
            max_text_length: Maximum text length to send to LLM
            temperature: Sampling temperature (0.0 = deterministic)
        """
        super().__init__(
            gemini_client=gemini_client,
            max_text_length=max_text_length,
            temperature=temperature,
        )

        logger.info("JobParser initialized")

    def get_prompt(self, text: str) -> str:
        """
        Generate job description parsing prompt.

        Args:
            text: Cleaned job description text

        Returns:
            str: Complete prompt for Gemini
        """
        return get_job_prompt(job_text=text, include_example=False)

    def get_schema_class(self) -> Type[BaseModel]:
        """
        Get JobDescription schema class for validation.

        Returns:
            Type[BaseModel]: JobDescription class
        """
        return JobDescription

    def post_process(self, job: JobDescription) -> JobDescription:
        """
        Post-process parsed job description.

        Operations:
            - Normalize skill names
            - Validate requirements
            - Apply business rules

        Args:
            job: Validated JobDescription object

        Returns:
            JobDescription: Post-processed job description
        """
        logger.debug(f"Post-processing job: {job.title}")

        # Log validation issues
        issues = job.validate_for_ml()
        if issues:
            logger.warning(f"Job validation issues: {issues}")

        return job

    def parse_from_file(self, file_path: str) -> JobDescription:
        """
        Parse job description from text file.

        Args:
            file_path: Path to text file containing job description

        Returns:
            JobDescription: Parsed job description

        Raises:
            ParsingError: If parsing fails
            FileNotFoundError: If file doesn't exist

        Example:
            parser = JobParser()
            job = parser.parse_from_file("job_description.txt")
        """
        logger.info(f"Parsing job description from: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ParsingError(f"Failed to read file {file_path}: {str(e)}") from e

        return self.parse(text)


# ==========================================
# Convenience Functions
# ==========================================

def parse_job_description(text: str) -> JobDescription:
    """
    Quick function to parse a job description.

    Args:
        text: Job description text

    Returns:
        JobDescription: Parsed job description

    Example:
        from src.parsers import parse_job_description

        job_text = "Senior Developer needed. 5+ years exp. Python, AWS."
        job = parse_job_description(job_text)

        print(f"Required years: {job.required_experience.get_target_years()}")
        print(f"Skills: {job.required_skills.get_all_required_skills()}")
    """
    parser = JobParser()
    return parser.parse(text)


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("JOB DESCRIPTION PARSER TEST")
    print("=" * 60)

    # Sample job description
    sample_job = """
Senior Python Developer

We are seeking an experienced Python developer to join our backend team.

Requirements:
- 5+ years of professional software development experience
- Strong expertise in Python and Django framework
- Experience with PostgreSQL and database optimization
- AWS cloud platform experience
- Strong understanding of RESTful API design
- Experience with Docker and containerization

Nice to have:
- React or frontend development experience
- Experience with microservices architecture
- Kubernetes knowledge

Education:
Bachelor's degree in Computer Science or related field

Location: San Francisco, CA
Remote work: Hybrid (3 days in office)

Salary: $120,000 - $160,000 per year

About the role:
You will be responsible for designing and implementing scalable backend services,
working closely with the frontend team, and mentoring junior developers.
    """

    print("\nJob Description Text:")
    print("-" * 60)
    print(sample_job)
    print()

    try:
        # Parse job description
        parser = JobParser()
        job = parser.parse(sample_job)

        # Display results
        print("✓ Parsing successful!\n")

        print("-" * 60)
        print("JOB INFORMATION")
        print("-" * 60)
        print(f"Title: {job.title}")
        print(f"Role: {job.role}")
        print(f"Company: {job.company_name or 'Not specified'}")

        print("\n" + "-" * 60)
        print("REQUIREMENTS")
        print("-" * 60)

        # Skills
        required_skills = job.required_skills.get_all_required_skills()
        nice_to_have = job.required_skills.get_all_nice_to_have_skills()
        print(f"\nRequired Skills ({len(required_skills)}):")
        print(f"  {', '.join(required_skills)}")
        if nice_to_have:
            print(f"\nNice-to-have Skills ({len(nice_to_have)}):")
            print(f"  {', '.join(nice_to_have)}")

        # Experience
        print(f"\nExperience Required:")
        print(f"  Minimum: {job.required_experience.min_years} years")
        if job.required_experience.max_years:
            print(f"  Maximum: {job.required_experience.max_years} years")
        print(f"  Target: {job.required_experience.get_target_years()} years")

        # Education
        print(f"\nEducation Required:")
        print(f"  Level: {job.education_requirement.min_level.value}")
        if job.education_requirement.field:
            print(f"  Field: {job.education_requirement.field}")

        # Location
        print(f"\nLocation:")
        print(f"  Location: {job.location_requirement.location}")
        print(f"  Remote Allowed: {job.location_requirement.remote_allowed}")
        print(f"  Hybrid: {job.location_requirement.hybrid}")

        # Salary
        if job.salary_range:
            print(f"\nSalary Range: {job.salary_range}")

        print("\n" + "-" * 60)
        print("RESPONSIBILITIES")
        print("-" * 60)
        for i, resp in enumerate(job.responsibilities[:5], 1):  # Show first 5
            print(f"{i}. {resp}")
        if len(job.responsibilities) > 5:
            print(f"   ... and {len(job.responsibilities) - 5} more")

        print("\n" + "-" * 60)
        print("ML FEATURES (for scoring)")
        print("-" * 60)
        features = job.extract_features()
        print(f"Required Skills: {len(features['required_skills'])}")
        print(f"Required Years: {features['required_years']}")
        print(f"Education Level (numeric): {features['required_education_level']}")
        print(f"Remote Allowed: {features['remote_allowed']}")

        print("\n" + "-" * 60)
        print("VALIDATION")
        print("-" * 60)
        issues = job.validate_for_ml()
        if issues:
            print(f"⚠ Validation Issues: {', '.join(issues)}")
        else:
            print("✓ No validation issues - ready for ML scoring!")

        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        summary = job.get_requirements_summary()
        print(summary)

        print("\n" + "-" * 60)
        print("PARSER STATISTICS")
        print("-" * 60)
        stats = parser.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"\n✗ Parsing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
