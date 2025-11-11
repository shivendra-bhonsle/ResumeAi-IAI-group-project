"""
Resume parser using Gemini API.

This module provides the ResumeParser class that combines:
- Text extraction from .docx files
- LLM-based parsing using Gemini
- Validation against Resume schema
- ML-ready output

Usage:
    parser = ResumeParser()
    resume = parser.parse_from_docx("resume.docx")
    features = resume.extract_features()  # For ML scoring
"""

from typing import List, Optional, Type
from pathlib import Path
import logging

from pydantic import BaseModel

from src.parsers.base_parser import BaseParser, ParsingError
from src.parsers.docx_extractor import DocxExtractor
from src.parsers.prompt_templates import get_resume_prompt
from src.parsers.gemini_client import GeminiClient
from src.models import Resume

logger = logging.getLogger(__name__)


class ResumeParser(BaseParser):
    """
    Parser for resume documents.

    Features:
        - Extracts text from .docx files
        - Parses using Gemini API
        - Returns ML-ready Resume objects
        - Handles batch processing
        - Validates and normalizes data

    Example:
        # Parse single resume
        parser = ResumeParser()
        resume = parser.parse_from_docx("resume.docx")

        # Access ML features
        features = resume.extract_features()
        skills = features["skills_list"]
        years_exp = features["years_experience"]

        # For semantic similarity
        text = resume.get_text_for_embedding()
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        max_text_length: int = 10000,
        temperature: float = 0.0,
        include_few_shot: bool = True,
    ):
        """
        Initialize resume parser.

        Args:
            gemini_client: Gemini API client (creates default if not provided)
            max_text_length: Maximum text length to send to LLM
            temperature: Sampling temperature (0.0 = deterministic)
            include_few_shot: Whether to include few-shot example in prompt
        """
        super().__init__(
            gemini_client=gemini_client,
            max_text_length=max_text_length,
            temperature=temperature,
        )

        # DOCX extractor
        self.docx_extractor = DocxExtractor(preserve_formatting=True)

        # Prompt configuration
        self.include_few_shot = include_few_shot

        logger.info("ResumeParser initialized")

    def get_prompt(self, text: str) -> str:
        """
        Generate resume parsing prompt.

        Args:
            text: Cleaned resume text

        Returns:
            str: Complete prompt for Gemini
        """
        return get_resume_prompt(
            resume_text=text,
            include_example=self.include_few_shot
        )

    def get_schema_class(self) -> Type[BaseModel]:
        """
        Get Resume schema class for validation.

        Returns:
            Type[BaseModel]: Resume class
        """
        return Resume

    def post_process(self, resume: Resume) -> Resume:
        """
        Post-process parsed resume.

        Operations:
            - Normalize skill names
            - Validate dates
            - Calculate derived fields
            - Apply business rules

        Args:
            resume: Validated Resume object

        Returns:
            Resume: Post-processed resume
        """
        # Skills are already normalized in the schema
        # Additional post-processing can be added here if needed

        logger.debug(f"Post-processing resume: {resume.personal_info.name}")

        # Log data quality
        completeness = resume.completeness_score()
        logger.debug(f"Resume completeness: {completeness:.1%}")

        # Log validation issues
        issues = resume.validate_for_ml()
        if issues:
            logger.warning(f"Resume validation issues: {issues}")

        return resume

    def parse_from_docx(self, file_path: str) -> Resume:
        """
        Parse resume from .docx file.

        This is the main entry point for parsing resume files.

        Args:
            file_path: Path to .docx file

        Returns:
            Resume: Parsed and validated resume object

        Raises:
            ParsingError: If parsing fails
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid

        Example:
            parser = ResumeParser()
            resume = parser.parse_from_docx("john_doe_resume.docx")

            # Use for ML scoring
            features = resume.extract_features()

            # Check data quality
            if resume.validate_for_ml():
                print("Resume has validation issues")
        """
        logger.info(f"Parsing resume from: {file_path}")

        # Extract text from DOCX
        try:
            text = self.docx_extractor.extract_text(file_path)
            logger.debug(f"Extracted {len(text)} characters from {file_path}")
        except Exception as e:
            raise ParsingError(f"Failed to extract text from {file_path}: {str(e)}") from e

        # Parse text using LLM
        try:
            resume = self.parse(text)
            logger.info(f"Successfully parsed resume: {resume.personal_info.name}")
            return resume
        except Exception as e:
            raise ParsingError(f"Failed to parse resume from {file_path}: {str(e)}") from e

    def parse_batch(self, file_paths: List[str]) -> List[Resume]:
        """
        Parse multiple resumes.

        Args:
            file_paths: List of .docx file paths

        Returns:
            List[Resume]: List of parsed resumes

        Note:
            Failed parses are logged but don't stop batch processing.
            Check return list length vs input list length to detect failures.

        Example:
            parser = ResumeParser()
            resumes = parser.parse_batch([
                "resume1.docx",
                "resume2.docx",
                "resume3.docx"
            ])

            # Process for ML
            for resume in resumes:
                features = resume.extract_features()
                # Send to scoring modules...
        """
        logger.info(f"Batch parsing {len(file_paths)} resumes")

        resumes = []
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Processing {i}/{len(file_paths)}: {Path(file_path).name}")
                resume = self.parse_from_docx(file_path)
                resumes.append(resume)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {str(e)}")
                # Continue with next file
                continue

        logger.info(
            f"Batch parsing complete: {len(resumes)}/{len(file_paths)} successful"
        )

        return resumes

    def parse_from_text(self, text: str, source: str = "unknown") -> Resume:
        """
        Parse resume from raw text.

        Useful for testing or when text is already extracted.

        Args:
            text: Raw resume text
            source: Source identifier (for logging)

        Returns:
            Resume: Parsed resume object

        Example:
            parser = ResumeParser()
            text = "John Doe\\nSoftware Engineer\\n..."
            resume = parser.parse_from_text(text, source="test")
        """
        logger.info(f"Parsing resume from text (source: {source})")

        try:
            resume = self.parse(text)
            return resume
        except Exception as e:
            raise ParsingError(f"Failed to parse text from {source}: {str(e)}") from e


# ==========================================
# Convenience Functions
# ==========================================

def parse_resume(file_path: str) -> Resume:
    """
    Quick function to parse a single resume.

    Args:
        file_path: Path to .docx file

    Returns:
        Resume: Parsed resume

    Example:
        from src.parsers import parse_resume

        resume = parse_resume("resume.docx")
        print(resume.personal_info.name)
        print(f"Experience: {resume.calculate_total_experience()} years")
    """
    parser = ResumeParser()
    return parser.parse_from_docx(file_path)


def parse_resumes_batch(file_paths: List[str]) -> List[Resume]:
    """
    Quick function to parse multiple resumes.

    Args:
        file_paths: List of .docx file paths

    Returns:
        List[Resume]: Parsed resumes

    Example:
        from src.parsers import parse_resumes_batch

        resumes = parse_resumes_batch(["r1.docx", "r2.docx"])
        for resume in resumes:
            features = resume.extract_features()
    """
    parser = ResumeParser()
    return parser.parse_batch(file_paths)


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <path_to_resume.docx>")
        print("\nExample:")
        print("  python resume_parser.py sample_resume.docx")
        sys.exit(1)

    file_path = sys.argv[1]

    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("RESUME PARSER TEST")
    print("=" * 60)
    print(f"File: {file_path}\n")

    try:
        # Parse resume
        parser = ResumeParser()
        resume = parser.parse_from_docx(file_path)

        # Display results
        print("✓ Parsing successful!\n")

        print("-" * 60)
        print("PERSONAL INFO")
        print("-" * 60)
        print(f"Name: {resume.personal_info.name}")
        print(f"Email: {resume.personal_info.email}")
        print(f"Location: {resume.personal_info.location.city}, {resume.personal_info.location.country}")
        print(f"Summary: {resume.personal_info.summary[:100]}..." if len(resume.personal_info.summary) > 100 else f"Summary: {resume.personal_info.summary}")

        print("\n" + "-" * 60)
        print("EXPERIENCE")
        print("-" * 60)
        total_exp = resume.calculate_total_experience()
        print(f"Total Experience: {total_exp:.1f} years")
        print(f"Number of Jobs: {len(resume.experience)}")
        for i, exp in enumerate(resume.experience[:3], 1):  # Show first 3
            print(f"\n{i}. {exp.title} at {exp.company}")
            print(f"   Duration: {exp.get_duration_years():.1f} years")
            print(f"   Responsibilities: {len(exp.responsibilities)}")

        print("\n" + "-" * 60)
        print("EDUCATION")
        print("-" * 60)
        print(f"Highest Level: {resume.get_highest_education_level()}")
        for edu in resume.education:
            print(f"- {edu.degree.level.value}: {edu.degree.field}")
            print(f"  {edu.institution.name}")

        print("\n" + "-" * 60)
        print("SKILLS")
        print("-" * 60)
        skills = resume.skills.get_all_skills_flat()
        print(f"Total Skills: {len(skills)}")
        print(f"Skills: {', '.join(skills[:15])}" + ("..." if len(skills) > 15 else ""))

        print("\n" + "-" * 60)
        print("ML FEATURES (for scoring)")
        print("-" * 60)
        features = resume.extract_features()
        print(f"Years Experience: {features['years_experience']}")
        print(f"Number of Skills: {features['num_skills']}")
        print(f"Education Level: {features['education_level']}")
        print(f"Completeness Score: {features['completeness']:.1%}")

        print("\n" + "-" * 60)
        print("VALIDATION")
        print("-" * 60)
        issues = resume.validate_for_ml()
        if issues:
            print(f"⚠ Validation Issues: {', '.join(issues)}")
        else:
            print("✓ No validation issues - ready for ML scoring!")

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
