"""
Document parsing module using Gemini API.

This module provides parsers for extracting structured information from
resumes and job descriptions using LLM-based parsing.

Usage:
    from src.parsers import ResumeParser, JobParser

    # Parse resume
    resume_parser = ResumeParser()
    resume = resume_parser.parse_from_docx("resume.docx")

    # Parse job description
    job_parser = JobParser()
    job = job_parser.parse("Job description text...")

    # Quick parsing
    from src.parsers import parse_resume, parse_job_description

    resume = parse_resume("resume.docx")
    job = parse_job_description("Job text...")
"""

# Main parsers
from src.parsers.resume_parser import ResumeParser, parse_resume, parse_resumes_batch
from src.parsers.job_parser import JobParser, parse_job_description

# Supporting classes
from src.parsers.gemini_client import GeminiClient, create_client
from src.parsers.docx_extractor import DocxExtractor

# Base classes (for advanced usage)
from src.parsers.base_parser import BaseParser, ParsingError

__all__ = [
    # Main parsers
    "ResumeParser",
    "JobParser",

    # Convenience functions
    "parse_resume",
    "parse_resumes_batch",
    "parse_job_description",

    # Clients and extractors
    "GeminiClient",
    "create_client",
    "DocxExtractor",

    # Base classes
    "BaseParser",
    "ParsingError",
]
