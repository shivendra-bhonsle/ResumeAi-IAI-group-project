"""
Base parser class with common parsing logic.

This module provides an abstract base class that all parsers inherit from,
ensuring consistent behavior and reducing code duplication.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
import logging
from pydantic import BaseModel, ValidationError

from src.parsers.gemini_client import GeminiClient, GeminiError
from src.utils.text_utils import clean_text, truncate_text

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Raised when parsing fails"""
    pass


class ParserValidationError(Exception):
    """Raised when validation fails"""
    pass


class BaseParser(ABC):
    """
    Abstract base class for all parsers.

    Provides common functionality:
        - Text cleaning and preprocessing
        - LLM-based parsing with retry
        - JSON validation against Pydantic schema
        - Post-processing and normalization
        - Error handling and logging

    Subclasses must implement:
        - get_prompt(text) - Generate parsing prompt
        - get_schema_class() - Return Pydantic model class
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        max_text_length: int = 10000,
        temperature: float = 0.0,
    ):
        """
        Initialize base parser.

        Args:
            gemini_client: Gemini API client (creates default if not provided)
            max_text_length: Maximum text length to send to LLM
            temperature: Sampling temperature for generation
        """
        # Gemini client
        if gemini_client:
            self.client = gemini_client
        else:
            from src.parsers.gemini_client import create_client
            self.client = create_client()

        # Configuration
        self.max_text_length = max_text_length
        self.temperature = temperature

        # Statistics
        self.total_parses = 0
        self.successful_parses = 0
        self.failed_parses = 0

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def get_prompt(self, text: str) -> str:
        """
        Generate parsing prompt for the given text.

        Args:
            text: Cleaned text to parse

        Returns:
            str: Complete prompt for LLM

        Note: Must be implemented by subclasses
        """
        raise NotImplementedError

    @abstractmethod
    def get_schema_class(self) -> Type[BaseModel]:
        """
        Get Pydantic model class for validation.

        Returns:
            Type[BaseModel]: Pydantic model class (Resume or JobDescription)

        Note: Must be implemented by subclasses
        """
        raise NotImplementedError

    def parse(self, text: str) -> BaseModel:
        """
        Parse text into structured data.

        This is the main method that orchestrates the entire parsing process:
        1. Preprocess text
        2. Generate prompt
        3. Call LLM
        4. Parse JSON response
        5. Validate against schema
        6. Post-process
        7. Return structured object

        Args:
            text: Raw text to parse

        Returns:
            BaseModel: Parsed and validated data (Resume or JobDescription)

        Raises:
            ParsingError: If parsing fails
            ValidationError: If validation fails
        """
        self.total_parses += 1

        try:
            # Step 1: Preprocess text
            cleaned_text = self.preprocess_text(text)

            if not cleaned_text:
                raise ParsingError("Text is empty after preprocessing")

            # Step 2: Generate prompt
            prompt = self.get_prompt(cleaned_text)

            logger.debug(f"Generated prompt ({len(prompt)} chars)")

            # Step 3: Call LLM and parse JSON
            try:
                parsed_dict = self.client.generate_and_parse_json(
                    prompt=prompt,
                    temperature=self.temperature,
                )
            except GeminiError as e:
                raise ParsingError(f"LLM generation failed: {str(e)}") from e

            logger.debug(f"Received JSON response with {len(parsed_dict)} top-level keys")

            # Step 4: Validate against schema
            try:
                validated_obj = self.validate_output(parsed_dict)
            except ValidationError as e:
                raise ParserValidationError(f"Validation failed: {str(e)}") from e

            # Step 5: Post-process
            final_obj = self.post_process(validated_obj)

            # Success
            self.successful_parses += 1
            logger.info(f"Successfully parsed text ({self.successful_parses}/{self.total_parses} success rate)")

            return final_obj

        except Exception as e:
            self.failed_parses += 1
            logger.error(f"Parsing failed: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before parsing.

        Operations:
            - Clean text (remove extra whitespace, fix encoding)
            - Truncate to max length
            - Additional preprocessing (can be overridden)

        Args:
            text: Raw text

        Returns:
            str: Cleaned and preprocessed text
        """
        # Clean text
        cleaned = clean_text(text)

        # Truncate if too long
        if len(cleaned) > self.max_text_length:
            logger.warning(
                f"Text length ({len(cleaned)}) exceeds maximum ({self.max_text_length}). "
                "Truncating..."
            )
            cleaned = truncate_text(cleaned, self.max_text_length)

        return cleaned

    def normalize_enum_values(self, data: Any) -> Any:
        """
        Recursively normalize enum string values to lowercase.

        LLMs sometimes return "Unknown" instead of "unknown" for enum fields.
        This method recursively walks through the data structure and lowercases
        all string values to match enum definitions.

        Args:
            data: Dictionary, list, or primitive value

        Returns:
            Normalized data with lowercase string values
        """
        if isinstance(data, dict):
            return {key: self.normalize_enum_values(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.normalize_enum_values(item) for item in data]
        elif isinstance(data, str):
            return data.lower()
        else:
            return data

    def validate_output(self, parsed_dict: Dict[str, Any]) -> BaseModel:
        """
        Validate parsed dictionary against Pydantic schema.

        Args:
            parsed_dict: Dictionary from LLM

        Returns:
            BaseModel: Validated Pydantic object

        Raises:
            ValidationError: If validation fails
        """
        schema_class = self.get_schema_class()

        try:
            # Normalize enum values (lowercase all strings)
            normalized_dict = self.normalize_enum_values(parsed_dict)

            # Pydantic validation
            validated_obj = schema_class(**normalized_dict)
            logger.debug("Validation successful")
            return validated_obj

        except ValidationError as e:
            logger.error(f"Validation errors: {e.errors()}")
            # Format error details for better debugging
            error_details = []
            for err in e.errors()[:5]:  # Show first 5 errors
                loc = ".".join(str(x) for x in err.get('loc', []))
                msg = err.get('msg', 'Unknown error')
                error_details.append(f"{loc}: {msg}")

            error_msg = (
                f"Schema validation failed with {len(e.errors())} error(s):\n"
                + "\n".join(f"  - {detail}" for detail in error_details)
            )
            raise ParserValidationError(error_msg) from e

    def post_process(self, obj: BaseModel) -> BaseModel:
        """
        Post-process validated object.

        Override this method in subclasses for custom post-processing:
            - Additional normalization
            - Derived field calculation
            - Business logic application

        Args:
            obj: Validated Pydantic object

        Returns:
            BaseModel: Post-processed object
        """
        # Default: no post-processing
        # Subclasses can override
        return obj

    def get_stats(self) -> Dict[str, Any]:
        """
        Get parser statistics.

        Returns:
            Dict with parsing statistics
        """
        return {
            "total_parses": self.total_parses,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "success_rate": (
                self.successful_parses / self.total_parses * 100
                if self.total_parses > 0
                else 0.0
            ),
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.total_parses = 0
        self.successful_parses = 0
        self.failed_parses = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_text_length={self.max_text_length}, "
            f"temperature={self.temperature})"
        )


# ==========================================
# Helper Functions
# ==========================================

def safe_parse(
    parser: BaseParser,
    text: str,
    fallback_value: Optional[BaseModel] = None,
) -> Optional[BaseModel]:
    """
    Safely parse text with fallback on error.

    Args:
        parser: Parser instance
        text: Text to parse
        fallback_value: Value to return if parsing fails

    Returns:
        Optional[BaseModel]: Parsed object or fallback value
    """
    try:
        return parser.parse(text)
    except Exception as e:
        logger.error(f"Parsing failed, using fallback: {e}")
        return fallback_value


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    print("BaseParser is an abstract class and cannot be instantiated directly.")
    print("Use ResumeParser or JobParser instead.")
    print("\nExample:")
    print("  from src.parsers import ResumeParser")
    print("  parser = ResumeParser()")
    print("  resume = parser.parse_from_docx('resume.docx')")
