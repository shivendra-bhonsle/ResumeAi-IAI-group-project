"""
Gemini API client with retry logic and error handling.

This module provides a robust client for interacting with Google's Gemini API,
specifically optimized for document parsing with structured JSON output.
"""

import json
import time
import logging
from typing import Optional, Dict, Any
from enum import Enum

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google-generativeai is not installed. "
        "Install it with: pip install google-generativeai"
    )

import config

# Set up logging
logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Base exception for Gemini API errors"""
    pass


class GeminiRateLimitError(GeminiError):
    """Raised when rate limit is exceeded"""
    pass


class GeminiTimeoutError(GeminiError):
    """Raised when request times out"""
    pass


class GeminiInvalidResponseError(GeminiError):
    """Raised when response is invalid or malformed"""
    pass


class RetryStrategy(Enum):
    """Retry strategies for failed requests"""
    EXPONENTIAL_BACKOFF = "exponential"
    LINEAR_BACKOFF = "linear"
    NO_RETRY = "none"


class GeminiClient:
    """
    Robust Gemini API client with retry logic.

    Features:
        - Automatic retry with exponential backoff
        - Rate limiting handling
        - Timeout management
        - JSON response parsing and validation
        - Comprehensive error handling
        - Request logging

    Example:
        client = GeminiClient(api_key="your_key")
        response = client.generate_with_retry(
            prompt="Extract name from: John Doe",
            temperature=0.0
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    ):
        """
        Initialize Gemini API client.

        Args:
            api_key: Gemini API key (uses config.GEMINI_API_KEY if not provided)
            model_name: Model name (uses config.GEMINI_MODEL if not provided)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            retry_strategy: Strategy for retry delays
        """
        # Get API key from config or parameter
        self.api_key = api_key or config.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY in .env or pass api_key parameter."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Model configuration
        self.model_name = model_name or config.GEMINI_MODEL
        self.model = genai.GenerativeModel(self.model_name)

        # Retry configuration
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_strategy = retry_strategy

        # Statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0

        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    def generate_with_retry(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate response with automatic retry logic.

        Args:
            prompt: The prompt to send to Gemini
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens in response
            system_instruction: System-level instruction for the model

        Returns:
            str: Generated text response

        Raises:
            GeminiRateLimitError: If rate limit exceeded after retries
            GeminiTimeoutError: If request times out after retries
            GeminiError: For other API errors
        """
        self.total_requests += 1

        for attempt in range(self.max_retries + 1):
            try:
                # Log attempt
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{self.max_retries}")
                    self.retried_requests += 1

                # Configure generation
                generation_config = {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                }

                if max_output_tokens:
                    generation_config["max_output_tokens"] = max_output_tokens

                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )

                # Check if response has text
                if not response.text:
                    raise GeminiInvalidResponseError("Empty response from Gemini")

                logger.debug(f"Successfully generated response (attempt {attempt + 1})")
                return response.text

            except Exception as e:
                error_message = str(e).lower()

                # Rate limit error
                if "429" in error_message or "quota" in error_message or "rate limit" in error_message:
                    if attempt < self.max_retries:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Rate limit hit. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        self.failed_requests += 1
                        raise GeminiRateLimitError(
                            f"Rate limit exceeded after {self.max_retries} retries"
                        ) from e

                # Timeout error
                elif "timeout" in error_message:
                    if attempt < self.max_retries:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Timeout error. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        self.failed_requests += 1
                        raise GeminiTimeoutError(
                            f"Request timeout after {self.max_retries} retries"
                        ) from e

                # Server error (500, 503)
                elif "500" in error_message or "503" in error_message or "server" in error_message:
                    if attempt < self.max_retries:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"Server error. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        self.failed_requests += 1
                        raise GeminiError(
                            f"Server error after {self.max_retries} retries: {e}"
                        ) from e

                # Other errors - don't retry
                else:
                    self.failed_requests += 1
                    logger.error(f"Gemini API error: {e}")
                    raise GeminiError(f"API error: {e}") from e

        # Should not reach here
        self.failed_requests += 1
        raise GeminiError("Max retries exceeded")

    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from Gemini response.

        Handles cases where LLM adds text before/after JSON.

        Args:
            response_text: Raw response text from Gemini

        Returns:
            Dict: Parsed JSON object

        Raises:
            GeminiInvalidResponseError: If JSON cannot be parsed
        """
        # Try direct parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        if "```json" in response_text:
            try:
                # Extract content between ```json and ```
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to find JSON object in text
        try:
            # Look for { ... } pattern
            start = response_text.find("{")
            end = response_text.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Failed to parse
        logger.error(f"Could not parse JSON from response: {response_text[:200]}...")
        raise GeminiInvalidResponseError(
            "Response does not contain valid JSON. "
            "Response preview: " + response_text[:200]
        )

    def generate_and_parse_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate response and parse as JSON (convenience method).

        Args:
            prompt: The prompt to send
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens

        Returns:
            Dict: Parsed JSON object

        Raises:
            GeminiError: If generation fails
            GeminiInvalidResponseError: If response is not valid JSON
        """
        # Generate response
        response_text = self.generate_with_retry(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Parse JSON
        return self.parse_json_response(response_text)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay based on attempt number and strategy.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            float: Delay in seconds
        """
        if self.retry_strategy == RetryStrategy.NO_RETRY:
            return 0.0

        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            # Linear: 2s, 4s, 6s, 8s...
            return 2.0 * (attempt + 1)

        else:  # EXPONENTIAL_BACKOFF (default)
            # Exponential: 2s, 4s, 8s, 16s...
            return min(2.0 ** (attempt + 1), 30.0)  # Cap at 30 seconds

    def get_stats(self) -> Dict[str, int]:
        """
        Get client statistics.

        Returns:
            Dict with request statistics
        """
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "retried_requests": self.retried_requests,
            "success_rate": (
                (self.total_requests - self.failed_requests) / self.total_requests * 100
                if self.total_requests > 0
                else 0.0
            ),
        }

    def reset_stats(self):
        """Reset statistics counters"""
        self.total_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0

    def __repr__(self) -> str:
        return (
            f"GeminiClient(model={self.model_name}, "
            f"max_retries={self.max_retries}, "
            f"timeout={self.timeout})"
        )


# ==========================================
# Convenience Functions
# ==========================================

def create_client(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GeminiClient:
    """
    Create a Gemini client with default configuration.

    Args:
        api_key: Optional API key (uses config if not provided)
        model_name: Optional model name (uses config if not provided)

    Returns:
        GeminiClient: Configured client instance
    """
    return GeminiClient(
        api_key=api_key,
        model_name=model_name,
        max_retries=3,
        timeout=30,
    )


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    # Test the client
    import sys

    # Check if API key is configured
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in config")
        print("Set it in your .env file")
        sys.exit(1)

    # Create client
    client = create_client()
    print(f"Created client: {client}")

    # Test simple generation
    print("\n" + "=" * 50)
    print("Test 1: Simple text generation")
    print("=" * 50)

    try:
        response = client.generate_with_retry(
            prompt="Say 'Hello, World!' and nothing else.",
            temperature=0.0,
        )
        print(f"Response: {response}")
    except GeminiError as e:
        print(f"Error: {e}")

    # Test JSON generation
    print("\n" + "=" * 50)
    print("Test 2: JSON generation and parsing")
    print("=" * 50)

    try:
        response = client.generate_and_parse_json(
            prompt="""
Extract information from this text and return as JSON:

Text: "John Doe is a software engineer with 5 years of experience in Python."

Return JSON with fields: name, role, years_experience

Return ONLY the JSON object, nothing else.
            """,
            temperature=0.0,
        )
        print(f"Parsed JSON: {json.dumps(response, indent=2)}")
    except GeminiError as e:
        print(f"Error: {e}")

    # Show statistics
    print("\n" + "=" * 50)
    print("Client Statistics:")
    print("=" * 50)
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
