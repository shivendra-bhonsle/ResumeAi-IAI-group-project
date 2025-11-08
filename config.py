"""
ResumeAI Central Configuration
This file contains all configuration settings for the resume ranking system.
Loads environment variables from .env file and provides defaults.
"""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===========================================
# Project Paths
# ===========================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_RESUMES_DIR = PROJECT_ROOT / "data" / "sample_resumes"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# ===========================================
# API Keys
# ===========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables!")
    print("Please set it in your .env file. See .env.example for reference.")

# ===========================================
# Model Configuration
# ===========================================
# Gemini model for document parsing
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Sentence transformer model for semantic similarity
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ===========================================
# Scoring Weights
# ===========================================
# These weights determine the importance of each scoring component
# Must sum to 1.0
WEIGHTS: Dict[str, float] = {
    "skills": float(os.getenv("WEIGHT_SKILLS", "0.30")),
    "experience": float(os.getenv("WEIGHT_EXPERIENCE", "0.25")),
    "semantic": float(os.getenv("WEIGHT_SEMANTIC", "0.25")),
    "education": float(os.getenv("WEIGHT_EDUCATION", "0.15")),
    "location": float(os.getenv("WEIGHT_LOCATION", "0.05")),
}

# Validate weights sum to 1.0
weights_sum = sum(WEIGHTS.values())
if not 0.99 <= weights_sum <= 1.01:  # Allow small floating point errors
    print(f"WARNING: Scoring weights sum to {weights_sum}, not 1.0")
    print("Weights will be normalized automatically.")
    # Normalize weights
    total = sum(WEIGHTS.values())
    WEIGHTS = {k: v / total for k, v in WEIGHTS.items()}


# ===========================================
# Helper Functions
# ===========================================
def get_config_summary() -> Dict:
    """Return a summary of current configuration."""
    return {
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "weights": WEIGHTS,
    }


def validate_config() -> bool:
    """Validate configuration settings."""
    issues = []

    # Check API key
    if not GEMINI_API_KEY:
        issues.append("GEMINI_API_KEY is not set")

    # Check weights
    if not 0.99 <= sum(WEIGHTS.values()) <= 1.01:
        issues.append(f"Weights sum to {sum(WEIGHTS.values())}, not 1.0")

    if issues:
        print("Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


if __name__ == "__main__":
    # Run when executed directly to validate configuration
    print("ResumeAI Configuration")
    print("=" * 50)

    import json
    summary = get_config_summary()
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 50)
    print("Validating configuration...")
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has issues")
