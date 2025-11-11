# ResumeAI Integration Guide
**Document Parsing & LLM Integration Module - Complete**

> **Author**: Person 1 (Shivendra)
> **Status**: ✅ Ready for Team Integration
> **Last Updated**: November 2025

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Data Models (Schemas)](#data-models-schemas)
5. [Parser Usage](#parser-usage)
6. [Integration Guide by Team Member](#integration-guide-by-team-member)
7. [Testing](#testing)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This module provides **complete document parsing** functionality for the ResumeAI system using the **Gemini API**. It extracts structured data from:
- **Resumes** (`.docx` format)
- **Job Descriptions** (`.txt` format or raw text)

### Key Features
✅ LLM-based parsing with Gemini 2.5 Flash-Lite
✅ ML-ready data schemas with feature extraction methods
✅ Robust error handling with retry logic
✅ Batch processing support (15 resumes in parallel)
✅ Data validation and quality scoring
✅ Easy integration for ML scoring modules

### Performance
- **Speed**: ~9 seconds per resume (gemini-2.5-flash-lite)
- **Batch**: 50 resumes in ~7.5 minutes sequential, ~30-40 seconds parallel
- **Cost**: ~$0.01 for 50 resumes (essentially free)
- **Accuracy**: 83%+ data completeness

---

## Quick Start

### Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Test the parsers
python tests/test_parser/test_parsing.py tests/test_parser/test_resume/Shivendra_Resume.docx
python tests/test_job_parser/test_job_parsing.py tests/test_job_parser/sample_job_description.txt
```

### Basic Usage
```python
from src.parsers import ResumeParser, JobParser

# Parse a resume
resume_parser = ResumeParser()
resume = resume_parser.parse_from_docx("path/to/resume.docx")

# Parse a job description
job_parser = JobParser()
job = job_parser.parse("job description text here...")

# Extract ML features
resume_features = resume.extract_features()
job_features = job.extract_features()
```

---

## Project Structure

```
ResumeAi/
├── src/
│   ├── models/              # Data schemas (Pydantic models)
│   │   ├── __init__.py
│   │   ├── base_schema.py   # Base models, enums, utilities
│   │   ├── resume_schema.py # Resume data structure
│   │   └── job_schema.py    # Job description data structure
│   │
│   ├── parsers/             # LLM-based parsers
│   │   ├── __init__.py
│   │   ├── gemini_client.py    # Gemini API client with retry
│   │   ├── docx_extractor.py   # Word document text extraction
│   │   ├── prompt_templates.py # LLM prompts
│   │   ├── base_parser.py      # Abstract base parser
│   │   ├── resume_parser.py    # Resume parser (main entry point)
│   │   └── job_parser.py       # Job description parser
│   │
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── text_utils.py    # Text preprocessing
│       └── validation.py    # Data validation helpers
│
├── tests/
│   ├── test_parser/         # Resume parser tests
│   │   ├── test_parsing.py
│   │   ├── README.md
│   │   └── test_resume/
│   │       └── Shivendra_Resume.docx
│   │
│   └── test_job_parser/     # Job parser tests
│       ├── test_job_parsing.py
│       ├── README.md
│       └── sample_job_description.txt
│
├── config.py                # Central configuration
├── .env.example             # Environment variables template
└── requirements.txt         # Python dependencies
```

---

## Data Models (Schemas)

All data models inherit from `MLReadyBaseModel` which provides:
- `extract_features()` - ML-ready features
- `to_flat_dict()` - Flattened dictionary
- `get_text_for_embedding()` - Text for semantic similarity
- `validate_for_ml()` - Data quality validation
- `completeness_score()` - Data completeness percentage

### Resume Schema

**File**: `src/models/resume_schema.py`

```python
from src.parsers import ResumeParser

parser = ResumeParser()
resume = parser.parse_from_docx("resume.docx")

# Access parsed data
print(resume.personal_info.name)
print(resume.personal_info.email)
print(resume.skills.get_all_skills_flat())
print(resume.calculate_total_experience())  # years

# ML features for scoring
features = resume.extract_features()
# Returns:
{
    "years_experience": 5.2,
    "num_previous_jobs": 3,
    "num_skills": 42,
    "education_level": 4,  # 0=unknown, 1=hs, 2=assoc, 3=bachelor, 4=master, 5=phd
    "location_city": "Pittsburgh",
    "location_country": "United States",
    "skills_list": ["python", "java", "aws", ...],
    "experience_text": "Software Engineer at ...",  # for embedding
    # ... more fields
}
```

**Key Methods for Team Integration:**
```python
# Person 2 (Scoring Modules)
features = resume.extract_features()
skills = resume.skills.get_all_skills_flat()
years_exp = resume.calculate_total_experience()
edu_level = resume.get_highest_education_level()

# Person 3 (Ranking Engine)
text_for_embedding = resume.get_text_for_embedding()
flat_data = resume.to_flat_dict()

# Person 4 (UI)
display_data = resume.dict()  # Clean JSON
validation_issues = resume.validate_for_ml()
completeness = resume.completeness_score()
```

### Job Description Schema

**File**: `src/models/job_schema.py`

```python
from src.parsers import JobParser

parser = JobParser()
job = parser.parse("Job description text...")

# Access parsed data
print(job.title)
print(job.company)
print(job.required_skills.must_have)
print(job.required_experience.min_years)

# ML features for scoring
features = job.extract_features()
# Returns:
{
    "required_skills": ["python", "aws", ...],
    "nice_to_have_skills": ["docker", ...],
    "num_required_skills": 15,
    "required_years": 3.0,
    "min_years": 2.0,
    "max_years": 5.0,
    "required_education_level": 3,  # same scale as resume
    "location": "San Francisco, CA",
    "remote_allowed": True,
    "hybrid": False,
    "description_text": "...",  # for embedding
    # ... more fields
}
```

**Key Methods for Team Integration:**
```python
# Person 2 (Scoring Modules)
features = job.extract_features()
required_skills = job.required_skills.get_all_required_skills()
required_years = job.required_experience.get_target_years()

# Person 3 (Ranking Engine)
text_for_embedding = job.get_text_for_embedding()
requirements_summary = job.get_requirements_summary()

# Person 4 (UI)
display_data = job.dict()
```

---

## Parser Usage

### Resume Parser

**File**: `src/parsers/resume_parser.py`

```python
from src.parsers import ResumeParser

# Initialize
parser = ResumeParser()

# Parse single resume
resume = parser.parse_from_docx("resume.docx")

# Parse multiple resumes (batch processing - FAST!)
resume_files = ["resume1.docx", "resume2.docx", "resume3.docx"]
resumes = parser.parse_batch(resume_files)  # Parallel processing

# Check parser stats
stats = parser.get_stats()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

### Job Description Parser

**File**: `src/parsers/job_parser.py`

```python
from src.parsers import JobParser

# Initialize
parser = JobParser()

# Parse from text
job_text = """
Software Engineer
Company: Tech Corp
Location: San Francisco
...
"""
job = parser.parse(job_text)

# Parse from file
with open("job_description.txt", "r") as f:
    job_text = f.read()
job = parser.parse(job_text)
```

---

## Integration Guide by Team Member

### Person 2: Scoring Modules

**Your Task**: Implement scoring algorithms (skills matching, experience scoring, etc.)

**What You Need from Parsing Module:**

```python
from src.parsers import ResumeParser, JobParser

# Parse resume and job
resume_parser = ResumeParser()
job_parser = JobParser()

resume = resume_parser.parse_from_docx("resume.docx")
job = job_parser.parse("job description...")

# Get ML features
resume_features = resume.extract_features()
job_features = job.extract_features()

# Skills Matching
resume_skills = resume_features["skills_list"]  # List[str]
required_skills = job_features["required_skills"]  # List[str]
# → Implement your skills matching algorithm

# Experience Scoring
resume_exp_years = resume_features["years_experience"]  # float
job_required_years = job_features["required_years"]  # float
# → Implement your experience scoring

# Education Scoring
resume_edu_level = resume_features["education_level"]  # int (0-5)
job_edu_level = job_features["required_education_level"]  # int (0-5)
# → Implement your education scoring

# Location Scoring
resume_location = resume_features["location_city"]
job_location = job_features["location_city"]
job_remote_allowed = job_features["remote_allowed"]  # bool
# → Implement your location scoring
```

**Helper Methods:**
```python
# Direct access to specific data
skills = resume.skills.get_all_skills_flat()
total_exp = resume.calculate_total_experience()
edu_level = resume.get_highest_education_level()
```

### Person 3: Ranking Engine

**Your Task**: Implement semantic similarity and final ranking

**What You Need from Parsing Module:**

```python
from src.parsers import ResumeParser, JobParser

resume = ResumeParser().parse_from_docx("resume.docx")
job = JobParser().parse("job description...")

# Get text for embedding (semantic similarity)
resume_text = resume.get_text_for_embedding()
job_text = job.get_text_for_embedding()

# Example with sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

resume_embedding = model.encode(resume_text)
job_embedding = model.encode(job_text)
# → Compute cosine similarity

# Get all scores from Person 2
from src.scoring import SkillsScorer, ExperienceScorer  # Your modules
skills_score = SkillsScorer().score(resume, job)
exp_score = ExperienceScorer().score(resume, job)
semantic_score = compute_similarity(resume_embedding, job_embedding)

# Weighted final score
from config import WEIGHTS
final_score = (
    WEIGHTS["skills"] * skills_score +
    WEIGHTS["experience"] * exp_score +
    WEIGHTS["semantic"] * semantic_score +
    # ... other scores
)
```

**Helper Methods:**
```python
# Flattened data for ranking dataframe
resume_flat = resume.to_flat_dict()
job_flat = job.to_flat_dict()

# Data quality checks
issues = resume.validate_for_ml()
completeness = resume.completeness_score()
```

### Person 4: UI (Streamlit)

**Your Task**: Build user interface for resume upload and ranking display

**What You Need from Parsing Module:**

```python
import streamlit as st
from src.parsers import ResumeParser, JobParser

# File upload
uploaded_files = st.file_uploader("Upload Resumes", type="docx", accept_multiple_files=True)
job_text = st.text_area("Paste Job Description")

# Parse
if st.button("Analyze"):
    # Parse resumes (batch processing)
    parser = ResumeParser()
    resumes = []
    for file in uploaded_files:
        # Save to temp location
        temp_path = f"/tmp/{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        resume = parser.parse_from_docx(temp_path)
        resumes.append(resume)

    # Parse job
    job = JobParser().parse(job_text)

    # Display parsed data
    for resume in resumes:
        st.subheader(resume.personal_info.name)
        st.write(f"Email: {resume.personal_info.email}")
        st.write(f"Experience: {resume.calculate_total_experience():.1f} years")
        st.write(f"Skills: {', '.join(resume.skills.get_all_skills_flat()[:10])}")

        # Data quality indicator
        completeness = resume.completeness_score()
        st.progress(completeness)

        # Get ranking from Person 3
        # score = RankingEngine().rank(resume, job)
        # st.metric("Match Score", f"{score:.1%}")
```

**Clean JSON for API:**
```python
# For REST API endpoints
resume_json = resume.dict()  # Pydantic to dict
job_json = job.dict()

# Return as JSON response
return {"resume": resume_json, "job": job_json}
```

---

## Testing

### Test Resume Parser
```bash
# Test with sample resume
python tests/test_parser/test_parsing.py tests/test_parser/test_resume/Shivendra_Resume.docx

# Add your own resume
cp your_resume.docx tests/test_parser/test_resume/
python tests/test_parser/test_parsing.py tests/test_parser/test_resume/your_resume.docx
```

### Test Job Parser
```bash
# 1. Edit the sample job description
nano tests/test_job_parser/sample_job_description.txt
# Paste a job description

# 2. Run test
python tests/test_job_parser/test_job_parsing.py tests/test_job_parser/sample_job_description.txt
```

### Expected Output
Both tests should show:
- ✅ Parsing successful
- Personal/Job information extracted
- Requirements extracted
- ML features displayed
- Data quality validation
- Completeness score > 70%

---

## Configuration

### Environment Variables (`.env`)

```bash
# API Key (REQUIRED)
GEMINI_API_KEY=your_api_key_here

# Model Selection
GEMINI_MODEL=gemini-2.5-flash-lite  # Fast and cheap
# Alternatives:
#   gemini-2.5-flash      - More accurate but 3x slower
#   gemini-2.5-pro        - Most accurate but 10x slower

# Embedding Model (for Person 3)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Scoring Weights (for Person 3)
WEIGHT_SKILLS=0.30
WEIGHT_EXPERIENCE=0.25
WEIGHT_SEMANTIC=0.25
WEIGHT_EDUCATION=0.15
WEIGHT_LOCATION=0.05
```

### Model Comparison

| Model | Speed | Accuracy | Cost | Use Case |
|-------|-------|----------|------|----------|
| gemini-2.5-flash-lite | ⚡⚡⚡ 9s | ⭐⭐⭐⭐ | 💰 Very Low | **Production** (recommended) |
| gemini-2.5-flash | ⚡⚡ 33s | ⭐⭐⭐⭐⭐ | 💰 Very Low | High accuracy needed |
| gemini-2.5-pro | ⚡ 90s+ | ⭐⭐⭐⭐⭐ | 💰💰 Low | Critical applications |

---

## API Reference

### ResumeParser

```python
class ResumeParser(BaseParser):
    def parse_from_docx(file_path: str) -> Resume:
        """Parse a .docx resume file."""

    def parse_batch(file_paths: List[str]) -> List[Resume]:
        """Parse multiple resumes in parallel."""

    def get_stats() -> Dict:
        """Get parsing statistics."""
```

### JobParser

```python
class JobParser(BaseParser):
    def parse(text: str) -> JobDescription:
        """Parse job description text."""
```

### Resume Model

```python
class Resume(MLReadyBaseModel):
    # Properties
    personal_info: PersonalInfo
    experience: List[Experience]
    education: List[Education]
    skills: Skills
    projects: List[Project]

    # Methods
    def extract_features() -> Dict[str, Any]:
        """Extract ML-ready features."""

    def get_text_for_embedding() -> str:
        """Get text for semantic similarity."""

    def calculate_total_experience() -> float:
        """Calculate total years of experience."""

    def get_highest_education_level() -> int:
        """Get education level (0-5 scale)."""

    def to_flat_dict() -> Dict[str, Any]:
        """Flatten to simple dict."""

    def validate_for_ml() -> List[str]:
        """Validate data quality, return issues."""

    def completeness_score() -> float:
        """Data completeness percentage (0-1)."""
```

### JobDescription Model

```python
class JobDescription(MLReadyBaseModel):
    # Properties
    title: str
    company: str
    description: str
    required_skills: RequiredSkills
    required_experience: ExperienceRequirement
    education_requirement: EducationRequirement
    location_requirement: LocationRequirement

    # Methods
    def extract_features() -> Dict[str, Any]:
        """Extract ML-ready features."""

    def get_text_for_embedding() -> str:
        """Get text for semantic similarity."""

    def get_requirements_summary() -> str:
        """Get summary of all requirements."""
```

---

## Troubleshooting

### Common Issues

**1. "GEMINI_API_KEY not found"**
```bash
# Solution: Set API key in .env file
echo "GEMINI_API_KEY=your_key_here" >> .env
```

**2. "No module named 'src'"**
```bash
# Solution: Run from project root
cd /path/to/ResumeAi
python tests/test_parser/test_parsing.py ...
```

**3. "Model not found: gemini-xxx"**
```bash
# Solution: Use correct model name in .env
GEMINI_MODEL=gemini-2.5-flash-lite
```

**4. Parsing is too slow**
```bash
# Solution: Switch to faster model
GEMINI_MODEL=gemini-2.5-flash-lite  # 3x faster
```

**5. "Rate limit exceeded"**
```python
# Solution: Use batch processing with delays
parser = ResumeParser()
resumes = parser.parse_batch(files)  # Handles rate limits automatically
```

### Getting Help

1. Check test scripts in `tests/` for examples
2. Read schema documentation in `src/models/`
3. Review prompt templates in `src/parsers/prompt_templates.py`
4. Check logs for detailed error messages

---

## Performance Tips

### Batch Processing (IMPORTANT!)

```python
# ❌ SLOW - Sequential parsing
parser = ResumeParser()
resumes = []
for file in files:
    resume = parser.parse_from_docx(file)  # 9 seconds each
    resumes.append(resume)
# 50 files = 7.5 minutes

# ✅ FAST - Parallel parsing
parser = ResumeParser()
resumes = parser.parse_batch(files)  # 15 at a time
# 50 files = 30-40 seconds!
```

### Caching Results

```python
import pickle

# Parse once, save results
resumes = parser.parse_batch(files)
with open("parsed_resumes.pkl", "wb") as f:
    pickle.dump(resumes, f)

# Load for development/testing
with open("parsed_resumes.pkl", "rb") as f:
    resumes = pickle.load(f)
```

---

## Next Steps

### For Person 2 (Scoring Modules)
1. ✅ Review `Resume.extract_features()` and `JobDescription.extract_features()`
2. Create `src/scoring/` folder
3. Implement scoring algorithms:
   - `skills_matcher.py` - Skills overlap scoring
   - `experience_scorer.py` - Experience years scoring
   - `education_scorer.py` - Education level scoring
   - `location_scorer.py` - Location matching
4. Add your dependencies to `requirements.txt`

### For Person 3 (Ranking Engine)
1. ✅ Review `get_text_for_embedding()` method
2. Create `src/ranking/` folder
3. Implement:
   - `semantic_similarity.py` - Embedding-based similarity
   - `ranking_engine.py` - Combine all scores with weights
4. Add sentence-transformers to `requirements.txt`

### For Person 4 (UI)
1. ✅ Review `Resume.dict()` and `JobDescription.dict()`
2. Create `app/` folder for Streamlit app
3. Implement:
   - `app.py` - Main Streamlit interface
   - File upload for resumes
   - Text input for job description
   - Results display with scores
4. Add streamlit, plotly to `requirements.txt`

---

## Summary

**✅ What's Complete:**
- Resume parsing from .docx files
- Job description parsing from text
- ML-ready data schemas with feature extraction
- Batch processing support
- Data validation and quality scoring
- Comprehensive testing

**📦 Ready for Integration:**
- All schemas are ML-ready
- Feature extraction methods implemented
- Text embedding methods available
- Clean JSON serialization
- Batch processing for performance

**🚀 You Can Start:**
- Person 2: Implementing scoring algorithms
- Person 3: Building ranking engine
- Person 4: Creating UI

---

**Questions?** Check the test scripts or reach out to Person 1 (Shivendra).

**Good luck with the project! 🎉**
