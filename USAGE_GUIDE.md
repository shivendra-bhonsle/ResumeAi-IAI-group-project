# ResumeAI Usage Guide

Complete guide for using the ResumeAI resume ranking system.

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Usage Methods](#usage-methods)
   - [Web Interface (Streamlit)](#web-interface-streamlit)
   - [Command Line Interface](#command-line-interface)
   - [Python API](#python-api)
4. [Understanding Results](#understanding-results)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** activated
3. **Dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```
4. **Gemini API key** configured in `.env` file

### 5-Minute Demo

```bash
# 1. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Run web interface
streamlit run frontend/app.py

# 3. Open browser to http://localhost:8501

# 4. Upload job description and resumes

# 5. Click "Rank Candidates" and view results!
```

---

## ⚙️ Configuration

### Setting Up `.env` File

Copy `.env.example` to `.env` and configure:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Model Configuration
GEMINI_MODEL=gemini-2.5-flash-lite
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Scoring Weights (must sum to 1.0)
WEIGHT_SKILLS=0.33
WEIGHT_EXPERIENCE=0.27
WEIGHT_SEMANTIC=0.27
WEIGHT_EDUCATION=0.13
WEIGHT_LOCATION=0.00  # Set to 0 to exclude
```

### Current Weight Distribution

**Default Configuration** (Location excluded):
- **Skills**: 33% - Matching of technical and soft skills
- **Experience**: 27% - Years of relevant experience
- **Semantic**: 27% - AI-based similarity of resume to job description
- **Education**: 13% - Educational qualification match
- **Location**: 0% - Geographic matching (disabled)

**Total**: 100%

### Verifying Configuration

```bash
python config.py
```

Expected output:
```
ResumeAI Configuration
==================================================
{
  "gemini_model": "gemini-2.5-flash-lite",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "weights": {
    "skills": 0.33,
    "experience": 0.27,
    "semantic": 0.27,
    "education": 0.13,
    "location": 0.0
  }
}

==================================================
Validating configuration...
✓ Configuration is valid
```

---

## 💻 Usage Methods

### Method 1: Web Interface (Streamlit)

**Best for**: Non-technical users, interactive exploration

#### Starting the App

```bash
streamlit run frontend/app.py
```

#### Using the Interface

1. **Upload Job Description**
   - Paste the complete job description in the text area
   - Include requirements, responsibilities, qualifications

2. **Upload Resumes**
   - Click "Browse files" or drag & drop
   - Select up to 10 `.docx` files
   - Multiple files can be uploaded at once

3. **Configure Output** (optional)
   - Check "Save as CSV" to export results
   - Check "Save as JSON" for programmatic access

4. **Rank Candidates**
   - Click "🔎 Rank Candidates"
   - Wait 1-2 minutes for processing (depends on number of resumes)

5. **View Results**
   - See ranked table with scores
   - Expand candidate details for breakdown
   - View score visualizations
   - Download results as CSV or JSON

#### Features

- **Real-time Rankings**: Instant visual feedback
- **Score Breakdown**: See individual component scores
- **Candidate Details**: Expandable view for each candidate
- **Export Options**: Download as CSV, JSON
- **Weight Display**: See current scoring weights in sidebar

---

### Method 2: Command Line Interface

**Best for**: Batch processing, automation, scripting

#### Basic Usage

```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx
```

#### Command Options

```bash
python run_ranking.py [OPTIONS]

Required:
  --job, -j PATH          Path to job description file (.txt)
  --resumes, -r PATH...   Paths to resume files (.docx), supports wildcards

Optional:
  --output, -o PATH       Output file path (auto-generates if not specified)
  --format, -f FORMAT     Output format: csv, json, excel (default: csv)
  --no-save               Don't save to file, only print to console
  --top, -n N             Show only top N candidates
  --verbose, -v           Show detailed progress information
  --help                  Show help message
```

#### Examples

**Example 1: Basic Usage**
```bash
python run_ranking.py --job job.txt --resumes resume1.docx resume2.docx
```

**Example 2: Use Wildcards**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx
```

**Example 3: Save as JSON**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx --format json
```

**Example 4: Custom Output Path**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx --output my_results.csv
```

**Example 5: Show Top 5 Only**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx --top 5
```

**Example 6: Verbose Mode**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx --verbose
```

**Example 7: Don't Save (Print Only)**
```bash
python run_ranking.py --job job.txt --resumes resumes/*.docx --no-save
```

---

### Method 3: Python API

**Best for**: Integration with other systems, custom workflows

#### Quick Example

```python
from src.pipeline import rank_candidates

# Simple usage
results = rank_candidates(
    job_text="Software Engineer with 3+ years in Python...",
    resume_files=["resume1.docx", "resume2.docx"]
)

print(results.head())
```

#### Advanced Usage

```python
from src.pipeline import ResumePipeline
import pandas as pd

# Initialize pipeline
pipeline = ResumePipeline()

# Read job description
with open("job.txt", "r") as f:
    job_text = f.read()

# Get resume files
resume_files = ["resume1.docx", "resume2.docx", "resume3.docx"]

# Run complete pipeline
results_df, output_path = pipeline.run(
    job_text=job_text,
    resume_files=resume_files,
    save_output=True,
    output_format='csv',
    return_format='dataframe'
)

# Work with results
print(f"Total candidates: {len(results_df)}")
print(f"Top candidate: {results_df.iloc[0]['name']}")
print(f"Top score: {results_df.iloc[0]['final_score']:.1%}")

# Filter excellent matches (>80%)
excellent = results_df[results_df['final_score'] >= 0.8]
print(f"Excellent matches: {len(excellent)}")
```

#### Step-by-Step Control

```python
from src.pipeline import ResumePipeline

pipeline = ResumePipeline()

# Step 1: Parse job
job = pipeline.parse_job(job_text)

# Step 2: Parse resumes
resumes = pipeline.parse_resumes(resume_files)

# Step 3: Rank candidates
ranked_df = pipeline.rank_candidates(
    job=job,
    resumes=resumes,
    return_format='dataframe'
)

# Step 4: Save results (optional)
output_path = pipeline.save_results(
    ranked_df,
    format='csv'
)
```

---

## 📊 Understanding Results

### Result Format

#### CSV Output

```csv
rank,name,email,final_score,skills_score,experience_score,semantic_score,education_score,location_score,years_experience,education_level,num_skills,location,completeness,has_issues
1,John Doe,john@example.com,0.852,0.90,0.85,0.88,1.00,0.50,5.2,4,42,"San Francisco, USA",0.91,False
2,Jane Smith,jane@example.com,0.745,0.75,0.80,0.70,0.75,0.00,3.5,3,35,"Unknown, Unknown",0.83,False
```

#### JSON Output

```json
{
  "candidates": [
    {
      "rank": 1,
      "name": "John Doe",
      "email": "john@example.com",
      "final_score": 0.852,
      "skills_score": 0.90,
      "experience_score": 0.85,
      "semantic_score": 0.88,
      "education_score": 1.00,
      "location_score": 0.50,
      "years_experience": 5.2,
      "education_level": 4,
      "num_skills": 42,
      "location": "San Francisco, USA",
      "completeness": 0.91,
      "has_issues": false
    }
  ],
  "metadata": {
    "total_count": 10,
    "weights": {
      "skills": 0.33,
      "experience": 0.27,
      "semantic": 0.27,
      "education": 0.13,
      "location": 0.00
    },
    "timestamp": "2025-11-23T10:30:00",
    "model_config": {
      "gemini_model": "gemini-2.5-flash-lite",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

### Column Descriptions

| Column | Description | Range |
|--------|-------------|-------|
| `rank` | Ranking position | 1, 2, 3, ... |
| `name` | Candidate name | String |
| `email` | Contact email | String |
| `final_score` | Weighted total score | 0-1 (0-100%) |
| `skills_score` | Skills matching score | 0-1 |
| `experience_score` | Experience match score | 0-1 |
| `semantic_score` | AI similarity score | 0-1 |
| `education_score` | Education match score | 0-1 |
| `location_score` | Location match score | 0-1 |
| `years_experience` | Total years of experience | Float |
| `education_level` | Highest education (0-5) | 0=Unknown, 3=Bachelor, 4=Master, 5=PhD |
| `num_skills` | Number of skills listed | Integer |
| `location` | Candidate location | String |
| `completeness` | Resume data quality | 0-1 (higher is better) |
| `has_issues` | Data validation issues | Boolean |

### Score Interpretation

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 0.80 - 1.00 | Excellent Match | Highly qualified, strong candidate |
| 0.60 - 0.79 | Good Match | Qualified, consider for interview |
| 0.40 - 0.59 | Average Match | Partially qualified, may need review |
| 0.00 - 0.39 | Poor Match | Not well-suited for role |

### Score Calculation

```
Final Score = (
    0.33 × Skills Score +
    0.27 × Experience Score +
    0.27 × Semantic Score +
    0.13 × Education Score +
    0.00 × Location Score
)
```

**Example**:
```
Skills: 0.85 (85%)
Experience: 0.75 (75%)
Semantic: 0.80 (80%)
Education: 1.00 (100%)
Location: 0.50 (50%, but weight=0)

Final = 0.33(0.85) + 0.27(0.75) + 0.27(0.80) + 0.13(1.00) + 0.00(0.50)
      = 0.2805 + 0.2025 + 0.2160 + 0.1300 + 0.0000
      = 0.829 (82.9%)
```

---

## 🎨 Customization

### Adjusting Weights

Edit `.env` file to change scoring priorities:

**Example 1: Prioritize Experience**
```bash
WEIGHT_SKILLS=0.25
WEIGHT_EXPERIENCE=0.40  # Increased
WEIGHT_SEMANTIC=0.20
WEIGHT_EDUCATION=0.15
WEIGHT_LOCATION=0.00
```

**Example 2: Balance All Components**
```bash
WEIGHT_SKILLS=0.28
WEIGHT_EXPERIENCE=0.28
WEIGHT_SEMANTIC=0.28
WEIGHT_EDUCATION=0.16
WEIGHT_LOCATION=0.00
```

**Example 3: Enable Location**
```bash
WEIGHT_SKILLS=0.30
WEIGHT_EXPERIENCE=0.25
WEIGHT_SEMANTIC=0.20
WEIGHT_EDUCATION=0.15
WEIGHT_LOCATION=0.10  # Enabled
```

**Important**: Weights must sum to 1.0!

### Changing Models

Edit `.env` file:

**For Faster Parsing** (9s per resume):
```bash
GEMINI_MODEL=gemini-2.5-flash-lite
```

**For More Accurate Parsing** (33s per resume):
```bash
GEMINI_MODEL=gemini-2.5-flash
```

**For Better Semantic Matching** (larger model):
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Fuzzy Matching Threshold

In code (`src/scoring/skills_matcher.py`):

```python
# Default: 85% similarity required
scorer = SkillsScorer(fuzzy_threshold=85)

# More strict: 90% similarity
scorer = SkillsScorer(fuzzy_threshold=90)

# More lenient: 80% similarity
scorer = SkillsScorer(fuzzy_threshold=80)
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: "GEMINI_API_KEY not found"

**Solution**: Set API key in `.env`:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

#### Issue 2: "ModuleNotFoundError"

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

#### Issue 3: "No valid resume files found"

**Causes**:
- Files are not `.docx` format
- File paths are incorrect
- Files are corrupted

**Solution**: Verify files are valid `.docx` format

#### Issue 4: Slow Processing

**Causes**:
- Using slower Gemini model
- Many resumes to process
- Slow internet connection

**Solutions**:
- Use `gemini-2.5-flash-lite` (fastest)
- Process fewer resumes at once
- Check internet connection

#### Issue 5: Low Scores for Good Candidates

**Causes**:
- Resume format issues
- Missing key skills
- Wrong weight configuration

**Solutions**:
- Check resume quality and completeness
- Review skills matching
- Adjust weights in `.env`

---

## ❓ FAQ

### General

**Q: How many resumes can I process at once?**
A: Up to 10 via web interface, unlimited via CLI.

**Q: What file formats are supported?**
A: Only `.docx` files for resumes, `.txt` for job descriptions.

**Q: How long does processing take?**
A: ~9-10 seconds per resume with `gemini-2.5-flash-lite` model.

**Q: Can I use this offline?**
A: No, requires internet for Gemini API calls.

### Scoring

**Q: Why is location score always 0?**
A: Location weight is set to 0 in current configuration (excluded from scoring).

**Q: Can I change the scoring weights?**
A: Yes, edit the `.env` file and restart the application.

**Q: What does semantic similarity measure?**
A: AI-based similarity between resume text and job description using embeddings.

**Q: How is experience scored?**
A: Based on years of experience vs. job requirements (proportional scoring).

### Technical

**Q: Can I integrate this into my system?**
A: Yes, use the Python API (see Method 3 above).

**Q: Can I export results programmatically?**
A: Yes, results are returned as pandas DataFrame for easy manipulation.

**Q: Does it work on Windows/Mac/Linux?**
A: Yes, cross-platform compatible.

**Q: Can I batch process multiple jobs?**
A: Yes, use CLI with a script to loop through jobs.

---

## 📞 Support

For issues, feature requests, or questions:
- Check [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for technical details
- Check [PERSON2_INTEGRATION_SUMMARY.md](PERSON2_INTEGRATION_SUMMARY.md) for scoring module info
- Review [README.md](README.md) for project overview

---

## 🎓 Best Practices

1. **Use consistent resume formats** for better parsing
2. **Provide complete job descriptions** with clear requirements
3. **Review top 5-10 candidates** manually before decisions
4. **Adjust weights** based on your hiring priorities
5. **Check data completeness scores** for resume quality
6. **Save results** for audit trails and analysis
7. **Test with known candidates** first to validate scoring

---

**Last Updated**: November 23, 2025
**Version**: 1.0.0
