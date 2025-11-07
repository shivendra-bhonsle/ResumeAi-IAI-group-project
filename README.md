# ResumeAI - AI-Powered Resume Screening System

An intelligent resume ranking system that automates candidate screening using LLM-based parsing and multi-factor scoring algorithms.

## Overview

ResumeAI takes a job description and multiple resume files as input, then automatically ranks candidates from best to worst fit based on:
- Skills matching (30%)
- Experience matching (25%)
- Semantic similarity (25%)
- Education matching (15%)
- Location matching (5%)

## Key Features

- **Intelligent Parsing**: Uses Gemini API to extract structured data from unstructured resumes and job descriptions
- **Multi-Factor Scoring**: Combines 5 different scoring algorithms with configurable weights
- **Semantic Understanding**: Uses sentence transformers for deep contextual matching beyond keywords
- **Explainability**: Provides clear reasoning for each ranking with strengths and gaps
- **Fast Processing**: Ranks 10 resumes in 5-10 seconds
- **User-Friendly UI**: Streamlit-based interface for non-technical users

## System Architecture

```
Input (Job Description + Resumes)
    ↓
Document Parsing (Gemini API)
    ↓
Information Extraction (Structured JSON)
    ↓
Multi-Factor Scoring (5 Modules)
    ↓
Ranking & Aggregation
    ↓
Explainability Generation
    ↓
Output (Ranked Candidates with Scores)
```

## Team Structure

- **Person 1**: Document Parsing & LLM Integration (Gemini API)
- **Person 2**: Scoring Modules Development
- **Person 3**: Ranking Engine & Explainability
- **Person 4**: Streamlit UI Development

## Tech Stack

- **Language**: Python 3.9+
- **LLM**: Google Gemini API (for document parsing)
- **ML Models**: Sentence Transformers (semantic similarity)
- **Backend**: FastAPI (REST API)
- **Frontend**: Streamlit
- **Text Processing**: RapidFuzz (fuzzy matching), python-docx (document parsing)

## Project Structure

```
ResumeAi/
├── src/                    # Main source code
│   ├── parsers/           # Document parsing (Gemini API)
│   ├── models/            # Data schemas and models
│   ├── scoring/           # Scoring modules (5 algorithms)
│   ├── ranking/           # Ranking engine & explainability
│   ├── api/               # REST API endpoints
│   ├── utils/             # Shared utilities
│   └── pipeline/          # End-to-end orchestration
├── frontend/              # Streamlit UI
│   └── components/        # Reusable UI components
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks for experiments
├── scripts/               # Utility scripts
├── data/                  # Datasets
│   ├── job_descriptions.csv
│   ├── master_resumes.jsonl
│   ├── sample_resumes/    # Test .docx files
│   └── sample_job_descriptions/
├── outputs/               # Generated results
│   ├── rankings/          # JSON/CSV outputs
│   └── logs/              # Application logs
└── documentation/         # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ResumeAi
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file and add your GEMINI_API_KEY
   ```


