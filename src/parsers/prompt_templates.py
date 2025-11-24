"""
Prompt templates for Gemini API document parsing.

This module contains carefully crafted prompts for:
- Resume parsing
- Job description parsing

These prompts are optimized for:
- High extraction accuracy
- Structured JSON output
- Handling missing information
- Consistency across documents
"""

import json
from typing import Dict, Any


# ==========================================
# Resume Parsing Prompts
# ==========================================

RESUME_SYSTEM_PROMPT = """You are an expert resume parser with deep understanding of various resume formats and structures.

Your task is to extract structured information from resume text and return it as valid JSON.

CRITICAL RULES:
1. Output ONLY valid JSON matching the exact schema provided
2. Do NOT add any explanations, comments, or text outside the JSON
3. If information is missing or unclear, use "Unknown" for strings, empty arrays [] for lists, or null for optional fields
4. Normalize all skill names to lowercase
5. For dates, preserve the original format if clear, otherwise use "Unknown"
6. Calculate experience duration from dates when possible
7. Extract ALL technical skills mentioned (programming languages, frameworks, databases, tools)
8. Be conservative - only extract information explicitly stated in the resume
9. For experience, extract company name, job title, dates, and key responsibilities
10. Distinguish between education levels: Bachelor, Master, PhD, etc.

degree.level must be one of: 'high_school', 'diploma', 'associate', 'bachelor', 'master', 'doctorate', 'unknown'. Do NOT output strings like 'high school'; use the exact enum values.
IMPORTANT: Return ONLY the JSON object. No markdown code blocks, no explanations."""


RESUME_USER_PROMPT_TEMPLATE = """Extract structured information from the following resume and return it as JSON matching this exact schema:

SCHEMA:
{schema}

RESUME TEXT:
{resume_text}

Remember:
- Return ONLY valid JSON
- Use "Unknown" for missing information
- Normalize skill names (lowercase, no spaces in compound words like "nodejs")
- Extract experience duration if dates are provided
- List all technical skills found

OUTPUT (JSON only):"""


# Example schema for documentation
RESUME_SCHEMA_EXAMPLE = {
    "personal_info": {
        "name": "string",
        "email": "string",
        "phone": "string",
        "location": {
            "city": "string",
            "country": "string",
            "remote_preference": "remote|hybrid|onsite|unknown"
        },
        "summary": "string",
        "linkedin": "string or null",
        "github": "string or null"
    },
    "experience": [
        {
            "company": "string",
            "company_info": {
                "industry": "string",
                "size": "string"
            },
            "title": "string",
            "level": "entry|mid|senior|lead|executive|unknown",
            "employment_type": "full-time|part-time|contract|intern|unknown",
            "dates": {
                "start": "string",
                "end": "string (or 'Present')",
                "duration": "string (e.g., '2 years 3 months')"
            },
            "responsibilities": ["string"],
            "technical_environment": {
                "technologies": ["string"],
                "methodologies": ["string"],
                "tools": ["string"]
            }
        }
    ],
    "education": [
        {
            "degree": {
                "level": "bachelor|master|doctorate|associate|diploma|unknown",
                "field": "string",
                "major": "string"
            },
            "institution": {
                "name": "string",
                "location": "string",
                "accreditation": "string"
            },
            "dates": {
                "start": "string",
                "end": "string",
                "duration": "string"
            },
            "achievements": {
                "gpa": "number or null",
                "honors": "string",
                "relevant_coursework": ["string"]
            }
        }
    ],
    "skills": {
        "technical": {
            "programming_languages": [
                {"name": "string", "level": "beginner|intermediate|advanced|expert"}
            ],
            "frameworks": [
                {"name": "string", "level": "beginner|intermediate|advanced|expert"}
            ],
            "databases": [
                {"name": "string", "level": "beginner|intermediate|advanced|expert"}
            ],
            "cloud": [
                {"name": "string", "level": "beginner|intermediate|advanced|expert"}
            ],
            "tools": [
                {"name": "string", "level": "beginner|intermediate|advanced|expert"}
            ]
        },
        "languages": [
            {"name": "string", "level": "native|fluent|intermediate|basic"}
        ]
    },
    "projects": [
        {
            "name": "string",
            "description": "string",
            "technologies": ["string"],
            "role": "string",
            "url": "string or null",
            "impact": "string"
        }
    ],
    "certifications": "string (comma-separated if multiple)"
}


# ==========================================
# Job Description Parsing Prompts
# ==========================================

JOB_SYSTEM_PROMPT = """You are an expert at analyzing job descriptions and extracting key requirements.

Your task is to extract structured job requirements from job description text and return it as valid JSON.

CRITICAL RULES:
1. Output ONLY valid JSON matching the exact schema provided
2. Do NOT add any explanations, comments, or text outside the JSON
3. Carefully distinguish between "must-have" (required) and "nice-to-have" (preferred) skills
4. Parse experience requirements:
   - "5+ years" → min_years: 5, max_years: null
   - "3-5 years" → min_years: 3, max_years: 5
   - "At least 5 years" → min_years: 5, max_years: null
5. Identify minimum education level (Bachelor, Master, PhD, etc.)
6. Extract location and determine if remote work is allowed
7. Normalize all skill names to lowercase
8. Separate responsibilities from requirements
9. If salary range is mentioned, extract it
10. Be conservative - only extract explicitly stated requirements

IMPORTANT: Return ONLY the JSON object. No markdown code blocks, no explanations."""


JOB_USER_PROMPT_TEMPLATE = """Extract structured requirements from the following job description and return it as JSON matching this exact schema:

SCHEMA:
{schema}

JOB DESCRIPTION TEXT:
{job_text}

Remember:
- Distinguish required vs nice-to-have skills
- Parse experience requirements (e.g., "5+ years" or "3-5 years")
- Extract minimum education level
- Identify location and remote work policy
- Normalize skill names (lowercase)
- Separate responsibilities from requirements

OUTPUT (JSON only):"""


# Example schema for documentation
JOB_SCHEMA_EXAMPLE = {
    "job_id": "string or null",
    "title": "string (job title)",
    "company": "string (company name)",
    "role": "string or null",
    "description": "string (full job description)",
    "responsibilities": ["string (key responsibilities)"],
    "required_skills": {
        "must_have": ["string (required skills)"],
        "nice_to_have": ["string (preferred/nice-to-have skills)"]
    },
    "required_experience": {
        "min_years": "number",
        "max_years": "number or null",
        "preferred_years": "number or null"
    },
    "education_requirement": {
        "min_level": "bachelor|master|doctorate|associate|diploma|unknown",
        "preferred_level": "bachelor|master|doctorate|associate|diploma|unknown or null",
        "field": "string (required field of study) or null"
    },
    "location_requirement": {
        "location": "string (city, country)",
        "remote_allowed": "boolean",
        "hybrid": "boolean"
    },
    "salary_range": "string or null",
    "benefits": ["string"],
    "company_info": {}
}


# ==========================================
# Few-Shot Examples (for improved accuracy)
# ==========================================

RESUME_FEW_SHOT_EXAMPLE = """
EXAMPLE INPUT:
---
John Doe
Email: john.doe@email.com | Phone: +1-555-0123
San Francisco, CA | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Senior Software Engineer with 6 years of experience in full-stack development.

EXPERIENCE
Software Engineer | Tech Corp | Jan 2020 - Present
- Developed RESTful APIs using Python and Django
- Managed PostgreSQL databases and optimized queries
- Led team of 3 junior developers

EDUCATION
Bachelor of Science in Computer Science
University of California | 2014 - 2018

SKILLS
Python, Django, JavaScript, React, PostgreSQL, AWS, Docker
---

EXAMPLE OUTPUT:
{
  "personal_info": {
    "name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "+1-555-0123",
    "location": {
      "city": "San Francisco",
      "country": "USA",
      "remote_preference": "unknown"
    },
    "summary": "Senior Software Engineer with 6 years of experience in full-stack development.",
    "linkedin": "linkedin.com/in/johndoe",
    "github": null
  },
  "experience": [
    {
      "company": "Tech Corp",
      "title": "Software Engineer",
      "level": "senior",
      "employment_type": "full-time",
      "dates": {
        "start": "2020-01",
        "end": "Present",
        "duration": "4 years"
      },
      "responsibilities": [
        "Developed RESTful APIs using Python and Django",
        "Managed PostgreSQL databases and optimized queries",
        "Led team of 3 junior developers"
      ],
      "technical_environment": {
        "technologies": ["python", "django", "postgresql"],
        "methodologies": [],
        "tools": []
      }
    }
  ],
  "education": [
    {
      "degree": {
        "level": "bachelor",
        "field": "Computer Science",
        "major": "Computer Science"
      },
      "institution": {
        "name": "University of California",
        "location": "California",
        "accreditation": "Unknown"
      },
      "dates": {
        "start": "2014",
        "end": "2018",
        "duration": "4 years"
      },
      "achievements": {
        "gpa": null,
        "honors": "Unknown",
        "relevant_coursework": []
      }
    }
  ],
  "skills": {
    "technical": {
      "programming_languages": [
        {"name": "python", "level": "advanced"},
        {"name": "javascript", "level": "intermediate"}
      ],
      "frameworks": [
        {"name": "django", "level": "advanced"},
        {"name": "react", "level": "intermediate"}
      ],
      "databases": [
        {"name": "postgresql", "level": "advanced"}
      ],
      "cloud": [
        {"name": "aws", "level": "intermediate"}
      ],
      "tools": [
        {"name": "docker", "level": "intermediate"}
      ]
    },
    "languages": []
  },
  "projects": [],
  "certifications": ""
}
"""


# ==========================================
# Helper Functions
# ==========================================

def get_resume_prompt(resume_text: str, include_example: bool = False) -> str:
    """
    Generate complete resume parsing prompt.

    Args:
        resume_text: Raw resume text
        include_example: Whether to include few-shot example

    Returns:
        str: Complete prompt for Gemini
    """
    schema_str = json.dumps(RESUME_SCHEMA_EXAMPLE, indent=2)

    user_prompt = RESUME_USER_PROMPT_TEMPLATE.format(
        schema=schema_str,
        resume_text=resume_text
    )

    if include_example:
        full_prompt = f"{RESUME_SYSTEM_PROMPT}\n\n{RESUME_FEW_SHOT_EXAMPLE}\n\nNow parse this resume:\n\n{user_prompt}"
    else:
        full_prompt = f"{RESUME_SYSTEM_PROMPT}\n\n{user_prompt}"

    return full_prompt


def get_job_prompt(job_text: str, include_example: bool = False) -> str:
    """
    Generate complete job description parsing prompt.

    Args:
        job_text: Raw job description text
        include_example: Whether to include few-shot example

    Returns:
        str: Complete prompt for Gemini
    """
    schema_str = json.dumps(JOB_SCHEMA_EXAMPLE, indent=2)

    user_prompt = JOB_USER_PROMPT_TEMPLATE.format(
        schema=schema_str,
        job_text=job_text
    )

    full_prompt = f"{JOB_SYSTEM_PROMPT}\n\n{user_prompt}"

    return full_prompt


# ==========================================
# Prompt Optimization Tips
# ==========================================

"""
TIPS FOR OPTIMIZING PROMPTS:

1. Be Explicit:
   - Specify exact output format
   - Provide schema/examples
   - State what to do with missing data

2. Use Few-Shot Examples:
   - Include 1-2 perfect examples
   - Show edge cases (missing fields, etc.)

3. Temperature Settings:
   - Use 0.0 for deterministic, structured output
   - Use 0.0-0.2 for data extraction tasks

4. Iterative Improvement:
   - Test with real resumes
   - Identify common failure patterns
   - Add specific instructions for those cases

5. Error Handling:
   - Instruct model on how to handle ambiguity
   - Provide fallback values
   - Request specific format for dates, skills, etc.

6. Token Optimization:
   - Keep system prompt concise but complete
   - Schema examples should be clear but not verbose
   - Test with different prompt lengths

7. A/B Testing:
   - Try variations of prompts
   - Measure extraction accuracy
   - Keep the best performing version
"""


# ==========================================
# Main for Testing
# ==========================================

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("RESUME PARSING PROMPT")
    print("=" * 60)

    sample_resume = """
John Smith
john.smith@email.com | +1-234-567-8900
New York, NY

SUMMARY
Full-stack developer with 4 years of experience.

EXPERIENCE
Senior Developer | ABC Tech | 2020 - Present
- Built web applications using React and Node.js
- Worked with MongoDB and PostgreSQL

EDUCATION
B.S. Computer Science | MIT | 2016-2020

SKILLS
JavaScript, React, Node.js, Python, MongoDB, PostgreSQL, Docker
    """

    prompt = get_resume_prompt(sample_resume, include_example=False)
    print(prompt[:1000] + "...\n")

    print("=" * 60)
    print("JOB DESCRIPTION PARSING PROMPT")
    print("=" * 60)

    sample_job = """
Senior Python Developer

We are looking for an experienced Python developer with 5+ years of experience.

Requirements:
- Strong Python programming skills
- Experience with Django or Flask
- PostgreSQL database knowledge
- AWS cloud experience is a plus

Education: Bachelor's degree in Computer Science or related field

Location: San Francisco, CA (Remote OK)
    """

    prompt = get_job_prompt(sample_job)
    print(prompt[:1000] + "...")
