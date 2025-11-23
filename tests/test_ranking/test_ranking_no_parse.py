"""
Test the ranking engine in isolation with mock data loaded from JSON files.
This bypasses the parser and tests ONLY the ranking engine logic.

===============================================================================
QUICK START GUIDE
===============================================================================

This test script allows you to test the ranking engine without needing the 
parser module. It loads resume and job data from JSON files and tests the
ranking logic in isolation.

-------------------------------------------------------------------------------
PREREQUISITES
-------------------------------------------------------------------------------
1. JSON resume files in a directory (e.g., /path/to/resumes/)
2. JSON job description files in a directory (e.g., /path/to/jobs/)
3. Python environment with all dependencies installed

-------------------------------------------------------------------------------
TEST MODES
-------------------------------------------------------------------------------

MODE 1: Single Resume vs Single Job
------------------------------------
Test one specific resume against one specific job.

Usage:
    python test_ranking_no_parse.py \
      --resume /path/to/resume.json \
      --job /path/to/job.json

Example:
    python test_ranking_no_parse.py \
      -r /home/shlok/test_resumes/json_files/resumes/shlok_kalekar.json \
      -j /home/shlok/test_resumes/json_files/jobs/junior_backend.json

Output:
    - Detailed score breakdown
    - Individual module scores (semantic, skills, experience, education, location)
    - Weighted contributions
    - Data quality metrics


MODE 2: One Resume vs Multiple Jobs (Find Best Job Fit)
--------------------------------------------------------
Test one resume against all jobs in a directory to find the best matches.

Usage:
    python test_ranking_no_parse.py \
      --resume /path/to/resume.json \
      --jobs-dir /path/to/jobs/

Example:
    python test_ranking_no_parse.py \
      -r /home/shlok/test_resumes/json_files/resumes/shlok_kalekar.json \
      --jobs-dir /home/shlok/test_resumes/json_files/jobs/

Output:
    - Ranked list of jobs
    - Best matching job with score
    - Job titles and companies


MODE 3: One Job vs Multiple Resumes (Find Best Candidates)
-----------------------------------------------------------
Test one job against all resumes in a directory to find the best candidates.

Usage:
    python test_ranking_no_parse.py \
      --job /path/to/job.json \
      --resumes-dir /path/to/resumes/

Example:
    python test_ranking_no_parse.py \
      -j /home/shlok/test_resumes/json_files/jobs/junior_backend.json \
      --resumes-dir /home/shlok/test_resumes/json_files/resumes/

Output:
    - Detailed breakdown per candidate
    - Module scores with visual bars
    - Final ranked list of candidates
    - Score analysis (range, spread, average, std deviation)
    - Average component scores across all candidates


MODE 4: All Combinations (Matrix Testing)
------------------------------------------
Test all resumes against all jobs and generate a complete matrix.

Usage:
    python test_ranking_no_parse.py \
      --resumes-dir /path/to/resumes/ \
      --jobs-dir /path/to/jobs/ \
      --output results.csv

Example:
    python test_ranking_no_parse.py \
      --resumes-dir /home/shlok/test_resumes/json_files/resumes/ \
      --jobs-dir /home/shlok/test_resumes/json_files/jobs/ \
      -o ranking_results.csv

Output:
    - Top 10 matches across all combinations
    - CSV file with all results (if --output specified)
    - Scores for each resume-job pair

-------------------------------------------------------------------------------
COMMON USAGE EXAMPLES
-------------------------------------------------------------------------------

# Test your resume against a specific job
python test_ranking_no_parse.py \
  -r /home/shlok/test_resumes/json_files/resumes/shlok_kalekar.json \
  -j /home/shlok/test_resumes/json_files/jobs/junior_backend.json

# Find best candidates for junior backend role
python test_ranking_no_parse.py \
  -j /home/shlok/test_resumes/json_files/jobs/junior_backend.json \
  --resumes-dir /home/shlok/test_resumes/json_files/resumes/

# Test all 6 resumes against all 3 jobs (18 combinations)
python test_ranking_no_parse.py \
  --resumes-dir /home/shlok/test_resumes/json_files/resumes/ \
  --jobs-dir /home/shlok/test_resumes/json_files/jobs/ \
  -o full_results.csv

-------------------------------------------------------------------------------
UNDERSTANDING THE OUTPUT
-------------------------------------------------------------------------------

Module Scores:
    - Semantic: Text similarity between resume and job description (0-1)
    - Skills: Skills matching score (0-1)
    - Experience: Experience level matching (0-1)
    - Education: Education level matching (0-1)
    - Location: Location/remote preference matching (0-1)

Visual Bars:
    ████████████████████ = 1.0 (100% match)
    ██████████░░░░░░░░░░ = 0.5 (50% match)
    ░░░░░░░░░░░░░░░░░░░░ = 0.0 (0% match)

Weighted Contribution:
    Each module score is multiplied by its weight to get the contribution:
    - Semantic: 0.25 weight
    - Skills: 0.30 weight
    - Experience: 0.25 weight
    - Education: 0.15 weight
    - Location: 0.05 weight
    
    Final Score = sum of all weighted contributions

Data Quality:
    - Completeness: How complete the resume data is (0-100%)
    - Issues: List of validation warnings (e.g., "education_level_unknown")

-------------------------------------------------------------------------------
TROUBLESHOOTING
-------------------------------------------------------------------------------

Error: "File not found"
    → Check that file paths are correct and files exist

Error: "JSON syntax error"
    → Validate JSON files using: python -m json.tool your_file.json

Error: "No resumes loaded successfully"
    → Check JSON format and ensure files are in correct directory

Error: "Module X has score 0.000"
    → That scoring module may not be implemented yet (Person 2's responsibility)

-------------------------------------------------------------------------------
FILE LOCATIONS (Example Project Structure)
-------------------------------------------------------------------------------

ResumeAI-IAI-group-project/
├── tests/
│   └── test_ranking/
│       └── test_ranking_no_parse.py  ← This file
└── test_resumes/
    └── json_files/
        ├── resumes/
        │   ├── shlok_kalekar.json
        │   ├── content_designer.json
        │   ├── data_analyst.json
        │   ├── data_scientist.json
        │   ├── devops.json
        │   └── frontend.json
        └── jobs/
            ├── junior_backend.json
            ├── senior_backend.json
            └── tiktok_content_designer.json

-------------------------------------------------------------------------------
SAMPLE INPUT FILES
-------------------------------------------------------------------------------

Sample Resume JSON (resume.json):
{
  "personal_info": {
    "name": "John Doe",
    "email": "john@email.com",
    "location": {"city": "Pittsburgh", "country": "United States"}
  },
  "experience": [{
    "company": "Tech Corp",
    "title": "Software Engineer",
    "level": "mid",
    "employment_type": "full_time",
    "dates": {"start": "2022-01", "end": "Present"},
    "responsibilities": ["Built REST APIs", "Wrote unit tests"]
  }],
  "education": [{
    "degree": {"level": "bachelor", "field": "Computer Science"},
    "institution": {"name": "State University"}
  }],
  "skills": {
    "technical": {
      "programming_languages": [{"name": "Python"}, {"name": "Java"}],
      "frameworks": [{"name": "Spring Boot"}],
      "databases": [{"name": "PostgreSQL"}],
      "cloud": [{"name": "AWS"}],
      "tools": [{"name": "Git"}]
    }
  },
  "projects": []
}

Sample Job JSON (job.json):
{
  "job_id": "JOB-001",
  "title": "Backend Engineer",
  "company": "StartupCo",
  "description": "Looking for a backend engineer with Java experience",
  "responsibilities": ["Develop REST APIs", "Write tests"],
  "required_skills": {
    "must_have": ["Java", "Spring Boot", "SQL"],
    "nice_to_have": ["AWS", "Docker"]
  },
  "required_experience": {
    "min_years": 2.0,
    "max_years": 5.0
  },
  "education_requirement": {
    "min_level": "bachelor",
    "field": "Computer Science"
  },
  "location_requirement": {
    "location": "San Francisco, CA",
    "remote_allowed": true,
    "hybrid": true
  },
  "salary_range": "$100,000 - $130,000"
}

Note: See actual resume/job JSON files in test_resumes/json_files/ for 
      complete examples with all fields.

===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ranking import RankingEngine, SemanticSimilarityScorer
from src.models.resume_schema import (
    Resume, PersonalInfo, Skills, TechnicalSkills, Skill,
    Experience, Education, Degree, Institution, Project,
    CompanyInfo, TechnicalEnvironment
)
from src.models.job_schema import (
    JobDescription, RequiredSkills, ExperienceRequirement,
    EducationRequirement, LocationRequirement
)
from src.models.base_schema import (
    Location, DateRange, EducationLevel, ExperienceLevel,
    EmploymentType, RemotePreference
)


# ============================================================================
# CONVERSION FUNCTIONS: JSON → Schema Objects
# ============================================================================

def parse_education_level(level_str: str) -> EducationLevel:
    """Convert string to EducationLevel enum."""
    if not level_str:
        return EducationLevel.BACHELOR
    
    level_lower = level_str.lower().strip()
    
    # Map to available enum values
    if level_lower in ["phd", "doctorate", "doctoral"]:
        return EducationLevel.DOCTORATE
    elif level_lower in ["master", "masters", "ms", "ma", "mba", "msc"]:
        return EducationLevel.MASTER
    elif level_lower in ["bachelor", "bachelors", "bs", "ba", "bsc", "undergraduate"]:
        return EducationLevel.BACHELOR
    elif level_lower in ["associate", "associates", "as", "aa"]:
        return EducationLevel.ASSOCIATE
    elif level_lower in ["high_school", "highschool", "high school", "diploma"]:
        return EducationLevel.HIGH_SCHOOL
    else:
        # Default to BACHELOR if unknown
        return EducationLevel.BACHELOR


def parse_experience_level(level_str: str) -> ExperienceLevel:
    """Convert string to ExperienceLevel enum."""
    if not level_str:
        return ExperienceLevel.MID
    
    level_lower = level_str.lower().strip()
    
    # Map all variations to the three main levels: ENTRY, MID, SENIOR
    if level_lower in ["entry", "junior", "intern", "graduate"]:
        return ExperienceLevel.ENTRY
    elif level_lower in ["mid", "intermediate", "mid-level"]:
        return ExperienceLevel.MID
    elif level_lower in ["senior", "lead", "principal", "staff", "architect"]:
        return ExperienceLevel.SENIOR
    else:
        # Default to MID if unknown
        return ExperienceLevel.MID


def parse_employment_type(type_str: str) -> EmploymentType:
    """Convert string to EmploymentType enum."""
    if not type_str:
        return EmploymentType.FULL_TIME
    
    type_lower = type_str.lower().strip().replace("_", " ").replace("-", " ")
    
    # Map to available enum values
    if any(x in type_lower for x in ["full time", "fulltime", "full"]):
        return EmploymentType.FULL_TIME
    elif any(x in type_lower for x in ["part time", "parttime", "part"]):
        return EmploymentType.PART_TIME
    elif any(x in type_lower for x in ["contract", "contractor", "freelance", "consulting"]):
        return EmploymentType.CONTRACT
    elif any(x in type_lower for x in ["intern", "internship"]):
        return EmploymentType.INTERN
    elif any(x in type_lower for x in ["temporary", "temp"]):
        return EmploymentType.TEMPORARY
    else:
        # Default to FULL_TIME if unknown
        return EmploymentType.FULL_TIME


def load_resume_from_json(filepath: Path) -> Resume:
    """Load a resume from JSON file and convert to Resume object."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON syntax error in {filepath.name}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {filepath.name}: {e}")
    
    # Parse personal info
    pi = data.get("personal_info", {})
    personal_info = PersonalInfo(
        name=pi.get("name", "Unknown"),
        email=pi.get("email", "unknown@email.com"),
        phone=pi.get("phone", "+1-000-000-0000"),
        location=Location(
            city=pi.get("location", {}).get("city", "Unknown"),
            state=pi.get("location", {}).get("state"),
            country=pi.get("location", {}).get("country", "Unknown"),
            remote_preference=pi.get("location", {}).get("remote_preference", "hybrid")
        ),
        summary=pi.get("summary", ""),
        linkedin=pi.get("linkedin"),
        github=pi.get("github")
    )
    
    # Parse experience
    experiences = []
    for exp_data in data.get("experience", []):
        company_info = None
        if exp_data.get("company_info"):
            company_info = CompanyInfo(
                industry=exp_data["company_info"].get("industry"),
                size=exp_data["company_info"].get("size")
            )
        
        tech_env = None
        if exp_data.get("technical_environment"):
            tech_env = TechnicalEnvironment(
                technologies=exp_data["technical_environment"].get("technologies", []),
                methodologies=exp_data["technical_environment"].get("methodologies", []),
                tools=exp_data["technical_environment"].get("tools", [])
            )
        
        dates = None
        if exp_data.get("dates"):
            dates = DateRange(
                start=exp_data["dates"].get("start"),
                end=exp_data["dates"].get("end", "Present")
            )
        
        experience = Experience(
            company=exp_data.get("company", "Unknown"),
            company_info=company_info,
            title=exp_data.get("title", "Unknown"),
            level=parse_experience_level(exp_data.get("level", "mid")),
            employment_type=parse_employment_type(exp_data.get("employment_type", "full_time")),
            dates=dates,
            responsibilities=exp_data.get("responsibilities", []),
            technical_environment=tech_env
        )
        experiences.append(experience)
    
    # Parse education
    educations = []
    for edu_data in data.get("education", []):
        degree_data = edu_data.get("degree", {})
        degree = Degree(
            level=parse_education_level(degree_data.get("level", "bachelor")),
            field=degree_data.get("field", "Unknown"),
            major=degree_data.get("major")
        )
        
        inst_data = edu_data.get("institution", {})
        institution = Institution(
            name=inst_data.get("name", "Unknown"),
            location=inst_data.get("location"),
            accreditation=inst_data.get("accreditation")
        )
        
        dates = None
        if edu_data.get("dates"):
            dates = DateRange(
                start=edu_data["dates"].get("start"),
                end=edu_data["dates"].get("end")
            )
        
        education = Education(
            degree=degree,
            institution=institution,
            dates=dates,
            achievements=edu_data.get("achievements")
        )
        educations.append(education)
    
    # Parse skills
    skills_data = data.get("skills", {})
    tech_data = skills_data.get("technical", {})
    
    def parse_skills_list(skill_list):
        return [Skill(name=s["name"], level=s.get("level", "intermediate")) 
                for s in skill_list]
    
    technical_skills = TechnicalSkills(
        programming_languages=parse_skills_list(tech_data.get("programming_languages", [])),
        frameworks=parse_skills_list(tech_data.get("frameworks", [])),
        databases=parse_skills_list(tech_data.get("databases", [])),
        cloud=parse_skills_list(tech_data.get("cloud", [])),
        tools=parse_skills_list(tech_data.get("tools", []))
    )
    
    skills = Skills(
        technical=technical_skills,
        languages=skills_data.get("languages", [])
    )
    
    # Parse projects
    projects = []
    for proj_data in data.get("projects", []):
        project = Project(
            name=proj_data.get("name", "Unknown"),
            description=proj_data.get("description", ""),
            technologies=proj_data.get("technologies", []),
            role=proj_data.get("role"),
            url=proj_data.get("url"),
            impact=proj_data.get("impact")
        )
        projects.append(project)
    
    return Resume(
        personal_info=personal_info,
        experience=experiences,
        education=educations,
        skills=skills,
        projects=projects if projects else None,
        certifications=data.get("certifications", "")
    )


def load_job_from_json(filepath: Path) -> JobDescription:
    """Load a job description from JSON file and convert to JobDescription object."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse required skills
    req_skills_data = data.get("required_skills", {})
    required_skills = RequiredSkills(
        must_have=req_skills_data.get("must_have", []),
        nice_to_have=req_skills_data.get("nice_to_have", [])
    )
    
    # Parse experience requirement
    exp_req_data = data.get("required_experience", {})
    required_experience = ExperienceRequirement(
        min_years=exp_req_data.get("min_years", 0.0),
        max_years=exp_req_data.get("max_years"),
        preferred_years=exp_req_data.get("preferred_years")
    )
    
    # Parse education requirement
    edu_req_data = data.get("education_requirement", {})
    education_requirement = EducationRequirement(
        min_level=parse_education_level(edu_req_data.get("min_level", "bachelor")),
        preferred_level=parse_education_level(edu_req_data.get("preferred_level", "master")) 
            if edu_req_data.get("preferred_level") else None,
        field=edu_req_data.get("field")
    )
    
    # Parse location requirement
    loc_req_data = data.get("location_requirement", {})
    location_requirement = LocationRequirement(
        location=loc_req_data.get("location", "Unknown"),
        remote_allowed=loc_req_data.get("remote_allowed", False),
        hybrid=loc_req_data.get("hybrid", False)
    )
    
    return JobDescription(
        job_id=data.get("job_id", "UNKNOWN"),
        title=data.get("title", "Unknown Position"),
        company=data.get("company", "Unknown Company"),
        role=data.get("role"),
        description=data.get("description", ""),
        responsibilities=data.get("responsibilities", []),
        required_skills=required_skills,
        required_experience=required_experience,
        education_requirement=education_requirement,
        location_requirement=location_requirement,
        salary_range=data.get("salary_range"),
        benefits=data.get("benefits", []),
        company_name=data.get("company_name", data.get("company", "Unknown")),
        company_info=data.get("company_info", {})
    )


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_single_pair(resume_path: Path, job_path: Path, verbose: bool = True):
    """Test a single resume-job pair."""
    if verbose:
        print("=" * 80)
        print(f"Testing: {resume_path.name} vs {job_path.name}")
        print("=" * 80)
    
    # Load data
    resume = load_resume_from_json(resume_path)
    job = load_job_from_json(job_path)
    
    if verbose:
        print(f"\nResume: {resume.personal_info.name}")
        print(f"  Experience: {resume.calculate_total_experience():.1f} years")
        print(f"  Education: {resume.get_highest_education_level()}")
        print(f"  Skills: {len(resume.skills.get_all_skills_flat())} total")
        
        print(f"\nJob: {job.title} at {job.company_name}")
        print(f"  Required Experience: {job.required_experience.min_years}+ years")
        print(f"  Required Education: {job.education_requirement.min_level}")
        print(f"  Required Skills: {len(job.required_skills.must_have)} must-have")
    
    # Rank
    engine = RankingEngine()
    result = engine.rank_single_resume(resume, job)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"Final Score: {result['final_score']:.3f} ({result['final_score']*100:.1f}%)")
        
        print(f"\nScore Breakdown:")
        for component, score in sorted(result['individual_scores'].items(), 
                                      key=lambda x: x[1], reverse=True):
            weight = result['weights_used'][component]
            contrib = score * weight
            print(f"  {component:12} {score:.3f} × {weight:.2f} = {contrib:.3f}")
        
        print(f"\nData Quality: {result['data_quality']['completeness_score']:.1%}")
        print("=" * 80)
    
    return result


def test_one_resume_vs_many_jobs(resume_path: Path, jobs_dir: Path):
    """Test one resume against multiple jobs to find best fit."""
    print("=" * 80)
    print(f"Testing Resume: {resume_path.name}")
    print(f"Against All Jobs in: {jobs_dir}")
    print("=" * 80)
    
    resume = load_resume_from_json(resume_path)
    print(f"\nCandidate: {resume.personal_info.name}")
    print(f"  Experience: {resume.calculate_total_experience():.1f} years")
    print(f"  Education: {resume.get_highest_education_level()}")
    
    # Load all jobs
    job_files = sorted(jobs_dir.glob("*.json"))
    
    if not job_files:
        print(f"\nERROR: No job files found in {jobs_dir}")
        return
    
    print(f"\nTesting against {len(job_files)} job(s)...")
    
    engine = RankingEngine()
    results = []
    
    for job_file in job_files:
        job = load_job_from_json(job_file)
        result = engine.rank_single_resume(resume, job)
        results.append({
            'job_file': job_file.name,
            'job_title': job.title,
            'company': job.company_name,
            'score': result['final_score'],
            'result': result
        })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display results
    print("\n" + "=" * 80)
    print("BEST JOB MATCHES (Ranked)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<8} {'Company':<25} {'Job Title':<30} {'File'}")
    print("-" * 80)
    
    for rank, r in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        print(f"{medal:<6} {r['score']:<8.3f} {r['company']:<25} {r['job_title']:<30} {r['job_file']}")
    
    print("\n" + "=" * 80)
    print(f"✓ Best Match: {results[0]['job_title']} at {results[0]['company']} ({results[0]['score']:.1%})")
    print("=" * 80)


def test_one_job_vs_many_resumes(job_path: Path, resumes_dir: Path):
    """Test one job against multiple resumes to find best candidates."""
    print("=" * 80)
    print(f"Testing Job: {job_path.name}")
    print(f"Against All Resumes in: {resumes_dir}")
    print("=" * 80)
    
    job = load_job_from_json(job_path)
    print(f"\nPosition: {job.title} at {job.company_name}")
    print(f"  Required Experience: {job.required_experience.min_years}+ years")
    print(f"  Required Education: {job.education_requirement.min_level}")
    print(f"  Required Skills: {', '.join(job.required_skills.must_have[:5])}...")
    
    # Load all resumes
    resume_files = sorted(resumes_dir.glob("*.json"))
    
    if not resume_files:
        print(f"\nERROR: No resume files found in {resumes_dir}")
        return
    
    print(f"\nTesting against {len(resume_files)} resume(s)...")
    
    engine = RankingEngine()
    resumes = []
    detailed_results = []
    
    for resume_file in resume_files:
        try:
            resume = load_resume_from_json(resume_file)
            resumes.append(resume)
        except Exception as e:
            print(f"  ✗ Error loading {resume_file.name}: {e}")
    
    if not resumes:
        print("\nERROR: No resumes loaded successfully")
        return
    
    # Process each resume and collect detailed scores
    print("\n" + "=" * 80)
    print("DETAILED SCORE BREAKDOWN PER CANDIDATE")
    print("=" * 80)
    
    for idx, resume in enumerate(resumes, 1):
        print(f"\n{'─' * 80}")
        print(f"[{idx}/{len(resumes)}] {resume.personal_info.name}")
        print(f"{'─' * 80}")
        
        result = engine.rank_single_resume(resume, job)
        detailed_results.append(result)
        
        # Show candidate profile
        print(f"Profile:")
        print(f"  Experience: {resume.calculate_total_experience():.1f} years")
        print(f"  Education: {resume.get_highest_education_level()}")
        print(f"  Skills: {len(resume.skills.get_all_skills_flat())} total")
        print(f"  Location: {resume.personal_info.location.city}, {resume.personal_info.location.country}")
        
        # Show individual module scores
        print(f"\nModule Scores:")
        for component, score in sorted(result['individual_scores'].items(), key=lambda x: x[1], reverse=True):
            weight = result['weights_used'][component]
            contribution = score * weight
            
            # Create visual bar (20 chars max)
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"  {component.capitalize():12} {bar} {score:.3f} (weight: {weight:.2f}, contrib: {contribution:.3f})")
        
        # Show final score
        print(f"\n  {'Final Score':12} {'█' * int(result['final_score'] * 20)}{' ' * (20 - int(result['final_score'] * 20))} {result['final_score']:.3f} ({result['final_score']*100:.1f}%)")
        
        # Show data quality
        completeness = result['data_quality']['completeness_score']
        issues = result['data_quality'].get('validation_issues', [])
        print(f"\n  Data Quality: {completeness:.1%} complete")
        if issues:
            print(f"  Issues: {', '.join(issues)}")
    
    # Rank all resumes
    print("\n" + "=" * 80)
    print("RANKING ALL CANDIDATES")
    print("=" * 80)
    
    ranked_df = engine.rank_multiple_resumes(resumes, job)
    
    # Display final rankings
    print(f"\n{'Rank':<6} {'Score':<8} {'Name':<30} {'Exp (yrs)':<12} {'Edu':<8} {'Skills':<8}")
    print("-" * 80)
    
    for idx, row in ranked_df.iterrows():
        if idx == 0:
            medal = "🥇"
        elif idx == 1:
            medal = "🥈"
        elif idx == 2:
            medal = "🥉"
        else:
            medal = f"#{idx+1}"
        
        print(f"{medal:<6} {row['final_score']:<8.3f} {row['name']:<30} "
              f"{row['years_experience']:<12.1f} {row['education_level']:<8} "
              f"{row['num_skills']:<8}")
    
    # Show score distribution
    print("\n" + "=" * 80)
    print("SCORE ANALYSIS")
    print("=" * 80)
    
    scores = ranked_df['final_score'].values
    print(f"Score Range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Score Spread: {scores.max() - scores.min():.3f}")
    print(f"Average Score: {scores.mean():.3f}")
    print(f"Std Deviation: {scores.std():.3f}")
    
    # Show component score averages
    print(f"\nAverage Component Scores:")
    all_scores = {
        'semantic': [],
        'skills': [],
        'experience': [],
        'education': [],
        'location': []
    }
    
    for result in detailed_results:
        for component in all_scores.keys():
            if component in result['individual_scores']:
                all_scores[component].append(result['individual_scores'][component])
    
    for component, scores_list in all_scores.items():
        if scores_list:
            avg = sum(scores_list) / len(scores_list)
            print(f"  {component.capitalize():12} {avg:.3f}")
    
    print("\n" + "=" * 80)
    print(f"✓ Top Candidate: {ranked_df.iloc[0]['name']} with {ranked_df.iloc[0]['final_score']:.1%} match")
    print("=" * 80)


def test_all_combinations(resumes_dir: Path, jobs_dir: Path, output_file: Path = None):
    """Test all resume-job combinations and generate a matrix."""
    print("=" * 80)
    print("TESTING ALL COMBINATIONS")
    print("=" * 80)
    
    resume_files = sorted(resumes_dir.glob("*.json"))
    job_files = sorted(jobs_dir.glob("*.json"))
    
    print(f"\nResumes: {len(resume_files)}")
    print(f"Jobs: {len(job_files)}")
    print(f"Total combinations: {len(resume_files) * len(job_files)}")
    
    if not resume_files or not job_files:
        print("\nERROR: Need at least one resume and one job")
        return
    
    # Load all data
    resumes = {}
    for rf in resume_files:
        try:
            resumes[rf.stem] = load_resume_from_json(rf)
        except Exception as e:
            print(f"✗ Error loading {rf.name}: {e}")
    
    jobs = {}
    for jf in job_files:
        try:
            jobs[jf.stem] = load_job_from_json(jf)
        except Exception as e:
            print(f"✗ Error loading {jf.name}: {e}")
    
    print(f"\n✓ Loaded {len(resumes)} resumes and {len(jobs)} jobs")
    print("\nRunning all combinations...")
    
    # Run all combinations
    engine = RankingEngine()
    results = []
    
    for resume_name, resume in resumes.items():
        for job_name, job in jobs.items():
            result = engine.rank_single_resume(resume, job)
            results.append({
                'resume': resume_name,
                'candidate': resume.personal_info.name,
                'job': job_name,
                'position': job.title,
                'company': job.company_name,
                'score': result['final_score'],
                'semantic': result['individual_scores'].get('semantic', 0),
                'skills': result['individual_scores'].get('skills', 0),
                'experience': result['individual_scores'].get('experience', 0),
                'education': result['individual_scores'].get('education', 0),
            })
    
    # Display summary
    print("\n" + "=" * 80)
    print("TOP 10 MATCHES ACROSS ALL COMBINATIONS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<8} {'Candidate':<25} {'Position':<30}")
    print("-" * 80)
    
    top_matches = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    for rank, match in enumerate(top_matches, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        print(f"{medal:<6} {match['score']:<8.3f} {match['candidate']:<25} {match['position']:<30}")
    
    # Save to file if requested
    if output_file:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda x: x['score'], reverse=True))
        print(f"\n✓ Detailed results saved to: {output_file}")
    
    print("=" * 80)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test ranking engine in isolation with JSON mock data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single resume-job pair
  python test_ranking_no_parse.py \\
    --resume /path/to/resumes/senior_dev.json \\
    --job /path/to/jobs/backend_role.json
  
  # Test one resume against all jobs (find best job fit)
  python test_ranking_no_parse.py \\
    --resume /path/to/resumes/senior_dev.json \\
    --jobs-dir /path/to/jobs/
  
  # Test one job against all resumes (find best candidates)
  python test_ranking_no_parse.py \\
    --job /path/to/jobs/backend_role.json \\
    --resumes-dir /path/to/resumes/
  
  # Test all combinations
  python test_ranking_no_parse.py \\
    --resumes-dir /path/to/resumes/ \\
    --jobs-dir /path/to/jobs/ \\
    --output results.csv
        """
    )
    
    parser.add_argument('-r', '--resume', type=str, help='Path to single resume JSON file')
    parser.add_argument('-j', '--job', type=str, help='Path to single job JSON file')
    parser.add_argument('--resumes-dir', type=str, help='Path to directory with resume JSON files')
    parser.add_argument('--jobs-dir', type=str, help='Path to directory with job JSON files')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file for all combinations')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        # Test single pair
        if args.resume and args.job:
            test_single_pair(Path(args.resume), Path(args.job), verbose=True)
        
        # Test one resume vs many jobs
        elif args.resume and args.jobs_dir:
            test_one_resume_vs_many_jobs(Path(args.resume), Path(args.jobs_dir))
        
        # Test one job vs many resumes
        elif args.job and args.resumes_dir:
            test_one_job_vs_many_resumes(Path(args.job), Path(args.resumes_dir))
        
        # Test all combinations
        elif args.resumes_dir and args.jobs_dir:
            output_path = Path(args.output) if args.output else None
            test_all_combinations(Path(args.resumes_dir), Path(args.jobs_dir), output_path)
        
        else:
            parser.print_help()
            print("\nERROR: Invalid argument combination")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()