"""
Test scoring modules integration with ranking engine.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.parsers import ResumeParser, JobParser
from src.scoring import SkillsScorer, ExperienceScorer, EducationScorer, LocationScorer
from src.ranking import RankingEngine


def test_scoring_modules():
    """Test all scoring modules with sample resume and job."""
    print("=" * 70)
    print("TESTING SCORING MODULES INTEGRATION")
    print("=" * 70)

    # Parse sample resume and job
    resume_parser = ResumeParser()
    job_parser = JobParser()

    resume_path = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    job_path = project_root / "tests" / "test_job_parser" / "sample_job_description.txt"

    if not resume_path.exists():
        print(f"ERROR: Resume not found at {resume_path}")
        return

    if not job_path.exists():
        print(f"ERROR: Job description not found at {job_path}")
        return

    print("\n1. Parsing resume and job description...")
    resume = resume_parser.parse_from_docx(str(resume_path))

    with open(job_path, 'r') as f:
        job_text = f.read()
    job = job_parser.parse(job_text)

    print(f"   ✓ Parsed resume: {resume.personal_info.name}")
    print(f"   ✓ Parsed job: {job.title}")

    # Test individual scorers
    print("\n2. Testing individual scorers...")

    # Skills
    print("\n   [Skills Scorer]")
    skills_scorer = SkillsScorer()
    skills_score = skills_scorer.score(resume, job)
    skills_detail = skills_scorer.score_detailed(resume, job)
    print(f"   Score: {skills_score:.3f} ({skills_score*100:.1f}/100)")
    print(f"   Matched: {skills_detail['matched_required']}/{skills_detail['total_required']} required skills")
    print(f"   Exact matches: {len(skills_detail['exact_matches'])}")
    print(f"   Fuzzy matches: {len(skills_detail['fuzzy_matches'])}")
    print(f"   Nice-to-have: {skills_detail['nice_to_have_count']}")

    # Experience
    print("\n   [Experience Scorer]")
    exp_scorer = ExperienceScorer()
    exp_score = exp_scorer.score(resume, job)
    exp_detail = exp_scorer.score_detailed(resume, job)
    print(f"   Score: {exp_score:.3f} ({exp_score*100:.1f}/100)")
    print(f"   Status: {exp_detail['status']}")
    print(f"   {exp_detail['message']}")

    # Education
    print("\n   [Education Scorer]")
    edu_scorer = EducationScorer()
    edu_score = edu_scorer.score(resume, job)
    edu_detail = edu_scorer.score_detailed(resume, job)
    print(f"   Score: {edu_score:.3f} ({edu_score*100:.1f}/100)")
    print(f"   Status: {edu_detail['status']}")
    print(f"   {edu_detail['message']}")

    # Location
    print("\n   [Location Scorer]")
    loc_scorer = LocationScorer()
    loc_score = loc_scorer.score(resume, job)
    loc_detail = loc_scorer.score_detailed(resume, job)
    print(f"   Score: {loc_score:.3f} ({loc_score*100:.1f}/100)")
    print(f"   Status: {loc_detail['status']}")
    print(f"   {loc_detail['message']}")

    # Test ranking engine integration
    print("\n3. Testing Ranking Engine Integration...")
    engine = RankingEngine()

    # Create scoring modules dict
    scoring_modules = {
        'skills': skills_scorer,
        'experience': exp_scorer,
        'education': edu_scorer,
        'location': loc_scorer
    }

    result = engine.rank_single_resume(resume, job, scoring_modules)

    print(f"\n   Final Ranking Result:")
    print(f"   Candidate: {result['resume_data']['name']}")
    print(f"   Final Score: {result['final_score']:.3f}")

    print(f"\n   Score Breakdown:")
    for component, score in result['individual_scores'].items():
        weight = result['weights_used'][component]
        contribution = score * weight
        print(f"      {component.capitalize():12} {score:.3f} × {weight:.2f} = {contribution:.3f}")

    print(f"\n   Data Quality:")
    print(f"      Completeness: {result['data_quality']['completeness_score']:.1%}")
    if result['data_quality']['validation_issues']:
        print(f"      Issues: {', '.join(result['data_quality']['validation_issues'])}")
    else:
        print(f"      Issues: None")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_scoring_modules()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
