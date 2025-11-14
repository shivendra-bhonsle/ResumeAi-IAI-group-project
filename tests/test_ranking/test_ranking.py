"""
Test the ranking engine module.
"""

import sys
from pathlib import Path

# Add project root to path
# Current file is at: tests/test_ranking/test_ranking.py
# Project root is 2 levels up
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Debug: Project root = {project_root}")
print(f"Debug: sys.path[0] = {sys.path[0]}")

from src.parsers import ResumeParser, JobParser
from src.ranking import RankingEngine, compute_semantic_similarity


def test_semantic_similarity():
    """Test semantic similarity computation."""
    print("=" * 60)
    print("TEST 1: Semantic Similarity")
    print("=" * 60)
    
    # Parse sample resume and job
    resume_parser = ResumeParser()
    job_parser = JobParser()
    
    # Use the test resume from Person 1
    resume_path = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    
    if not resume_path.exists():
        print(f"ERROR: Resume file not found at {resume_path}")
        print("Please provide a valid resume path.")
        return
    
    resume = resume_parser.parse_from_docx(str(resume_path))
    
    job_text = """
    Software Engineer
    
    We're looking for a Software Engineer with 3+ years of experience in Python, AWS, and distributed systems.
    
    Requirements:
    - Bachelor's degree in Computer Science
    - 3-5 years of software development experience
    - Strong skills in Python, Java, and cloud platforms
    - Experience with databases and APIs
    """
    job = job_parser.parse(job_text)
    
    # Compute similarity
    print("\nComputing semantic similarity...")
    similarity = compute_semantic_similarity(resume, job)
    
    print(f"\n✓ Semantic Similarity Score: {similarity:.3f}")
    print("=" * 60)


def test_ranking_engine():
    """Test the full ranking engine."""
    print("\n" + "=" * 60)
    print("TEST 2: Ranking Engine")
    print("=" * 60)
    
    # Parse resumes and job
    resume_parser = ResumeParser()
    job_parser = JobParser()
    
    # Parse a sample resume
    resume_path = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    
    if not resume_path.exists():
        print(f"ERROR: Resume file not found at {resume_path}")
        return
    
    resume = resume_parser.parse_from_docx(str(resume_path))
    
    job_text = """
    Senior Software Engineer
    
    Looking for an experienced engineer with cloud expertise.
    
    Requirements:
    - 5+ years experience
    - Master's degree preferred
    - Python, AWS, Docker, Kubernetes
    - Remote work available
    """
    job = job_parser.parse(job_text)
    
    # Initialize ranking engine
    print("\nInitializing ranking engine...")
    engine = RankingEngine()
    
    # Rank single resume (without Person 2's modules for now)
    print("Ranking candidate...")
    result = engine.rank_single_resume(resume, job)
    
    print(f"\nCandidate: {result['resume_data']['name']}")
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"\nScore Breakdown:")
    for component, score in result['individual_scores'].items():
        weight = result['weights_used'][component]
        print(f"  {component.capitalize():12} {score:.3f} (weight: {weight:.2f})")
    
    print(f"\nData Quality:")
    print(f"  Completeness: {result['data_quality']['completeness_score']:.1%}")
    if result['data_quality']['validation_issues']:
        print(f"  Issues: {', '.join(result['data_quality']['validation_issues'])}")
    else:
        print(f"  Issues: None")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_semantic_similarity()
        test_ranking_engine()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()