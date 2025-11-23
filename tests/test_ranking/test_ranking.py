"""
Test the ranking engine module with multiple resumes.
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
# Current file is at: tests/test_ranking/test_ranking.py
# Project root is 2 levels up
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Debug: Project root = {project_root}")
print(f"Debug: sys.path[0] = {sys.path[0]}")

from src.parsers import ResumeParser, JobParser
from src.ranking import RankingEngine, compute_semantic_similarity


def read_job_from_file(job_file_path):
    """Read job description from a text file."""
    with open(job_file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_resume_files(resume_dir: Path) -> List[Path]:
    """Get all resume files from a directory."""
    resume_files = []
    
    # Support common resume formats
    for ext in ['*.docx', '*.pdf', '*.doc']:
        resume_files.extend(resume_dir.glob(ext))
    
    return sorted(resume_files)


def test_multiple_resumes_similarity(resume_dir: Path, job_path: Path):
    """Test semantic similarity for multiple resumes."""
    print("=" * 80)
    print("TEST 1: Semantic Similarity for Multiple Resumes")
    print("=" * 80)
    
    # Parse job description
    job_parser = JobParser()
    
    if not job_path.exists():
        print(f"ERROR: Job description file not found at {job_path}")
        return
    
    print(f"\nReading job description from: {job_path}")
    job_text = read_job_from_file(job_path)
    job = job_parser.parse(job_text)
    
    # Get all resume files
    resume_files = get_resume_files(resume_dir)
    
    if not resume_files:
        print(f"ERROR: No resume files found in {resume_dir}")
        return
    
    print(f"\nFound {len(resume_files)} resume(s) to process")
    print("-" * 80)
    
    # Parse and compute similarity for each resume
    resume_parser = ResumeParser()
    results = []
    
    for idx, resume_path in enumerate(resume_files, 1):
        print(f"\n[{idx}/{len(resume_files)}] Processing: {resume_path.name}")
        
        try:
            # Parse resume
            resume = resume_parser.parse_from_docx(str(resume_path))
            
            # Compute similarity
            similarity = compute_semantic_similarity(resume, job)
            
            results.append({
                'file': resume_path.name,
                'name': resume.name or 'Unknown',
                'similarity': similarity
            })
            
            print(f"    Candidate: {resume.name or 'Unknown'}")
            print(f"    Similarity Score: {similarity:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error processing resume: {e}")
            continue
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIMILARITY RANKING SUMMARY")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<8} {'Candidate':<30} {'File'}")
    print("-" * 80)
    
    for rank, result in enumerate(results, 1):
        print(f"{rank:<6} {result['similarity']:<8.3f} {result['name']:<30} {result['file']}")
    
    print("=" * 80)


def test_ranking_engine_multiple(resume_dir: Path, job_path: Path):
    """Test the full ranking engine with multiple resumes."""
    print("\n" + "=" * 80)
    print("TEST 2: Full Ranking Engine for Multiple Resumes")
    print("=" * 80)
    
    # Parse job description
    job_parser = JobParser()
    
    if not job_path.exists():
        print(f"ERROR: Job description file not found at {job_path}")
        return
    
    print(f"\nReading job description from: {job_path}")
    job_text = read_job_from_file(job_path)
    job = job_parser.parse(job_text)
    
    # Get all resume files
    resume_files = get_resume_files(resume_dir)
    
    if not resume_files:
        print(f"ERROR: No resume files found in {resume_dir}")
        return
    
    print(f"\nFound {len(resume_files)} resume(s) to process")
    print("-" * 80)
    
    # Initialize ranking engine
    print("\nInitializing ranking engine...")
    engine = RankingEngine()
    
    # Parse and rank each resume
    resume_parser = ResumeParser()
    results = []
    
    for idx, resume_path in enumerate(resume_files, 1):
        print(f"\n[{idx}/{len(resume_files)}] Processing: {resume_path.name}")
        
        try:
            # Parse resume
            resume = resume_parser.parse_from_docx(str(resume_path))
            
            # Rank resume
            result = engine.rank_single_resume(resume, job)
            
            results.append({
                'file': resume_path.name,
                'result': result
            })
            
            print(f"    Candidate: {result['resume_data']['name']}")
            print(f"    Final Score: {result['final_score']:.3f}")
            print(f"    Completeness: {result['data_quality']['completeness_score']:.1%}")
            
        except Exception as e:
            print(f"    ❌ Error processing resume: {e}")
            continue
    
    # Sort by final score (descending)
    results.sort(key=lambda x: x['result']['final_score'], reverse=True)
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("FINAL RANKING RESULTS")
    print("=" * 80)
    
    for rank, item in enumerate(results, 1):
        result = item['result']
        print(f"\n{'='*80}")
        print(f"RANK #{rank}: {result['resume_data']['name']}")
        print(f"File: {item['file']}")
        print(f"{'='*80}")
        print(f"Final Score: {result['final_score']:.3f}")
        
        print(f"\nScore Breakdown:")
        for component, score in result['individual_scores'].items():
            weight = result['weights_used'][component]
            contribution = score * weight
            print(f"  {component.capitalize():12} {score:.3f} × {weight:.2f} = {contribution:.3f}")
        
        print(f"\nData Quality:")
        print(f"  Completeness: {result['data_quality']['completeness_score']:.1%}")
        if result['data_quality']['validation_issues']:
            print(f"  Issues: {', '.join(result['data_quality']['validation_issues'])}")
        else:
            print(f"  Issues: None")
    
    # Print final ranking table
    print("\n" + "=" * 80)
    print("RANKING SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<8} {'Candidate':<30} {'Completeness':<13} {'File'}")
    print("-" * 80)
    
    for rank, item in enumerate(results, 1):
        result = item['result']
        completeness = result['data_quality']['completeness_score']
        print(f"{rank:<6} {result['final_score']:<8.3f} {result['resume_data']['name']:<30} {completeness:<13.1%} {item['file']}")
    
    print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test the ranking engine with multiple resumes and a job description.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specify resume directory and job file
  python test_ranking.py -d path/to/resumes/ -j path/to/job.txt
  
  # Process specific resume file
  python test_ranking.py -r path/to/resume.docx -j path/to/job.txt
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        default=None,
        help='Path to directory containing resume files (required if -r not specified)'
    )
    
    parser.add_argument(
        '-r', '--resume',
        type=str,
        default=None,
        help='Path to single resume file (.docx format) (required if -d not specified)'
    )
    
    parser.add_argument(
        '-j', '--job',
        type=str,
        required=True,
        help='Path to job description file (.txt format) [REQUIRED]'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        
        # Validate that either directory or resume is provided
        if not args.directory and not args.resume:
            print("ERROR: You must specify either -d/--directory or -r/--resume")
            print("Usage examples:")
            print("  python test_ranking.py -d path/to/resumes/ -j path/to/job.txt")
            print("  python test_ranking.py -r path/to/resume.docx -j path/to/job.txt")
            sys.exit(1)
        
        # Set job path (required argument)
        job_path = Path(args.job)
        
        if not job_path.exists():
            print(f"ERROR: Job description file not found: {job_path}")
            sys.exit(1)
        
        # Set resume directory
        if args.directory:
            resume_dir = Path(args.directory)
        else:
            # Single resume - use its parent directory
            resume_path = Path(args.resume)
            if not resume_path.exists():
                print(f"ERROR: Resume file not found: {resume_path}")
                sys.exit(1)
            resume_dir = resume_path.parent
        
        if not resume_dir.exists():
            print(f"ERROR: Resume directory not found: {resume_dir}")
            sys.exit(1)
        
        # Run tests
        test_multiple_resumes_similarity(resume_dir, job_path)
        test_ranking_engine_multiple(resume_dir, job_path)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()