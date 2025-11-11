"""
Quick test script for job description parsing.

Usage:
    python test_job_parsing.py <path_to_job_description.txt>

This script will:
1. Parse the job description using JobParser
2. Display extracted requirements
3. Show ML features
4. Validate data quality
"""

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

def test_job_parsing(file_path: str):
    """Test job description parsing with a .txt file"""

    print("=" * 70)
    print("RESUMEAI - JOB DESCRIPTION PARSING TEST")
    print("=" * 70)
    print(f"\nFile: {file_path}")

    # Check if file exists
    if not Path(file_path).exists():
        print(f"\n❌ ERROR: File not found: {file_path}")
        return False

    # Check file extension
    if not file_path.endswith('.txt'):
        print(f"\n❌ ERROR: File must be a .txt file")
        print(f"   Got: {Path(file_path).suffix}")
        return False

    try:
        # Read job description text
        print("\n📥 Reading job description text...")
        with open(file_path, 'r', encoding='utf-8') as f:
            job_text = f.read()

        if not job_text.strip():
            print("\n❌ ERROR: File is empty!")
            print("   Please paste a job description into the file and try again.")
            return False

        # Import parser
        print("📥 Importing JobParser...")
        from src.parsers import JobParser

        # Create parser
        print("🔧 Initializing parser...")
        parser = JobParser()

        # Parse job description
        print(f"🤖 Parsing job description with Gemini API...")
        print("   (This may take a few seconds...)")
        job = parser.parse(job_text)

        print("\n✅ PARSING SUCCESSFUL!\n")

        # Display results
        display_results(job)

        # Show ML features
        display_ml_features(job)

        # Validate
        validate_job(job)

        return True

    except ImportError as e:
        print(f"\n❌ IMPORT ERROR: {e}")
        print("\n💡 Make sure you've installed dependencies:")
        print("   pip install -r requirements.txt")
        return False

    except Exception as e:
        print(f"\n❌ PARSING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_results(job):
    """Display parsed job description information"""

    print("─" * 70)
    print("JOB INFORMATION")
    print("─" * 70)
    print(f"💼 Title:    {job.title}")
    print(f"🏢 Company:  {job.company}")

    # Location from location_requirement
    if job.location_requirement and job.location_requirement.location:
        loc_str = job.location_requirement.location
        if loc_str and loc_str != "Unknown":
            print(f"📍 Location: {loc_str}")

    if job.location_requirement:
        if job.location_requirement.remote_allowed:
            print(f"🏠 Remote:   Allowed")
        if job.location_requirement.hybrid:
            print(f"🏠 Hybrid:   Yes")

    if job.description:
        desc = job.description[:200] + "..." if len(job.description) > 200 else job.description
        print(f"\n📝 Description:\n   {desc}")

    print("\n" + "─" * 70)
    print("REQUIREMENTS")
    print("─" * 70)

    # Experience
    if job.required_experience and (job.required_experience.min_years or job.required_experience.max_years):
        print(f"\n⏱️  Experience Required:")
        exp = job.required_experience
        if exp.min_years or exp.max_years:
            min_str = f"{exp.min_years}" if exp.min_years else "0"
            max_str = f"{exp.max_years}" if exp.max_years else "+"
            print(f"   Years: {min_str}-{max_str} years")
        if exp.preferred_years:
            print(f"   Preferred: {exp.preferred_years} years")

    # Education
    if job.education_requirement and (job.education_requirement.min_level or job.education_requirement.field):
        print(f"\n🎓 Education Required:")
        edu = job.education_requirement
        if edu.min_level and edu.min_level != "unknown":
            print(f"   Minimum Level: {edu.min_level.replace('_', ' ').title()}")
        if edu.preferred_level and edu.preferred_level != "unknown":
            print(f"   Preferred Level: {edu.preferred_level.replace('_', ' ').title()}")
        if edu.field:
            print(f"   Field: {edu.field}")

    # Skills
    if job.required_skills and (job.required_skills.must_have or job.required_skills.nice_to_have):
        print(f"\n🛠️  Skills Required:")

        if job.required_skills.must_have:
            must_have = job.required_skills.must_have[:15]
            print(f"   Must Have: {', '.join(must_have)}" + ("..." if len(job.required_skills.must_have) > 15 else ""))

        if job.required_skills.nice_to_have:
            nice_to_have = job.required_skills.nice_to_have[:10]
            print(f"   Nice to Have: {', '.join(nice_to_have)}" + ("..." if len(job.required_skills.nice_to_have) > 10 else ""))

    # Salary and Benefits
    if job.salary_range or job.benefits:
        print("\n" + "─" * 70)
        print("COMPENSATION & BENEFITS")
        print("─" * 70)

        if job.salary_range:
            print(f"💰 Salary Range: {job.salary_range}")

        if job.benefits:
            print(f"\n🎁 Benefits:")
            for benefit in job.benefits[:8]:
                print(f"   • {benefit}")
            if len(job.benefits) > 8:
                print(f"   ... and {len(job.benefits) - 8} more")


def display_ml_features(job):
    """Display ML-ready features"""

    print("\n" + "─" * 70)
    print("ML FEATURES (For Scoring Modules)")
    print("─" * 70)

    features = job.extract_features()

    print("\n📊 Numerical Features:")
    print(f"   • Required Years Experience: {features['required_years']}")
    print(f"   • Min Years: {features['min_years']}")
    print(f"   • Max Years: {features['max_years'] if features['max_years'] else 'N/A'}")
    print(f"   • Required Skills Count: {features['num_required_skills']}")
    print(f"   • Education Level (numeric): {features['required_education_level']}")

    print("\n🏷️  Categorical Features:")
    print(f"   • Location: {features['location']}")
    print(f"   • Location City: {features['location_city']}")
    print(f"   • Location Country: {features['location_country']}")
    print(f"   • Remote Allowed: {features['remote_allowed']}")
    print(f"   • Hybrid: {features['hybrid']}")

    print("\n💻 Required Skills List (for matching):")
    skills = features['required_skills'][:15]
    print(f"   {', '.join(skills)}" + ("..." if len(features['required_skills']) > 15 else ""))

    if features['nice_to_have_skills']:
        nice = features['nice_to_have_skills'][:10]
        print(f"\n   Nice to Have: {', '.join(nice)}" + ("..." if len(features['nice_to_have_skills']) > 10 else ""))

    print("\n📝 Text for Embedding (first 200 chars):")
    text = features['description_text'][:200] + "..."
    print(f"   {text}")


def validate_job(job):
    """Validate data quality"""

    print("\n" + "─" * 70)
    print("DATA QUALITY VALIDATION")
    print("─" * 70)

    # Completeness score
    completeness = job.completeness_score()
    print(f"\n📈 Completeness Score: {completeness:.1%}")

    if completeness >= 0.8:
        print("   ✅ Excellent data quality!")
    elif completeness >= 0.6:
        print("   ⚠️  Good, but some fields missing")
    else:
        print("   ❌ Poor data quality - many fields missing")

    # Validation issues
    issues = job.validate_for_ml()
    if issues:
        print(f"\n⚠️  Validation Issues:")
        for issue in issues:
            print(f"   • {issue.replace('_', ' ').title()}")
    else:
        print("\n✅ No validation issues - Ready for ML scoring!")


def print_usage():
    """Print usage instructions"""
    print("""
🧪 RESUMEAI JOB DESCRIPTION PARSING TEST

Usage:
    python test_job_parsing.py <path_to_job_description.txt>

Example:
    python test_job_parsing.py tests/test_job_parser/sample_job_description.txt

Requirements:
    1. Create a .txt file with the job description text
    2. Set GEMINI_API_KEY in your .env file
    3. Install dependencies: pip install -r requirements.txt

How to use:
    1. Copy the job description text from a job posting
    2. Paste it into tests/test_job_parser/sample_job_description.txt
    3. Run this test script
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    file_path = sys.argv[1]
    success = test_job_parsing(file_path)

    if success:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Job Parser is working correctly!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED - See errors above")
        print("=" * 70)
        sys.exit(1)
