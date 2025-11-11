"""
Quick test script for document parsing.

Usage:
    python test_parsing.py <path_to_resume.docx>

This script will:
1. Parse the resume using ResumeParser
2. Display extracted information
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

def test_resume_parsing(file_path: str):
    """Test resume parsing with a .docx file"""

    print("=" * 70)
    print("RESUMEAI - DOCUMENT PARSING TEST")
    print("=" * 70)
    print(f"\nFile: {file_path}")

    # Check if file exists
    if not Path(file_path).exists():
        print(f"\n❌ ERROR: File not found: {file_path}")
        return False

    # Check file extension
    if not file_path.endswith('.docx'):
        print(f"\n❌ ERROR: File must be a .docx file")
        print(f"   Got: {Path(file_path).suffix}")
        return False

    try:
        # Import parser
        print("\n📥 Importing ResumeParser...")
        from src.parsers import ResumeParser

        # Create parser
        print("🔧 Initializing parser...")
        parser = ResumeParser()

        # Parse resume
        print(f"🤖 Parsing resume with Gemini API...")
        print("   (This may take a few seconds...)")
        resume = parser.parse_from_docx(file_path)

        print("\n✅ PARSING SUCCESSFUL!\n")

        # Display results
        display_results(resume)

        # Show ML features
        display_ml_features(resume)

        # Validate
        validate_resume(resume)

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


def display_results(resume):
    """Display parsed resume information"""

    print("─" * 70)
    print("PERSONAL INFORMATION")
    print("─" * 70)
    print(f"👤 Name:     {resume.personal_info.name}")
    print(f"📧 Email:    {resume.personal_info.email}")
    print(f"📱 Phone:    {resume.personal_info.phone}")
    print(f"📍 Location: {resume.personal_info.location.city}, {resume.personal_info.location.country}")

    if resume.personal_info.linkedin:
        print(f"🔗 LinkedIn: {resume.personal_info.linkedin}")
    if resume.personal_info.github:
        print(f"💻 GitHub:   {resume.personal_info.github}")

    if resume.personal_info.summary:
        summary = resume.personal_info.summary[:150] + "..." if len(resume.personal_info.summary) > 150 else resume.personal_info.summary
        print(f"\n📝 Summary:\n   {summary}")

    print("\n" + "─" * 70)
    print("EXPERIENCE")
    print("─" * 70)
    total_exp = resume.calculate_total_experience()
    print(f"⏱️  Total Experience: {total_exp:.1f} years")
    print(f"🏢 Number of Jobs: {len(resume.experience)}")

    for i, exp in enumerate(resume.experience[:3], 1):  # Show first 3
        print(f"\n{i}. {exp.title} at {exp.company}")
        print(f"   📅 Duration: {exp.get_duration_years():.1f} years ({exp.dates.start} - {exp.dates.end})")
        print(f"   📋 Responsibilities: {len(exp.responsibilities)}")
        if exp.responsibilities:
            print(f"      • {exp.responsibilities[0][:80]}...")

    if len(resume.experience) > 3:
        print(f"\n   ... and {len(resume.experience) - 3} more jobs")

    print("\n" + "─" * 70)
    print("EDUCATION")
    print("─" * 70)
    highest_level = resume.get_highest_education_level()
    level_names = {0: "Unknown", 1: "High School", 2: "Associate", 3: "Bachelor", 4: "Master", 5: "Doctorate"}
    print(f"🎓 Highest Level: {level_names.get(highest_level, 'Unknown')}")

    for i, edu in enumerate(resume.education, 1):
        # Handle both enum and string values
        level_str = edu.degree.level.value if hasattr(edu.degree.level, 'value') else str(edu.degree.level)
        print(f"\n{i}. {level_str.replace('_', ' ').title()}: {edu.degree.field}")
        print(f"   🏛️  {edu.institution.name}")
        if edu.achievements and edu.achievements.gpa:
            print(f"   📊 GPA: {edu.achievements.gpa}")

    print("\n" + "─" * 70)
    print("SKILLS")
    print("─" * 70)
    skills = resume.skills.get_all_skills_flat()
    print(f"🛠️  Total Skills: {len(skills)}")

    # Show by category
    tech = resume.skills.technical
    if tech.programming_languages:
        langs = [s.name for s in tech.programming_languages]
        print(f"\n   Languages: {', '.join(langs[:10])}" + ("..." if len(langs) > 10 else ""))

    if tech.frameworks:
        frameworks = [s.name for s in tech.frameworks]
        print(f"   Frameworks: {', '.join(frameworks[:10])}" + ("..." if len(frameworks) > 10 else ""))

    if tech.databases:
        dbs = [s.name for s in tech.databases]
        print(f"   Databases: {', '.join(dbs[:10])}")

    if tech.cloud:
        cloud = [s.name for s in tech.cloud]
        print(f"   Cloud: {', '.join(cloud[:10])}")

    if resume.projects:
        print("\n" + "─" * 70)
        print("PROJECTS")
        print("─" * 70)
        print(f"💼 Total Projects: {len(resume.projects)}")
        for i, proj in enumerate(resume.projects[:2], 1):
            if proj.name != "Unknown":
                print(f"\n{i}. {proj.name}")
                if proj.technologies:
                    print(f"   Technologies: {', '.join(proj.technologies[:5])}")


def display_ml_features(resume):
    """Display ML-ready features"""

    print("\n" + "─" * 70)
    print("ML FEATURES (For Scoring Modules)")
    print("─" * 70)

    features = resume.extract_features()

    print("\n📊 Numerical Features:")
    print(f"   • Years of Experience: {features['years_experience']:.1f}")
    print(f"   • Number of Jobs: {features['num_previous_jobs']}")
    print(f"   • Number of Skills: {features['num_skills']}")
    print(f"   • Education Level (numeric): {features['education_level']}")

    print("\n🏷️  Categorical Features:")
    print(f"   • Location: {features['location_city']}, {features['location_country']}")
    print(f"   • Remote Preference: {features['remote_preference']}")

    print("\n💻 Skills List (for matching):")
    skills = features['skills_list'][:15]
    print(f"   {', '.join(skills)}" + ("..." if len(features['skills_list']) > 15 else ""))

    print("\n📝 Text for Embedding (first 200 chars):")
    text = features['experience_text'][:200] + "..."
    print(f"   {text}")


def validate_resume(resume):
    """Validate data quality"""

    print("\n" + "─" * 70)
    print("DATA QUALITY VALIDATION")
    print("─" * 70)

    # Completeness score
    completeness = resume.completeness_score()
    print(f"\n📈 Completeness Score: {completeness:.1%}")

    if completeness >= 0.8:
        print("   ✅ Excellent data quality!")
    elif completeness >= 0.6:
        print("   ⚠️  Good, but some fields missing")
    else:
        print("   ❌ Poor data quality - many fields missing")

    # Validation issues
    issues = resume.validate_for_ml()
    if issues:
        print(f"\n⚠️  Validation Issues:")
        for issue in issues:
            print(f"   • {issue.replace('_', ' ').title()}")
    else:
        print("\n✅ No validation issues - Ready for ML scoring!")


def print_usage():
    """Print usage instructions"""
    print("""
🧪 RESUMEAI PARSING TEST

Usage:
    python test_parsing.py <path_to_resume.docx>

Example:
    python test_parsing.py data/sample_resumes/john_doe_resume.docx

Requirements:
    1. Place a .docx resume file somewhere in your project
    2. Set GEMINI_API_KEY in your .env file
    3. Install dependencies: pip install -r requirements.txt

Suggested test locations:
    • data/sample_resumes/
    • Any path to a .docx file
    """)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    file_path = sys.argv[1]
    success = test_resume_parsing(file_path)

    if success:
        print("\n" + "=" * 70)
        print("✅ TEST PASSED - Parser is working correctly!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED - See errors above")
        print("=" * 70)
        sys.exit(1)
