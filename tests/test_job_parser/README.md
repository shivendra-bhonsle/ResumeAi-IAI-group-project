# Job Description Parser Testing

This folder contains test scripts and sample data for testing the Job Description Parser.

## Quick Start

### 1. Prepare Job Description

Copy a job description text and paste it into `sample_job_description.txt`:

```bash
# Open the file and paste your job description
nano sample_job_description.txt
# or use any text editor
```

### 2. Run the Test

```bash
# From project root
python tests/test_job_parser/test_job_parsing.py tests/test_job_parser/sample_job_description.txt
```

## What the Test Does

The test script will:
1. ✅ Read the job description text from the file
2. ✅ Parse it using the Gemini API
3. ✅ Extract structured information:
   - Job title, company, location
   - Required experience (years, level, areas)
   - Required education (degrees, fields)
   - Required skills (technical & soft skills)
   - Compensation details
   - Benefits
4. ✅ Display ML-ready features for scoring modules
5. ✅ Validate data quality

## Example Output

```
======================================================================
RESUMEAI - JOB DESCRIPTION PARSING TEST
======================================================================

File: tests/test_job_parser/sample_job_description.txt

📥 Reading job description text...
📥 Importing JobParser...
🔧 Initializing parser...
🤖 Parsing job description with Gemini API...
   (This may take a few seconds...)

✅ PARSING SUCCESSFUL!

──────────────────────────────────────────────────────────────────────
JOB INFORMATION
──────────────────────────────────────────────────────────────────────
💼 Title:    Senior Software Engineer
🏢 Company:  Tech Corp
📍 Location: San Francisco, United States
🏠 Remote:   Hybrid
📋 Type:     Full Time

📝 Description:
   We are looking for an experienced software engineer...

──────────────────────────────────────────────────────────────────────
REQUIREMENTS
──────────────────────────────────────────────────────────────────────

⏱️  Experience Required:
   Years: 5-10 years
   Level: Senior
   Areas: Software Development, Cloud Architecture

🎓 Education Required:
   Degrees: Bachelor, Master
   Fields: Computer Science, Software Engineering

🛠️  Skills Required:
   Languages: Python, Java, Go
   Frameworks: Django, Spring Boot, React
   Databases: PostgreSQL, MongoDB
   Cloud: AWS, GCP

──────────────────────────────────────────────────────────────────────
ML FEATURES (For Scoring Modules)
──────────────────────────────────────────────────────────────────────

📊 Numerical Features:
   • Min Years Experience: 5
   • Max Years Experience: 10
   • Required Skills Count: 12
   • Education Level (numeric): 3

✅ TEST PASSED - Job Parser is working correctly!
```

## Troubleshooting

### File is Empty Error
```
❌ ERROR: File is empty!
   Please paste a job description into the file and try again.
```
**Solution**: Open `sample_job_description.txt` and paste a job description

### Import Error
```
❌ IMPORT ERROR: No module named 'src'
```
**Solution**: Make sure you're running from the project root directory

### API Key Error
```
WARNING: GEMINI_API_KEY not found in environment variables!
```
**Solution**: Set your Gemini API key in the `.env` file

## File Structure

```
tests/test_job_parser/
├── README.md                          # This file
├── test_job_parsing.py                # Test script
└── sample_job_description.txt         # Paste job description here
```

## Tips

1. **Copy-Paste Format**: Job descriptions from any source work (LinkedIn, Indeed, company websites, etc.)
2. **Multiple Tests**: You can create multiple .txt files and test them individually
3. **Data Quality**: The script will show you completeness score and validation issues

## Next Steps

After successful parsing:
1. The parsed data is ready for the Scoring Modules (Person 2)
2. The ML features can be used for Ranking (Person 3)
3. The structured data can be displayed in the UI (Person 4)

## Support

If you encounter issues:
1. Check that `GEMINI_API_KEY` is set in `.env`
2. Verify dependencies are installed: `pip install -r requirements.txt`
3. Ensure the job description file is not empty
4. Check the logs for detailed error messages
