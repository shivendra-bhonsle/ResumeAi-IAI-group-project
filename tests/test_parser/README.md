# Testing the Document Parser

This guide shows you how to test the resume parsing functionality before committing.

## Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key:
     ```bash
     cp .env.example .env
     # Edit .env and set: GEMINI_API_KEY=your_actual_key_here
     ```

3. **Get a Resume File**
   - Place a `.docx` resume file in `data/sample_resumes/`
   - Or use any path to a `.docx` resume file

## How to Test

### Option 1: Using the Test Script (Easiest)

```bash
python tests/test_parser/test_parsing.py path/to/your/resume.docx
```

Example:
```bash
# If you put resume in data/sample_resumes/
python tests/test_parser/test_parsing.py data/sample_resumes/my_resume.docx

# Or any other path
python tests/test_parser/test_parsing.py ~/Downloads/john_doe_resume.docx
```

**What it does:**
- Extracts text from the .docx file
- Sends to Gemini API for parsing
- Displays all extracted information
- Shows ML features
- Validates data quality

### Option 2: Using the Parser Directly

```bash
# Test resume parser
python src/parsers/resume_parser.py path/to/resume.docx

# Test job parser (will use sample text)
python src/parsers/job_parser.py
```

### Option 3: Interactive Python

```python
# Start Python from project root
python

# Import and test
from src.parsers import ResumeParser

parser = ResumeParser()
resume = parser.parse_from_docx("data/sample_resumes/resume.docx")

# Check results
print(f"Name: {resume.personal_info.name}")
print(f"Experience: {resume.calculate_total_experience()} years")
print(f"Skills: {resume.skills.get_all_skills_flat()}")

# ML features
features = resume.extract_features()
print(features)
```

## Expected Output

If successful, you should see:

```
======================================================================
RESUMEAI - DOCUMENT PARSING TEST
======================================================================

File: data/sample_resumes/resume.docx

📥 Importing ResumeParser...
🔧 Initializing parser...
🤖 Parsing resume with Gemini API...
   (This may take a few seconds...)

✅ PARSING SUCCESSFUL!

──────────────────────────────────────────────────────────────────────
PERSONAL INFORMATION
──────────────────────────────────────────────────────────────────────
👤 Name:     John Doe
📧 Email:    john.doe@email.com
📱 Phone:    +1-555-0123
📍 Location: San Francisco, USA

... (more output)

✅ TEST PASSED - Parser is working correctly!
```

## Common Issues

### 1. Missing API Key
```
Error: GEMINI_API_KEY not found
```
**Solution**: Set `GEMINI_API_KEY` in `.env` file

### 2. Module Not Found
```
ModuleNotFoundError: No module named 'google.generativeai'
```
**Solution**: Run `pip install -r requirements.txt`

### 3. File Not Found
```
ERROR: File not found: resume.docx
```
**Solution**: Check the file path is correct

### 4. Rate Limit Error
```
GeminiRateLimitError: Rate limit exceeded
```
**Solution**: Wait a moment and try again (free tier has limits)

## Creating a Test Resume

If you don't have a resume file, create a simple Word document with:

```
John Doe
Email: john@example.com
Phone: 555-0123

Experience:
Software Engineer at Tech Corp
2020 - Present
- Developed web applications
- Used Python and Django

Education:
Bachelor of Science in Computer Science
MIT, 2020

Skills:
Python, Django, JavaScript, PostgreSQL
```

Save as `.docx` in `data/sample_resumes/` and test!

## What the Parser Extracts

The parser will extract:

✅ Personal Info (name, email, phone, location)
✅ Work Experience (companies, titles, dates, responsibilities)
✅ Education (degrees, institutions, dates)
✅ Skills (categorized by type)
✅ Projects (if present)
✅ Certifications (if present)

## Next Steps After Testing

Once you verify it works:

1. ✅ Parser is working
2. Commit the code to Git
3. Share with team members
4. Person 2 can start using `resume.extract_features()` for scoring
5. Person 3 can use `resume.get_text_for_embedding()` for semantic similarity

## Questions?

Check the main README.md or the inline code documentation.
