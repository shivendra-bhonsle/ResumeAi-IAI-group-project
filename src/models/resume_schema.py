"""
Resume data schema with ML-friendly methods.

This module defines the complete structure for parsed resume data,
optimized for integration with ML scoring modules.

Team Integration:
- Person 2 (Scoring): Use extract_features() for all scoring inputs
- Person 3 (Ranking): Use get_text_for_embedding() for semantic similarity
- Person 4 (UI): Use to_dict() for API responses, to_flat_dict() for tables
"""

from typing import List, Dict, Any, Optional
from pydantic import Field
from src.models.base_schema import (
    MLReadyBaseModel,
    EducationLevel,
    ExperienceLevel,
    EmploymentType,
    Location,
    DateRange,
    normalize_text,
)


# ==========================================
# Personal Information
# ==========================================

class PersonalInfo(MLReadyBaseModel):
    """Personal and contact information"""
    name: str = Field(default="Unknown", description="Full name")
    email: str = Field(default="Unknown", description="Email address")
    phone: str = Field(default="Unknown", description="Phone number")
    location: Location = Field(default_factory=Location, description="Location details")
    summary: str = Field(default="", description="Professional summary")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github: Optional[str] = Field(default=None, description="GitHub profile URL")

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}name": self.name,
            f"{prefix}email": self.email,
            f"{prefix}phone": self.phone,
            f"{prefix}location_city": self.location.city,
            f"{prefix}location_country": self.location.country,
            f"{prefix}summary": self.summary,
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "has_linkedin": bool(self.linkedin and self.linkedin != "Unknown"),
            "has_github": bool(self.github and self.github != "Unknown"),
            **self.location.extract_features(),
        }


# ==========================================
# Skills
# ==========================================

class Skill(MLReadyBaseModel):
    """Individual skill with proficiency level"""
    name: str = Field(..., description="Skill name (e.g., 'Python', 'AWS')")
    level: Optional[str] = Field(default="intermediate", description="Proficiency level")

    def to_normalized_name(self) -> str:
        """Return lowercase, normalized skill name"""
        return normalize_text(self.name)


class TechnicalSkills(MLReadyBaseModel):
    """Technical skills categorized by type"""
    programming_languages: List[Skill] = Field(default_factory=list, description="Programming languages")
    frameworks: List[Skill] = Field(default_factory=list, description="Frameworks and libraries")
    databases: List[Skill] = Field(default_factory=list, description="Database technologies")
    cloud: List[Skill] = Field(default_factory=list, description="Cloud platforms")
    tools: Optional[List[Skill]] = Field(default_factory=list, description="Development tools")

    def get_all_skills_flat(self) -> List[str]:
        """
        Return flat list of ALL skills (normalized, lowercase, deduplicated).

        Returns:
            List[str]: Sorted list of unique skill names

        Example:
            ["python", "django", "postgresql", "aws", "docker"]

        Team Usage:
            Person 2 (Skills Matcher): Direct input for skills matching algorithm
        """
        all_skills = []

        # Collect from all categories
        for skill in self.programming_languages:
            all_skills.append(skill.to_normalized_name())

        for skill in self.frameworks:
            all_skills.append(skill.to_normalized_name())

        for skill in self.databases:
            all_skills.append(skill.to_normalized_name())

        for skill in self.cloud:
            all_skills.append(skill.to_normalized_name())

        if self.tools:
            for skill in self.tools:
                all_skills.append(skill.to_normalized_name())

        # Remove empty strings, deduplicate, sort
        all_skills = [s for s in all_skills if s]
        return sorted(list(set(all_skills)))

    def get_skill_count_by_category(self) -> Dict[str, int]:
        """
        Count skills by category (feature for ML).

        Returns:
            Dict with category counts

        Team Usage:
            Person 2: Feature for assessing skill diversity
        """
        return {
            "num_languages": len(self.programming_languages),
            "num_frameworks": len(self.frameworks),
            "num_databases": len(self.databases),
            "num_cloud": len(self.cloud),
            "num_tools": len(self.tools) if self.tools else 0,
        }


class Skills(MLReadyBaseModel):
    """All skills (technical and non-technical)"""
    technical: TechnicalSkills = Field(default_factory=TechnicalSkills, description="Technical skills")
    languages: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Spoken languages")

    def get_all_skills_flat(self) -> List[str]:
        """Get all technical skills as flat list"""
        return self.technical.get_all_skills_flat()

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}skills": ",".join(self.get_all_skills_flat()),
            f"{prefix}num_skills": len(self.get_all_skills_flat()),
            **{f"{prefix}{k}": v for k, v in self.technical.get_skill_count_by_category().items()},
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "skills_list": self.get_all_skills_flat(),
            "num_skills": len(self.get_all_skills_flat()),
            **self.technical.get_skill_count_by_category(),
        }


# ==========================================
# Experience
# ==========================================

class CompanyInfo(MLReadyBaseModel):
    """Company information"""
    industry: Optional[str] = Field(default="Unknown", description="Industry sector")
    size: Optional[str] = Field(default="Unknown", description="Company size")


class TechnicalEnvironment(MLReadyBaseModel):
    """Technical environment for a role"""
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    methodologies: Optional[List[str]] = Field(default_factory=list, description="Methodologies (Agile, etc.)")
    tools: Optional[List[str]] = Field(default_factory=list, description="Tools used")


class Experience(MLReadyBaseModel):
    """Work experience entry"""
    company: str = Field(..., description="Company name")
    company_info: Optional[CompanyInfo] = Field(default_factory=CompanyInfo, description="Company details")
    title: str = Field(..., description="Job title")
    level: ExperienceLevel = Field(default=ExperienceLevel.UNKNOWN, description="Seniority level")
    employment_type: EmploymentType = Field(default=EmploymentType.FULL_TIME, description="Employment type")
    dates: DateRange = Field(default_factory=DateRange, description="Employment dates")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities and achievements")
    technical_environment: Optional[TechnicalEnvironment] = Field(default_factory=TechnicalEnvironment, description="Tech stack used")

    def get_duration_years(self) -> float:
        """
        Calculate years in this role.

        Team Usage:
            Person 2 (Experience Scorer): Input for experience calculations
        """
        return self.dates.calculate_duration_years()

    def get_experience_text(self) -> str:
        """
        Combined text for semantic analysis.

        Returns:
            str: All responsibilities joined as single text

        Team Usage:
            Person 3 (Semantic Scorer): Input for embedding generation
        """
        return " ".join(self.responsibilities)

    def extract_skills_from_experience(self) -> List[str]:
        """Extract skills mentioned in this experience"""
        if not self.technical_environment or not self.technical_environment.technologies:
            return []

        return [normalize_text(tech) for tech in self.technical_environment.technologies if tech]

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}company": self.company,
            f"{prefix}title": self.title,
            f"{prefix}level": self.level.value if isinstance(self.level, ExperienceLevel) else self.level,
            f"{prefix}duration_years": self.get_duration_years(),
            f"{prefix}num_responsibilities": len(self.responsibilities),
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "duration_years": self.get_duration_years(),
            "level_numeric": self.level.to_numeric() if isinstance(self.level, ExperienceLevel) else 0,
            "num_responsibilities": len(self.responsibilities),
            "experience_text": self.get_experience_text(),
        }


# ==========================================
# Education
# ==========================================

class Degree(MLReadyBaseModel):
    """Degree information"""
    level: EducationLevel = Field(default=EducationLevel.UNKNOWN, description="Degree level")
    field: str = Field(default="Unknown", description="Field of study")
    major: Optional[str] = Field(default=None, description="Major/specialization")


class Institution(MLReadyBaseModel):
    """Educational institution"""
    name: str = Field(default="Unknown", description="Institution name")
    location: Optional[str] = Field(default="Unknown", description="Location")
    accreditation: Optional[str] = Field(default="Unknown", description="Accreditation status")


class Achievements(MLReadyBaseModel):
    """Academic achievements"""
    gpa: Optional[float] = Field(default=None, description="GPA")
    honors: Optional[str] = Field(default=None, description="Honors/awards")
    relevant_coursework: Optional[List[str]] = Field(default_factory=list, description="Relevant courses")


class Education(MLReadyBaseModel):
    """Education entry"""
    degree: Degree = Field(default_factory=Degree, description="Degree information")
    institution: Institution = Field(default_factory=Institution, description="Institution information")
    dates: DateRange = Field(default_factory=DateRange, description="Attendance dates")
    achievements: Optional[Achievements] = Field(default_factory=Achievements, description="Academic achievements")

    def get_level_numeric(self) -> int:
        """
        Get numeric education level.

        Team Usage:
            Person 2 (Education Scorer): Direct input for education comparison
        """
        if isinstance(self.degree.level, EducationLevel):
            return self.degree.level.to_numeric()
        return 0

    def to_flat_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}degree_level": self.degree.level.value if isinstance(self.degree.level, EducationLevel) else self.degree.level,
            f"{prefix}degree_field": self.degree.field,
            f"{prefix}institution": self.institution.name,
            f"{prefix}gpa": self.achievements.gpa if self.achievements else None,
        }

    def extract_features(self) -> Dict[str, Any]:
        return {
            "level_numeric": self.get_level_numeric(),
            "degree_level": self.degree.level.value if isinstance(self.degree.level, EducationLevel) else self.degree.level,
            "field": self.degree.field,
            "has_gpa": bool(self.achievements and self.achievements.gpa),
        }


# ==========================================
# Projects
# ==========================================

class Project(MLReadyBaseModel):
    """Project entry"""
    name: str = Field(default="Unknown", description="Project name")
    description: str = Field(default="", description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    role: Optional[str] = Field(default="Unknown", description="Role in project")
    url: Optional[str] = Field(default=None, description="Project URL")
    impact: Optional[str] = Field(default=None, description="Impact/results")

    def get_project_text(self) -> str:
        """Combined text for semantic analysis"""
        texts = [self.description]
        if self.impact:
            texts.append(self.impact)
        return " ".join(texts)

    def extract_skills_from_project(self) -> List[str]:
        """Extract skills from project"""
        return [normalize_text(tech) for tech in self.technologies if tech]


# ==========================================
# Complete Resume Model
# ==========================================

class Resume(MLReadyBaseModel):
    """
    Complete resume data model with ML-ready methods.

    This is the main class that parsers will return and other team members will use.

    Team Integration:
        Person 2 (Scoring Modules):
            - resume.extract_features() → All features needed for scoring
            - resume.calculate_total_experience() → Total years
            - resume.skills.get_all_skills_flat() → Skills list

        Person 3 (Ranking Engine):
            - resume.get_text_for_embedding() → Text for semantic similarity
            - resume.validate_for_ml() → Check data completeness
            - resume.to_flat_dict() → Convert to DataFrame

        Person 4 (UI):
            - resume.dict() → Clean JSON for API
            - resume.to_flat_dict() → Table display format
            - resume.completeness_score() → Data quality indicator
    """
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo, description="Personal information")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education history")
    skills: Skills = Field(default_factory=Skills, description="Skills")
    projects: List[Project] = Field(default_factory=list, description="Projects")
    certifications: Optional[str] = Field(default="", description="Certifications")

    # ==========================================
    # ML-FRIENDLY METHODS
    # ==========================================

    def calculate_total_experience(self) -> float:
        """
        Calculate total years of experience across all jobs.

        Returns:
            float: Total years of experience

        Team Usage:
            Person 2 (Experience Scorer): Primary input for experience matching
        """
        return sum(exp.get_duration_years() for exp in self.experience)

    def get_text_for_embedding(self) -> str:
        """
        Combined text for semantic similarity analysis.

        Returns:
            str: All experience descriptions + project descriptions

        Team Usage:
            Person 3 (Semantic Scorer): Input for sentence transformer embedding
        """
        texts = []

        # Professional summary
        if self.personal_info.summary:
            texts.append(self.personal_info.summary)

        # All experience descriptions
        for exp in self.experience:
            texts.append(exp.get_experience_text())

        # All project descriptions
        for proj in self.projects:
            texts.append(proj.get_project_text())

        return " ".join(texts)

    def get_all_skills_including_experience(self) -> List[str]:
        """
        Get all skills including those mentioned in experience and projects.

        Returns:
            List[str]: Comprehensive skills list

        Team Usage:
            Person 2 (Skills Matcher): Enhanced skills list for better matching
        """
        all_skills = set(self.skills.get_all_skills_flat())

        # Add skills from experience
        for exp in self.experience:
            all_skills.update(exp.extract_skills_from_experience())

        # Add skills from projects
        for proj in self.projects:
            all_skills.update(proj.extract_skills_from_project())

        return sorted(list(all_skills))

    def get_highest_education_level(self) -> int:
        """
        Get highest education level achieved (numeric).

        Returns:
            int: Numeric education level (0-5)

        Team Usage:
            Person 2 (Education Scorer): Primary input for education matching
        """
        if not self.education:
            return 0

        return max(edu.get_level_numeric() for edu in self.education)

    def extract_features(self) -> Dict[str, Any]:
        """
        Extract ALL features needed for ML scoring modules.

        Returns:
            Dict containing:
                - Numerical features (years_experience, num_skills, etc.)
                - Categorical features (skills_list, location, etc.)
                - Text features (experience_text, summary)

        Team Usage:
            Person 2: PRIMARY method - contains all inputs for all scoring modules:
                - Skills Matcher: features["skills_list"]
                - Experience Scorer: features["years_experience"]
                - Education Scorer: features["education_level"]
                - Location Scorer: features["location_city"], features["location_country"]

            Person 3: Use features["experience_text"] for semantic similarity
        """
        return {
            # Numerical features
            "years_experience": self.calculate_total_experience(),
            "num_previous_jobs": len(self.experience),
            "num_skills": len(self.skills.get_all_skills_flat()),
            "num_projects": len(self.projects),
            "education_level": self.get_highest_education_level(),
            "has_certifications": bool(self.certifications and self.certifications != ""),

            # Categorical/List features
            "skills_list": self.skills.get_all_skills_flat(),
            "skills_with_experience": self.get_all_skills_including_experience(),
            "location_city": self.personal_info.location.city,
            "location_country": self.personal_info.location.country,
            "remote_preference": self.personal_info.location.remote_preference,

            # Text features (for semantic analysis)
            "experience_text": self.get_text_for_embedding(),
            "summary": self.personal_info.summary,

            # Metadata
            "completeness": self.completeness_score(),
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary for pandas DataFrame.

        Returns:
            Dict with flat structure (no nesting)

        Team Usage:
            Person 3: Easy DataFrame conversion - pd.DataFrame([resume.to_flat_dict()])
            Person 4: Table display in UI
        """
        return {
            "name": self.personal_info.name,
            "email": self.personal_info.email,
            "location": f"{self.personal_info.location.city}, {self.personal_info.location.country}",
            "years_experience": self.calculate_total_experience(),
            "education_level": self.get_highest_education_level(),
            "skills": ",".join(self.skills.get_all_skills_flat()),
            "num_skills": len(self.skills.get_all_skills_flat()),
            "num_jobs": len(self.experience),
            "num_projects": len(self.projects),
            "has_certifications": bool(self.certifications),
            "completeness_score": self.completeness_score(),
        }

    def validate_for_ml(self) -> List[str]:
        """
        Validate if resume has sufficient data for ML scoring.

        Returns:
            List of validation issues. Empty list = ready for scoring.

        Team Usage:
            Person 2 & 3: Check data quality before scoring
            Person 4: Show validation warnings in UI
        """
        issues = []

        # Check experience
        if not self.experience:
            issues.append("no_experience_data")
        elif self.calculate_total_experience() == 0:
            issues.append("cannot_calculate_experience_duration")

        # Check skills
        if not self.skills.get_all_skills_flat():
            issues.append("no_skills_data")

        # Check education
        if not self.education:
            issues.append("no_education_data")
        elif self.get_highest_education_level() == 0:
            issues.append("education_level_unknown")

        # Check completeness
        if self.completeness_score() < 0.5:
            issues.append("low_data_completeness")

        return issues

    def format_for_display(self) -> Dict[str, Any]:
        """
        Format resume for UI display.

        Returns:
            Dict with human-readable format

        Team Usage:
            Person 4: Display resume summary in UI
        """
        return {
            "name": self.personal_info.name,
            "summary": self.personal_info.summary,
            "experience_summary": f"{self.calculate_total_experience():.1f} years of experience",
            "education_summary": f"{len(self.education)} degree(s)",
            "skills_summary": f"{len(self.skills.get_all_skills_flat())} skills",
            "top_skills": self.skills.get_all_skills_flat()[:10],  # Top 10
            "recent_companies": [exp.company for exp in self.experience[:3]],  # Recent 3
            "data_quality": f"{self.completeness_score():.0%}",
        }
