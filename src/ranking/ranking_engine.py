"""
Ranking Engine for ResumeAI
Combines all scoring components and ranks candidates.

Team Integration:
- Integrates with Person 2's scoring modules
- Uses semantic similarity from semantic_similarity.py
- Applies weighted aggregation from config.WEIGHTS
- Outputs ranked results for Person 4's UI
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from src.ranking.semantic_similarity import SemanticSimilarityScorer
import config


class RankingEngine:
    """
    Main ranking engine that combines all scores and ranks candidates.
    
    This class orchestrates:
    1. Semantic similarity scoring (Person 3's module)
    2. Integration with Person 2's scoring modules
    3. Weighted score aggregation
    4. Final ranking and result formatting
    """
    
    def __init__(self):
        """Initialize the ranking engine with improved semantic scorer."""
        # Initialize semantic scorer with cross-encoder re-ranking
        self.semantic_scorer = SemanticSimilarityScorer(
            use_cross_encoder=config.USE_CROSS_ENCODER,
            cross_encoder_model=config.CROSS_ENCODER_MODEL,
            rerank_top_k=config.RERANK_TOP_K
        )
        self.weights = config.WEIGHTS

        # Validate weights
        if not 0.99 <= sum(self.weights.values()) <= 1.01:
            print("WARNING: Weights don't sum to 1.0. Normalizing...")
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def compute_semantic_score(self, resume, job) -> float:
        """
        Compute semantic similarity score.
        
        Args:
            resume: Resume object
            job: JobDescription object
            
        Returns:
            float: Semantic similarity score (0-1)
        """
        return self.semantic_scorer.score_resume_job_pair(resume, job)
    
    def rank_single_resume(
        self,
        resume,
        job,
        scoring_modules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Rank a single resume against a job description.
        
        Args:
            resume: Resume object
            job: JobDescription object
            scoring_modules: Dict containing Person 2's scoring modules
                            Format: {
                                'skills': SkillsScorer(),
                                'experience': ExperienceScorer(),
                                'education': EducationScorer(),
                                'location': LocationScorer()
                            }
        
        Returns:
            Dict containing:
                - final_score: Weighted final score (0-1)
                - individual_scores: Dict of component scores
                - resume_data: Resume metadata
        """
        # Initialize scores dict
        scores = {}
        
        # 1. Compute semantic similarity (Person 3's responsibility)
        scores['semantic'] = self.compute_semantic_score(resume, job)
        
        # 2. Get scores from Person 2's modules (if available)
        if scoring_modules:
            if 'skills' in scoring_modules:
                scores['skills'] = scoring_modules['skills'].score(resume, job)
            
            if 'experience' in scoring_modules:
                scores['experience'] = scoring_modules['experience'].score(resume, job)
            
            if 'education' in scoring_modules:
                scores['education'] = scoring_modules['education'].score(resume, job)
            
            if 'location' in scoring_modules:
                scores['location'] = scoring_modules['location'].score(resume, job)
        
        # If Person 2 hasn't finished yet, use placeholder scores
        if 'skills' not in scores:
            scores['skills'] = 0.0
        if 'experience' not in scores:
            scores['experience'] = 0.0
        if 'education' not in scores:
            scores['education'] = 0.0
        if 'location' not in scores:
            scores['location'] = 0.0
        
        # 3. Compute weighted final score
        final_score = (
            self.weights['skills'] * scores['skills'] +
            self.weights['experience'] * scores['experience'] +
            self.weights['semantic'] * scores['semantic'] +
            self.weights['education'] * scores['education'] +
            self.weights['location'] * scores['location']
        )
        
        # 4. Validate resume data quality
        validation_issues = resume.validate_for_ml()
        completeness = resume.completeness_score()
        
        # 5. Return comprehensive result
        return {
            'final_score': final_score,
            'individual_scores': scores,
            'resume_data': {
                'name': resume.personal_info.name,
                'email': resume.personal_info.email,
                'years_experience': resume.calculate_total_experience(),
                'education_level': resume.get_highest_education_level(),
                'num_skills': len(resume.skills.get_all_skills_flat()),
                'location': f"{resume.personal_info.location.city}, {resume.personal_info.location.country}",
            },
            'data_quality': {
                'completeness_score': completeness,
                'validation_issues': validation_issues,
            },
            'weights_used': self.weights,
        }
    
    def rank_multiple_resumes(
        self,
        resumes: List,
        job,
        scoring_modules: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Rank multiple resumes against a job description.
        
        Args:
            resumes: List of Resume objects
            job: JobDescription object
            scoring_modules: Dict of Person 2's scoring modules (optional)
        
        Returns:
            pandas DataFrame with ranked results, sorted by final_score descending
        """
        results = []
        
        print(f"Ranking {len(resumes)} resumes...")
        for i, resume in enumerate(resumes):
            print(f"  Processing resume {i+1}/{len(resumes)}: {resume.personal_info.name}")
            result = self.rank_single_resume(resume, job, scoring_modules)
            results.append(result)
        
        # Convert to DataFrame for easy manipulation
        df_data = []
        for result in results:
            row = {
                'name': result['resume_data']['name'],
                'email': result['resume_data']['email'],
                'final_score': result['final_score'],
                'skills_score': result['individual_scores']['skills'],
                'experience_score': result['individual_scores']['experience'],
                'semantic_score': result['individual_scores']['semantic'],
                'education_score': result['individual_scores']['education'],
                'location_score': result['individual_scores']['location'],
                'years_experience': result['resume_data']['years_experience'],
                'education_level': result['resume_data']['education_level'],
                'num_skills': result['resume_data']['num_skills'],
                'location': result['resume_data']['location'],
                'completeness': result['data_quality']['completeness_score'],
                'has_issues': len(result['data_quality']['validation_issues']) > 0,
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort by final score (descending)
        df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
        
        # Add rank column
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        print(f"✓ Ranking complete! Top candidate: {df.iloc[0]['name']} (score: {df.iloc[0]['final_score']:.3f})")
        
        return df
    
    def get_top_n_candidates(
        self,
        resumes: List,
        job,
        n: int = 5,
        scoring_modules: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top N candidates with detailed breakdown.
        
        Args:
            resumes: List of Resume objects
            job: JobDescription object
            n: Number of top candidates to return
            scoring_modules: Dict of scoring modules (optional)
        
        Returns:
            List of top N candidate dicts with full details
        """
        df = self.rank_multiple_resumes(resumes, job, scoring_modules)
        
        # Get top N
        top_df = df.head(n)
        
        # Convert to list of dicts for Person 4's UI
        return top_df.to_dict('records')


# Convenience function for quick ranking
def rank_resumes(resumes: List, job, scoring_modules: Optional[Dict] = None) -> pd.DataFrame:
    """
    Quick helper function to rank resumes.
    
    Args:
        resumes: List of Resume objects
        job: JobDescription object
        scoring_modules: Optional dict of scoring modules from Person 2
    
    Returns:
        DataFrame with ranked results
    """
    engine = RankingEngine()
    return engine.rank_multiple_resumes(resumes, job, scoring_modules)