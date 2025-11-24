"""
Main Pipeline for ResumeAI
End-to-end orchestration for resume ranking.

This module ties together all components:
- Document parsing (Person 1)
- Scoring modules (Person 2)
- Ranking engine (Person 3)

Usage:
    from src.pipeline.main_pipeline import rank_candidates

    results = rank_candidates(
        job_text="Software Engineer job description...",
        resume_files=["resume1.docx", "resume2.docx"]
    )
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Import parsers
from src.parsers import ResumeParser, JobParser

# Import scoring modules
from src.scoring import (
    SkillsScorer,
    ExperienceScorer,
    EducationScorer,
    LocationScorer
)

# Import ranking engine
from src.ranking import RankingEngine

# Import config
import config


class ResumePipeline:
    """
    End-to-end pipeline for resume ranking.

    This class orchestrates the entire process:
    1. Parse job description
    2. Parse resumes (batch processing)
    3. Initialize scoring modules
    4. Run ranking engine
    5. Format and save results
    """

    def __init__(self, use_location: bool = None):
        """
        Initialize the pipeline.

        Args:
            use_location: Whether to use location scoring.
                         If None, uses config.WEIGHTS['location'] > 0
        """
        self.resume_parser = ResumeParser()
        self.job_parser = JobParser()

        # Check if location should be used
        if use_location is None:
            use_location = config.WEIGHTS['location'] > 0

        self.use_location = use_location

        # Initialize scoring modules
        self.scoring_modules = {
            'skills': SkillsScorer(fuzzy_threshold=85),
            'experience': ExperienceScorer(),
            'education': EducationScorer(),
            'location': LocationScorer() if use_location else LocationScorer()  # Always create but may not use
        }

        # Initialize ranking engine
        self.ranking_engine = RankingEngine()

        print(f"✓ Pipeline initialized")
        print(f"  Location scoring: {'Enabled' if use_location else 'Disabled (weight=0)'}")
        print(f"  Weights: Skills={config.WEIGHTS['skills']}, " +
              f"Experience={config.WEIGHTS['experience']}, " +
              f"Semantic={config.WEIGHTS['semantic']}, " +
              f"Education={config.WEIGHTS['education']}, " +
              f"Location={config.WEIGHTS['location']}")

    def parse_job(self, job_text: str):
        """
        Parse job description.

        Args:
            job_text: Job description as string

        Returns:
            JobDescription object
        """
        print("\n[1/4] Parsing job description...")
        job = self.job_parser.parse(job_text)
        print(f"  ✓ Parsed job: {job.title}")
        return job

    def parse_resumes(self, resume_files: List[str]):
        """
        Parse multiple resumes.

        Args:
            resume_files: List of paths to resume files (.docx)

        Returns:
            List of Resume objects
        """
        print(f"\n[2/4] Parsing {len(resume_files)} resume(s)...")

        # Use batch processing for speed
        resumes = self.resume_parser.parse_batch(resume_files)

        print(f"  ✓ Successfully parsed {len(resumes)}/{len(resume_files)} resumes")

        # Show any parsing failures
        if len(resumes) < len(resume_files):
            print(f"  ⚠ Warning: {len(resume_files) - len(resumes)} resumes failed to parse")

        return resumes

    def rank_candidates(
        self,
        job,
        resumes: List,
        return_format: str = 'dataframe'
    ) -> Union[pd.DataFrame, List[Dict], Dict]:
        """
        Rank candidates against job.

        Args:
            job: JobDescription object
            resumes: List of Resume objects
            return_format: Output format ('dataframe', 'list', 'dict')

        Returns:
            Ranked results in specified format
        """
        print(f"\n[3/4] Ranking {len(resumes)} candidate(s)...")

        # Run ranking engine
        ranked_df = self.ranking_engine.rank_multiple_resumes(
            resumes,
            job,
            self.scoring_modules
        )

        print(f"  ✓ Ranking complete!")
        print(f"  Top candidate: {ranked_df.iloc[0]['name']} " +
              f"(score: {ranked_df.iloc[0]['final_score']:.3f})")

        # Return in requested format
        if return_format == 'dataframe':
            return ranked_df
        elif return_format == 'list':
            return ranked_df.to_dict('records')
        elif return_format == 'dict':
            return {
                'candidates': ranked_df.to_dict('records'),
                'total_count': len(ranked_df),
                'weights': config.WEIGHTS,
                'timestamp': datetime.now().isoformat()
            }
        else:
            raise ValueError(f"Unknown format: {return_format}")

    def save_results(
        self,
        ranked_df: pd.DataFrame,
        output_path: Optional[str] = None,
        format: str = 'csv'
    ) -> str:
        """
        Save ranking results to file.

        Args:
            ranked_df: DataFrame with ranked results
            output_path: Path to save file. If None, auto-generates in outputs/rankings/
            format: Output format ('csv', 'json', 'excel')

        Returns:
            Path to saved file
        """
        print(f"\n[4/4] Saving results...")

        # Auto-generate path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = config.OUTPUT_DIR / "rankings"
            output_dir.mkdir(exist_ok=True, parents=True)

            if format == 'csv':
                output_path = output_dir / f"ranking_results_{timestamp}.csv"
            elif format == 'json':
                output_path = output_dir / f"ranking_results_{timestamp}.json"
            elif format == 'excel':
                output_path = output_dir / f"ranking_results_{timestamp}.xlsx"
            else:
                raise ValueError(f"Unknown format: {format}")

        # Save file
        if format == 'csv':
            ranked_df.to_csv(output_path, index=False)
        elif format == 'json':
            # Include metadata
            output_data = {
                'candidates': ranked_df.to_dict('records'),
                'metadata': {
                    'total_count': len(ranked_df),
                    'weights': config.WEIGHTS,
                    'timestamp': datetime.now().isoformat(),
                    'model_config': {
                        'gemini_model': config.GEMINI_MODEL,
                        'embedding_model': config.EMBEDDING_MODEL
                    }
                }
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        elif format == 'excel':
            ranked_df.to_excel(output_path, index=False, sheet_name='Rankings')

        print(f"  ✓ Results saved to: {output_path}")
        return str(output_path)

    def run(
        self,
        job_text: str,
        resume_files: List[str],
        save_output: bool = True,
        output_format: str = 'csv',
        return_format: str = 'dataframe'
    ) -> Union[pd.DataFrame, tuple]:
        """
        Run complete end-to-end pipeline.

        Args:
            job_text: Job description text
            resume_files: List of paths to resume files
            save_output: Whether to save results to file
            output_format: Format for saved file ('csv', 'json', 'excel')
            return_format: Format for return value ('dataframe', 'list', 'dict')

        Returns:
            Ranked results (and optionally output path if saved)
        """
        print("=" * 70)
        print("RESUMEAI - CANDIDATE RANKING PIPELINE")
        print("=" * 70)

        # Step 1: Parse job
        job = self.parse_job(job_text)

        # Step 2: Parse resumes
        resumes = self.parse_resumes(resume_files)

        if not resumes:
            raise ValueError("No resumes were successfully parsed!")

        # Step 3: Rank candidates
        ranked_results = self.rank_candidates(job, resumes, return_format=return_format)

        # Step 4: Save results (optional)
        output_path = None
        if save_output:
            # Convert to dataframe if needed
            df = ranked_results if isinstance(ranked_results, pd.DataFrame) else pd.DataFrame(ranked_results)
            output_path = self.save_results(df, format=output_format)

        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 70)

        if save_output:
            return ranked_results, output_path
        else:
            return ranked_results


# Convenience function for quick usage
def rank_candidates(
    job_text: str,
    resume_files: List[str],
    save_output: bool = True,
    output_format: str = 'csv'
) -> pd.DataFrame:
    """
    Quick helper function to rank candidates.

    Args:
        job_text: Job description text
        resume_files: List of paths to resume files
        save_output: Whether to save results to file
        output_format: Format for output file ('csv', 'json', 'excel')

    Returns:
        DataFrame with ranked candidates

    Example:
        >>> results = rank_candidates(
        ...     job_text="Software Engineer with 3+ years...",
        ...     resume_files=["resume1.docx", "resume2.docx"]
        ... )
        >>> print(results.head())
    """
    pipeline = ResumePipeline()
    result = pipeline.run(
        job_text=job_text,
        resume_files=resume_files,
        save_output=save_output,
        output_format=output_format,
        return_format='dataframe'
    )

    if save_output:
        return result[0]  # Return just the dataframe
    else:
        return result


if __name__ == "__main__":
    # Example usage / testing
    print("ResumePipeline module loaded successfully!")
    print(f"Configuration: {config.get_config_summary()}")
