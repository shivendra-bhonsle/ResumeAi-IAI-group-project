"""
Semantic Similarity Module for ResumeAI
Computes embedding-based similarity between resumes and job descriptions.

Team Integration:
- Uses Resume.get_text_for_embedding() and JobDescription.get_text_for_embedding()
- Returns similarity score (0-1) for ranking engine
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import config


class SemanticSimilarityScorer:
    """
    Compute semantic similarity between resume and job description using embeddings.
    
    Uses sentence-transformers to generate embeddings and cosine similarity for comparison.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the semantic similarity scorer.
        
        Args:
            model_name: Name of sentence-transformer model to use.
                       Defaults to config.EMBEDDING_MODEL
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("✓ Model loaded successfully")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text (e.g., resume text)
            text2: Second text (e.g., job description text)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Generate embeddings
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        
        # Reshape for sklearn
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Ensure output is in [0, 1] range
        # Cosine similarity is in [-1, 1], normalize to [0, 1]
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)
    
    def score_resume_job_pair(self, resume, job) -> float:
        """
        Score a single resume against a job description.
        
        Args:
            resume: Resume object (from Person 1's parser)
            job: JobDescription object (from Person 1's parser)
            
        Returns:
            float: Semantic similarity score (0-1)
        """
        resume_text = resume.get_text_for_embedding()
        job_text = job.get_text_for_embedding()
        
        return self.compute_similarity(resume_text, job_text)
    
    def score_multiple_resumes(self, resumes: List, job) -> List[float]:
        """
        Score multiple resumes against a single job description.
        
        Args:
            resumes: List of Resume objects
            job: JobDescription object
            
        Returns:
            List of similarity scores (same order as input resumes)
        """
        job_text = job.get_text_for_embedding()
        job_embedding = self.generate_embedding(job_text)
        
        scores = []
        for resume in resumes:
            resume_text = resume.get_text_for_embedding()
            resume_embedding = self.generate_embedding(resume_text)
            
            # Compute similarity
            resume_emb = resume_embedding.reshape(1, -1)
            job_emb = job_embedding.reshape(1, -1)
            similarity = cosine_similarity(resume_emb, job_emb)[0][0]
            
            # Normalize to [0, 1]
            normalized_similarity = (similarity + 1) / 2
            scores.append(float(normalized_similarity))
        
        return scores


# Convenience function for quick scoring
def compute_semantic_similarity(resume, job) -> float:
    """
    Quick helper function to compute semantic similarity.
    
    Args:
        resume: Resume object
        job: JobDescription object
        
    Returns:
        float: Similarity score (0-1)
    """
    scorer = SemanticSimilarityScorer()
    return scorer.score_resume_job_pair(resume, job)