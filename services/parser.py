"""
Unified parsing service.
Exposes APIs for parsing candidate resumes and job descriptions.
"""
import os
import logging
from core.parser import parse_resume
from core.pdf_parser import extract_text_from_pdf

logger = logging.getLogger(__name__)

def parse_resume_file(file_path: str) -> str:
    """
    Extracts raw text from a candidate's resume (PDF or DOCX).
    """
    logger.info(f"Parsing resume file: {file_path}")
    return parse_resume(file_path)

def parse_jd_file(file_path: str) -> str:
    """
    Extracts raw text from a Job Description PDF.
    """
    logger.info(f"Parsing job description: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Job Description file not found at: {file_path}")
    
    _, ext = os.path.splitext(file_path.lower())
    if ext != ".pdf":
        raise ValueError("Only PDF format is currently supported for Job Descriptions.")
        
    return extract_text_from_pdf(file_path)
