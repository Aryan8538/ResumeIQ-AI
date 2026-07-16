"""
Unified parser routing module.
Detects the file type, performs format validation, and routes
to the appropriate sub-parser (PDF or DOCX).
"""

import os
import logging
from core.constants import SUPPORTED_EXTENSIONS
from core.pdf_parser import extract_text_from_pdf
from core.docx_parser import extract_text_from_docx

logger = logging.getLogger(__name__)

def parse_resume(file_path: str) -> str:
    """
    Detects the file type and extracts plain text from the resume file.
    Validates that the file exists and has a supported extension.
    
    Args:
        file_path (str): Absolute or relative path to the resume file.
        
    Returns:
        str: Extracted plain text content from the file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Resume file not found at: {file_path}")

    _, ext = os.path.splitext(file_path.lower())
    
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error(f"Unsupported file extension '{ext}' for file: {file_path}")
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Routing file to corresponding parser based on extension '{ext}': {file_path}")
    
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    
    raise ValueError(f"Extension '{ext}' matched SUPPORTED_EXTENSIONS but had no routing rule.")
