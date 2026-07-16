"""
PDF Document parser module.
Extracts raw text from PDF files using PyMuPDF (fitz) with a fallback to pdfplumber.
"""

import logging
import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts and returns plain text from a PDF document.
    First attempts extraction using PyMuPDF (fitz).
    If PyMuPDF returns no text (e.g. empty or layout-blocked),
    falls back to pdfplumber.
    """
    text_content = []
    
    try:
        # 1. Try PyMuPDF
        logger.info(f"Attempting PDF text extraction with PyMuPDF for: {pdf_path}")
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text_content.append(page_text)
                    
        extracted_text = "\n".join(text_content).strip()
        
        # If text is extracted, return it
        if len(extracted_text) > 50:
            logger.info(f"Successfully extracted {len(extracted_text)} characters using PyMuPDF.")
            return extracted_text
            
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}. Retrying with pdfplumber.")

    # 2. Fallback to pdfplumber
    text_content = []
    try:
        logger.info(f"Attempting fallback PDF text extraction with pdfplumber for: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                    
        extracted_text = "\n".join(text_content).strip()
        logger.info(f"Successfully extracted {len(extracted_text)} characters using pdfplumber.")
        return extracted_text
        
    except Exception as e:
        logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
        raise RuntimeError(f"Failed to extract text from PDF: {e}") from e
