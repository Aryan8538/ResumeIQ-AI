"""
Helper functions for text cleaning, preprocessing, and section segmentation.
"""

import re
import string

def clean_text(text: str) -> str:
    """
    Cleans raw text by:
    - Normalizing whitespace (replacing tabs, multiple spaces, etc. with single spaces)
    - Removing non-printable characters
    - Preserving single newlines for line-based layout parsing
    """
    if not text:
        return ""
    
    # Replace carriage returns and horizontal tabs
    text = text.replace("\r", "\n").replace("\t", " ")
    
    # Filter out non-printable ASCII/Unicode characters
    # (except newlines and standard printable characters)
    printable = set(string.printable + "‘’“”–—•●◦▪▪▪▪")
    text = "".join(filter(lambda x: x in printable or ord(x) > 127, text))
    
    # Clean multiple spaces on each line while preserving line structure
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        lines.append(cleaned_line)
        
    return "\n".join(lines)

def extract_lines(text: str) -> list[str]:
    """
    Splits text into non-empty, stripped lines.
    """
    if not text:
        return []
    return [line.strip() for line in text.split("\n") if line.strip()]

def segment_sections(text: str, section_headers_dict: dict[str, list[str]]) -> dict[str, str]:
    """
    Segments the resume text into sections based on defined header keywords.
    Returns a dictionary mapping standardized section keys to the corresponding text content.
    Text preceding the first detected section header is categorized under 'contact_info'.
    """
    lines = text.split("\n")
    sections = {}
    current_section = "contact_info"
    section_content = []

    # Map header keyword variations to their standardized keys
    header_mapping = {}
    for standard_key, variations in section_headers_dict.items():
        for var in variations:
            header_mapping[var.lower().strip()] = standard_key

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        
        # Strip punctuation to normalize headers like "Education:" or "Experience."
        normalized_line = re.sub(r'[^\w\s]', '', stripped_line).lower().strip()
        
        # Check if line looks like a section header:
        # 1. The normalized line exists in our header mapping
        # 2. It is short (typically 1-4 words)
        is_header = False
        if normalized_line in header_mapping and len(normalized_line.split()) <= 4:
            is_header = True
            new_section = header_mapping[normalized_line]
            
        if is_header:
            # Save accumulated content for the section we are leaving
            if section_content:
                content_str = "\n".join(section_content).strip()
                if content_str:
                    if current_section in sections:
                        sections[current_section] += "\n" + content_str
                    else:
                        sections[current_section] = content_str
                section_content = []
            current_section = new_section
        else:
            section_content.append(stripped_line)

    # Save the final section
    if section_content:
        content_str = "\n".join(section_content).strip()
        if content_str:
            if current_section in sections:
                sections[current_section] += "\n" + content_str
            else:
                sections[current_section] = content_str

    return sections
