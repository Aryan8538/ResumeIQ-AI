"""
Information extraction module.
Responsible for extracting structured fields from resume text using spaCy NER,
regex patterns, and layout-based section segmentation.
"""

import re
import logging
import spacy
from core.constants import SECTION_HEADERS, SOFT_SKILLS, LANGUAGES_LIST
from core.regex_patterns import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    LINKEDIN_PATTERN,
    GITHUB_PATTERN,
    GENERIC_URL_PATTERN,
    DATE_RANGE_PATTERN,
    YEAR_PATTERN
)
from utils.helpers import segment_sections, clean_text, extract_lines

logger = logging.getLogger(__name__)

# Load spaCy English model globally
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model 'en_core_web_sm': {e}")
    raise

def extract_name(text: str) -> str:
    """
    Extracts the candidate's name.
    Looks at the first few lines of the resume (contact area) and uses
    spaCy NER (PERSON label) and heuristics to find the name.
    """
    lines = extract_lines(text)
    if not lines:
        return ""
        
    # Analyze the top 4 non-empty lines
    candidate_lines = lines[:4]
    
    # Heuristic 1: Run spaCy NER line-by-line on the first few lines
    for line in candidate_lines:
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned_name = ent.text.strip()
                # Validate name shape (2-4 words, no digits, no email symbols)
                words = cleaned_name.split()
                if 2 <= len(words) <= 4:
                    if not any(char.isdigit() for char in cleaned_name) and "@" not in cleaned_name and "/" not in cleaned_name:
                        return cleaned_name
                    
    # Heuristic 2: Fallback to the first line that doesn't look like contact info
    for line in candidate_lines:
        # Check that line is short (2-4 words) and doesn't contain typical contact info
        words = line.split()
        if 2 <= len(words) <= 4:
            if not any(char.isdigit() for char in line) and "@" not in line and "http" not in line.lower() and ".com" not in line.lower():
                # Clean punctuation
                cleaned = re.sub(r'[^\w\s-]', '', line).strip()
                if cleaned:
                    return cleaned
                    
    # Heuristic 3: Default to first line of text
    return lines[0] if lines else ""

def extract_email(text: str) -> str:
    """
    Extracts the first matching email address.
    """
    match = EMAIL_PATTERN.search(text)
    return match.group(0).strip().lower() if match else ""

def extract_phone(text: str) -> str:
    """
    Extracts the first matching phone number.
    """
    match = PHONE_PATTERN.search(text)
    return match.group(0).strip() if match else ""

def extract_links(text: str) -> tuple[str, str, str]:
    """
    Extracts LinkedIn, GitHub, and Portfolio URLs.
    Extracts all generic URLs, then classifies them.
    """
    linkedin = ""
    github = ""
    portfolio = ""
    
    # Find all URLs in the text
    urls = GENERIC_URL_PATTERN.findall(text)
    seen_urls = set()
    
    for url in urls:
        # Clean trailing slashes/punctuation
        url_clean = url.strip().rstrip("/.,;")
        if url_clean in seen_urls:
            continue
        seen_urls.add(url_clean)
        
        # Check classification
        if LINKEDIN_PATTERN.search(url_clean):
            if not linkedin:
                linkedin = url_clean
        elif GITHUB_PATTERN.search(url_clean):
            if not github:
                github = url_clean
        else:
            # Exclude standard common domains to isolate personal portfolios
            ignored_domains = {
                "linkedin.com",
                "github.com",
                "google.com",
                "youtube.com",
                "facebook.com",
                "twitter.com",
                "medium.com",
                "gmail.com",
                "yahoo.com",
                "outlook.com"
            }
            if not any(domain in url_clean.lower() for domain in ignored_domains):
                if not portfolio:
                    portfolio = url_clean
                    
    return linkedin, github, portfolio

def extract_technical_skills(text: str, skills_db: dict[str, list[str]]) -> list[str]:
    """
    Matches and extracts technical skills from the text based on a predefined database.
    Ensures word boundaries are respected to avoid false positive substring matches.
    """
    extracted = []
    
    # We search the plain text (case-insensitively)
    for category, skills in skills_db.items():
        for skill in skills:
            # Prepare regex pattern depending on special chars
            # For skills like C++, C#, .NET, Node.js we need relaxed boundaries
            if "+" in skill or "#" in skill or "." in skill:
                # Custom boundary matching: preceding space or line start, following space, comma, newline, or end
                pattern = rf'(?:^|\s|[,;])({re.escape(skill)})(?:\s|[,;]|$)'
            else:
                pattern = rf'\b({re.escape(skill)})\b'
                
            # Perform search
            if re.search(pattern, text, re.IGNORECASE):
                # Save clean name (unescaping regex escapes if any)
                clean_skill = skill.replace("\\+", "+")
                if clean_skill not in extracted:
                    extracted.append(clean_skill)
                    
    return sorted(extracted)

def extract_soft_skills(text: str) -> list[str]:
    """
    Matches and extracts soft skills from the text.
    """
    extracted = []
    for skill in SOFT_SKILLS:
        pattern = rf'\b{re.escape(skill)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            if skill not in extracted:
                extracted.append(skill)
    return sorted(extracted)

def extract_languages(text: str) -> list[str]:
    """
    Extracts spoken/written languages from the text.
    """
    extracted = []
    for lang in LANGUAGES_LIST:
        pattern = rf'\b{re.escape(lang)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            # Format nicely
            formatted_lang = lang.capitalize()
            if formatted_lang not in extracted:
                extracted.append(formatted_lang)
    return sorted(extracted)

def extract_education(sections: dict[str, str]) -> list[dict[str, str]]:
    """
    Parses structural education entries from the 'education' text segment.
    Extracts institution name, degree, and duration/year.
    """
    edu_text = sections.get("education", "")
    if not edu_text:
        return []
        
    entries = []
    lines = extract_lines(edu_text)
    
    # Key signals for education mapping
    inst_keywords = {"university", "college", "institute", "school", "academy", "polytechnic"}
    degree_patterns = [
        r'\bb\.?\s*s\.?\b', r'\bb\.?\s*a\.?\b', r'\bb\.?\s*e\.?\b', r'\bb\.?\s*tech\b',
        r'\bm\.?\s*s\.?\b', r'\bm\.?\s*a\.?\b', r'\bm\.?\s*tech\b', r'\bm\.?\s*b\.?\s*a\b',
        r'\bph\.?\s*d\b', r'\bbachelor\b', r'\bmaster\b', r'\bphd\b', r'\bassociate\b',
        r'\bdiploma\b', r'\bdegree\b'
    ]
    
    current_entry = {}
    
    for i, line in enumerate(lines):
        # 1. Look for Date/Duration in current line
        date_match = DATE_RANGE_PATTERN.search(line) or YEAR_PATTERN.search(line)
        duration = date_match.group(0).strip() if date_match else ""
        
        # 2. Look for Institution name
        # Heuristic A: Line contains a university keyword
        institution = ""
        if any(keyword in line.lower() for keyword in inst_keywords):
            institution = line
        else:
            # Heuristic B: Run spaCy NER on the line to see if ORG exists
            doc = nlp(line)
            for ent in doc.ents:
                if ent.label_ == "ORG" and any(k in ent.text.lower() for k in inst_keywords):
                    institution = ent.text.strip()
                    break
        
        # 3. Look for Degree
        degree = ""
        for dp in degree_patterns:
            match = re.search(dp, line, re.IGNORECASE)
            if match:
                degree = line  # Often the whole line contains degree detail (e.g., "Bachelor of Science in CS")
                break
                
        # Aggregate heuristics into entries
        if institution or degree:
            # If we already have something in current_entry and we are encountering new values,
            # save the old one first.
            if current_entry and (institution and current_entry.get("institution") or degree and current_entry.get("degree")):
                entries.append({
                    "institution": current_entry.get("institution", ""),
                    "degree": current_entry.get("degree", ""),
                    "duration": current_entry.get("duration", "")
                })
                current_entry = {}
                
            if institution:
                current_entry["institution"] = institution
            if degree:
                current_entry["degree"] = degree
            if duration:
                current_entry["duration"] = duration
        elif duration and current_entry:
            # If we just find a duration, attach it to current entry
            current_entry["duration"] = duration
            
        # Check next line: if next line is short and has duration, attach it.
        if duration and not current_entry:
            # Search nearby lines to bind duration
            pass
            
    if current_entry:
        entries.append({
            "institution": current_entry.get("institution", ""),
            "degree": current_entry.get("degree", ""),
            "duration": current_entry.get("duration", "")
        })
        
    # Clean output entries
    cleaned_entries = []
    for entry in entries:
        inst = entry["institution"].strip()
        deg = entry["degree"].strip()
        dur = entry["duration"].strip()
        
        # Clean separators and dates from institution
        if "|" in inst:
            parts = inst.split("|")
            for part in parts:
                if any(keyword in part.lower() for keyword in inst_keywords):
                    inst = part.strip()
                    break
        inst_date = DATE_RANGE_PATTERN.search(inst) or YEAR_PATTERN.search(inst)
        if inst_date:
            inst = inst.replace(inst_date.group(0), "").strip().strip(",;|–— ")
            
        # If institution is empty but degree has content, try parsing institution from degree text
        # (Often "B.S. Computer Science, Stanford University" is one line)
        if not inst and deg:
            for kw in inst_keywords:
                if kw in deg.lower():
                    # Split at comma or similar if possible
                    parts = re.split(r'[,|at]\s*', deg)
                    for part in parts:
                        if kw in part.lower():
                            inst = part.strip()
                            deg = deg.replace(part, "").strip().strip(",;| ")
                            break
                            
        # Clean separators and dates from degree
        if "|" in deg:
            parts = deg.split("|")
            for part in parts:
                if not any(keyword in part.lower() for keyword in inst_keywords) and not YEAR_PATTERN.search(part):
                    deg = part.strip()
                    break
            if "|" in deg:
                deg = deg.split("|")[0].strip()
                
        deg_date = DATE_RANGE_PATTERN.search(deg) or YEAR_PATTERN.search(deg)
        if deg_date:
            deg = deg.replace(deg_date.group(0), "").strip().strip(",;|–— ")
                            
        # Clean any overlapping text
        if inst.lower() in deg.lower():
            deg = deg.replace(inst, "").strip().strip(",;| ")
            
        if inst or deg:
            cleaned_entries.append({
                "institution": inst or "Not Specified",
                "degree": deg or "Not Specified",
                "duration": dur or "Not Specified"
            })
            
    return cleaned_entries

def extract_experience(sections: dict[str, str]) -> list[dict[str, str]]:
    """
    Parses job entries from the 'experience' text segment.
    Uses date ranges to partition jobs, extracting company, role, duration, and details.
    """
    exp_text = sections.get("experience", "")
    if not exp_text:
        return []
        
    entries = []
    lines = extract_lines(exp_text)
    
    current_entry = None
    role_keywords = {"engineer", "developer", "manager", "intern", "analyst", "consultant", "lead", "architect", "specialist"}
    
    for line in lines:
        date_match = DATE_RANGE_PATTERN.search(line)
        is_role_line = any(re.search(rf"\b{kw}\b", line, re.IGNORECASE) for kw in role_keywords)
        
        # 1. Check if line contains a date range
        if date_match:
            duration = date_match.group(0).strip()
            remaining_text = line.replace(duration, "").strip().strip(",;|–— ")
            
            # Heuristic: If we already have an entry without duration, and this line is just the date range
            if current_entry and current_entry["duration"] == "Not Specified" and not remaining_text:
                current_entry["duration"] = duration
                continue
                
            # Otherwise, this starts a new experience entry
            if current_entry:
                entries.append(current_entry)
                
            company = "Not Specified"
            role = "Not Specified"
            
            if remaining_text:
                if " at " in remaining_text.lower():
                    parts = re.split(r'\s+at\s+', remaining_text, flags=re.IGNORECASE)
                    role = parts[0].strip()
                    company = parts[1].strip()
                elif "|" in remaining_text:
                    parts = remaining_text.split("|")
                    role = parts[0].strip()
                    company = parts[1].strip()
                elif "," in remaining_text:
                    parts = remaining_text.split(",")
                    role = parts[0].strip()
                    company = ", ".join(parts[1:]).strip()
                else:
                    role = remaining_text
                    
            current_entry = {
                "company": company,
                "role": role,
                "duration": duration,
                "description_lines": []
            }
            
        # 2. Check if it's a role line
        elif is_role_line:
            if current_entry:
                entries.append(current_entry)
                
            company = "Not Specified"
            role = line
            
            if " at " in line.lower():
                parts = re.split(r'\s+at\s+', line, flags=re.IGNORECASE)
                role = parts[0].strip()
                company = parts[1].strip()
            elif "|" in line:
                parts = line.split("|")
                role = parts[0].strip()
                company = parts[1].strip()
            elif "," in line:
                parts = line.split(",")
                role = parts[0].strip()
                company = ", ".join(parts[1:]).strip()
                
            current_entry = {
                "company": company,
                "role": role,
                "duration": "Not Specified",
                "description_lines": []
            }
            
        # 3. Otherwise, append description
        elif current_entry:
            current_entry["description_lines"].append(line)
            
    if current_entry:
        entries.append(current_entry)
        
    formatted_entries = []
    for entry in entries:
        desc = "\n".join(entry["description_lines"]).strip()
        formatted_entries.append({
            "company": entry["company"],
            "role": entry["role"],
            "duration": entry["duration"],
            "description": desc
        })
        
    return formatted_entries

def extract_projects(sections: dict[str, str]) -> list[dict[str, str]]:
    """
    Parses project entries from the 'projects' text segment.
    Identifies projects based on line structure and aggregates descriptions.
    """
    proj_text = sections.get("projects", "")
    if not proj_text:
        return []
        
    entries = []
    lines = extract_lines(proj_text)
    
    current_entry = None
    
    for line in lines:
        date_match = DATE_RANGE_PATTERN.search(line) or YEAR_PATTERN.search(line)
        is_bullet = line.startswith(("-", "*", "•", "●"))
        
        # 1. If line has a date range
        if date_match:
            duration = date_match.group(0).strip()
            remaining = line.replace(duration, "").strip().strip(",;|–— ")
            
            # Heuristic: If we already have a project entry without duration, and this line is just the date range
            if current_entry and current_entry["duration"] == "Not Specified" and not remaining:
                current_entry["duration"] = duration
                continue
                
            # Otherwise, new project entry
            if current_entry:
                entries.append(current_entry)
                
            current_entry = {
                "title": remaining or "Project Entry",
                "duration": duration,
                "description_lines": []
            }
            
        # 2. If it's a short non-bullet line (likely a title)
        elif len(line.split()) < 6 and not is_bullet:
            if current_entry:
                entries.append(current_entry)
                
            current_entry = {
                "title": line,
                "duration": "Not Specified",
                "description_lines": []
            }
            
        # 3. Otherwise, append description
        elif current_entry:
            current_entry["description_lines"].append(line)
            
    if current_entry:
        entries.append(current_entry)
        
    formatted_projects = []
    for entry in entries:
        desc = "\n".join(entry["description_lines"]).strip()
        formatted_projects.append({
            "title": entry["title"],
            "duration": entry["duration"],
            "description": desc
        })
        
    return formatted_projects

def extract_certifications(sections: dict[str, str]) -> list[str]:
    """
    Extracts a list of certification strings.
    """
    cert_text = sections.get("certifications", "")
    if not cert_text:
        return []
    
    # Return non-empty lines from the certification section
    return [line.strip() for line in cert_text.split("\n") if line.strip()]

def extract_resume_info(raw_text: str) -> dict:
    """
    Master function to parse plain resume text into the unified structured dictionary format.
    """
    # 1. Clean the raw text
    cleaned_text = clean_text(raw_text)
    
    # 2. Segment into sections
    sections = segment_sections(cleaned_text, SECTION_HEADERS)
    
    # Import technical skills database
    from core.skills import TECHNICAL_SKILLS_DB
    
    # 3. Extract individual fields
    name = extract_name(cleaned_text)
    email = extract_email(cleaned_text)
    phone = extract_phone(cleaned_text)
    linkedin, github, portfolio = extract_links(cleaned_text)
    
    # Section-based extraction
    summary = extract_summary(cleaned_text, sections)
    skills = extract_technical_skills(cleaned_text, TECHNICAL_SKILLS_DB)
    soft_skills = extract_soft_skills(cleaned_text)
    languages = extract_languages(cleaned_text)
    
    education = extract_education(sections)
    experience = extract_experience(sections)
    projects = extract_projects(sections)
    certifications = extract_certifications(sections)
    
    # 4. Construct unified response dictionary
    parsed_resume = {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "portfolio": portfolio,
        "summary": summary,
        "skills": skills,
        "soft_skills": soft_skills,
        "education": education,
        "experience": experience,
        "projects": projects,
        "certifications": certifications,
        "languages": languages
    }
    
    return parsed_resume

def extract_summary(text: str, sections: dict[str, str]) -> str:
    """
    Extracts professional summary from sections dictionary.
    """
    return sections.get("summary", "").strip()
