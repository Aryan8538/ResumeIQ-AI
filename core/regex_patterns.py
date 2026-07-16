"""
Regex patterns for extracting structured data from resume texts.
All regexes are pre-compiled and kept centralized to avoid hardcoding in business logic.
"""

import re

# Email Address Pattern
# Matches standard email addresses: user@domain.com
EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    re.IGNORECASE
)

# Phone Number Pattern
# Matches:
# - +1-555-555-5555
# - (555) 555-5555
# - 555.555.5555
# - +91 99999 99999
# - +919999999999
# - 09876543210
PHONE_PATTERN = re.compile(
    r'(?:'
    r'(?:\+?\d{1,3}[-.\s]*)?'         # Optional country code
    r'(?:\(\d{2,4}\)|\d{2,4})[-.\s]*'  # Area code (with or without parens)
    r'\d{3,4}[-.\s]*\d{3,4}'          # Main number sequence
    r')',
    re.IGNORECASE
)

# LinkedIn Profile URL Pattern
LINKEDIN_PATTERN = re.compile(
    r'https?://(?:[a-z]{2,3}\.)?linkedin\.com/in/[a-zA-Z0-9-_]+/?',
    re.IGNORECASE
)

# GitHub Profile URL Pattern
GITHUB_PATTERN = re.compile(
    r'https?://(?:www\.)?github\.com/[a-zA-Z0-9-_]+/?',
    re.IGNORECASE
)

# General URL Pattern (used to find portfolios, personal sites, etc.)
# We will filter out LinkedIn, GitHub, and email domains from these matches
GENERIC_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
    re.IGNORECASE
)

# Date/Period Patterns for Education & Experience duration extraction
# Matches: "2018 - 2022", "May 2019 - Present", "08/2020 - 12/2022", etc.
MONTHS_ABBR = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
YEAR = r'(?:19\d{2}|20\d{2})'
DATE_TERM = rf'(?:{MONTHS_ABBR}\s+{YEAR}|\d{{1,2}}/\d{{4}}|\d{{1,2}}/{YEAR}|{YEAR}|Present|Current)'

# Pattern to capture a range, e.g., "Jan 2018 - Present" or "2018-2022"
DATE_RANGE_PATTERN = re.compile(
    rf'{DATE_TERM}\s*(?:-|to|–|—)\s*{DATE_TERM}',
    re.IGNORECASE
)

# Pattern to check if a single token represents a year (e.g. 2021)
YEAR_PATTERN = re.compile(r'\b(?:19\d{2}|20\d{2})\b')
