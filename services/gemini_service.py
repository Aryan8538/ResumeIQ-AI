"""
Gemini service layer for interacting with Google Gemini 2.5 Flash model.
Provides methods for summary generation, resume vs JD analysis, and interview question generation.
"""
import os
import time
import json
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from services.prompts import (
    RESUME_SUMMARY_PROMPT,
    RESUME_VS_JD_PROMPT,
    INTERVIEW_QUESTIONS_PROMPT
)

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# Fetch API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY environment variable is not set.")
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please add it to your .env file.")

# Initialize the Gemini Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    logger.error(f"Failed to initialize Gemini Client: {e}")
    raise RuntimeError(f"Failed to initialize Gemini Client: {e}") from e

def clean_json_response(response_text: str) -> dict:
    """
    Cleans raw response text from Gemini by stripping markdown wrappers
    and parsing it into a dictionary.
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse json string from LLM: {text}")
        raise ValueError(f"Invalid JSON string returned: {e}") from e

def generate_resume_summary(resume_text: str) -> str:
    """
    Generates a professional resume summary from the candidate's parsed text.
    """
    logger.info("Generating resume summary using Gemini API...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[RESUME_SUMMARY_PROMPT, f"Candidate Resume Text:\n{resume_text}"]
            )
            if response.text:
                return response.text.strip()
            raise ValueError("Empty response returned from Gemini API.")
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient Gemini API error during summary (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Gemini resume summary generation failed: {e}")
            return f"Failed to generate summary: {e}"

def analyze_resume_vs_jd(resume_text: str, jd_text: str) -> dict:
    """
    Performs matching analysis between candidate resume and job description.
    Returns:
        dict: containing match_percentage, hiring_recommendation, missing_skills,
              strengths, weaknesses, and match_explanation.
    """
    logger.info("Performing Resume vs JD analysis...")
    formatted_prompt = RESUME_VS_JD_PROMPT.replace("{resume_text}", resume_text).replace("{jd_text}", jd_text)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            if response.text:
                return clean_json_response(response.text)
            raise ValueError("Empty response returned from Gemini API.")
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient Gemini API error during analysis (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Resume vs JD comparison failed: {e}")
            return {
                "match_percentage": 0,
                "hiring_recommendation": "Unsuitable",
                "missing_skills": [],
                "strengths": [],
                "weaknesses": [f"Analysis failed: {e}"],
                "match_explanation": f"Failed to execute match analysis: {e}"
            }

def generate_interview_questions(resume_text: str, jd_text: str) -> dict:
    """
    Generates custom interview questions based on the candidate's experience and the JD.
    Returns:
        dict: containing lists of technical, behavioral, and HR questions with answer guides.
    """
    logger.info("Generating custom interview questions...")
    formatted_prompt = INTERVIEW_QUESTIONS_PROMPT.replace("{resume_text}", resume_text).replace("{jd_text}", jd_text)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            if response.text:
                return clean_json_response(response.text)
            raise ValueError("Empty response returned from Gemini API.")
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient Gemini API error during questions (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Interview question generation failed: {e}")
            return {
                "technical": [],
                "behavioral": [],
                "hr": []
            }
