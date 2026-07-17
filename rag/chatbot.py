"""
Chatbot service coordinating RAG (Retrieval-Augmented Generation) pipeline for ResumeIQ AI.
Retrieves context, formats conversational history, and calls Gemini with strict grounding guidelines.
"""
import os
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from services.prompts import RAG_SYSTEM_PROMPT
from rag.retriever import retrieve_relevant_chunks

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Lazy initialization of Gemini Client
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is not defined.")
            raise ValueError("GEMINI_API_KEY is required.")
        _client = genai.Client(api_key=api_key)
    return _client

def generate_rag_response(query: str, chat_history: list[dict] = None) -> tuple[str, list[dict]]:
    """
    Executes the RAG lifecycle:
    1. Retrieves relevant document chunks.
    2. Constructs a context-grounded system instruction.
    3. Builds message payload incorporating recent chat history.
    4. Calls Gemini and returns the generated response and source citations.
    
    Args:
        query (str): The recruiter's query/chat message.
        chat_history (list[dict]): Optional list of past chat messages of form {'role': 'user'|'assistant', 'content': str}.
        
    Returns:
        tuple[str, list[dict]]: The chatbot answer string and list of chunks retrieved.
    """
    if not query or not query.strip():
        return "Please ask a question.", []
        
    client = _get_client()
    
    # 1. Retrieve context
    logger.info(f"RAG Chat: Retrieving chunks for user query: '{query}'")
    chunks = retrieve_relevant_chunks(query, n_results=5)
    
    # Format retrieved context
    if chunks:
        context_blocks = []
        for idx, chunk in enumerate(chunks):
            source_lbl = "Resume" if chunk["source"] == "resume" else "Job Description"
            context_blocks.append(f"[{source_lbl} Segment #{idx+1}]:\n{chunk['text']}")
        context_str = "\n\n".join(context_blocks)
    else:
        context_str = "No specific context could be retrieved from the resume or job description documents."
        
    # 2. Formulate system instruction with retrieved context
    system_instruction = RAG_SYSTEM_PROMPT.replace("{context}", context_str)
    
    # 3. Format conversational history (keep last 6 turns to keep context brief and relevant)
    history_str = ""
    if chat_history:
        recent_history = chat_history[-6:]
        for msg in recent_history:
            role = "Recruiter" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
            
    # 4. Construct content list for model call
    contents = []
    if history_str:
        contents.append(f"Recent Conversation History:\n{history_str}")
    contents.append(f"Recruiter's Current Question: {query}")
    
    # 5. Generate content using Gemini 2.0 Flash
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info("Calling Gemini 2.0 Flash for RAG grounded generation...")
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2 # Lower temperature for factual accuracy and grounding
                )
            )
            
            answer = response.text.strip() if response.text else "Sorry, I could not generate an answer."
            return answer, chunks
            
        except Exception as e:
            if ("503" in str(e) or "429" in str(e)) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient Gemini API error in chatbot (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Failed to generate RAG response: {e}")
            return f"An error occurred while answering your query: {e}", []
