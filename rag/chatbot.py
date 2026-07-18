"""
Chatbot Grounding Service for ResumeIQ AI.
Uses an OpenAI-compatible Groq client to query Llama 3.3 70B with RAG context chunks.
"""
import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import retrieve_relevant_chunks

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Groq Client
def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not defined.")
        raise ValueError("GROQ_API_KEY environment variable is missing. Set it in the .env file.")
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

def generate_rag_response(query: str, chat_history: list[dict] = None) -> tuple[str, list[dict]]:
    """
    Executes the RAG lifecycle:
    1. Retrieves relevant document chunks.
    2. Constructs a context-grounded system instruction.
    3. Builds message payload incorporating recent chat history.
    4. Calls Groq and returns the generated response and source citations.
    """
    if not query or not query.strip():
        return "Please ask a question.", []
        
    client = _get_client()
    
    # 1. Retrieve most relevant context chunks (up to 5)
    try:
        logger.info(f"Retrieving candidate chunks for chatbot query: '{query}'")
        chunks = retrieve_relevant_chunks(query, n_results=5)
    except Exception as retrieval_err:
        logger.error(f"Retrieval step failed in chatbot: {retrieval_err}")
        chunks = []
        
    # 2. Build context text
    context_text = ""
    if chunks:
        context_text = "\n\n".join([
            f"--- Chunk {i+1} [Source: {c['source']}] ---\n{c['text']}" 
            for i, c in enumerate(chunks)
        ])
    else:
        context_text = "No relevant context found in candidate files."
        
    # 3. Construct System Instruction
    system_instruction = (
        "You are an AI Recruiter Screening Assistant for Aryan Sharma. "
        "Your task is to answer the recruiter's questions regarding the candidate using ONLY the provided candidate resume and job description context below.\n\n"
        f"Grounded Candidate Context:\n{context_text}\n\n"
        "Guidelines:\n"
        "1. Answer truthfully based only on the provided context.\n"
        "2. If the answer cannot be found in the context, politely state that the information is not present in the candidate files.\n"
        "3. Keep your answers concise, professional, and directly relevant to the recruiter's inquiry."
    )
    
    # 4. Compile messages payload with history
    messages = [{"role": "system", "content": system_instruction}]
    
    if chat_history:
        # Keep only the last 6 messages to preserve token window context
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": query})
    
    # 5. Generate content using Groq
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logger.info("Calling Llama 3.3 (Groq) for RAG grounded generation...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.2 # Lower temperature for factual accuracy and grounding
            )
            
            answer = response.choices[0].message.content.strip() if response.choices[0].message.content else "Sorry, I could not generate an answer."
            return answer, chunks
            
        except Exception as e:
            if ("503" in str(e) or "429" in str(e) or "rate_limit" in str(e).lower()) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient Groq API error in chatbot (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Failed to generate RAG response: {e}")
            return f"An error occurred while answering your query: {e}", []
