"""
Chatbot Grounding Service with dual-engine fallback (Groq Llama 3.3 / Gemini 2.0 Flash).
Uses retrieve_relevant_chunks from vector store to answer context-grounded queries.
"""
import os
import time
import logging
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
from rag.retriever import retrieve_relevant_chunks

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize active client
def _get_client():
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if groq_key:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        ), True
    elif gemini_key:
        return genai.Client(api_key=gemini_key), False
    else:
        logger.error("Neither GROQ_API_KEY nor GEMINI_API_KEY is configured.")
        raise ValueError("Missing API credentials. Set GROQ_API_KEY or GEMINI_API_KEY.")

def generate_rag_response(query: str, chat_history: list[dict] = None) -> tuple[str, list[dict]]:
    """
    Executes the RAG chatbot lifecycle with dual-engine fallback.
    """
    if not query or not query.strip():
        return "Please ask a question.", []
        
    try:
        client, is_groq = _get_client()
    except Exception as credential_err:
        logger.error(f"Failed to initialize chatbot client: {credential_err}")
        return f"Authentication error: {credential_err}", []
        
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
    
    # 4. Generate response based on client engine
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if is_groq:
                # Compile messages list for OpenAI-compatible client
                messages = [{"role": "system", "content": system_instruction}]
                if chat_history:
                    for msg in chat_history[-6:]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": query})
                
                logger.info("Calling Llama 3.3 (Groq) for RAG grounded generation...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip() if response.choices[0].message.content else "Sorry, I could not generate an answer."
            else:
                # Compile string contents list for Gemini client
                contents = []
                contents.append(f"Grounded Candidate Context:\n{context_text}")
                if chat_history:
                    for msg in chat_history[-6:]:
                        role_label = "Recruiter" if msg["role"] == "user" else "Assistant"
                        contents.append(f"{role_label}: {msg['content']}")
                contents.append(f"Recruiter's Current Question: {query}")
                
                logger.info("Calling Gemini 2.0 Flash for RAG grounded generation...")
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2
                    )
                )
                answer = response.text.strip() if response.text else "Sorry, I could not generate an answer."
                
            return answer, chunks
            
        except Exception as e:
            if ("503" in str(e) or "429" in str(e) or "rate_limit" in str(e).lower()) and attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 4
                logger.warning(f"Transient LLM API error in chatbot (attempt {attempt+1}/{max_retries}): {e}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
                continue
            logger.error(f"Failed to generate RAG response: {e}")
            return f"An error occurred while answering your query: {e}", []
