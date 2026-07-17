"""
Embeddings generator service for ResumeIQ AI.
Communicates with Gemini Embeddings API (gemini-embedding-2) to vectorize document chunks.
"""
import os
import logging
from dotenv import load_dotenv
from google import genai

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
            logger.error("GEMINI_API_KEY is not defined in the environment.")
            raise ValueError("GEMINI_API_KEY environment variable is missing. Set it in the .env file.")
        try:
            _client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to instantiate GenAI client in embeddings module: {e}")
            raise RuntimeError(f"Embedding service initialization failure: {e}") from e
    return _client

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Computes vector embeddings for a list of strings.
    
    Args:
        texts (list[str]): List of texts/chunks to vectorize.
        
    Returns:
        list[list[float]]: List of float vectors representing the texts.
    """
    if not texts:
        return []
        
    client = _get_client()
    try:
        logger.info(f"Requesting embeddings for {len(texts)} chunks from gemini-embedding-2...")
        embeddings = []
        for text in texts:
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
            )
            if not response.embeddings or not response.embeddings[0].values:
                raise ValueError("No embeddings returned by Gemini Embeddings API for a text chunk.")
            embeddings.append(response.embeddings[0].values)
        return embeddings
        
    except Exception as e:
        logger.error(f"Gemini embedding API call failed: {e}")
        raise RuntimeError(f"Embedding generation error: {e}") from e

def get_single_embedding(text: str) -> list[float]:
    """
    Helper to embed a single string query.
    """
    res = get_embeddings([text])
    return res[0] if res else []
