"""
Offline Embedding Service for ResumeIQ AI.
Uses ChromaDB's built-in ONNX MiniLM-L6-V2 model to generate 384-dimensional vector embeddings locally.
"""
import logging
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Initialize local ONNX embedding function (downloads once, runs 100% offline)
try:
    _onnx_ef = embedding_functions.ONNXMiniLM_L6_V2()
except Exception as e:
    logger.error(f"Failed to initialize local ONNX embedding function: {e}")
    raise RuntimeError(f"Embedding initialization failure: {e}") from e

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Computes local vector embeddings for a list of strings.
    """
    if not texts:
        return []
        
    try:
        logger.info(f"Generating local ONNX embeddings for {len(texts)} chunks...")
        embeddings = _onnx_ef(texts)
        return embeddings
    except Exception as e:
        logger.error(f"Local embedding generation failed: {e}")
        raise RuntimeError(f"Embedding generation error: {e}") from e

def get_single_embedding(text: str) -> list[float]:
    """
    Helper to embed a single query string.
    """
    res = get_embeddings([text])
    return res[0] if res else []
