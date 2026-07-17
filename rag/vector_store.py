"""
Vector Database interaction layer for ResumeIQ-AI.
Wraps ChromaDB PersistentClient to save and load document chunks with explicit embeddings.
"""
import os
import logging
import chromadb
from rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Path to persist database within the workspace
CHROMA_PATH = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "resume_jd_collection"

_chroma_client = None

def _get_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"Initializing Persistent ChromaDB Client at: {CHROMA_PATH}")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

def get_or_create_collection():
    """
    Fetches or creates the standard collection.
    """
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)

def reset_db():
    """
    Resets the database by deleting and recreating the collection.
    Used when a new candidate or job description is uploaded.
    """
    client = _get_client()
    logger.info("Resetting ChromaDB collection...")
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.info(f"Collection '{COLLECTION_NAME}' did not exist or could not be deleted: {e}")
        
    # Recreate
    client.create_collection(COLLECTION_NAME)
    logger.info(f"Created collection: {COLLECTION_NAME}")

def index_document_chunks(chunks: list[dict], source_type: str):
    """
    Computes embeddings for chunks and adds them to ChromaDB.
    
    Args:
        chunks (list[dict]): List of chunks containing 'text'.
        source_type (str): Either 'resume' or 'job_description'.
    """
    if not chunks:
        logger.warning(f"No chunks provided to index for source: {source_type}")
        return
        
    collection = get_or_create_collection()
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    
    ids = [f"{source_type}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_type, "index": i} for i in range(len(chunks))]
    
    logger.info(f"Adding {len(chunks)} chunks to collection '{COLLECTION_NAME}' for {source_type}...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )
    logger.info("Successfully indexed chunks.")
