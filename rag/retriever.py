"""
Retriever service for ResumeIQ AI.
Generates a query embedding and queries ChromaDB to find the most relevant document chunks.
"""
import logging
from rag.embeddings import get_single_embedding
from rag.vector_store import get_or_create_collection

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(query: str, n_results: int = 5) -> list[dict]:
    """
    Finds the most relevant chunks in ChromaDB for a given text query.
    
    Args:
        query (str): The search query.
        n_results (int): Number of chunks to retrieve.
        
    Returns:
        list[dict]: List of retrieved chunks, each containing 'text' and 'source'.
    """
    if not query or not query.strip():
        return []
        
    try:
        # 1. Embed query
        query_vector = get_single_embedding(query)
        if not query_vector:
            logger.warning("Empty query embedding generated.")
            return []
            
        # 2. Get database collection
        collection = get_or_create_collection()
        
        # Check if database has elements
        count = collection.count()
        if count == 0:
            logger.info("ChromaDB collection is empty. Returning 0 results.")
            return []
            
        # Adjust n_results if it exceeds total elements in the database
        effective_n = min(n_results, count)
        
        # 3. Query
        logger.info(f"Querying ChromaDB for: '{query}' (limit={effective_n})")
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=effective_n
        )
        
        # 4. Parse results
        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(docs)
            
            for doc, meta in zip(docs, metadatas):
                retrieved_chunks.append({
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "index": meta.get("index", 0)
                })
                
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks
        
    except Exception as e:
        logger.error(f"Failed to retrieve chunks from ChromaDB: {e}")
        return []
