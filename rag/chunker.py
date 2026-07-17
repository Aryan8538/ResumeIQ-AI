"""
Document chunking utility for ResumeIQ-AI RAG pipeline.
Splits text into smaller, overlapping chunks suitable for semantic search embeddings.
"""

def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> list[dict]:
    """
    Splits text into chunks of specific size and overlap.
    
    Args:
        text (str): Raw string content of the document.
        chunk_size (int): Max number of characters per chunk.
        chunk_overlap (int): Overlap size in characters.
        
    Returns:
        list[dict]: List of chunk dictionaries containing 'text', 'start_idx', and 'end_idx'.
    """
    if not text:
        return []
        
    chunks = []
    text_len = len(text)
    
    # Clean excessive newlines to create more cohesive text chunks
    cleaned_text = " ".join(text.split())
    cleaned_len = len(cleaned_text)
    
    start = 0
    while start < cleaned_len:
        end = min(start + chunk_size, cleaned_len)
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "start_idx": start,
                "end_idx": end
            })
        
        # Advance the window
        if end == cleaned_len:
            break
        start += chunk_size - chunk_overlap
        
    return chunks
