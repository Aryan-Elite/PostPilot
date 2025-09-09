from langchain_openai import OpenAIEmbeddings
import numpy as np
from typing import List, Optional, Dict, Any

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using OpenAI embeddings.
    """
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    if not a or not b:
        return 0.0
        
    a = np.array(a)
    b = np.array(b)
        
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
        
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))

def batch_get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch.
    More efficient than calling get_embedding() individually.
    """
    try:
        return embeddings.embed_documents(texts)
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return []