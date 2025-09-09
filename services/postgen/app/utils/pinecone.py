from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import uuid
import time
import re
from collections import Counter
import math
from app.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from app.utils.embeddings import get_embedding

class PineconeHybridSearchService:
    """Optimized Pinecone service with hybrid search - production ready"""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize Pinecone service
        
        Args:
            alpha: Weight for semantic vs keyword search (0.0-1.0)
        """
        self.pc = None
        self.index = None
        self.alpha = alpha
        self.initialize_pinecone()
    
    def initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            existing_indexes = [i["name"] for i in self.pc.list_indexes()]

            if PINECONE_INDEX_NAME not in existing_indexes:
                print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    ),
                )
                while PINECONE_INDEX_NAME not in [i["name"] for i in self.pc.list_indexes()]:
                    time.sleep(1)

            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    def _preprocess_text(self, text: str) -> List[str]:
        """Extract keywords from text"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if len(word) > 2]

    def _calculate_bm25_score(self, query_terms: List[str], doc_text: str) -> float:
        """Calculate BM25 score for keyword matching"""
        k1, b = 1.5, 0.75
        doc_terms = self._preprocess_text(doc_text)
        doc_length = len(doc_terms)
        doc_term_freq = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                idf = math.log(1000 / 2)  # Simplified IDF
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / 100.0))
                score += idf * (numerator / denominator)
        
        return min(score / 10.0, 1.0)  # Normalize to 0-1

    def store_user_posts(self, username: str, posts: List[Dict[str, Any]]) -> bool:
        """Store user posts with embeddings"""
        try:
            vectors_to_upsert = []
            
            for post in posts:
                post_content = post.get("content", "") or post.get("text", "") or post.get("post_text", "")
                
                if not post_content or len(post_content.strip()) < 10:
                    continue
                
                embedding = get_embedding(post_content)
                if not embedding:
                    continue
                
                post_id = f"{username}_{post.get('post_id', str(uuid.uuid4()))}"
                keywords = self._preprocess_text(post_content)
                
                metadata = {
                    "username": username,
                    "content": post_content[:1000],
                    "post_id": post.get("post_id", ""),
                    "likes": post.get("likes", 0),
                    "comments": post.get("comments", 0),
                    "reposts": post.get("reposts", 0),
                    "keywords": " ".join(keywords[:50])
                }
                
                vectors_to_upsert.append({
                    "id": post_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            if vectors_to_upsert:
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
                print(f"Successfully stored {len(vectors_to_upsert)} posts for {username}")
                return True
            
            return False
                
        except Exception as e:
            print(f"Error storing posts for {username}: {e}")
            return False

    def hybrid_search(self, query: str, username: str = None, top_k: int = 5, 
                     alpha: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Main search method - hybrid semantic + keyword search
        """
        if alpha is None:
            alpha = self.alpha
            
        try:
            # Get query embedding
            query_embedding = get_embedding(query)
            if not query_embedding:
                return []
            
            # Build filter
            filter_dict = {"username": username} if username else None
            search_k = min(top_k * 3, 50)
            
            # Semantic search
            semantic_results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=search_k,
                include_metadata=True
            )
            
            # Combine with keyword scoring
            query_terms = self._preprocess_text(query)
            combined_results = {}
            
            for match in semantic_results.matches:
                doc_content = match.metadata.get("content", "")
                semantic_score = float(match.score)
                keyword_score = self._calculate_bm25_score(query_terms, doc_content)
                combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
                
                combined_results[match.id] = {
                    "id": match.id,
                    "content": doc_content,
                    "username": match.metadata.get("username", ""),
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "combined_score": combined_score,
                    "post_id": match.metadata.get("post_id", ""),
                    "likes": match.metadata.get("likes", 0),
                    "comments": match.metadata.get("comments", 0),
                    "reposts": match.metadata.get("reposts", 0)
                }
            
            # Sort by combined score
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )
            
            return sorted_results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def delete_user_posts(self, username: str) -> bool:
        """Delete all posts for a user"""
        try:
            query_response = self.index.query(
                vector=[0.0] * 1536,
                filter={"username": username},
                top_k=10000,
                include_metadata=True
            )
            
            if query_response.matches:
                post_ids = [match.id for match in query_response.matches]
                
                batch_size = 1000
                for i in range(0, len(post_ids), batch_size):
                    batch = post_ids[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                print(f"Deleted {len(post_ids)} posts for user {username}")
            
            return True
                
        except Exception as e:
            print(f"Error deleting posts for {username}: {e}")
            return False

    def update_user_posts(self, username: str, new_posts: List[Dict[str, Any]]) -> bool:
        """Replace all posts for a user"""
        try:
            self.delete_user_posts(username)
            return self.store_user_posts(username, new_posts)
        except Exception as e:
            print(f"Error updating posts for {username}: {e}")
            return False

# Global instance
pinecone_service = PineconeHybridSearchService(alpha=0.5)