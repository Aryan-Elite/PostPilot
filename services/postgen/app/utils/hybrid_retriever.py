from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List, Optional
from pydantic import BaseModel, PrivateAttr
from app.utils.pinecone import pinecone_service

class PineconeHybridRetriever(BaseRetriever, BaseModel):
    """LangChain retriever for hybrid Pinecone search"""

    _pinecone_service: "pinecone_service" = PrivateAttr()
    alpha: float = 0.5
    top_k: int = 5
    username: Optional[str] = None

    def __init__(self, pinecone_service, alpha=0.5, top_k=5, username=None, **kwargs):
        super().__init__(**kwargs)
        self._pinecone_service = pinecone_service
        self.alpha = alpha
        self.top_k = top_k
        self.username = username

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            results = self._pinecone_service.hybrid_search(
                query=query,
                username=self.username,
                top_k=self.top_k,
                alpha=self.alpha
            )
            documents = []
            for result in results:
                documents.append(
                    Document(
                        page_content=result.get("content", ""),
                        metadata=result
                    )
                )
            return documents
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)

    def set_user_filter(self, username: Optional[str]):
        self.username = username

    def set_search_params(self, alpha: float = None, top_k: int = None):
        if alpha is not None and 0.0 <= alpha <= 1.0:
            self.alpha = alpha
        if top_k is not None and top_k > 0:
            self.top_k = top_k

# Factory function
def create_retriever(
    alpha: float = 0.5,
    top_k: int = 5,
    username: Optional[str] = None
) -> PineconeHybridRetriever:
    """Create a retriever without exposing pinecone_service"""
    return PineconeHybridRetriever(
        pinecone_service=pinecone_service,
        alpha=alpha,
        top_k=top_k,
        username=username
    )
