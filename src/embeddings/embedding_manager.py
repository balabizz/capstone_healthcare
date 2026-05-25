"""Embedding generation and management."""

from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from src.config import EMBEDDING_MODEL


class EmbeddingManager:
    """Manage embeddings for documents."""

    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize embedding manager.
        
        Args:
            model: Embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(model=model)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
