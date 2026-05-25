"""FAISS vector store integration."""

from typing import List, Dict, Any
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from src.config import FAISS_INDEX_PATH
import os


class FAISSStore:
    """Manage FAISS vector store for fast similarity search."""

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        """
        Initialize FAISS store.
        
        Args:
            index_path: Path to store FAISS index
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_store(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """
        Create a new FAISS vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
        """
        os.makedirs(self.index_path, exist_ok=True)
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        self.vectorstore.save_local(self.index_path)

    def load_store(self):
        """Load existing FAISS store."""
        self.vectorstore = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embeddings
        )
        return self.vectorstore

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        if self.vectorstore is None:
            self.load_store()
        
        results = self.vectorstore.similarity_search_with_scores(query, k=k)
        return results

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """
        Add new documents to the store.
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
        """
        if self.vectorstore is None:
            self.load_store()
        
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.save_local(self.index_path)
