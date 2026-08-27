"""ChromaDB vector store integration for document chunks used by RAG."""

from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from src.config import CHROMADB_PATH


class ChromaDBStore:
    """Manage ChromaDB vector store for semantic search."""

    def __init__(self, persist_directory: str = CHROMADB_PATH):
        """
        Initialize ChromaDB store.
        
        Args:
            persist_directory: Path to store ChromaDB data
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def create_store(self, texts: List[str], metadatas: List[Dict] = None) -> None:
        """
        Create a new ChromaDB vector store.
        
        Args:
            texts: List of text documents
            metadatas: Optional metadata for each document
        """
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()

    def load_store(self):
        """Load existing ChromaDB store."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings.embed_query
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
        self.vectorstore.persist()
