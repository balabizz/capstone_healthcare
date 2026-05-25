"""Text processing and chunking utilities."""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextProcessor:
    """Process and chunk text documents."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize text processor with chunk parameters.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def process_documents(self, documents: dict) -> List[dict]:
        """
        Process multiple documents and create chunks with metadata.
        
        Args:
            documents: Dictionary of filename: content pairs
            
        Returns:
            List of chunks with metadata
        """
        processed_chunks = []
        
        for filename, content in documents.items():
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                processed_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": filename,
                        "chunk_index": chunk_idx
                    }
                })
        
        return processed_chunks
