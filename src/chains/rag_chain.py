"""RAG (Retrieval Augmented Generation) chain implementation."""

from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from src.config import LLM_MODEL, LLM_TEMPERATURE


class RAGChain:
    """Retrieval Augmented Generation chain."""

    def __init__(self, vectorstore, model: str = LLM_MODEL, 
                 temperature: float = LLM_TEMPERATURE):
        """
        Initialize RAG chain.
        
        Args:
            vectorstore: Vector store for retrieval
            model: LLM model to use
            temperature: Temperature for generation
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG chain.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with question, answer, and source documents
        """
        result = self.chain({"query": question})
        return {
            "question": question,
            "answer": result["result"],
            "source_documents": result.get("source_documents", [])
        }

    def query_with_history(self, question: str, chat_history: List[tuple]) -> str:
        """
        Query with conversation history.
        
        Args:
            question: Current question
            chat_history: List of (question, answer) tuples
            
        Returns:
            Generated answer
        """
        # Build context from history
        history_context = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in chat_history
        ])
        
        enhanced_question = f"Chat history:\n{history_context}\n\nNew question: {question}"
        result = self.chain({"query": enhanced_question})
        
        return result["result"]
