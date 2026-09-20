"""RAG (Retrieval Augmented Generation) chain implementation."""

from typing import List, Dict, Any
from src.config import LLM_MODEL, LLM_TEMPERATURE
from src.llm.task_prompts import reference_qa_prompt, NO_REFERENCE_ANSWER


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
        from langchain.chains import RetrievalQA
        from langchain.chat_models import ChatOpenAI
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model, temperature=temperature)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": reference_qa_prompt()},
            return_source_documents=True,
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
            "answer": result["result"] if result.get("source_documents") else NO_REFERENCE_ANSWER,
            "source_documents": result.get("source_documents", [])
        }

    def query_with_history(self, question: str, chat_history: List[dict], *,
                           check_access=None, context_client=None) -> Dict[str, Any]:
        """Resolve dialogue references, then retrieve evidence using only the resolved question."""
        from src.llm.conversation_context import resolve_question
        if check_access:
            check_access()
        if not chat_history:
            result = self.query(question)
        else:
            resolved = resolve_question(question, chat_history, client=context_client)
            if check_access:
                check_access()
            if resolved['clarification']:
                return {'question': question, 'answer': resolved['clarification'],
                        'source_documents': [], 'needs_clarification': True}
            result = self.query(resolved['question'])
            result['standalone_question'] = resolved['question']
            result['question'] = question
        if check_access:
            check_access()
        return result
