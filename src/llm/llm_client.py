"""LLM client for generating responses."""

from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from src.config import LLM_MODEL, LLM_TEMPERATURE


class LLMClient:
    """Interface for LLM interactions."""

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initialize LLM client.
        
        Args:
            model: Model name to use
            temperature: Temperature for generation
        """
        self.model = ChatOpenAI(model_name=model, temperature=temperature)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response for a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.model(messages)
        return response.content

    def generate_with_context(self, query: str, context: List[str], 
                            system_prompt: str = None) -> str:
        """
        Generate response using context from retrieval.
        
        Args:
            query: User query
            context: List of context documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        context_text = "\n\n".join(context)
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        return self.generate(full_prompt, system_prompt)
