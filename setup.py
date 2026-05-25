"""Setup configuration for the healthcare capstone project."""

from setuptools import setup, find_packages

setup(
    name="healthcare-rag",
    version="0.1.0",
    description="Healthcare RAG system using LangChain and LangGraph",
    author="Healthcare Capstone Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "langchain==0.1.9",
        "langgraph==0.0.11",
        "faiss-cpu==1.7.4",
        "chromadb==0.4.22",
        "pypdf==4.0.1",
        "streamlit==1.31.1",
        "python-dotenv==1.0.0",
        "openai==1.3.8",
    ],
)
