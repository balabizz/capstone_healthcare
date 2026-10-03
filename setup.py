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
        "langchain-community==0.0.38",
        "langgraph==0.0.11",
        "faiss-cpu==1.8.0",
        "pypdf==4.0.1",
        "streamlit==1.55.0",
        "python-dotenv==1.0.0",
        "openai==1.3.8",
        "requests==2.31.0",
        "pandas==2.1.3",
        "numpy==1.26.4",
        "pydantic==2.7.4",
        "pyyaml==6.0.1",
    ],
)
