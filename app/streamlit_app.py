"""Streamlit web application for healthcare RAG system."""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.llm_client import LLMClient
from src.vector_store.chromadb_store import ChromaDBStore
from src.chains.rag_chain import RAGChain
from src.graphs.workflow_graph import HealthcareWorkflow


def main():
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Healthcare RAG System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Healthcare RAG System")
    st.markdown("Retrieve and generate healthcare information using AI")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        db_type = st.radio(
            "Vector Database",
            ("ChromaDB", "FAISS"),
            help="Choose the vector database backend"
        )
        
        retrieval_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask a Question")
        query = st.text_area(
            "Enter your healthcare question:",
            placeholder="e.g., What are the symptoms of diabetes?",
            height=100
        )
        
        if st.button("Search", key="search_btn"):
            if query:
                with st.spinner("Searching and generating response..."):
                    # Initialize RAG components
                    llm_client = LLMClient()
                    
                    # Initialize vector store
                    if db_type == "ChromaDB":
                        from src.vector_store.chromadb_store import ChromaDBStore
                        vectorstore = ChromaDBStore().load_store()
                    else:
                        from src.vector_store.faiss_store import FAISSStore
                        vectorstore = FAISSStore().load_store()
                    
                    # Create RAG chain
                    rag_chain = RAGChain(vectorstore)
                    
                    # Get response
                    result = rag_chain.query(query)
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": result["answer"]
                    })
                    
                    # Display result
                    st.success("Response generated!")
                    st.markdown("### Answer")
                    st.write(result["answer"])
                    
                    if result["source_documents"]:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.markdown(f"**Document {i}:**")
                                st.write(doc.page_content[:500] + "...")
            else:
                st.warning("Please enter a question")
    
    with col2:
        st.header("Chat History")
        if st.session_state.chat_history:
            for i, interaction in enumerate(st.session_state.chat_history[-5:], 1):
                with st.expander(f"Q{i}: {interaction['query'][:50]}..."):
                    st.write(interaction["answer"])
            
            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No chat history yet")


if __name__ == "__main__":
    main()
