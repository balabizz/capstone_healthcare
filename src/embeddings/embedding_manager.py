"""Batched OpenAI embeddings shared by reference ingestion and retrieval."""
from langchain_core.embeddings import Embeddings
from src.config import EMBEDDING_MODEL
from src.vector_store.patient_summaries import SummaryEmbeddings


class EmbeddingManager(Embeddings):
    def __init__(self, model=EMBEDDING_MODEL, client=None):
        self.model = model
        self.client = client or SummaryEmbeddings(model=model)

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), 32):
            vectors.extend(self.client.embed(texts[start:start+32]))
        return vectors

    def embed_query(self, text):
        return self.client.embed([text])[0]

    def embed_text(self, text):
        return self.embed_query(text)
