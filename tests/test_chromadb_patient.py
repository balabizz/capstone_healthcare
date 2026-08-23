import pytest
from typing import Any, Dict, List

from src.models.patient_vo import PatientVO
import src.vector_store.chromadb_store as chroma_module
from src.vector_store.chromadb_store import ChromaDBStore


class FakeVectorStore:
    def __init__(self):
        self.persist_called = False
        self.added_texts = []
        self.added_metadatas = []
        self._search_results = []

    def persist(self):
        self.persist_called = True

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        self.added_texts.extend(texts)
        if metadatas:
            self.added_metadatas.extend(metadatas)

    def similarity_search_with_scores(self, query: str, k: int = 5):
        # Return the pre-configured search results
        return self._search_results


class FakeChroma:
    """Fake replacement for langchain.vectorstores.Chroma used in tests."""

    def __init__(self, persist_directory: str = None, embedding_function: Any = None):
        # constructor used by load_store
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function
        # instance that mimics the real vectorstore behaviour
        self._store = FakeVectorStore()

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any = None, metadatas: List[Dict] = None, persist_directory: str = None):
        # create a FakeChroma instance and attach a FakeVectorStore as .vectorstore
        inst = cls(persist_directory=persist_directory, embedding_function=getattr(embedding, 'embed_query', None))
        # store the texts/metadatas on the fake store to assert later
        inst._store.add_texts(texts, metadatas)
        inst._store.persist()
        # Return an object that matches the real API: methods add_texts, persist, similarity_search_with_scores
        return inst._store


class FakeEmbeddings:
    def __init__(self):
        pass

    def embed_documents(self, docs: List[str]):
        # Return deterministic small vectors
        return [[len(d), 0.0] for d in docs]

    def embed_query(self, query: str):
        return [len(query), 0.0]


def test_create_store_calls_chroma_from_texts(monkeypatch, tmp_path):
    # Patch Chroma and OpenAIEmbeddings used in chromadb_store
    monkeypatch.setattr(chroma_module, "Chroma", FakeChroma)
    monkeypatch.setattr(chroma_module, "OpenAIEmbeddings", FakeEmbeddings)

    # Prepare a PatientVO and texts/metadatas
    vo = PatientVO.from_dict({
        "patient_id": "p-1",
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": "1990-01-01",
        "address": "123 Main St",
        "mobile_number": "+1000000000",
    })

    texts = ["Patient record for John Doe"]
    metadatas = [vo.to_metadata()]

    store = ChromaDBStore(persist_directory=str(tmp_path))
    # create_store should call our FakeChroma.from_texts and return the fake vectorstore
    store.create_store(texts=texts, metadatas=metadatas)

    # After create_store, vectorstore should be set and persist should have been called
    assert store.vectorstore is not None
    # The fake vectorstore records texts and metadata when created
    assert texts[0] in store.vectorstore.added_texts
    assert metadatas[0] in store.vectorstore.added_metadatas
    assert store.vectorstore.persist_called is True


def test_add_and_search_documents(monkeypatch, tmp_path):
    # Patch embeddings (so constructor doesn't fail) but don't patch Chroma.from_texts here
    monkeypatch.setattr(chroma_module, "OpenAIEmbeddings", FakeEmbeddings)

    # Create store and attach a FakeVectorStore
    store = ChromaDBStore(persist_directory=str(tmp_path))
    fake = FakeVectorStore()

    # Preconfigure search results to return a metadata that includes patient metadata
    vo = PatientVO.from_dict({
        "patient_id": "p-2",
        "first_name": "Alice",
        "last_name": "Smith",
        "date_of_birth": "1985-06-15",
        "address": "10 Downing St",
        "mobile_number": "+1999999999",
    })
    fake._search_results = [({"document": "Patient record for Alice Smith"}, 0.01), (vo.to_metadata(), 0.02)]

    # Attach fake vectorstore
    store.vectorstore = fake

    # Add a document
    texts = ["Patient record for Alice Smith"]
    metadatas = [vo.to_metadata()]
    store.add_documents(texts=texts, metadatas=metadatas)

    # Ensure texts and metadata were added to fake store
    assert texts[0] in fake.added_texts
    assert metadatas[0] in fake.added_metadatas

    # Search should return the configured results
    results = store.search(query="Alice", k=2)
    assert results == fake._search_results
