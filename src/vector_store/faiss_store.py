"""Shared reference FAISS index with atomic index/metadata persistence (no pickle)."""
import json
import sqlite3
from pathlib import Path
from uuid import uuid4
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from src.config import FAISS_INDEX_PATH, MAX_DOCUMENTS
from src.embeddings.embedding_manager import EmbeddingManager


class FAISSStore:
    def __init__(self, index_path=FAISS_INDEX_PATH, embeddings=None):
        self.index_path = str(index_path)
        self.embeddings = embeddings or EmbeddingManager()
        self.vectorstore = None

    def _connect(self):
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(Path(self.index_path) / 'references.sqlite'), timeout=2)
        c.row_factory = sqlite3.Row
        c.execute('CREATE TABLE IF NOT EXISTS reference_index (id INTEGER PRIMARY KEY CHECK(id=1), model TEXT, vectors BLOB, documents TEXT, manifest TEXT)')
        return c

    def _decode(self, row):
        if row['model'] != self.embeddings.model:
            raise ValueError('Reference embedding model differs from the saved index. Use the original model or a new index directory.')
        index = faiss.deserialize_index(np.frombuffer(row['vectors'], dtype='uint8').copy())
        docs = json.loads(row['documents'])
        mapping = {i: d['id'] for i, d in enumerate(docs)}
        store = InMemoryDocstore({d['id']: Document(page_content=d['text'], metadata=d['metadata']) for d in docs})
        return FAISS(self.embeddings, index, store, mapping)

    def _save(self, c, store, manifest):
        docs = []
        for i in range(store.index.ntotal):
            key = store.index_to_docstore_id[i]
            document = store.docstore.search(key)
            docs.append({'id': key, 'text': document.page_content, 'metadata': document.metadata})
        c.execute('INSERT OR REPLACE INTO reference_index VALUES (1,?,?,?,?)',
                  (self.embeddings.model, faiss.serialize_index(store.index).tobytes(), json.dumps(docs), json.dumps(manifest)))

    def load_store(self):
        with self._connect() as c:
            row = c.execute('SELECT * FROM reference_index WHERE id=1').fetchone()
        if row is None:
            raise ValueError('No reference index is available. Upload PDFs in Reference documents or run python -m scripts.ingest_documents. Legacy index.faiss/index.pkl files must be rebuilt from source PDFs.')
        self.vectorstore = self._decode(row)
        return self.vectorstore

    def list_documents(self):
        with self._connect() as c:
            row = c.execute('SELECT manifest FROM reference_index WHERE id=1').fetchone()
        return json.loads(row['manifest']) if row else []

    def _append(self, texts, metadatas, entry, replace=False):
        # Serialize writers; failure at any stage rolls back both vectors and metadata.
        with self._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT * FROM reference_index WHERE id=1').fetchone()
            manifest = json.loads(row['manifest']) if row and not replace else []
            if entry.get('sha256') and any(d.get('sha256') == entry['sha256'] for d in manifest):
                return {**entry, 'status': 'duplicate'}
            if len(manifest) >= MAX_DOCUMENTS:
                raise ValueError('Reference document limit reached.')
            old = self._decode(row) if row and not replace else None
            if not texts or len(texts) != len(metadatas):
                raise ValueError('No valid reference chunks supplied.')
            vectors = np.asarray(self.embeddings.embed_documents(texts), dtype='float32')
            if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
                raise ValueError('Invalid reference embedding response.')
            if old and old.index.d != vectors.shape[1]:
                raise ValueError('Reference embedding dimensions changed.')
            addition = FAISS.from_embeddings(zip(texts, vectors.tolist()), self.embeddings, metadatas=metadatas)
            if old:
                old.merge_from(addition)
                addition = old
            manifest.append(entry)
            self._save(c, addition, manifest)
        self.vectorstore = addition
        return {**entry, 'status': 'indexed'}

    def ingest_chunks(self, chunks, entry):
        return self._append([d['content'] for d in chunks], [d['metadata'] for d in chunks], entry)

    def create_store(self, texts, metadatas=None):
        return self._append(texts, metadatas or [{} for _ in texts],
                            {'document_id': uuid4().hex, 'source': 'manual', 'chunks': len(texts)}, replace=True)

    def add_documents(self, texts, metadatas=None):
        return self._append(texts, metadatas or [{} for _ in texts],
                            {'document_id': uuid4().hex, 'source': 'manual', 'chunks': len(texts)})

    def search(self, query, k=5):
        if not query.strip() or len(query) > 2000 or not 1 <= k <= 20:
            raise ValueError('Enter a search query up to 2000 characters; k must be 1–20.')
        return self.load_store().similarity_search_with_score(query, k=k)
