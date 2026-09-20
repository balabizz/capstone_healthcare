"""PDF -> page extraction -> chunks -> OpenAI embeddings -> shared FAISS index."""
import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath
from src.data_processing.pdf_loader import PDFLoader
from src.data_processing.text_processing import TextProcessor
from src.vector_store.faiss_store import FAISSStore
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class DocumentIngestion:
    def __init__(self, store=None, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        if not 100 <= chunk_size <= 4000 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError('Chunk size must be 100–4000; overlap must be smaller than chunk size.')
        self.store = store or FAISSStore()
        self.processor = TextProcessor(chunk_size, chunk_overlap)
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap

    def ingest(self, filename, data):
        source = PurePosixPath(filename.replace('\\', '/')).name
        if not source.lower().endswith('.pdf'):
            raise ValueError('Only PDF files are supported.')
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError('PDF must be nonempty and at most 20 MB.')
        digest = hashlib.sha256(data).hexdigest()
        for entry in self.store.list_documents():
            if entry.get('sha256') == digest:
                return {**entry, 'status': 'duplicate'}
        pages = PDFLoader.load_pages(data)
        chunks = []
        for page in pages:
            for text in self.processor.chunk_text(page['text']):
                chunks.append({'content': text, 'metadata': {'source': source, 'page': page['page'],
                    'chunk_index': len(chunks), 'document_id': digest}})
                if len(chunks) > 5000:
                    raise ValueError('PDF exceeds the 5000-chunk limit.')
        entry = {'document_id': digest, 'sha256': digest, 'source': source,
                 'pages': len(pages), 'empty_pages': sum(not p['text'] for p in pages),
                 'chunks': len(chunks), 'chunk_size': self.chunk_size, 'chunk_overlap': self.chunk_overlap,
                 'indexed_at': datetime.now(timezone.utc).isoformat()}
        return self.store.ingest_chunks(chunks, entry)
