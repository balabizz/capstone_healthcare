"""Authorized patient summary pipeline; native FAISS indexes persisted atomically in SQLite."""
import json
import requests
from src.config import OPENAI_API_KEY, PATIENT_SUMMARY_EMBEDDING_MODEL


class SummaryEmbeddings:
    def __init__(self, model=PATIENT_SUMMARY_EMBEDDING_MODEL, session=None):
        self.model = model
        self.session = session or requests.Session()

    def embed(self, texts):
        if not OPENAI_API_KEY:
            raise ValueError('Configure OPENAI_API_KEY to index/search summaries.')
        try:
            response = self.session.post('https://api.openai.com/v1/embeddings',
                headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
                json={'model': self.model, 'input': texts, 'encoding_format': 'float'}, timeout=(10, 60))
            response.raise_for_status()
            rows = sorted(response.json()['data'], key=lambda row: row['index'])
            if [r['index'] for r in rows] != list(range(len(texts))):
                raise ValueError('Invalid embedding response.')
            return [r['embedding'] for r in rows]
        except (requests.RequestException, KeyError, TypeError) as error:
            raise ValueError('Summary embedding request failed. Check configuration and connection.') from error


class PatientSummaryStore:
    def __init__(self, history, embeddings=None):
        self.history = history
        self.embeddings = embeddings or SummaryEmbeddings()
        with history.store._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS patient_summary_vectors (
                patient_id TEXT PRIMARY KEY REFERENCES patients(patient_id) ON DELETE CASCADE,
                fingerprint TEXT NOT NULL, model TEXT NOT NULL, snapshot_json TEXT NOT NULL,
                chunks_json TEXT NOT NULL, faiss_index BLOB NOT NULL)''')

    def _snapshot(self, requester, patient):
        return self.history.retrieve(requester_patient_id=requester, patient_id=patient)

    @staticmethod
    def _matrix(values, count):
        import numpy as np
        import faiss
        matrix = np.asarray(values, dtype='float32')
        if matrix.ndim != 2 or matrix.shape[0] != count or not matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError('Invalid summary vectors.')
        if (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError('Empty summary vector.')
        faiss.normalize_L2(matrix)
        return matrix

    def index(self, *, requester_patient_id, bundle, summary):
        return self._index(bundle, summary, lambda: self._snapshot(requester_patient_id, bundle['subject_patient_id']))

    def rebuild_for_staff(self, *, staff_type, staff_id, patient_id, summarizer=None):
        from src.llm.planning_client import PlanningClient
        def snapshot():
            return self.history.retrieve_for_staff(staff_type=staff_type, staff_id=staff_id, patient_id=patient_id)
        bundle = snapshot()
        summary = (summarizer or PlanningClient().summarize_history)(bundle)
        return self._index(bundle, summary, snapshot)

    def _index(self, bundle, summary, snapshot):
        import faiss
        patient = bundle['subject_patient_id']
        current = snapshot()
        if current['source_fingerprint'] != bundle['source_fingerprint']:
            raise ValueError('Medical records changed. Rebuild the summary.')
        if not isinstance(summary, str) or not summary.strip() or len(summary) > 100000:
            raise ValueError('Invalid patient summary.')
        # Replace the entire patient index: repeated indexing never accumulates duplicates.
        chunks = [summary[i:i+1800] for i in range(0, len(summary), 1800)]
        vectors = self._matrix(self.embeddings.embed(chunks), len(chunks))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        if snapshot()['source_fingerprint'] != bundle['source_fingerprint']:
            raise ValueError('Medical records changed during indexing. Rebuild the summary.')
        with self.history.store._connect() as c:
            c.execute('INSERT OR REPLACE INTO patient_summary_vectors VALUES (?,?,?,?,?,?)',
                (patient, bundle['source_fingerprint'], self.embeddings.model,
                 json.dumps(bundle), json.dumps(chunks), faiss.serialize_index(index).tobytes()))
        return len(chunks)

    def rebuild(self, *, requester_patient_id, patient_id, summarizer=None):
        from src.llm.planning_client import PlanningClient
        bundle = self._snapshot(requester_patient_id, patient_id)
        summary = (summarizer or PlanningClient().summarize_history)(bundle)
        return self.index(requester_patient_id=requester_patient_id, bundle=bundle, summary=summary)

    def search(self, *, requester_patient_id, patient_id, query, k=5):
        self.history.require_access(requester_patient_id, patient_id)
        result = self._search(patient_id, query, k)
        self.history.require_access(requester_patient_id, patient_id)
        return result

    def search_for_staff(self, *, staff_type, staff_id, patient_id, query, k=5):
        self.history.require_staff_access(staff_type, staff_id, patient_id)
        result = self._search(patient_id, query, k)
        self.history.require_staff_access(staff_type, staff_id, patient_id)
        return result

    def _search(self, patient_id, query, k=5):
        import faiss
        import numpy as np
        current = self.history._retrieve_unchecked(patient_id)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or not 1 <= k <= 20:
            raise ValueError('Enter a search query up to 2000 characters; k must be 1–20.')
        with self.history.store._connect() as c:
            row = c.execute('SELECT * FROM patient_summary_vectors WHERE patient_id=?', (patient_id,)).fetchone()
        if row is None:
            return {'status': 'missing', 'matches': []}
        if row['fingerprint'] != current['source_fingerprint'] or row['model'] != self.embeddings.model:
            return {'status': 'stale', 'matches': []}
        index = faiss.deserialize_index(np.frombuffer(row['faiss_index'], dtype='uint8').copy())
        vector = self._matrix(self.embeddings.embed([query]), 1)
        if vector.shape[1] != index.d:
            raise ValueError('Embedding dimensions changed. Rebuild the summary.')
        scores, ids = index.search(vector, min(k, index.ntotal))
        latest = self.history._retrieve_unchecked(patient_id)
        if latest['source_fingerprint'] != row['fingerprint']:
            return {'status': 'stale', 'matches': []}
        chunks = json.loads(row['chunks_json'])
        return {'status': 'ready', 'patient_id': patient_id, 'snapshot': json.loads(row['snapshot_json']),
                'matches': [{'text': chunks[int(i)], 'score': float(s), 'chunk': int(i)}
                            for s, i in zip(scores[0], ids[0]) if i >= 0]}
