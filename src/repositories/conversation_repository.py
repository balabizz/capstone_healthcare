"""Durable conversations scoped to both the speaker and selected patient."""
import re
from uuid import uuid4


class ConversationRepository:
    def __init__(self, history):
        self.history = history
        with history.store._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS patient_conversations (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT UNIQUE NOT NULL,
                requester_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
                query TEXT NOT NULL, answer TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))''')
            c.execute('CREATE INDEX IF NOT EXISTS conversation_scope ON patient_conversations(requester_id,patient_id,sequence)')

    def save(self, requester, patient, query, answer, turn_id=None):
        self.history.require_access(requester, patient)
        if not query.strip() or not answer.strip() or len(query) > 12000 or len(answer) > 100000:
            raise ValueError('Conversation exceeds storage limits or is empty.')
        with self.history.store._connect() as c:
            c.execute('INSERT OR IGNORE INTO patient_conversations(turn_id,requester_id,patient_id,query,answer) VALUES (?,?,?,?,?)',
                      (turn_id or uuid4().hex, requester, patient, query, answer))

    def retrieve(self, requester, patient, query='', limit=6):
        self.history.require_access(requester, patient)
        if not 1 <= limit <= 20:
            raise ValueError('Invalid conversation limit.')
        with self.history.store._connect() as c:
            recent = list(c.execute('SELECT * FROM patient_conversations WHERE requester_id=? AND patient_id=? ORDER BY sequence DESC LIMIT ?',
                                   (requester, patient, limit)))
            terms = list(dict.fromkeys(re.findall(r'[\w]{4,}', query.lower())))[:8]
            relevant = []
            if terms:
                clauses = ' OR '.join('(lower(query) LIKE ? OR lower(answer) LIKE ?)' for _ in terms)
                params = [value for term in terms for value in ('%'+term+'%', '%'+term+'%')]
                relevant = list(c.execute(f'SELECT * FROM patient_conversations WHERE requester_id=? AND patient_id=? AND ({clauses}) ORDER BY sequence DESC LIMIT ?',
                                         [requester, patient, *params, limit]))
        # Half the budget preserves immediate continuity; the remainder finds older related turns.
        selected = {r['sequence']: r for r in recent[:max(1, limit//2)]}
        reasons = {key:'recent_continuity' for key in selected}
        related_ids = {r['sequence'] for r in relevant}
        for row in relevant + recent:
            if len(selected) >= limit:
                break
            if row['sequence'] not in selected:
                reasons[row['sequence']] = 'keyword_match' if row['sequence'] in related_ids else 'recent_fallback'
            selected[row['sequence']] = row
        self.history.require_access(requester, patient)
        return [{'turn_id': r['turn_id'], 'created_at': r['created_at'],
                 'selection_reason': reasons[r['sequence']],
                 'query': r['query'][:1500], 'answer': r['answer'][:2500],
                 'truncated': len(r['query']) > 1500 or len(r['answer']) > 2500}
                for _, r in sorted(selected.items())]

    def get_turns(self, requester, patient, turn_ids):
        """Resolve trace IDs only within the same authorized conversation namespace."""
        self.history.require_access(requester,patient)
        if len(turn_ids)>20:
            raise ValueError('Too many memory references.')
        rows = []
        with self.history.store._connect() as c:
            for turn_id in turn_ids:
                row = c.execute('SELECT turn_id,created_at,query,answer FROM patient_conversations WHERE requester_id=? AND patient_id=? AND turn_id=?',
                                (requester,patient,turn_id)).fetchone()
                if row:
                    rows.append({**dict(row),'query':row['query'][:1500],'answer':row['answer'][:2500]})
        self.history.require_access(requester,patient)
        return rows

    def clear(self, requester, patient):
        self.history.require_access(requester, patient)
        with self.history.store._connect() as c:
            c.execute('DELETE FROM patient_conversations WHERE requester_id=? AND patient_id=?', (requester, patient))
