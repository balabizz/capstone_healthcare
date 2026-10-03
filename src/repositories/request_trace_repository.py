"""Patient-authorized request traces, separate from redacted operational events."""
import json


class RequestTraceRepository:
    def __init__(self, history):
        self.history = history
        with history.store._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS request_traces (
                request_id TEXT PRIMARY KEY, requester_id TEXT NOT NULL REFERENCES patients(patient_id),
                patient_id TEXT NOT NULL REFERENCES patients(patient_id), plan_json TEXT NOT NULL,
                memory_json TEXT NOT NULL, steps_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))''')

    def save(self, requester, patient, request_id, plan, memory, steps):
        self.history.require_access(requester,patient)
        # Persist execution metadata, not generated answers or copied memory text.
        execution = [{k:r[k] for k in ('id','goal','status','duration_ms','depends_on','blocked_by') if k in r} for r in steps]
        with self.history.store._connect() as c:
            c.execute('INSERT OR IGNORE INTO request_traces(request_id,requester_id,patient_id,plan_json,memory_json,steps_json) VALUES (?,?,?,?,?,?)',
                      (request_id,requester,patient,json.dumps(plan),json.dumps(memory),json.dumps(execution)))

    def list(self, requester, patient, limit=20):
        self.history.require_access(requester,patient)
        if not 1 <= limit <= 50:
            raise ValueError('Invalid trace limit.')
        with self.history.store._connect() as c:
            rows = c.execute('SELECT * FROM request_traces WHERE requester_id=? AND patient_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?',
                             (requester,patient,limit)).fetchall()
        self.history.require_access(requester,patient)
        return [{'request_id':r['request_id'],'created_at':r['created_at'],
                 'plan':json.loads(r['plan_json']),'memory':json.loads(r['memory_json']),
                 'steps':json.loads(r['steps_json'])} for r in rows]

    def clear(self, requester, patient):
        self.history.require_access(requester,patient)
        with self.history.store._connect() as c:
            c.execute('DELETE FROM request_traces WHERE requester_id=? AND patient_id=?',(requester,patient))
