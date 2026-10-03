"""Attendant-scoped medical record tools with optimistic edits and audit history.

attendant_id must come from the authenticated session. Every operation rechecks
that an active attendant account exists and the patient is assigned to it.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
import json

from src.config import SCHEDULE_TIMEZONE
from src.database.sqlite_store import SQLiteStore


class RecordConflictError(ValueError):
    pass


class MedicalRecordRepository:
    def __init__(self, database_path):
        self.store = SQLiteStore(database_path)

    def _authorize(self, connection, attendant_id, patient_id=None):
        active = connection.execute("SELECT 1 FROM login_details WHERE attendant_id = ? "
            "AND user_type = 'attendant' AND is_active = 1", (attendant_id,)).fetchone()
        assigned = patient_id is None or connection.execute('SELECT 1 FROM attendant_patients '
            'WHERE attendant_id = ? AND patient_id = ?', (attendant_id, patient_id)).fetchone()
        if not active or not assigned:
            raise PermissionError('An active attendant account assigned to this patient is required.')

    def list_patients(self, attendant_id):
        with self.store._connect() as c:
            self._authorize(c, attendant_id)
            return [dict(r) for r in c.execute('SELECT p.patient_id, p.first_name, p.last_name, p.date_of_birth '
                'FROM patients p JOIN attendant_patients a ON a.patient_id = p.patient_id '
                'WHERE a.attendant_id = ? ORDER BY p.last_name, p.first_name', (attendant_id,))]

    def list_records(self, attendant_id, patient_id):
        with self.store._connect() as c:
            self._authorize(c, attendant_id, patient_id)
            return [dict(r) for r in c.execute('SELECT * FROM medical_history WHERE patient_id = ? '
                'ORDER BY COALESCE(updated_at,recorded_at) DESC, history_id', (patient_id,))]

    def revisions(self, attendant_id, patient_id, history_id):
        with self.store._connect() as c:
            self._authorize(c, attendant_id, patient_id)
            rows = c.execute('SELECT r.* FROM medical_record_revisions r JOIN medical_history h '
                'ON h.history_id = r.history_id WHERE h.patient_id = ? AND r.history_id = ? '
                'ORDER BY r.version DESC', (patient_id, history_id)).fetchall()
            return [{**dict(r), 'before': json.loads(r['before_json']) if r['before_json'] else None,
                     'after': json.loads(r['after_json'])} for r in rows]

    def save(self, *, attendant_id, patient_id, record_type, condition_name='', diagnosis_date=None,
             treatment='', notes='', doctor_id=None, history_id=None, expected_version=None,
             change_reason, request_id):
        """Create or explicitly update a record; patient ownership is immutable."""
        fields = {'condition_name': condition_name, 'treatment': treatment, 'notes': notes,
                  'change_reason': change_reason}
        for key, value in fields.items():
            if not isinstance(value, str) or len(value) > (500 if key in ('condition_name','change_reason') else 20000):
                raise ValueError(f'Invalid or excessively long {key}.')
        fields = {key: value.strip() for key, value in fields.items()}
        if record_type not in ('diagnosis', 'note') or not fields['change_reason']:
            raise ValueError('Choose a record type and provide a reason for the change.')
        if record_type == 'diagnosis' and not fields['condition_name']:
            raise ValueError('A diagnosis/condition is required.')
        if record_type == 'note' and (not fields['notes'] or fields['condition_name'] or diagnosis_date or fields['treatment']):
            raise ValueError('A standalone clinical note requires notes only; use a diagnosis entry for diagnosis/treatment.')
        if diagnosis_date:
            parsed = date.fromisoformat(diagnosis_date)
            if parsed > datetime.now(ZoneInfo(SCHEDULE_TIMEZONE)).date():
                raise ValueError('Diagnosis date cannot be in the future.')
            diagnosis_date = parsed.isoformat()
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('A save request ID is required.')
        if history_id and (type(expected_version) is not int or expected_version < 1):
            raise ValueError('An existing record version is required for updates.')
        values = dict(record_type=record_type, condition_name=fields['condition_name'],
                      diagnosis_date=diagnosis_date or None, treatment=fields['treatment'],
                      notes=fields['notes'], doctor_id=doctor_id or None)
        payload = json.dumps(dict(patient_id=patient_id, history_id=history_id, expected_version=expected_version,
                                 change_reason=fields['change_reason'], **values), sort_keys=True)
        with self.store._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            self._authorize(c, attendant_id, patient_id)
            prior = c.execute('SELECT request_payload, after_json FROM medical_record_revisions '
                'WHERE attendant_id = ? AND request_id = ?', (attendant_id, request_id)).fetchone()
            if prior:
                if prior['request_payload'] != payload:
                    raise ValueError('This save request ID was used for different content.')
                return json.loads(prior['after_json'])
            if doctor_id and not c.execute('SELECT 1 FROM doctors WHERE doctor_id = ?', (doctor_id,)).fetchone():
                raise ValueError('The selected doctor does not exist.')
            before = None
            if history_id:
                row = c.execute('SELECT * FROM medical_history WHERE history_id = ? AND patient_id = ?',
                                (history_id, patient_id)).fetchone()
                if not row:
                    raise ValueError('Record not found for the selected patient.')
                before = dict(row)
                if row['version'] != expected_version:
                    raise RecordConflictError('This record changed since you opened it. Reload and review before saving.')
                c.execute('UPDATE medical_history SET record_type=?, condition_name=?, diagnosis_date=?, treatment=?, '
                    'notes=?, doctor_id=?, version=version+1, updated_at=CURRENT_TIMESTAMP, updated_by=? WHERE history_id=?',
                    (*values.values(), attendant_id, history_id))
            else:
                history_id = f'history-{uuid4().hex}'
                c.execute('INSERT INTO medical_history (record_type,condition_name,diagnosis_date,treatment,notes,doctor_id,'
                    'history_id,patient_id,updated_by,updated_at) VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                    (*values.values(), history_id, patient_id, attendant_id))
            after = dict(c.execute('SELECT * FROM medical_history WHERE history_id=?', (history_id,)).fetchone())
            c.execute('INSERT INTO medical_record_revisions (revision_id,history_id,attendant_id,request_id,request_payload,'
                'version,before_json,after_json,change_reason) VALUES (?,?,?,?,?,?,?,?,?)',
                (uuid4().hex, history_id, attendant_id, request_id, payload, after['version'],
                 json.dumps(before) if before else None, json.dumps(after), fields['change_reason']))
            c.execute('INSERT INTO agent_events (event_id,request_id,patient_id,goal_id,event_type,tool_name,status,details_json) '
                'VALUES (?,?,?,?,?,?,?,?)', (uuid4().hex, request_id, patient_id, 'medical_record_save',
                 'medical_record_updated' if before else 'medical_record_created', 'medical_records.save', 'success',
                 json.dumps({'attendant_id': attendant_id, 'history_id': history_id, 'version': after['version']})))
        return after

    def list_alerts(self, attendant_id, patient_id):
        with self.store._connect() as c:
            self._authorize(c, attendant_id, patient_id)
            return [dict(r) for r in c.execute('SELECT * FROM patient_alerts WHERE patient_id=? '
                "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END,recorded_at DESC", (patient_id,))]

    def add_alert(self, *, attendant_id, patient_id, alert_type, severity, description, request_id):
        if alert_type not in ('allergy','clinical','other') or severity not in ('low','moderate','high','critical'):
            raise ValueError('Choose a valid alert type and severity.')
        if not isinstance(description, str) or not description.strip() or len(description) > 4000:
            raise ValueError('An alert description of at most 4,000 characters is required.')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('A save request ID is required.')
        alert_id = f'alert-{request_id}'
        with self.store._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            self._authorize(c, attendant_id, patient_id)
            prior = c.execute('SELECT * FROM patient_alerts WHERE alert_id=?', (alert_id,)).fetchone()
            if prior:
                if (prior['patient_id'],prior['recorded_by'],prior['alert_type'],prior['severity'],prior['description']) != (
                        patient_id,attendant_id,alert_type,severity,description.strip()):
                    raise ValueError('The alert request ID was already used for different content.')
                return alert_id
            c.execute('INSERT INTO patient_alerts(alert_id,patient_id,alert_type,severity,description,recorded_by) '
                'VALUES (?,?,?,?,?,?)', (alert_id,patient_id,alert_type,severity,description.strip(),attendant_id))
            c.execute('INSERT INTO agent_events(event_id,request_id,patient_id,event_type,tool_name,status,details_json) '
                'VALUES (?,?,?,?,?,?,?)', (uuid4().hex,request_id,patient_id,'patient_alert_created','patient_alerts.add','success',
                 json.dumps({'attendant_id':attendant_id,'alert_id':alert_id})))
        return alert_id

    def resolve_alert(self, *, attendant_id, patient_id, alert_id, reason):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError('A resolution reason of at most 500 characters is required.')
        with self.store._connect() as c:
            c.execute('BEGIN IMMEDIATE')
            self._authorize(c, attendant_id, patient_id)
            row = c.execute('SELECT * FROM patient_alerts WHERE alert_id=? AND patient_id=?', (alert_id,patient_id)).fetchone()
            if not row:
                raise ValueError('Alert not found for the selected patient.')
            if row['status'] == 'resolved':
                if row['resolved_by'] == attendant_id and row['resolution_reason'] == reason.strip():
                    return
                raise ValueError('This alert is already resolved; its original resolution is preserved.')
            c.execute("UPDATE patient_alerts SET status='resolved',resolved_by=?,resolved_at=CURRENT_TIMESTAMP,resolution_reason=? "
                'WHERE alert_id=?', (attendant_id,reason.strip(),alert_id))
            c.execute('INSERT INTO agent_events(event_id,request_id,patient_id,event_type,tool_name,status,details_json) '
                'VALUES (?,?,?,?,?,?,?)', (uuid4().hex,uuid4().hex,patient_id,'patient_alert_resolved','patient_alerts.resolve','success',
                 json.dumps({'attendant_id':attendant_id,'alert_id':alert_id})))
