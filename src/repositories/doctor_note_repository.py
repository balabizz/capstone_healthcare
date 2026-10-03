"""Append-only doctor notes in the existing patient medical history."""
import json
from uuid import uuid4
from src.database.sqlite_store import SQLiteStore


class DoctorNoteRepository:
    def __init__(self, database_path):
        self.store = SQLiteStore(database_path)

    @staticmethod
    def _authorize(connection, doctor_id, patient_id):
        if not connection.execute(
            "SELECT 1 FROM login_details WHERE doctor_id=? AND user_type='doctor' AND is_active=1",
            (doctor_id,)).fetchone():
            raise PermissionError('An active doctor account is required.')
        if not connection.execute('SELECT 1 FROM patients WHERE patient_id=?', (patient_id,)).fetchone():
            raise ValueError('Patient record not found.')

    def list_notes(self, doctor_id, patient_id):
        with self.store._connect() as connection:
            self._authorize(connection, doctor_id, patient_id)
            return [dict(row) for row in connection.execute(
                "SELECT h.history_id,h.notes,h.recorded_at,h.doctor_id,d.first_name,d.last_name "
                "FROM medical_history h JOIN doctors d ON d.doctor_id=h.doctor_id "
                "WHERE h.patient_id=? AND TRIM(COALESCE(h.notes,''))<>'' "
                "ORDER BY h.recorded_at DESC,h.history_id DESC", (patient_id,))]

    def save(self, *, doctor_id, patient_id, notes, request_id):
        if not isinstance(notes, str) or not notes.strip() or len(notes) > 10000:
            raise ValueError('Enter a doctor note of 1 to 10,000 characters.')
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('A save request ID is required.')
        notes = notes.strip()
        history_id = 'doctor-note-' + doctor_id + '-' + request_id
        with self.store._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._authorize(connection, doctor_id, patient_id)
            prior = connection.execute('SELECT * FROM medical_history WHERE history_id=?', (history_id,)).fetchone()
            if prior:
                if (prior['patient_id'], prior['doctor_id'], prior['notes']) != (patient_id, doctor_id, notes):
                    raise ValueError('This save request was already used for a different note.')
                return dict(prior)
            connection.execute(
                "INSERT INTO medical_history(history_id,patient_id,doctor_id,record_type,condition_name,notes,updated_at) "
                "VALUES (?,?,?,'note','',?,CURRENT_TIMESTAMP)", (history_id,patient_id,doctor_id,notes))
            connection.execute(
                'INSERT INTO agent_events(event_id,request_id,patient_id,event_type,tool_name,status,details_json) '
                'VALUES (?,?,?,?,?,?,?)',
                (uuid4().hex,request_id,patient_id,'doctor_note_created','doctor_notes.save','success',
                 json.dumps({'doctor_id':doctor_id,'history_id':history_id})))
            return dict(connection.execute('SELECT * FROM medical_history WHERE history_id=?', (history_id,)).fetchone())
