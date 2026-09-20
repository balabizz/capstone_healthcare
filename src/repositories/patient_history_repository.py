"""Patient-scoped, bounded clinical evidence for LLM history summaries."""
from datetime import date, datetime
import json
import hashlib
from zoneinfo import ZoneInfo

from src.config import SCHEDULE_TIMEZONE
from src.database.sqlite_store import SQLiteStore
from src.repositories.dependent_repository import DependentRepository


class PatientHistoryRepository:
    def __init__(self, database_path, limit=50, text_limit=4000):
        if not 1 <= limit <= 200 or not 100 <= text_limit <= 10000:
            raise ValueError('Invalid history retrieval limits.')
        self.store = SQLiteStore(database_path)
        self.permissions = DependentRepository(database_path)
        self.limit, self.text_limit = limit, text_limit

    def require_access(self, requester_patient_id, patient_id):
        if not requester_patient_id or not patient_id:
            raise PermissionError('Sign in and select a patient before retrieving history.')
        self.permissions.require_access(requester_patient_id, patient_id, 'view_medical')
        if not self.store.get_patient(patient_id):
            raise ValueError('Patient record not found.')

    @staticmethod
    def prescription_timing(row, today):
        try:
            start = date.fromisoformat(row['start_date']) if row['start_date'] else None
            end = date.fromisoformat(row['end_date']) if row['end_date'] else None
            if start and end and end < start:
                return 'invalid_recorded_date_range'
            if start and start > today:
                return 'future_recorded_start'
            if end and end < today:
                return 'past_recorded_end'
            if start and end:
                return 'within_recorded_dates_not_confirmed_taking'
            return 'incomplete_dates_current_use_unknown'
        except (ValueError, TypeError):
            return 'invalid_dates_current_use_unknown'

    def retrieve(self, *, requester_patient_id, patient_id):
        self.require_access(requester_patient_id, patient_id)
        now = datetime.now(ZoneInfo(SCHEDULE_TIMEZONE))
        # One read transaction keeps category counts and rows at the same snapshot.
        with self.store._connect() as c:
            c.execute('BEGIN')
            queries = {
                'records': ('medical_history', 'history_id,record_type,condition_name,diagnosis_date,treatment,notes,'
                            'version,recorded_at,updated_at', 'COALESCE(updated_at,recorded_at) DESC,history_id'),
                'prescriptions': ('prescriptions', 'prescription_id,medication_name,dosage,frequency,start_date,end_date,'
                                  'instructions,prescribed_at', 'prescribed_at DESC,prescription_id'),
                'alerts': ('patient_alerts', 'alert_id,alert_type,severity,description,status,recorded_at,resolved_at,resolution_reason',
                           "CASE status WHEN 'active' THEN 0 ELSE 1 END, CASE severity WHEN 'critical' THEN 0 "
                           "WHEN 'high' THEN 1 WHEN 'moderate' THEN 2 ELSE 3 END,recorded_at DESC,alert_id"),
            }
            digest = hashlib.sha256(now.date().isoformat().encode())
            for table in ('medical_history', 'prescriptions', 'patient_alerts'):
                for row in c.execute(f'SELECT * FROM {table} WHERE patient_id=? ORDER BY 1', (patient_id,)):
                    digest.update(json.dumps(dict(row), sort_keys=True).encode())
            bundle = {'source_fingerprint': digest.hexdigest(), 'subject_patient_id': patient_id, 'retrieved_at': now.isoformat(), 'coverage': {}}
            for category, (table, fields, ordering) in queries.items():
                total = c.execute(f'SELECT COUNT(*) FROM {table} WHERE patient_id=?', (patient_id,)).fetchone()[0]
                rows = [dict(r) for r in c.execute(f'SELECT {fields} FROM {table} WHERE patient_id=? ORDER BY {ordering} LIMIT ?',
                                                  (patient_id, self.limit))]
                shortened, included, used = [], [], 0
                text_fields = {'condition_name','treatment','notes','medication_name','dosage','frequency',
                               'instructions','description','resolution_reason'}
                for row in rows:
                    record_id = row.get('history_id') or row.get('prescription_id') or row.get('alert_id')
                    row_shortened = []
                    for field, value in list(row.items()):
                        if field in text_fields and isinstance(value, str) and len(value) > self.text_limit:
                            row[field] = value[:self.text_limit] + ' [TRUNCATED]'
                            row_shortened.append(f'{str(record_id)[:200]}:{field}')
                    if category == 'prescriptions':
                        row['date_interpretation'] = self.prescription_timing(row, now.date())
                    size = len(json.dumps(row))
                    if used + size > 24000:
                        break
                    used += size
                    included.append(row)
                    shortened.extend(row_shortened)
                bundle[category] = included
                bundle['coverage'][category] = {'total': total, 'included': len(included),
                    'omitted': total - len(included), 'truncated_fields': shortened}
                if category == 'alerts':
                    active_total = c.execute("SELECT COUNT(*) FROM patient_alerts WHERE patient_id=? AND status='active'",
                                             (patient_id,)).fetchone()[0]
                    bundle['coverage'][category]['active_omitted'] = active_total - sum(r['status'] == 'active' for r in included)
        self.require_access(requester_patient_id, patient_id)
        return bundle
