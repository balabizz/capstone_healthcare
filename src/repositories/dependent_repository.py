"""Family relationships and explicit, revocable patient permissions."""

from uuid import uuid4
from src.database.sqlite_store import SQLiteStore

RELATIONSHIPS = ('father', 'mother', 'spouse', 'child', 'sibling', 'guardian', 'other')
PERMISSIONS = ('view_appointments', 'book_appointment', 'view_medical')


class DependentRepository:
    def __init__(self, database_path):
        self.store = SQLiteStore(database_path)

    def add_dependent(self, patient_id, name, relationship, dependent_patient_id=None):
        if relationship not in RELATIONSHIPS or not name.strip():
            raise ValueError('A name and valid relationship are required')
        if patient_id == dependent_patient_id:
            raise ValueError('A patient cannot be their own dependent')
        dependent_id = f'dependent-{uuid4().hex}'
        with self.store._connect() as connection:
            connection.execute(
                'INSERT INTO patient_dependents VALUES (?, ?, ?, ?, ?)',
                (dependent_id, patient_id, name.strip(), relationship, dependent_patient_id or None),
            )
        return dependent_id

    def list_dependents(self, patient_id):
        with self.store._connect() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM patient_dependents WHERE patient_id = ? ORDER BY name, dependent_id',
                (patient_id,),
            )]

    def set_permission(self, subject_patient_id, requester_patient_id, permission, enabled):
        """Trusted boundary: subject ID must come from the authenticated session.

        Only the record owner grants access here; verified guardian workflows
        must be implemented separately before enabling grants for minors.
        """
        if permission not in PERMISSIONS:
            raise ValueError('Invalid permission')
        with self.store._connect() as connection:
            if enabled:
                connection.execute(
                    'INSERT OR IGNORE INTO dependent_permissions VALUES (?, ?, ?)',
                    (subject_patient_id, requester_patient_id, permission),
                )
            else:
                connection.execute(
                    'DELETE FROM dependent_permissions WHERE subject_patient_id = ? '
                    'AND requester_patient_id = ? AND permission = ?',
                    (subject_patient_id, requester_patient_id, permission),
                )

    def require_access(self, requester_patient_id, subject_patient_id, permission):
        if permission not in PERMISSIONS:
            raise ValueError('Invalid permission')
        if requester_patient_id == subject_patient_id:
            return
        with self.store._connect() as connection:
            allowed = connection.execute(
                'SELECT 1 FROM dependent_permissions WHERE subject_patient_id = ? '
                'AND requester_patient_id = ? AND permission = ?',
                (subject_patient_id, requester_patient_id, permission),
            ).fetchone()
        if not allowed:
            raise PermissionError('This patient has not granted the required access.')

    def resolve(self, patient_id, relationship=None, dependent_id=None):
        matches = [row for row in self.list_dependents(patient_id)
                   if (not relationship or row['relationship'] == relationship)
                   and (not dependent_id or row['dependent_id'] == dependent_id)]
        if len(matches) != 1:
            raise ValueError('Add or select the family member in Family & dependents to identify one person.')
        if not matches[0]['dependent_patient_id']:
            raise ValueError('This family member needs a linked patient record before accessing records or booking.')
        return matches[0]['dependent_patient_id']
