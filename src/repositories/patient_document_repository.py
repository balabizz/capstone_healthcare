"""Original patient PDFs with database-enforced ownership and staff access checks."""
from contextlib import closing
import hashlib
from io import BytesIO
from pathlib import PurePosixPath
from uuid import uuid4

from pypdf import PdfReader
from src.database.sqlite_store import SQLiteStore


class PatientDocumentRepository:
    def __init__(self, database_path):
        self.store = SQLiteStore(database_path)

    @staticmethod
    def _save_history_text(connection, patient_id, document_id, filename, reader):
        """Preserve PDF text as attributed notes, without inferring clinical facts."""
        added, empty_pages = 0, []
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or '').strip()
            if not text:
                empty_pages.append(page_number)
                continue
            for offset in range(0, len(text), 3500):
                part = offset // 3500 + 1
                history_id = f'document-{document_id}-page-{page_number}-part-{part}'
                result = connection.execute(
                    "INSERT INTO medical_history "
                    "(history_id,patient_id,condition_name,notes,record_type,updated_at) "
                    "VALUES (?,?,?,?,'note',CURRENT_TIMESTAMP) "
                    "ON CONFLICT(history_id) DO NOTHING",
                    (history_id, patient_id,
                     f'Uploaded PDF: {filename} (page {page_number}, part {part})',
                     text[offset:offset + 3500]),
                )
                added += result.rowcount
        return {'history_notes_added': added, 'pages_without_text': empty_pages}

    def import_history(self, username, patient_id, document_id):
        """Backfill a previously stored PDF using the same staff authorization."""
        with closing(self.store._connect()) as c, c:
            c.execute('BEGIN IMMEDIATE')
            self._authorize(c, username, patient_id)
            row = c.execute(
                'SELECT filename,content FROM patient_documents WHERE patient_id=? AND document_id=?',
                (patient_id, document_id)).fetchone()
            if not row:
                raise ValueError('Document not found for this patient.')
            return self._save_history_text(c, patient_id, document_id, row['filename'],
                                           PdfReader(BytesIO(row['content'])))

    def _authorize(self, connection, username, patient_id=None):
        actor = connection.execute(
            "SELECT * FROM login_details WHERE username=? AND is_active=1 "
            "AND user_type IN ('doctor','attendant')", (username,)).fetchone()
        if not actor:
            raise PermissionError('An active staff account is required.')
        if patient_id is not None:
            if not connection.execute('SELECT 1 FROM patients WHERE patient_id=?', (patient_id,)).fetchone():
                raise ValueError('Patient not found.')
            if actor['user_type'] == 'attendant' and not connection.execute(
                'SELECT 1 FROM attendant_patients WHERE attendant_id=? AND patient_id=?',
                (actor['attendant_id'], patient_id)).fetchone():
                raise PermissionError('You are not assigned to this patient.')
        return actor

    def list_patients(self, username):
        with closing(self.store._connect()) as c, c:
            actor = self._authorize(c, username)
            sql = 'SELECT patient_id, first_name, last_name, date_of_birth FROM patients '
            params = ()
            if actor['user_type'] == 'attendant':
                sql += 'WHERE patient_id IN (SELECT patient_id FROM attendant_patients WHERE attendant_id=?) '
                params = (actor['attendant_id'],)
            return [dict(r) for r in c.execute(sql + 'ORDER BY last_name, first_name', params)]

    def save(self, username, patient_id, filename, data):
        filename = PurePosixPath(filename.replace('\\', '/')).name
        if not filename.lower().endswith('.pdf') or not data or len(data) > 20 * 1024 * 1024:
            raise ValueError('Choose a nonempty PDF of at most 20 MB.')
        if not data.startswith(b'%PDF-'):
            raise ValueError('The file is not a PDF.')
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
                raise ValueError('Use an unencrypted PDF with 1–200 pages.')
        except Exception as error:
            raise ValueError('Use a valid, unencrypted PDF with 1–200 pages.') from error
        digest = hashlib.sha256(data).hexdigest()
        with closing(self.store._connect()) as c, c:
            c.execute('BEGIN IMMEDIATE')
            actor = self._authorize(c, username, patient_id)
            existing = c.execute('SELECT document_id,filename FROM patient_documents WHERE patient_id=? AND sha256=?',
                                 (patient_id, digest)).fetchone()
            if existing:
                extracted = self._save_history_text(c, patient_id, existing['document_id'], existing['filename'], reader)
                return {'document_id': existing['document_id'], 'status': 'duplicate', **extracted}
            document_id = uuid4().hex
            c.execute('INSERT INTO patient_documents '
                      '(document_id,patient_id,filename,sha256,content,uploaded_by) VALUES (?,?,?,?,?,?)',
                      (document_id, patient_id, filename, digest, data, actor['login_id']))
            extracted = self._save_history_text(c, patient_id, document_id, filename, reader)
            return {'document_id': document_id, 'status': 'saved', **extracted}

    def list_documents(self, username, patient_id):
        with closing(self.store._connect()) as c, c:
            self._authorize(c, username, patient_id)
            return [dict(r) for r in c.execute(
                'SELECT document_id,filename,uploaded_at,length(content) AS size_bytes '
                'FROM patient_documents WHERE patient_id=? ORDER BY uploaded_at DESC', (patient_id,))]

    def get_document(self, username, patient_id, document_id):
        with closing(self.store._connect()) as c, c:
            self._authorize(c, username, patient_id)
            row = c.execute('SELECT filename,content FROM patient_documents WHERE patient_id=? AND document_id=?',
                            (patient_id, document_id)).fetchone()
            if not row:
                raise ValueError('Document not found for this patient.')
            return dict(row)
