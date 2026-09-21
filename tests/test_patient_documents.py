"""Patient document persistence and authorization regression tests."""
import gc
from contextlib import closing
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from src.repositories.patient_document_repository import PatientDocumentRepository


class PatientDocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(gc.collect)
        self.repo = PatientDocumentRepository(str(Path(self.temp.name) / 'clinic.db'))
        with closing(self.repo.store._connect()) as c, c:
            c.executemany('INSERT INTO patients (patient_id) VALUES (?)', [('p1',), ('p2',)])
            c.execute("INSERT INTO doctors (doctor_id) VALUES ('d')")
            c.execute("INSERT INTO attendants VALUES ('a','Attendant','One')")
            c.execute("INSERT INTO login_details (login_id,user_type,doctor_id,username,password_hash) VALUES ('ld','doctor','d','doctor','unused')")
            c.execute("INSERT INTO login_details (login_id,user_type,attendant_id,username,password_hash) VALUES ('la','attendant','a','attendant','unused')")
            c.execute("INSERT INTO attendant_patients VALUES ('a','p1')")
        buffer = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(buffer)
        self.pdf = buffer.getvalue()

    def test_patient_ownership_duplicates_and_original_bytes(self):
        first = self.repo.save('doctor', 'p1', 'report.pdf', self.pdf)
        self.assertEqual(first['status'], 'saved')
        self.assertEqual(self.repo.save('doctor', 'p1', 'copy.pdf', self.pdf)['status'], 'duplicate')
        second = self.repo.save('doctor', 'p2', 'report.pdf', self.pdf)
        self.assertNotEqual(first['document_id'], second['document_id'])
        self.assertEqual(self.repo.get_document('attendant', 'p1', first['document_id'])['content'], self.pdf)
        self.assertEqual(len(self.repo.list_documents('doctor', 'p1')), 1)
        with self.assertRaises(ValueError):
            self.repo.get_document('doctor', 'p2', first['document_id'])

    def test_assignment_and_disabled_account_checks(self):
        self.assertEqual([p['patient_id'] for p in self.repo.list_patients('attendant')], ['p1'])
        self.assertEqual(len(self.repo.list_patients('doctor')), 2)
        with self.assertRaises(PermissionError):
            self.repo.save('attendant', 'p2', 'report.pdf', self.pdf)
        with self.assertRaises(PermissionError):
            self.repo.list_documents('attendant', 'p2')
        with closing(self.repo.store._connect()) as c, c:
            c.execute("UPDATE login_details SET is_active=0 WHERE username='attendant'")
        with self.assertRaises(PermissionError):
            self.repo.list_documents('attendant', 'p1')

    def test_invalid_pdf_is_not_saved(self):
        with self.assertRaises(ValueError):
            self.repo.save('doctor', 'p1', 'report.pdf', b'not a PDF')
        self.assertEqual(self.repo.list_documents('doctor', 'p1'), [])


if __name__ == '__main__':
    unittest.main()
