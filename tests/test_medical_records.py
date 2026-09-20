import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest

from src.agents.goal_execution import GoalExecution
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.models.doctor_vo import DoctorVO
from src.repositories.medical_record_repository import RecordConflictError
from scripts.manage_attendants import create_attendant, manage_access


@pytest.fixture
def clinic(tmp_path):
    path = str(tmp_path / 'records.db')
    store = SQLiteStore(path)
    store.save_patient(PatientVO(patient_id='patient', first_name='Test'))
    store.save_patient(PatientVO(patient_id='other'))
    store.save_doctor(DoctorVO(doctor_id='doctor'))
    actor = create_attendant(path, 'test.attendant', 'Test', 'Attendant', 'local-test-password', ['patient'])
    service = GoalExecution(path)
    from tests.search_stub import OfflineMedicalSearch
    service.medical_search = OfflineMedicalSearch()
    return service, actor


def save(service, actor, **updates):
    args = dict(attendant_id=actor, patient_id='patient', record_type='diagnosis', condition_name='CKD',
        treatment='Documented treatment', notes='Documented notes', change_reason='Initial entry', request_id=uuid4().hex)
    args.update(updates)
    return service.medical_records.save(**args)


def test_attendant_login_and_assignments(clinic):
    service, actor = clinic
    assert service.authenticate_user(' TEST.ATTENDANT ', 'local-test-password', 'attendant')
    assert not service.authenticate_user('test.attendant', 'wrong', 'attendant')
    assert not service.authenticate_user('test.attendant', 'local-test-password', 'doctor')
    assert service.get_user_profile('test.attendant', 'attendant')['attendant_id'] == actor
    with service._connect() as c:
        stored = c.execute("SELECT password_hash FROM login_details WHERE user_type='attendant'").fetchone()[0]
        assert stored.startswith('pbkdf2_sha256$') and stored != 'local-test-password'
    assert [p['patient_id'] for p in service.medical_records.list_patients(actor)] == ['patient']


def test_create_update_revisions_and_private_audit(clinic):
    service, actor = clinic
    original = save(service, actor, doctor_id='doctor', diagnosis_date='2020-01-01')
    changed = save(service, actor, history_id=original['history_id'], expected_version=1,
        treatment='Corrected documented treatment', change_reason='Transcription correction')
    assert changed['version'] == 2
    assert changed['recorded_at'] == original['recorded_at']
    assert changed['updated_by'] == actor
    revisions = service.medical_records.revisions(actor, 'patient', original['history_id'])
    assert len(revisions) == 2
    assert revisions[0]['before']['treatment'] == 'Documented treatment'
    assert revisions[0]['after']['treatment'] == 'Corrected documented treatment'
    events = service.events.list_events(patient_id='patient')
    assert len(events) == 2
    assert all(e['details']['attendant_id'] == actor for e in events)
    assert 'Documented notes' not in json.dumps(events)
    assert 'Transcription correction' not in json.dumps(events)


def test_standalone_notes_do_not_invent_diagnosis(clinic):
    service, actor = clinic
    record = save(service, actor, record_type='note', condition_name='', treatment='', notes='Patient supplied a discharge letter.')
    assert record['record_type'] == 'note' and record['condition_name'] == ''
    assert record['diagnosis_date'] is None


@pytest.mark.parametrize('updates', [
    {'condition_name': ''}, {'change_reason': ''}, {'diagnosis_date': 'not-a-date'},
    {'diagnosis_date': '2999-01-01'}, {'doctor_id': 'unknown'}, {'notes': 'a' * 20001},
    {'record_type': 'note', 'condition_name': '', 'notes': '', 'treatment': ''},
    {'record_type': 'note'},
])
def test_invalid_records_do_not_write(clinic, updates):
    service, actor = clinic
    with pytest.raises(ValueError):
        save(service, actor, **updates)
    assert service.medical_records.list_records(actor, 'patient') == []
    assert service.events.list_events(patient_id='patient') == []


def test_scope_revocation_and_inactive_session(clinic):
    service, actor = clinic
    record = save(service, actor)
    for operation in (
        lambda: save(service, actor, patient_id='other'),
        lambda: service.medical_records.list_records(actor, 'other'),
        lambda: service.medical_records.revisions(actor, 'other', record['history_id']),
        lambda: save(service, 'patient'),
        lambda: save(service, 'doctor'),
    ):
        with pytest.raises(PermissionError):
            operation()
    manage_access(service.database_path, 'test.attendant', 'revoke', ['patient'])
    with pytest.raises(PermissionError):
        save(service, actor, history_id=record['history_id'], expected_version=1)
    manage_access(service.database_path, 'test.attendant', 'assign', ['patient'])
    manage_access(service.database_path, 'test.attendant', 'disable')
    with pytest.raises(PermissionError):
        service.medical_records.list_patients(actor)
    assert not service.authenticate_user('test.attendant', 'local-test-password', 'attendant')
    assert service.get_user_profile('test.attendant', 'attendant') is None


def test_patient_cannot_be_changed_by_editing_id(clinic):
    service, actor = clinic
    record = save(service, actor)
    manage_access(service.database_path, 'test.attendant', 'assign', ['other'])
    with pytest.raises(ValueError, match='not found'):
        save(service, actor, patient_id='other', history_id=record['history_id'], expected_version=1)
    assert service.medical_records.list_records(actor, 'other') == []


def test_duplicate_save_and_concurrent_edits(clinic):
    service, actor = clinic
    first = save(service, actor, request_id='create-once')
    again = save(service, actor, request_id='create-once')
    assert first == again
    assert len(service.medical_records.list_records(actor, 'patient')) == 1
    def edit(text):
        try:
            save(service, actor, history_id=first['history_id'], expected_version=1, notes=text)
            return 'saved'
        except RecordConflictError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(edit, ['edit one', 'edit two'])) == ['conflict', 'saved']
    assert len(service.medical_records.revisions(actor, 'patient', first['history_id'])) == 2
    with pytest.raises(ValueError, match='different content'):
        save(service, actor, request_id='create-once', notes='different content')


def test_revision_failure_rolls_back_record_and_event(clinic):
    service, actor = clinic
    with service._connect() as c:
        c.execute("CREATE TRIGGER reject_revision BEFORE INSERT ON medical_record_revisions BEGIN SELECT RAISE(ABORT,'test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        save(service, actor)
    assert service.medical_records.list_records(actor, 'patient') == []
    assert service.events.list_events(patient_id='patient') == []


def test_saved_note_flows_into_existing_patient_history(clinic):
    from src.agents.planner import Planner
    from src.agents.plan_execution import PlanExecution
    from tests.test_planning import sample
    service, actor = clinic
    save(service, actor, record_type='note', condition_name='', treatment='', notes='Patient reported follow-up completed.')
    payload = sample()
    payload['relationship'] = None
    captured = []
    def summarize(evidence):
        captured.extend(evidence)
        return 'Summary of stored history'
    PlanExecution(service, summarize).run(Planner.validate(payload), patient_id='patient')
    history = next(r for r in captured if r['goal'] == 'history_retrieval')
    assert history['records'][0]['record_type'] == 'note'
    assert history['records'][0]['notes'] == 'Patient reported follow-up completed.'


def test_legacy_schema_migrates_without_losing_logins_or_history(tmp_path):
    path = str(tmp_path / 'legacy.db')
    with sqlite3.connect(path) as c:
        c.executescript('''
            CREATE TABLE login_details (
                login_id TEXT PRIMARY KEY,user_type TEXT CHECK(user_type IN ('patient','doctor')),
                patient_id TEXT,doctor_id TEXT,username TEXT UNIQUE,password_hash TEXT,
                is_active INTEGER DEFAULT 1,last_login_at TEXT,created_at TEXT,updated_at TEXT);
            CREATE TABLE medical_history (
                history_id TEXT PRIMARY KEY,patient_id TEXT,doctor_id TEXT,condition_name TEXT NOT NULL,
                diagnosis_date TEXT,treatment TEXT,notes TEXT,recorded_at TEXT DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO login_details(login_id,user_type,patient_id,username,password_hash)
                VALUES ('old-login','patient','patient','legacy','old-password');
            INSERT INTO medical_history(history_id,patient_id,condition_name,notes)
                VALUES ('old-record','patient','Legacy diagnosis','Original clinical notes');
        ''')
    # Legacy rows refer to a patient present before schema rebuilding.
    with sqlite3.connect(path) as c:
        c.execute('CREATE TABLE patients (patient_id TEXT PRIMARY KEY,first_name TEXT,last_name TEXT,gender TEXT,date_of_birth TEXT,address TEXT,mobile_number TEXT,home_number TEXT,created_at TEXT,updated_at TEXT)')
        c.execute("INSERT INTO patients(patient_id) VALUES ('patient')")
    service = GoalExecution(path)
    assert service.authenticate_user('legacy', 'old-password', 'patient')
    SQLiteStore(path)  # repeat startup migration
    actor = create_attendant(path, 'staff', 'New', 'Staff', 'test-password-long', ['patient'])
    record = service.medical_records.list_records(actor, 'patient')[0]
    assert record['notes'] == 'Original clinical notes' and record['version'] == 1
    updated = save(service, actor, history_id='old-record', expected_version=1)
    assert updated['version'] == 2
    assert service.medical_records.revisions(actor, 'patient', 'old-record')[0]['before']['notes'] == 'Original clinical notes'
    with service._connect() as c:
        assert not c.execute('PRAGMA foreign_key_check').fetchall()


def test_invalid_assignment_rolls_back_new_account(clinic):
    service, actor = clinic
    with pytest.raises(sqlite3.IntegrityError):
        create_attendant(service.database_path, 'bad.staff', 'Bad', 'Staff', 'test-password-long', ['not-a-patient'])
    assert service.get_user_profile('bad.staff', 'attendant') is None
    with service._connect() as c:
        assert c.execute('SELECT COUNT(*) FROM attendants').fetchone()[0] == 1


def test_login_role_cannot_mix_patient_and_attendant(clinic):
    service, actor = clinic
    with service._connect() as c:
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO login_details(login_id,user_type,patient_id,attendant_id,username,password_hash) "
                      "VALUES ('bad','attendant','patient',?,'bad','bad')", (actor,))
