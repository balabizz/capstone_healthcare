import pytest
from datetime import date
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.models.doctor_vo import DoctorVO
from src.agents.goal_execution import GoalExecution
from src.agents.planner import Goal


@pytest.fixture
def service(tmp_path):
    path = str(tmp_path / 'family.db')
    store = SQLiteStore(path)
    for pid in ('caller', 'father', 'child1', 'child2', 'stranger'):
        store.save_patient(PatientVO(patient_id=pid))
    store.save_doctor(DoctorVO(doctor_id='doctor'))
    service = GoalExecution(path)
    from tests.search_stub import OfflineMedicalSearch
    service.medical_search = OfflineMedicalSearch()
    for weekday in range(7):
        service.calendars.set_hours("doctor", weekday, "09:00", "12:00")
    return service


def test_father_booking_and_revocation(service):
    repo = service.dependents
    repo.add_dependent('caller', 'Dad', 'father', 'father')
    goal = Goal('appointment', 'Book a nephrologist', 'father')
    with pytest.raises(PermissionError):
        service.execute(goal, patient_id='caller')
    repo.set_permission('father', 'caller', 'book_appointment', True)
    assert service.execute(goal, patient_id='caller')['subject_patient_id'] == 'father'
    assert service.book_appointment('father', 'doctor', date(2030, 1, 1), '09:00', 'Visit', requester_patient_id='caller')
    with pytest.raises(PermissionError):
        service.get_patient_appointments('father', requester_patient_id='caller')
    repo.set_permission('father', 'caller', 'view_appointments', True)
    assert len(service.get_patient_appointments('father', requester_patient_id='caller')) == 1
    assert service.get_patient_appointments('caller') == []
    repo.set_permission('father', 'caller', 'book_appointment', False)
    with pytest.raises(PermissionError):
        service.book_appointment('father', 'doctor', date(2030, 1, 1), '09:30', '', requester_patient_id='caller')


def test_ambiguous_unregistered_and_foreign_selection(service):
    repo = service.dependents
    first = repo.add_dependent('caller', 'First', 'child', 'child1')
    repo.add_dependent('caller', 'Second', 'child', 'child2')
    with pytest.raises(ValueError):
        repo.resolve('caller', 'child')
    assert repo.resolve('caller', 'child', first) == 'child1'
    with pytest.raises(ValueError):
        repo.resolve('stranger', dependent_id=first)
    repo.add_dependent('caller', 'Mother', 'mother')
    with pytest.raises(ValueError, match='linked patient'):
        repo.resolve('caller', 'mother')
    with pytest.raises(ValueError):
        repo.add_dependent('caller', 'Self', 'other', 'caller')


def test_relationships_and_medical_permission(service, monkeypatch):
    repo = service.dependents
    repo.add_dependent('caller', 'Partner', 'spouse', 'father')
    goal = Goal('medical_question', 'Kidney disease question', 'spouse')
    monkeypatch.setattr(service, 'answer_question', lambda question: {'answer': 'General information', 'source_documents': []})
    with pytest.raises(PermissionError):
        service.execute(goal, patient_id='caller')
    repo.set_permission('father', 'caller', 'view_medical', True)
    result = service.execute(goal, patient_id='caller')
    assert result['answer'].startswith('General information')
    assert result['patient_summary_status'] == 'missing'
