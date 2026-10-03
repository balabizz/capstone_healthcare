from unittest.mock import Mock
import pytest
from tests.test_medical_records import clinic
from tests.test_patient_summary_vectors import Embeddings
from src.vector_store.patient_summaries import PatientSummaryStore
from src.llm.planning_client import PlanningClient
from src.agents.goal_execution import GoalExecution


@pytest.fixture
def doctor_clinic(clinic, monkeypatch):
    service, _ = clinic
    with service._connect() as c:
        c.execute("INSERT INTO login_details(login_id,user_type,doctor_id,username,password_hash) VALUES ('doctor-login','doctor','doctor','doctor-user','unused')")
    service._patient_summaries = PatientSummaryStore(service.patient_history, Embeddings())
    monkeypatch.setattr(PlanningClient, 'summarize_history', lambda self, bundle:
                        '\n'.join(r['notes'] or '' for r in bundle['records']))
    return service


def save(service, text='First visit note', request='visit-one', patient='patient'):
    return service.save_doctor_note(doctor_id='doctor', patient_id=patient, notes=text, request_id=request)


def test_notes_persist_across_visits_and_update_patient_vectors(doctor_clinic):
    service = doctor_clinic
    assert save(service)['embedding_status'] == 'indexed'
    assert save(service, 'Follow-up visit note', 'visit-two')['embedding_status'] == 'indexed'
    restarted = GoalExecution(service.database_path)
    notes = restarted.doctor_notes.list_notes('doctor','patient')
    assert {r['notes'] for r in notes} == {'First visit note','Follow-up visit note'}
    assert all(r['doctor_id']=='doctor' and r['recorded_at'] for r in notes)
    assert restarted.doctor_notes.list_notes('doctor','other') == []
    reopened = PatientSummaryStore(restarted.patient_history, Embeddings())
    found = reopened.search_for_staff(staff_type='doctor',staff_id='doctor',patient_id='patient',query='visits')
    assert found['status']=='ready'
    assert 'First visit note' in found['matches'][0]['text']
    assert 'Follow-up visit note' in found['matches'][0]['text']
    assert reopened.search(requester_patient_id='patient',patient_id='patient',query='visits')['status']=='ready'


def test_duplicate_save_does_not_duplicate_note(doctor_clinic):
    first = save(doctor_clinic)
    assert save(doctor_clinic)['note']['history_id'] == first['note']['history_id']
    assert len(doctor_clinic.doctor_notes.list_notes('doctor','patient'))==1
    with pytest.raises(ValueError, match='different note'):
        save(doctor_clinic, 'Changed content')


def test_inactive_doctor_cannot_read_or_write(doctor_clinic):
    service = doctor_clinic
    with service._connect() as c:
        c.execute("UPDATE login_details SET is_active=0 WHERE doctor_id='doctor'")
    with pytest.raises(PermissionError): save(service)
    with pytest.raises(PermissionError): service.doctor_notes.list_notes('doctor','patient')


def test_index_failure_keeps_original_note_and_retry_updates_vectors(doctor_clinic, monkeypatch):
    service = doctor_clinic
    original = service.patient_summaries.embeddings.embed
    monkeypatch.setattr(service.patient_summaries.embeddings, 'embed', Mock(side_effect=RuntimeError('offline')))
    assert save(service)['embedding_status']=='failed'
    assert service.doctor_notes.list_notes('doctor','patient')[0]['notes']=='First visit note'
    monkeypatch.setattr(service.patient_summaries.embeddings, 'embed', original)
    service.patient_summaries.rebuild_for_staff(staff_type='doctor',staff_id='doctor',patient_id='patient')
    assert len(service.doctor_notes.list_notes('doctor','patient'))==1


def test_access_revoked_during_embedding_does_not_store_vectors(doctor_clinic):
    service = doctor_clinic
    def revoke():
        with service._connect() as c:
            c.execute("UPDATE login_details SET is_active=0 WHERE doctor_id='doctor'")
    service.patient_summaries.embeddings.callback = revoke
    with pytest.raises(PermissionError): save(service)
    with service._connect() as c:
        assert c.execute('SELECT COUNT(*) FROM patient_summary_vectors').fetchone()[0]==0
        assert c.execute('SELECT COUNT(*) FROM medical_history').fetchone()[0]==1


def test_existing_notes_by_other_doctors_are_visible(doctor_clinic):
    service=doctor_clinic
    with service._connect() as c:
        c.execute("INSERT INTO doctors(doctor_id,first_name,last_name) VALUES ('previous','Earlier','Doctor')")
        c.execute("INSERT INTO medical_history(history_id,patient_id,doctor_id,condition_name,notes) VALUES ('legacy','patient','previous','','Earlier visit')")
    assert service.doctor_notes.list_notes('doctor','patient')[0]['notes']=='Earlier visit'


@pytest.mark.parametrize('text',['', ' '*10, 'x'*10001])
def test_invalid_notes_rejected(doctor_clinic,text):
    with pytest.raises(ValueError): save(doctor_clinic,text)


def test_doctor_notes_view_save_and_patient_switch(doctor_clinic):
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string("""
import streamlit as st
from app.doctor_notes_view import render_doctor_notes
render_doctor_notes({'doctor_id':'doctor'}, st.session_state.test_execution)
""")
    app.session_state.test_execution=doctor_clinic
    app.run(timeout=30)
    app.selectbox[0].set_value('patient').run()
    app.text_area[0].input('Return visit recorded by doctor')
    next(b for b in app.button if b.label=='Save doctor note and update summary').click().run(timeout=30)
    assert not app.exception
    assert app.success
    assert any('Return visit recorded by doctor' in t.value for t in app.text)
    app.selectbox[0].set_value('other').run()
    assert not app.exception
    assert not any('Return visit recorded by doctor' in t.value for t in app.text)
    assert app.text_area[0].value==''
