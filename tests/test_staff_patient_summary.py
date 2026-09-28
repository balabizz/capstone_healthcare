import json
from unittest.mock import Mock
import pytest
from tests.test_medical_records import clinic
from tests.test_patient_summary_vectors import setup
from src.llm.planning_client import PlanningClient


def test_staff_summary_receives_database_and_vectors(clinic, monkeypatch):
    service, store, _, bundle = setup(clinic)
    actor = clinic[1]
    service._patient_summaries = store
    complete = Mock(return_value='Summary')
    monkeypatch.setattr(PlanningClient, '_complete', complete)
    result = service.summarize_patient_for_staff(staff_type='attendant', staff_id=actor, patient_id='patient', request='Summarize overall health')
    evidence = json.loads(complete.call_args.args[1])
    assert evidence['sqlite_patient_snapshot']['records']
    assert 'Recorded allergy' in str(evidence['patient_summary_vector_excerpts'])
    assert result['patient_summary']['status'] == 'ready'


def test_staff_summary_rejects_record_changes_during_generation(clinic, monkeypatch):
    service, store, _, bundle = setup(clinic)
    service._patient_summaries = store
    def generate(*args, **kwargs):
        with service._connect() as c:
            c.execute("UPDATE medical_history SET notes='Changed during summary' WHERE patient_id='patient'")
        return 'Outdated summary'
    monkeypatch.setattr(PlanningClient, '_complete', generate)
    with pytest.raises(ValueError, match='changed'):
        service.summarize_patient_for_staff(staff_type='attendant', staff_id=clinic[1], patient_id='patient', request='Summarize overall health')


def test_unknown_doctor_cannot_list_patients(clinic):
    service, _ = clinic
    with pytest.raises(PermissionError):
        service.list_staff_patients('doctor', 'nonexistent-doctor')


def test_doctor_listing_requires_active_account(clinic):
    service, _ = clinic
    with service._connect() as c:
        c.execute("INSERT INTO login_details(login_id,user_type,doctor_id,username,password_hash,is_active) VALUES ('doctor-login','doctor','doctor','doctor-user','unused',0)")
    with pytest.raises(PermissionError):
        service.list_staff_patients('doctor', 'doctor')
    with service._connect() as c:
        c.execute("UPDATE login_details SET is_active=1 WHERE login_id='doctor-login'")
    assert {p['patient_id'] for p in service.list_staff_patients('doctor', 'doctor')} == {'patient', 'other'}


def test_change_during_vector_search_prevents_generation(clinic, monkeypatch):
    service, store, embeddings, _ = setup(clinic)
    service._patient_summaries = store
    def change():
        with service._connect() as c:
            c.execute("UPDATE medical_history SET notes='Changed during retrieval' WHERE patient_id='patient'")
    embeddings.callback = change
    complete = Mock()
    monkeypatch.setattr(PlanningClient, '_complete', complete)
    with pytest.raises(ValueError, match='changed'):
        service.summarize_patient_for_staff(staff_type='attendant', staff_id=clinic[1], patient_id='patient', request='Summarize health')
    complete.assert_not_called()


def test_staff_revocation_during_generation_blocks_answer(clinic, monkeypatch):
    service, store, _, _ = setup(clinic)
    service._patient_summaries = store
    def revoke(*args, **kwargs):
        with service._connect() as c:
            c.execute('DELETE FROM attendant_patients WHERE attendant_id=?', (clinic[1],))
        return 'PRIVATE answer'
    monkeypatch.setattr(PlanningClient, '_complete', revoke)
    with pytest.raises(PermissionError):
        service.summarize_patient_for_staff(staff_type='attendant', staff_id=clinic[1], patient_id='patient', request='Summarize health')


@pytest.mark.parametrize('event', ['unchanged', 'records_changed', 'regeneration_failed', 'access_revoked', 'scope_changed'])
def test_staff_ui_checks_cached_summary_before_display(clinic, monkeypatch, event):
    from contextlib import nullcontext
    from tests.test_medical_records_view import SessionState
    import importlib.util
    import sys
    from pathlib import Path
    service, store, _, bundle = setup(clinic)
    actor = clinic[1]
    ui = Mock()
    cached = {'answer': 'PRIVATE cached summary', 'history_snapshot': bundle,
              'patient_summary': {'status': 'ready', 'matches': []}}
    ui.session_state = SessionState(staff_patient_summary=cached,
                                   staff_summary_scope=('attendant', actor, 'patient'))
    ui.selectbox.return_value = 'patient'
    ui.text_input.return_value = 'Summarize health'
    ui.button.return_value = event == 'regeneration_failed'
    ui.expander.side_effect = lambda *a, **k: nullcontext()
    if event == 'records_changed':
        with service._connect() as c:
            c.execute("UPDATE medical_history SET notes='Changed after summary' WHERE patient_id='patient'")
    elif event == 'access_revoked':
        with service._connect() as c:
            c.execute('DELETE FROM attendant_patients WHERE attendant_id=?', (actor,))
    elif event == 'regeneration_failed':
        service.summarize_patient_for_staff = Mock(side_effect=ValueError('Generation failed'))
    elif event == 'scope_changed':
        ui.session_state.staff_summary_scope = ('doctor', 'old-doctor', 'patient')
    monkeypatch.setitem(sys.modules, 'streamlit', ui)
    spec = importlib.util.spec_from_file_location('staff_summary_ui_test', Path('app/streamlit_app.py'))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.render_staff_patient_summary({'attendant_id': actor}, 'attendant', service)
    if event == 'unchanged':
        ui.write.assert_any_call('PRIVATE cached summary')
    else:
        assert 'staff_patient_summary' not in ui.session_state
        assert 'PRIVATE cached summary' not in str(ui.write.call_args_list)
