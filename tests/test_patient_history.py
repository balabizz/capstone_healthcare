import json
from datetime import date
from unittest.mock import Mock
import pytest

from tests.test_medical_records import clinic, save
from src.agents.planner import Planner
from src.agents.plan_execution import PlanExecution
from src.repositories.patient_history_repository import PatientHistoryRepository
from src.llm.history_summary import render_summary, coverage_notice
from src.llm.planning_client import PlanningClient


def history_plan(relationship=None):
    return Planner.validate({'relationship': relationship, 'clarification': None, 'steps': [
        {'id': 'p', 'name': 'patient_lookup', 'query': 'Resolve patient', 'specialty': None, 'depends_on': []},
        {'id': 'h', 'name': 'history_retrieval', 'query': 'Retrieve history, prescriptions, alerts', 'specialty': None, 'depends_on': ['p']},
        {'id': 's', 'name': 'final_summary', 'query': 'Summarize saved evidence', 'specialty': None, 'depends_on': ['p','h']}]})


def populate(service, actor):
    record = save(service, actor, notes='Reported by patient; not independently confirmed.')
    with service._connect() as c:
        c.executemany('INSERT INTO prescriptions(prescription_id,patient_id,doctor_id,medication_name,dosage,frequency,start_date,end_date,instructions) '
            'VALUES (?,?,?,?,?,?,?,?,?)', [
                ('rx-own','patient','doctor','Recorded medication','5 mg','daily','2020-01-01','2020-01-31','Recorded instructions'),
                ('rx-other','other','doctor','PRIVATE OTHER MEDICATION',None,None,None,None,None)])
    alert = service.medical_records.add_alert(attendant_id=actor,patient_id='patient',alert_type='allergy',severity='high',
        description='Recorded allergy; reaction documented in source notes.', request_id='own-alert')
    return record, alert


def valid_payload(bundle):
    result = {k: [] for k in ('diagnoses_and_treatments','clinical_notes','prescriptions','alerts')}
    for r in bundle['records']:
        section = 'clinical_notes' if r['record_type'] == 'note' else 'diagnoses_and_treatments'
        result[section].append({'text': r['notes'] or r['condition_name'], 'sources': ['history:' + r['history_id']]})
    for r in bundle['prescriptions']:
        result['prescriptions'].append({'text': r['medication_name'], 'sources': ['prescription:' + r['prescription_id']]})
    for r in bundle['alerts']:
        result['alerts'].append({'text': r['description'], 'sources': ['alert:' + r['alert_id']]})
    return result


def test_retrieves_all_categories_scoped_to_selected_patient(clinic):
    service, actor = clinic
    record, alert = populate(service, actor)
    bundle = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    assert bundle['records'][0]['history_id'] == record['history_id']
    assert bundle['records'][0]['treatment'] == 'Documented treatment'
    assert bundle['prescriptions'][0]['prescription_id'] == 'rx-own'
    assert bundle['prescriptions'][0]['date_interpretation'] == 'past_recorded_end'
    assert bundle['alerts'][0]['alert_id'] == alert
    assert 'PRIVATE OTHER MEDICATION' not in json.dumps(bundle)
    assert all(info['total'] == 1 for info in bundle['coverage'].values())


def test_dependent_requires_medical_permission_before_llm_receives_records(clinic):
    service, actor = clinic
    populate(service, actor)
    service.dependents.add_dependent('other', 'Father', 'father', 'patient')
    summary = Mock(return_value='No private data available')
    denied = PlanExecution(service, summary).run(history_plan('father'), patient_id='other')
    assert denied['steps'][1]['status'] == 'denied'
    assert 'rx-own' not in json.dumps(summary.call_args.args[0])
    service.dependents.set_permission('patient', 'other', 'view_medical', True)
    allowed = PlanExecution(service, summary).run(history_plan('father'), patient_id='other')
    assert allowed['history_snapshot']['subject_patient_id'] == 'patient'
    assert 'rx-own' in json.dumps(summary.call_args.args[0])


def test_revocation_during_summary_scrubs_result(clinic):
    service, actor = clinic
    populate(service, actor)
    service.dependents.add_dependent('other', 'Father', 'father', 'patient')
    service.dependents.set_permission('patient', 'other', 'view_medical', True)
    def revoke(evidence):
        service.dependents.set_permission('patient', 'other', 'view_medical', False)
        return 'Private generated summary'
    result = PlanExecution(service, revoke).run(history_plan('father'), patient_id='other')
    assert result['history_snapshot'] is None
    assert 'Private generated summary' not in json.dumps(result)
    assert 'rx-own' not in json.dumps(result)
    assert result['steps'][1]['status'] == 'denied'


def test_structured_llm_adapter_validates_sources_and_preserves_operational_outcomes(clinic):
    service, actor = clinic
    populate(service, actor)
    bundle = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    session = Mock()
    session.post.return_value.json.return_value = {'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(valid_payload(bundle))}}]}
    client = PlanningClient(api_key='test-placeholder', session=session)
    result = client.summarize([{'goal':'history_retrieval','history_snapshot':bundle},
                              {'goal':'appointment','status':'awaiting_confirmation','message':'Not booked yet.'}])
    assert 'prescription:rx-own' in result
    assert 'Not booked yet.' in result
    body = session.post.call_args.kwargs['json']
    assert body['response_format']['json_schema']['name'] == 'patient_history_summary'
    assert 'subject_patient_id' not in body['messages'][1]['content']
    assert 'NOT proof of current use' in body['messages'][0]['content']


def test_hallucinated_sources_and_omitted_active_alert_rejected(clinic):
    service, actor = clinic
    populate(service, actor)
    bundle = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    payload = valid_payload(bundle)
    payload['prescriptions'][0]['sources'] = ['prescription:rx-other']
    with pytest.raises(ValueError, match='source'):
        render_summary(payload, bundle)
    payload = valid_payload(bundle)
    payload['alerts'] = []
    with pytest.raises(ValueError):
        render_summary(payload, bundle)


def test_empty_history_explicitly_unknown(clinic):
    service, actor = clinic
    bundle = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    rendered = render_summary(valid_payload(bundle), bundle)
    assert 'does not establish absence' in rendered
    assert 'do not confirm' in rendered
    assert '0 of 0' in coverage_notice(bundle)


def test_prescription_dates_do_not_infer_current_use():
    timing = PatientHistoryRepository.prescription_timing
    today = date(2030,1,1)
    assert timing({'start_date':None, 'end_date':None}, today) == 'incomplete_dates_current_use_unknown'
    assert timing({'start_date':'bad', 'end_date':None}, today) == 'invalid_dates_current_use_unknown'
    assert timing({'start_date':'2031-01-01', 'end_date':None}, today) == 'future_recorded_start'
    assert timing({'start_date':'2030-01-01', 'end_date':'2029-01-01'}, today) == 'invalid_recorded_date_range'
    assert timing({'start_date':'2029-01-01', 'end_date':'2031-01-01'}, today) == 'within_recorded_dates_not_confirmed_taking'


def test_alert_priority_truncation_and_resolution_are_explicit(clinic):
    service, actor = clinic
    record = save(service, actor, notes='Long notes ' * 500)
    for i, severity in enumerate(('low','high','critical')):
        service.medical_records.add_alert(attendant_id=actor,patient_id='patient',alert_type='clinical',
            severity=severity,description='Recorded alert',request_id=f'a{i}')
    repo = PatientHistoryRepository(service.database_path, limit=1, text_limit=100)
    bundle = repo.retrieve(requester_patient_id='patient',patient_id='patient')
    assert bundle['alerts'][0]['severity'] == 'critical'
    assert bundle['coverage']['alerts']['active_omitted'] == 2
    assert bundle['coverage']['records']['truncated_fields']
    assert '2 active alerts' in coverage_notice(bundle)
    service.medical_records.resolve_alert(attendant_id=actor,patient_id='patient',alert_id='alert-a2',reason='Resolved by care team')
    bundle = repo.retrieve(requester_patient_id='patient',patient_id='patient')
    assert bundle['alerts'][0]['severity'] == 'high'
    assert bundle['coverage']['alerts']['active_omitted'] == 1


def test_snapshot_has_bounded_size_and_audit_has_no_clinical_text(clinic):
    service, actor = clinic
    for i in range(12):
        save(service, actor, notes='private clinical text' * 700, treatment='documented treatment' * 700)
    result = PlanExecution(service, lambda e: 'Summary').run(history_plan(), patient_id='patient', request_id='history-request')
    bundle = result['history_snapshot']
    assert len(json.dumps(bundle)) < 100000
    assert bundle['coverage']['records']['omitted'] > 0
    events = service.events.list_events(request_id='history-request')
    assert 'private clinical text' not in json.dumps(events)
    assert 'Incomplete records' in result['answer']


def test_alerts_cannot_be_written_for_unassigned_patient(clinic):
    service, actor = clinic
    with pytest.raises(PermissionError):
        service.medical_records.add_alert(attendant_id=actor,patient_id='other',alert_type='allergy',severity='high',description='Test',request_id='denied')
    populate(service, actor)
    with pytest.raises(PermissionError):
        service.medical_records.resolve_alert(attendant_id=actor,patient_id='other',alert_id='alert-own-alert',reason='Test')


def test_summary_schema_restricts_model_to_exact_typed_source_keys(clinic):
    from src.llm.history_summary import schema_for
    service, actor = clinic
    record, alert_id = populate(service, actor)
    bundle = service.patient_history.retrieve(requester_patient_id='patient',patient_id='patient')
    schema = schema_for(bundle)
    source_enums = {section: spec['items']['properties']['sources']['items']['enum']
                    for section, spec in schema['properties'].items()}
    assert source_enums['prescriptions'] == ['prescription:rx-own']
    assert source_enums['alerts'] == ['alert:' + alert_id]
    rendered = render_summary(valid_payload(bundle), bundle)
    assert 'Recorded active high allergy alert' in rendered
