"""Current records and retrieved evidence must reach the same medical answer."""
from unittest.mock import Mock
import json
import pytest
from tests.test_medical_records import clinic
from tests.test_patient_summary_vectors import setup
from tests.test_task_prompt_chaining import real_chain
from src.llm.planning_client import PlanningClient

QUESTION = 'I am suffering from frequent urination, book a GP appointment, suggest a treatment plan'

@pytest.mark.parametrize('vector_status', ['ready', 'missing', 'stale'])
def test_diabetes_reaches_answer_even_when_vector_does_not_mention_it(clinic, monkeypatch, vector_status):
    service, store, _, _ = setup(clinic)
    with service._connect() as c:
        c.execute("UPDATE medical_history SET condition_name='Diabetes' WHERE patient_id='patient'")
    snapshot = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    store.index(requester_patient_id='patient', bundle=snapshot, summary='Unrelated summary about a previous visit')
    if vector_status != 'ready':
        store.search = Mock(return_value={'status': vector_status, 'matches': []})
    service._patient_summaries = store
    chain, capture, _ = real_chain(monkeypatch, [])
    import src.vector_store.faiss_store as module
    monkeypatch.setattr(module.FAISSStore, 'load_store', lambda self: chain.vectorstore)
    monkeypatch.setattr(PlanningClient, 'medical_search_topic', lambda *a: 'frequent urination diabetes management')
    source = {'id': 'who:test', 'excerpt': 'Relevant publication excerpt', 'evidence_type': 'overview',
              'url': 'https://www.who.int/example', 'truncated': False}
    service.medical_search = Mock(search=Mock(return_value={'status': 'success', 'sources': [source]}))
    result = service.answer_patient_question(QUESTION, requester_patient_id='patient')
    human = capture.calls[0][1].content
    assert 'Diabetes' in human
    assert 'prescriptions' in human and 'alerts' in human and 'coverage' in human
    assert 'Relevant publication excerpt' in human
    assert QUESTION in human
    service.medical_search.search.assert_called_once_with('frequent urination diabetes management')
    assert result['patient_summary_status'] == vector_status
    assert 'authoritative' in capture.calls[0][0].content


def test_live_evidence_failure_still_uses_current_records(clinic, monkeypatch):
    service, store, _, _ = setup(clinic)
    service._patient_summaries = store
    chain, capture, _ = real_chain(monkeypatch, [])
    import src.vector_store.faiss_store as module
    def missing(self):
        raise ValueError('No reference index is available.')
    monkeypatch.setattr(module.FAISSStore, 'load_store', missing)
    monkeypatch.setattr(PlanningClient, 'medical_search_topic', Mock(side_effect=ValueError('unavailable')))
    result = service.answer_patient_question(QUESTION, requester_patient_id='patient')
    assert 'Current authorized database snapshot' in capture.calls[0][1].content
    assert 'No usable live medical evidence' in result['answer']
    assert 'medical_search' not in result


def test_search_topic_does_not_send_records_to_search_provider():
    client = PlanningClient()
    client.plan = Mock(return_value={'topic': 'frequent urination diabetes management'})
    topic = client.medical_search_topic(QUESTION, {'subject_patient_id': 'PRIVATE_ID',
        'records': [{'condition_name': 'Diabetes', 'notes': 'PRIVATE_NOTE'}],
        'prescriptions': [{'instructions': 'PRIVATE_PRESCRIPTION'}]})
    assert topic == 'frequent urination diabetes management'
    assert 'PRIVATE' not in client.plan.call_args.args[1]
    client.plan.return_value = {'topic': 'patient_id PRIVATE_ID'}
    with pytest.raises(ValueError):
        client.medical_search_topic(QUESTION, {})
