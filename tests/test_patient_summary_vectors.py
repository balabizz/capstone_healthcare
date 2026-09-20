import json
import pytest
from tests.test_medical_records import clinic
from tests.test_patient_history import populate, valid_payload, history_plan
from src.vector_store.patient_summaries import PatientSummaryStore
from src.llm.planning_client import PlanningClient
from src.agents.plan_execution import PlanExecution


class Embeddings:
    model = 'test-embeddings'
    def __init__(self):
        self.calls = []
        self.callback = None
    def embed(self, texts):
        self.calls.append(texts)
        if self.callback:
            self.callback()
        return [[1., float('allergy' in text.lower()), .5] for text in texts]


def setup(clinic):
    service, actor = clinic
    populate(service, actor)
    embeddings = Embeddings()
    store = PatientSummaryStore(service.patient_history, embeddings)
    bundle = service.patient_history.retrieve(requester_patient_id='patient', patient_id='patient')
    store.index(requester_patient_id='patient', bundle=bundle, summary='Recorded allergy [alert:own].')
    return service, store, embeddings, bundle


def test_real_faiss_persistence_replacement_and_patient_scope(clinic):
    service, store, embeddings, bundle = setup(clinic)
    reopened = PatientSummaryStore(service.patient_history, embeddings)
    result = reopened.search(requester_patient_id='patient', patient_id='patient', query='allergy')
    assert result['status'] == 'ready' and len(result['matches']) == 1
    assert 'Recorded allergy' in result['matches'][0]['text']
    assert reopened.search(requester_patient_id='other', patient_id='other', query='allergy')['status'] == 'missing'
    count = len(embeddings.calls)
    with pytest.raises(PermissionError):
        reopened.search(requester_patient_id='other', patient_id='patient', query='allergy')
    assert len(embeddings.calls) == count
    store.index(requester_patient_id='patient', bundle=bundle, summary='Replacement')
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['matches'][0]['text'] == 'Replacement'


@pytest.mark.parametrize('table,assignment', [('medical_history', "notes='Changed'"),
    ('prescriptions', "dosage='Changed'"), ('patient_alerts', "status='resolved'")])
def test_changed_sources_reject_stale_vectors(clinic, table, assignment):
    service, store, embeddings, bundle = setup(clinic)
    with service._connect() as c:
        c.execute(f"UPDATE {table} SET {assignment} WHERE patient_id='patient'")
    count = len(embeddings.calls)
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['status'] == 'stale'
    assert len(embeddings.calls) == count
    with pytest.raises(ValueError, match='changed'):
        store.index(requester_patient_id='patient', bundle=bundle, summary='Old')
    store.rebuild(requester_patient_id='patient', patient_id='patient', summarizer=lambda b: 'Updated')
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['status'] == 'ready'


def test_revoked_access_during_embedding_returns_no_summary(clinic):
    service, store, embeddings, bundle = setup(clinic)
    service.dependents.set_permission('patient', 'other', 'view_medical', True)
    embeddings.callback = lambda: service.dependents.set_permission('patient', 'other', 'view_medical', False)
    with pytest.raises(PermissionError):
        store.search(requester_patient_id='other', patient_id='patient', query='allergy')


def test_automatic_plan_pipeline_indexes_only_clinical_summary(clinic):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    client = PlanningClient()
    client._complete = lambda *args: json.dumps(valid_payload(bundle))
    result = PlanExecution(service, client.summarize).run(history_plan(), patient_id='patient')
    assert result['summary_index_status'] == 'indexed'
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['status'] == 'ready'


def test_embedding_failure_preserves_previous_index(clinic):
    service, store, embeddings, bundle = setup(clinic)
    embeddings.callback = lambda: (_ for _ in ()).throw(ValueError('offline'))
    with pytest.raises(ValueError):
        store.index(requester_patient_id='patient', bundle=bundle, summary='Replacement')
    embeddings.callback = None
    assert 'Recorded allergy' in store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['matches'][0]['text']


def test_source_change_during_search_does_not_return_stale_text(clinic):
    service, store, embeddings, bundle = setup(clinic)
    def change():
        with service._connect() as c:
            c.execute("UPDATE medical_history SET notes='new' WHERE patient_id='patient'")
    embeddings.callback = change
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy') == {'status': 'stale', 'matches': []}


def test_model_change_requires_rebuild_without_embedding(clinic):
    service, store, embeddings, bundle = setup(clinic)
    embeddings.model = 'different-model'
    count = len(embeddings.calls)
    assert store.search(requester_patient_id='patient', patient_id='patient', query='allergy')['status'] == 'stale'
    assert len(embeddings.calls) == count


def test_embedding_adapter_orders_and_validates_response(monkeypatch):
    from unittest.mock import Mock
    import src.vector_store.patient_summaries as module
    monkeypatch.setattr(module, 'OPENAI_API_KEY', 'test-key')
    session = Mock()
    session.post.return_value.json.return_value = {'data': [
        {'index': 1, 'embedding': [0, 1]}, {'index': 0, 'embedding': [1, 0]}]}
    adapter = module.SummaryEmbeddings(session=session)
    assert adapter.embed(['a', 'b']) == [[1, 0], [0, 1]]
    assert session.post.call_args.kwargs['json']['input'] == ['a', 'b']
    with pytest.raises(ValueError):
        adapter.embed(['a'])
