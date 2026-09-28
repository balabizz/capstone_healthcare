"""Real FAISS retrieval through the chat boundary and evidence-separated prompts."""
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from tests.test_medical_records import clinic
from tests.test_patient_summary_vectors import setup
from tests.test_task_prompt_chaining import real_chain


def test_chat_retrieves_saved_summary_and_supplies_it_to_prompt(clinic, monkeypatch):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    doc = Document(page_content='General reference facts', metadata={'source': 'reference.pdf'})
    chain, capture, retriever = real_chain(monkeypatch, [doc])
    import src.vector_store.faiss_store as module
    monkeypatch.setattr(module.FAISSStore, 'load_store', lambda self: chain.vectorstore)
    result = service.answer_patient_question('What allergy was recorded?', requester_patient_id='patient')
    assert embeddings.calls[-1] == ['What allergy was recorded?']
    assert 'Recorded allergy' in capture.calls[0][1].content
    assert 'General reference facts' in capture.calls[0][1].content
    assert retriever.queries == ['What allergy was recorded?']
    assert result['source_documents'] == [doc]
    assert result['patient_summary_status'] == 'ready'


def test_patient_only_rag_without_reference_index(clinic, monkeypatch):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    chain, capture, _ = real_chain(monkeypatch, [])
    import src.vector_store.faiss_store as module
    def missing(self):
        raise ValueError('No reference index is available.')
    monkeypatch.setattr(module.FAISSStore, 'load_store', missing)
    service.medical_search = Mock()
    result = service.answer_patient_question('Recorded allergy?', requester_patient_id='patient')
    assert 'Recorded allergy' in capture.calls[0][1].content
    assert 'No reference excerpts retrieved.' in capture.calls[0][1].content
    assert result['source_documents'] == []
    service.medical_search.search.assert_not_called()


def test_main_plan_uses_vector_context_in_medical_question(clinic, monkeypatch):
    from src.agents.plan_execution import PlanExecution
    from tests.test_medical_question_context import medical_plan
    service, store, _, _ = setup(clinic)
    service._patient_summaries = store
    chain, capture, _ = real_chain(monkeypatch, [Document(page_content='Reference evidence')])
    import src.vector_store.faiss_store as module
    monkeypatch.setattr(module.FAISSStore, 'load_store', lambda self: chain.vectorstore)
    result = PlanExecution(service, lambda rows: rows[-1]['message']).run(
        medical_plan(), patient_id='patient')
    assert result['steps'][0]['status'] == 'success'
    assert result['steps'][0]['patient_summary_status'] == 'ready'
    assert 'Recorded allergy' in capture.calls[0][1].content
    assert result['answer'].startswith('Evidence-based answer')


@pytest.mark.parametrize('status', ['missing', 'stale', 'unavailable'])
def test_unusable_summaries_never_enter_answer_prompt(clinic, status):
    service, _ = clinic
    service._patient_summaries = Mock()
    if status == 'unavailable':
        service._patient_summaries.search.side_effect = ValueError('embedding offline')
    else:
        service._patient_summaries.search.return_value = {'status': status, 'matches': []}
    service.answer_question = Mock(return_value={'answer': 'Reference answer', 'source_documents': []})
    result = service.answer_patient_question('Question', requester_patient_id='patient')
    service.answer_question.assert_called_once_with('Question')
    assert result['patient_summary_status'] == status
    assert 'Rebuild patient summary' in result['answer']


def test_resolved_followup_is_used_for_patient_vector_search(clinic, monkeypatch):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    service.conversations.save('patient', 'patient', 'Recorded allergy?', 'Summary overview')
    import src.llm.conversation_context as module
    client = Mock()
    client.plan.return_value = {'question': 'Which allergy was recorded?', 'clarification': None}
    monkeypatch.setattr(module, 'PlanningClient', lambda: client)
    service.answer_question = Mock(return_value={'answer': 'Answer', 'source_documents': []})
    service.answer_patient_question('Which one?', requester_patient_id='patient')
    assert embeddings.calls[-1] == ['Which allergy was recorded?']


def test_selected_patient_never_receives_another_patients_summary(clinic):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    service.answer_question = Mock(return_value={'answer': 'General answer', 'source_documents': []})
    result = service.answer_patient_question('Allergy?', requester_patient_id='other', patient_id='other')
    service.answer_question.assert_called_once()
    context = service.answer_question.call_args.kwargs['patient_context']
    assert context['snapshot']['subject_patient_id'] == 'other'
    assert context['matches'] == []
    assert 'Recorded allergy' not in str(context)
    assert 'Recorded allergy' not in str(result)
    service.answer_question.reset_mock()
    with pytest.raises(PermissionError):
        service.answer_patient_question('Allergy?', requester_patient_id='other', patient_id='patient')
    service.answer_question.assert_not_called()


@pytest.mark.parametrize('change', ['records', 'access'])
def test_changes_during_generation_block_patient_answer(clinic, change):
    service, store, embeddings, bundle = setup(clinic)
    service._patient_summaries = store
    service.dependents.set_permission('patient', 'other', 'view_medical', True)
    def answer(*args, **kwargs):
        if change == 'access':
            service.dependents.set_permission('patient', 'other', 'view_medical', False)
        else:
            with service._connect() as c:
                c.execute("UPDATE medical_history SET notes='changed' WHERE patient_id='patient'")
        return {'answer': 'PRIVATE answer', 'source_documents': []}
    service.answer_question = answer
    with pytest.raises(PermissionError if change == 'access' else ValueError):
        service.answer_patient_question('Allergy?', requester_patient_id='other', patient_id='patient')
