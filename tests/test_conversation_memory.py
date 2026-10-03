from unittest.mock import Mock
import pytest
from tests.test_medical_records import clinic
from src.repositories.conversation_repository import ConversationRepository
from src.agents.conversation_flow import ConversationFlow
from src.agents.planner import Planner, Plan, PLANNING_PROMPT
from src.agents.plan_execution import PlanExecution
from src.llm.planning_client import PlanningClient


def recall_plan():
    return Planner.validate({'relationship': None, 'clarification': None, 'steps': [
        {'id': 'p', 'name': 'patient_lookup', 'query': 'patient', 'specialty': None, 'depends_on': []},
        {'id': 'm', 'name': 'conversation_recall', 'query': 'past discussion', 'specialty': None, 'depends_on': ['p']},
        {'id': 's', 'name': 'final_summary', 'query': 'recall', 'specialty': None, 'depends_on': ['p','m']}]})


def test_restart_isolation_and_idempotency(clinic):
    service, _ = clinic
    service.conversations.save('patient', 'patient', 'Prefer mornings', 'Preference noted', 'turn1')
    service.conversations.save('patient', 'patient', 'Prefer mornings', 'Preference noted', 'turn1')
    reopened = ConversationRepository(service.patient_history)
    assert len(reopened.retrieve('patient','patient')) == 1
    assert reopened.retrieve('other','other') == []
    with pytest.raises(PermissionError):
        reopened.retrieve('other','patient')
    service.dependents.set_permission('patient','other','view_medical',True)
    assert reopened.retrieve('other','patient') == []  # no sharing another speaker's dialogue
    reopened.save('other','patient','Father enquiry','Private caregiver conversation')
    service.dependents.set_permission('patient','other','view_medical',False)
    with pytest.raises(PermissionError):
        reopened.retrieve('other','patient')


def test_relevant_older_context_and_scoped_deletion(clinic):
    service, _ = clinic
    memory = service.conversations
    memory.save('patient','patient','Nephrology preference','Morning visits')
    for i in range(12):
        memory.save('patient','patient',f'Other topic {i}','Answer')
    memory.save('other','other','Other person','Private')
    result = memory.retrieve('patient','patient','nephrology',limit=6)
    assert len(result) == 6
    assert any('Nephrology' in r['query'] for r in result)
    memory.clear('patient','patient')
    assert memory.retrieve('patient','patient') == []
    assert len(memory.retrieve('other','other')) == 1


def test_recall_tool_and_provenance(clinic):
    service, _ = clinic
    service.conversations.save('patient','patient','Prefer mornings','No booking made','original')
    result = PlanExecution(service, PlanningClient().summarize).run(recall_plan(), patient_id='patient')
    assert 'Prefer mornings' in result['answer']
    assert '[turn:original]' in result['answer']
    assert 'not verified' in result['answer']
    assert result['summary_index_status'] is None


def test_context_reaches_planner_and_current_request_wins(clinic):
    service, _ = clinic
    service.conversations.save('patient','patient','Prefer mornings','Noted')
    planner = Mock()
    planner.plan.return_value = recall_plan()
    plan, subject = ConversationFlow(service, planner).plan('Actually use afternoons',requester='patient')
    assert subject == 'patient'
    assert planner.plan.call_args.kwargs['conversation_context'][0]['query'] == 'Prefer mornings'
    assert planner.plan.call_args.args[0] == 'Actually use afternoons'
    assert 'Explicit current requests' in PLANNING_PROMPT


def test_planner_serializes_memory_in_prompt():
    client = Mock()
    client.plan.return_value = {'relationship':None,'clarification':'Which appointment?', 'steps':[]}
    Planner(client).plan('same preference',conversation_context=[{'query':'Prefer mornings'}])
    assert 'Prefer mornings' in client.plan.call_args.args[1]


def test_revocation_during_summary_scrubs_recalled_data(clinic):
    service, _ = clinic
    service.dependents.set_permission('patient','other','view_medical',True)
    service.dependents.add_dependent('other','Father','father','patient')
    service.conversations.save('other','patient','Private text','Private reply')
    def summarize(evidence):
        service.dependents.set_permission('patient','other','view_medical',False)
        return 'Private generated answer'
    result = PlanExecution(service,summarize).run(recall_plan(),patient_id='other',
        dependent_id=service.dependents.list_dependents('other')[0]['dependent_id'])
    assert 'Private' not in str(result)
    assert result['steps'][-1]['status'] == 'denied'


def test_persist_and_reload_flow(clinic):
    service, _ = clinic
    flow = ConversationFlow(service, Mock())
    assert flow.remember('Question',{'answer':'Answer','steps':[]},requester='patient',subject='patient',request_id='r')
    assert ConversationRepository(service.patient_history).retrieve('patient','patient')[0]['answer'] == 'Answer'
    assert not flow.remember('Denied',{'answer':'No','steps':[{'status':'denied'}]},requester='patient',subject='patient',request_id='r2')
