"""Offline contract tests: no credentials, medical data, or paid calls required."""
from unittest.mock import Mock
import pytest

from src.agents.planner import Planner
from src.agents.plan_execution import PlanExecution
from tests.test_dependents import service


def sample():
    def step(i, name, deps, specialty=None):
        return dict(id=i, name=name, query=name, specialty=specialty, depends_on=deps)
    return {'relationship': 'father', 'clarification': None, 'steps': [
        step('p', 'patient_lookup', []), step('h', 'history_retrieval', ['p']),
        step('d', 'specialist_discovery', ['p'], 'Nephrology'),
        step('b', 'appointment', ['p', 'd']), step('s', 'medical_search', []),
        step('f', 'final_summary', ['p', 'h', 'd', 'b', 's'])]}


def test_planner_uses_model_for_semantic_decomposition():
    client = Mock()
    client.plan.return_value = sample()
    plan = Planner(client).plan('Please arrange a renal consultation for Dad and review current treatments.')
    assert [g.name for g in plan.goals] == ['patient_lookup', 'history_retrieval',
        'specialist_discovery', 'appointment', 'medical_search', 'final_summary']
    assert plan.goals[3].depends_on == ('p', 'd')
    assert plan.goals[2].specialty == 'Nephrology'
    assert client.plan.call_count == 1


@pytest.mark.parametrize('change', [
    lambda p: p['steps'][0].update(name='execute_sql'),
    lambda p: p['steps'][1].update(depends_on=['f']),
    lambda p: p['steps'][3].update(depends_on=['p']),
    lambda p: p['steps'][1].update(id='p'),
    lambda p: p['steps'][-1].update(depends_on=['p']),
    lambda p: p.update(patient_id='victim'),
    lambda p: p['steps'][0].update(query=''),
    lambda p: p['steps'][0].update(depends_on=[{}]),
])
def test_invalid_or_unsafe_plans_rejected(change):
    payload = sample()
    change(payload)
    with pytest.raises(ValueError):
        Planner.validate(payload)


def test_clarification_executes_no_tools(service):
    plan = Planner.validate({'relationship': None, 'clarification': 'Which child?', 'steps': []})
    service.answer_question = Mock(side_effect=AssertionError('must not run'))
    assert PlanExecution(service).run(plan)['answer'] == 'Which child?'
    service.answer_question.assert_not_called()


def prepare(service):
    service.dependents.add_dependent('caller', 'Dad', 'father', 'father')
    service.dependents.set_permission('father', 'caller', 'book_appointment', True)
    with service._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'Nephrology' WHERE doctor_id = 'doctor'")


def test_denied_history_does_not_block_independent_booking(service):
    prepare(service)
    summarize = Mock(return_value='A booking proposal is ready.')
    result = PlanExecution(service, summarize).run(Planner.validate(sample()),
        patient_id='caller', request_id='req')
    assert [s['status'] for s in result['steps']] == [
        'success', 'denied', 'success', 'awaiting_confirmation', 'failed', 'success']
    assert result['booking']['subject_patient_id'] == 'father'
    assert 'no current treatment claim can be verified' in result['answer']
    assert service.get_patient_appointments('father') == []
    assert len(service.events.list_events(request_id='req')) == 12


def test_missing_relative_blocks_dependent_tools(service):
    result = PlanExecution(service, lambda e: 'Missing relative').run(
        Planner.validate(sample()), patient_id='stranger')
    assert result['booking'] is None
    assert [s['status'] for s in result['steps']][:4] == ['needs_input', 'blocked', 'blocked', 'blocked']


def test_history_is_subject_scoped_and_used_by_summary(service):
    prepare(service)
    service.dependents.set_permission('father', 'caller', 'view_medical', True)
    with service._connect() as connection:
        connection.executemany('INSERT INTO medical_history(history_id, patient_id, condition_name) VALUES (?, ?, ?)',
            [('h1', 'father', 'CKD'), ('h2', 'stranger', 'Private other condition')])
    summarize = Mock(return_value='Saved history contains CKD.')
    result = PlanExecution(service, summarize).run(Planner.validate(sample()), patient_id='caller')
    evidence = summarize.call_args.args[0]
    assert evidence[1]['records'][0]['condition_name'] == 'CKD'
    assert 'Private other condition' not in str(evidence)
    assert result['steps'][1]['status'] == 'success'


def test_summary_failure_keeps_booking_and_step_results(service):
    prepare(service)
    result = PlanExecution(service, Mock(side_effect=RuntimeError())).run(
        Planner.validate(sample()), patient_id='caller')
    assert result['booking'] is not None
    assert 'Summary generation failed' in result['answer']


def test_general_question_without_patient(service):
    payload = {'relationship': None, 'clarification': None, 'steps': [
        dict(id='q', name='medical_question', query='What is CKD?', specialty=None, depends_on=[]),
        dict(id='f', name='final_summary', query='Summarize', specialty=None, depends_on=['q'])]}
    service.answer_question = Mock(return_value={'answer': 'Reference answer', 'source_documents': ['source']})
    result = PlanExecution(service, lambda e: e[0]['message']).run(Planner.validate(payload))
    assert result['answer'] == 'Reference answer'
    assert result['source_documents'] == ['source']


def test_selected_family_requires_lookup():
    client = Mock()
    client.plan.return_value = {'relationship': None, 'clarification': None, 'steps': [
        dict(id='q', name='medical_question', query='Question', specialty=None, depends_on=[]),
        dict(id='f', name='final_summary', query='Summarize', specialty=None, depends_on=['q'])]}
    with pytest.raises(ValueError, match='resolve'):
        Planner(client).plan('Explain this condition', selected_family=True)


def test_transport_sends_strict_schema_and_handles_refusal():
    from src.llm.planning_client import PlanningClient
    from src.agents.planner import PLAN_SCHEMA
    import json
    session = Mock()
    response = session.post.return_value
    response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(sample())}}]}
    client = PlanningClient(api_key='test-placeholder', session=session)
    assert client.plan('instructions', 'question', PLAN_SCHEMA) == sample()
    body = session.post.call_args.kwargs['json']
    assert body['response_format']['json_schema']['strict'] is True
    assert body['response_format']['json_schema']['schema'] == PLAN_SCHEMA
    response.json.return_value['choices'][0]['message'] = {'refusal': 'Cannot comply', 'content': None}
    with pytest.raises(ValueError, match='rephrase'):
        client.plan('instructions', 'question', PLAN_SCHEMA)


@pytest.mark.parametrize('failure', ['timeout', 'malformed', 'truncated'])
def test_transport_errors_never_fallback_to_keyword_execution(failure):
    from src.llm.planning_client import PlanningClient
    import requests
    session = Mock()
    response = session.post.return_value
    response.json.return_value = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'not JSON'}}]}
    if failure == 'timeout':
        session.post.side_effect = requests.Timeout()
    elif failure == 'truncated':
        response.json.return_value['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError):
        Planner(PlanningClient(api_key='test-placeholder', session=session)).plan('Book for father')


def test_missing_key_fails_before_network():
    from src.llm.planning_client import PlanningClient
    session = Mock()
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        Planner(PlanningClient(api_key='', session=session)).plan('Medical question')
    session.post.assert_not_called()


def test_invalid_model_plan_gets_one_repair_attempt():
    client = Mock()
    invalid = sample()
    invalid['steps'][0]['query'] = ''
    client.plan.side_effect = [invalid, sample()]
    plan = Planner(client).plan('Book a nephrologist for father and find latest treatments')
    assert plan.relationship == 'father'
    assert client.plan.call_count == 2
    assert 'Repair it' in client.plan.call_args.args[1]


def test_repeated_invalid_plan_stops_after_two_calls():
    client = Mock()
    invalid = sample()
    invalid['steps'][3]['depends_on'] = ['d']
    client.plan.return_value = invalid
    with pytest.raises(ValueError):
        Planner(client).plan('Book for father')
    assert client.plan.call_count == 2
