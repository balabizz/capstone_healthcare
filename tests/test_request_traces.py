from unittest.mock import Mock
import pytest
from tests.test_dependents import service
from tests.test_medical_question_context import medical_plan
from tests.test_planning import sample
from src.agents.conversation_flow import ConversationFlow
from src.agents.plan_execution import PlanExecution
from src.agents.planner import Planner
from src.agents.memory_trace import memory_trace
from src.agents.scenario_testing import run_scenario
from src.repositories.request_trace_repository import RequestTraceRepository


def test_actual_memory_selection_and_execution_trace(service):
    service.conversations.save('caller','caller','A private question','Private answer','turn1')
    planner = Mock()
    planner.plan.return_value = medical_plan()
    flow = ConversationFlow(service,planner)
    plan,subject = flow.plan('Follow-up',requester='caller')
    trace = flow.memory_traces[0]
    assert trace['turns'][0]['turn_id']=='turn1'
    assert trace['turns'][0]['selection_reason']=='recent_continuity'
    assert 'Private answer' not in str(trace)
    service.answer_question = Mock(return_value={'answer':'Reference response','source_documents':[]})
    result = PlanExecution(service,lambda rows:'Done').run(plan,patient_id='caller')
    assert result['memory_traces'][0]['stage']=='medical_question_context'
    assert all('duration_ms' in step and 'depends_on' in step for step in result['steps'])
    service.request_traces.save('caller',subject,'req1',plan.details(),flow.memory_traces+result['memory_traces'],result['steps'])
    reopened = RequestTraceRepository(service.patient_history)
    saved = reopened.list('caller','caller')[0]
    assert saved['plan']['goals'][0]['query']=='What are its complications?'
    assert saved['memory'][0]['turns'][0]['turn_id']=='turn1'
    assert 'Reference response' not in str(saved)


def test_trace_scope_and_deletion_do_not_leak_memory(service):
    service.dependents.set_permission('father','caller','view_medical',True)
    service.conversations.save('caller','father','Private question','Private answer','private-turn')
    turns = service.conversations.retrieve('caller','father')
    service.request_traces.save('caller','father','r',medical_plan().details(),[memory_trace('planner',turns)],[])
    assert service.request_traces.list('caller','caller')==[]
    assert service.request_traces.list('father','father')==[]
    assert service.conversations.get_turns('caller','caller',['private-turn'])==[]
    service.conversations.clear('caller','father')
    assert service.conversations.get_turns('caller','father',['private-turn'])==[]
    assert 'Private answer' not in str(service.request_traces.list('caller','father'))
    service.dependents.set_permission('father','caller','view_medical',False)
    with pytest.raises(PermissionError):
        service.request_traces.list('caller','father')


def test_blocked_subgoal_exposes_dependency_not_hidden_reasoning(service):
    result = PlanExecution(service,lambda rows:'Failed prerequisites').run(Planner.validate(sample()),patient_id='stranger')
    blocked = next(r for r in result['steps'] if r['goal']=='history_retrieval')
    assert blocked['status']=='blocked' and blocked['blocked_by']==['p']
    assert blocked['depends_on']==['p']


def test_planner_discarded_memory_is_labelled(service):
    service.conversations.save('caller','caller','Prior','Reply','t')
    service.dependents.add_dependent('caller','Dad','father','father')
    changed = Planner.validate(sample())
    planner = Mock()
    planner.plan.side_effect = [medical_plan(),changed]
    flow = ConversationFlow(service,planner)
    plan,subject = flow.plan('Ambiguous',requester='caller')
    assert plan.relationship is None and subject=='caller'
    assert flow.memory_traces[0]['status']=='discarded_subject_change'


def test_scenario_isolated_and_no_real_bookings(service):
    planner = Mock()
    planner.plan.return_value = medical_plan()
    before = service.get_patient_appointments('caller')
    result = run_scenario('Remembered follow-up',planner=planner)
    assert result['synthetic_bookings_written']==0
    assert result['missing_expected_goals']==[]
    assert result['memory'][0]['turns'][0]['turn_id']=='synthetic-turn'
    assert service.get_patient_appointments('caller')==before
    assert service.conversations.retrieve('caller','caller')==[]


def test_trace_view_renders_plan_memory_and_scope(service,monkeypatch):
    import sys,importlib.util
    from pathlib import Path
    from contextlib import nullcontext
    from types import SimpleNamespace
    rendered=[]
    ui=SimpleNamespace(expander=lambda *a,**k:nullcontext(),caption=lambda *a:None,
        dataframe=lambda rows,**k:rendered.append(rows),info=rendered.append,warning=rendered.append,
        text=rendered.append,json=rendered.append,write=rendered.append,
        selectbox=lambda *a,**k:0,checkbox=lambda *a,**k:True,button=lambda *a,**k:False)
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec=importlib.util.spec_from_file_location('request_trace_view_test',Path('app/request_trace_view.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service.conversations.save('caller','caller','Actual memory question','Actual memory answer','t')
    turns=service.conversations.retrieve('caller','caller')
    service.request_traces.save('caller','caller','req',medical_plan().details(),[memory_trace('planner',turns)],[])
    module.render_request_traces(service,'caller','caller')
    assert 'Actual memory answer' in str(rendered)
    assert 'depends_on' in str(rendered)


def test_real_streamlit_trace_panel_and_deleted_memory(service):
    from streamlit.testing.v1 import AppTest
    service.conversations.save('caller','caller','Remember this topic','Saved memory excerpt','real-ui-turn')
    turns=service.conversations.retrieve('caller','caller')
    service.request_traces.save('caller','caller','real-ui-request',medical_plan().details(),[memory_trace('planner',turns)],[])
    app=AppTest.from_string('''
import streamlit as st
from src.agents.goal_execution import GoalExecution
from app.request_trace_view import render_request_traces
render_request_traces(GoalExecution(st.session_state.test_database),'caller','caller')
''')
    app.session_state['test_database']=service.database_path
    app.run()
    assert not app.exception
    assert len(app.dataframe)>=2
    app.checkbox[0].check().run()
    assert not app.exception
    assert any('Saved memory excerpt' in x.value for x in app.text)
    service.conversations.clear('caller','caller')
    app.run()
    assert not app.exception
    assert not any('Saved memory excerpt' in x.value for x in app.text)
    assert any('deleted or are no longer available' in x.value for x in app.info)
