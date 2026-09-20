from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.util
from pathlib import Path
import sys
import pytest
from tests.test_dependents import service
from src.agents.goal_execution import GoalExecution
from src.config import APPOINTMENT_REFRESH_SECONDS


@pytest.fixture
def live_view(monkeypatch,service):
    rendered = {'rows':[], 'messages':[], 'interval':None}
    state = {'appointment_view':True,'authenticated_user':{'username':'caller','user_type':'patient'}}
    def fragment(*,run_every):
        rendered['interval'] = run_every
        return lambda fn:fn
    ui = SimpleNamespace(fragment=fragment,session_state=state,
        button=lambda *a,**k:False,info=rendered['messages'].append,warning=rendered['messages'].append,
        caption=rendered['messages'].append,subheader=rendered['messages'].append,
        dataframe=lambda rows,**kwargs:rendered['rows'].extend(rows))
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec = importlib.util.spec_from_file_location('live_appointments_test',Path('app/live_appointments_view.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    service.get_user_profile = Mock(side_effect=lambda name,role:{'patient_id':name} if role=='patient' else {'doctor_id':name})
    def tick():
        rendered['rows'].clear()
        rendered['messages'].clear()
        module.render_live_appointments(service)
        return rendered
    return tick,state,rendered


def test_another_connection_booking_appears_on_next_tick(service,live_view):
    tick,state,rendered = live_view
    assert tick()['rows']==[]
    other = GoalExecution(service.database_path)
    assert other.book_appointment('caller','doctor',date(2030,1,1),'09:00','New appointment')
    assert len(tick()['rows'])==1
    assert rendered['interval']==APPOINTMENT_REFRESH_SECONDS
    assert any('Last refreshed' in m for m in rendered['messages'])
    # Status updates are also read fresh; no cached table or session copy.
    with other._connect() as c:
        c.execute("UPDATE appointments SET status='cancelled' WHERE patient_id='caller'")
    assert tick()['rows'][0]['status']=='cancelled'


def test_doctor_date_and_role_are_read_on_each_tick(service,live_view):
    tick,state,rendered = live_view
    service.book_appointment('caller','doctor',date(2030,1,1),'09:00','New appointment')
    state['authenticated_user']={'username':'doctor','user_type':'doctor'}
    state['doctor_schedule_date']=date(2030,1,1)
    assert len(tick()['rows'])==1
    state['doctor_schedule_date']=date(2030,1,2)
    assert tick()['rows']==[]
    state['authenticated_user']={'username':'stranger','user_type':'patient'}
    assert tick()['rows']==[]


def test_permission_revocation_and_patient_switch_clear_previous_rows(service,live_view):
    tick,state,rendered = live_view
    dependent = service.dependents.add_dependent('caller','Dad','father','father')
    service.dependents.set_permission('father','caller','view_appointments',True)
    service.book_appointment('father','doctor',date(2030,1,1),'09:00','Father appointment')
    state['selected_dependent']=dependent
    assert len(tick()['rows'])==1
    service.dependents.set_permission('father','caller','view_appointments',False)
    assert tick()['rows']==[]
    assert rendered['messages']
    state['selected_dependent']=None
    assert tick()['rows']==[]


def test_polling_never_replays_booking_or_calls_models(service,live_view):
    tick,state,rendered = live_view
    service.book_appointment=Mock(side_effect=AssertionError('No writes from polling'))
    service.answer_question=Mock(side_effect=AssertionError('No LLM during polling'))
    state['appointment_reason']='Unsubmitted reason'
    state['booking_request_keys']={'existing':'keep'}
    tick();tick()
    assert state['appointment_reason']=='Unsubmitted reason'
    assert state['booking_request_keys']=={'existing':'keep'}
    service.book_appointment.assert_not_called()
    service.answer_question.assert_not_called()


def test_navigation_logout_and_errors_clear_panel(service,live_view):
    tick,state,rendered = live_view
    state['appointment_view']=False
    assert tick()['rows']==[]
    service.get_user_profile.assert_not_called()
    state['appointment_view']=True
    state['authenticated_user']=None
    assert tick()['rows']==[]
    state['authenticated_user']={'username':'caller','user_type':'patient'}
    service.get_patient_appointments=Mock(side_effect=RuntimeError('private database info'))
    assert tick()['rows']==[]
    assert 'private database info' not in str(rendered)
    assert any('Retrying automatically' in m for m in rendered['messages'])


@pytest.mark.parametrize('role,identity',[('patient','caller'),('doctor','doctor')])
def test_real_streamlit_panel_reads_other_connection_updates(service,role,identity):
    from streamlit.testing.v1 import AppTest
    with service._connect() as c:
        field = 'patient_id' if role=='patient' else 'doctor_id'
        c.execute(f'INSERT INTO login_details(login_id,user_type,{field},username,password_hash) VALUES (?,?,?,?,?)',
                  ('test-'+role,role,identity,identity,'test-password'))
    app = AppTest.from_string('''
import streamlit as st
from datetime import date
from src.agents.goal_execution import GoalExecution
from app.live_appointments_view import render_live_appointments
st.session_state.appointment_view = True
st.session_state.doctor_schedule_date = date(2030,1,1)
render_live_appointments(GoalExecution(st.session_state.test_database))
''')
    app.session_state['test_database']=service.database_path
    app.session_state['authenticated_user']={'username':identity,'user_type':role}
    app.run()
    assert not app.exception
    assert len(app.dataframe)==0
    writer = GoalExecution(service.database_path)
    assert writer.book_appointment('caller','doctor',date(2030,1,1),'09:00','Booked in another session')
    app.run()
    assert not app.exception
    assert len(app.dataframe)==1 and len(app.dataframe[0].value)==1
    with writer._connect() as c:
        c.execute("UPDATE appointments SET status='cancelled'")
    app.run()
    assert app.dataframe[0].value.iloc[0]['status']=='cancelled'
    with writer._connect() as c:
        c.execute('UPDATE login_details SET is_active=0 WHERE username=?',(identity,))
    app.run()
    assert not app.exception
    assert len(app.dataframe)==0
    assert any('no longer available' in w.value for w in app.warning)
