from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
import json
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from src.config import SCHEDULE_TIMEZONE
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.models.doctor_vo import DoctorVO
from src.repositories.schedule_repository import ScheduleRepository
from src.scheduling.api import create_app
from src.scheduling.client import ScheduleAPIClient


@pytest.fixture
def schedule(tmp_path):
    path = str(tmp_path / 'calendar.db')
    store = SQLiteStore(path)
    for pid in ('p1', 'p2', 'caller'):
        store.save_patient(PatientVO(patient_id=pid))
    for did, name in [('d1', 'First'), ('d2', 'Second')]:
        store.save_doctor(DoctorVO(doctor_id=did, first_name=name, last_name='Doctor', speciality='Nephrologist'))
    repo = ScheduleRepository(path, now=lambda: datetime(2030, 1, 7, 8, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE)))
    repo.set_hours('d1', 0, '09:00', '12:00')
    repo.set_hours('d1', 0, '14:00', '17:00')
    repo.set_hours('d2', 0, '10:00', '11:00')
    return repo


def discover(repo, **extra):
    return repo.discover(requester_patient_id='p1', patient_id='p1', specialty='Nephrology',
        date_from='2030-01-07', date_to='2030-01-07', **extra)


def booking(repo, patient='p1', doctor='d1', slot='09:00', key='one'):
    return repo.book(requester_patient_id=patient, patient_id=patient, doctor_id=doctor,
                     day='2030-01-07', slot=slot, idempotency_key=key)


def test_earliest_slots_use_alias_calendars_and_named_doctor(schedule):
    slots = discover(schedule)
    assert [(s['time'], s['doctor_id']) for s in slots[:4]] == [
        ('09:00', 'd1'), ('09:30', 'd1'), ('10:00', 'd1'), ('10:00', 'd2')]
    assert slots[0]['timezone'] == SCHEDULE_TIMEZONE
    named = discover(schedule, doctor_name='Dr. Second Doctor')
    assert [(s['time'], s['doctor_id']) for s in named] == [('10:00', 'd2'), ('10:30', 'd2')]
    assert discover(schedule, doctor_name='Unknown Doctor') == []


def test_general_practice_aliases_match_general_physician(schedule):
    with schedule.store._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'General Physician'")
    for specialty in ('General Physician', 'General Practice', 'General Practitioner', 'General Medicine', 'Family Physician', 'GP'):
        assert len(schedule.specialists(specialty)) == 2


def test_gp_query_discovers_available_general_physician_slots(schedule):
    with schedule.store._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'General Physician'")
    slots = schedule.discover(requester_patient_id='p1', patient_id='p1', specialty='GP',
                              date_from='2030-01-07', date_to='2030-01-07')
    assert slots
    assert all(slot['speciality'] == 'General Physician' for slot in slots)


def test_lunch_time_off_and_no_calendar_are_unavailable(schedule):
    assert discover(schedule, time_from='12:00', time_to='14:00') == []
    assert booking(schedule, slot='12:00')['status'] == 'slot_unavailable'
    schedule.set_day_off('d1', '2030-01-07')
    assert all(s['doctor_id'] == 'd2' for s in discover(schedule))
    assert booking(schedule)['status'] == 'slot_unavailable'
    schedule.set_day_off('d1', '2030-01-07', False)
    for hours in schedule.list_hours('d2'):
        schedule.remove_hours('d2', hours['window_id'])
    assert all(s['doctor_id'] == 'd1' for s in discover(schedule))


def test_stale_slot_patient_conflicts_and_idempotency(schedule):
    first = booking(schedule)
    replay = booking(schedule)
    assert first['appointment_id'] == replay['appointment_id']
    assert replay['replayed']
    assert booking(schedule, patient='p2', key='two')['status'] == 'slot_unavailable'
    schedule.set_hours('d2', 0, '09:00', '10:00')
    assert booking(schedule, doctor='d2', key='three')['status'] == 'slot_unavailable'
    with pytest.raises(ValueError, match='different booking'):
        booking(schedule, slot='09:30')
    with schedule.store._connect() as c:
        assert c.execute('SELECT COUNT(*) FROM appointments').fetchone()[0] == 1


def test_two_concurrent_patients_cannot_book_same_slot(schedule):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(booking, schedule, patient=p, key=p) for p in ('p1', 'p2')]
        assert sorted(f.result()['status'] for f in futures) == ['booked', 'slot_unavailable']


def test_overlapping_legacy_appointment_blocks_slot(schedule):
    with schedule.store._connect() as c:
        c.execute("INSERT INTO appointments (appointment_id,patient_id,doctor_id,appointment_datetime) "
                  "VALUES ('old','p2','d1','2030-01-07 09:15')")
    assert booking(schedule, slot='09:00')['status'] == 'slot_unavailable'
    assert booking(schedule, slot='09:30')['status'] == 'slot_unavailable'
    assert booking(schedule, slot='10:00')['status'] == 'booked'


def test_permission_checked_on_discovery_and_confirmation(schedule):
    with pytest.raises(PermissionError):
        schedule.discover(requester_patient_id='caller', patient_id='p1')
    schedule.permissions.set_permission('p1', 'caller', 'book_appointment', True)
    assert schedule.discover(requester_patient_id='caller', patient_id='p1', date_from='2030-01-07', date_to='2030-01-07')
    schedule.permissions.set_permission('p1', 'caller', 'book_appointment', False)
    with pytest.raises(PermissionError):
        schedule.book(requester_patient_id='caller', patient_id='p1', doctor_id='d1', day='2030-01-07', slot='09:00', idempotency_key='revoked')


def test_bad_ranges_and_past_times(schedule):
    with pytest.raises(ValueError):
        discover(schedule, time_from='14:00', time_to='09:00')
    with pytest.raises(ValueError):
        schedule.discover(requester_patient_id='p1', patient_id='p1', date_from='2029-01-01')
    schedule.now = lambda: datetime(2030, 1, 7, 9, 15, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    assert discover(schedule)[0]['time'] == '09:30'


def test_api_contract_authentication_and_unknown_fields(schedule, monkeypatch):
    # Exercise API dispatch and real repository without opening network ports.
    import src.scheduling.api as api
    monkeypatch.setattr(api, 'ScheduleRepository', lambda path: schedule)
    app = create_app(str(schedule.store.database_path), token='x' * 32)
    def call(path, payload, authorized=True):
        data = json.dumps(payload).encode()
        headers = []
        response = app({'REQUEST_METHOD': 'POST', 'PATH_INFO': path, 'CONTENT_LENGTH': str(len(data)),
                        'wsgi.input': BytesIO(data), 'HTTP_AUTHORIZATION': 'Bearer ' + 'x' * 32 if authorized else ''},
                       lambda status, fields: headers.append(status))
        return headers[0], json.loads(b''.join(response))
    assert call('/specialists', {}, False)[0].startswith('403')
    assert call('/specialists', {'specialty': 'Nephrology'})[1][0]['doctor_id'] == 'd1'
    assert call('/slots', {'requester_patient_id': 'p1', 'patient_id': 'p1',
                           'date_from': '2030-01-07', 'date_to': '2030-01-07'})[1][0]['time'] == '09:00'
    payload = dict(requester_patient_id='p1', patient_id='p1', doctor_id='d1', day='2030-01-07', slot='09:00', idempotency_key='api')
    assert call('/bookings', payload)[1]['status'] == 'booked'
    assert call('/bookings', payload)[1]['replayed']
    assert call('/bookings', {**payload, 'sql': 'untrusted'})[0].startswith('400')


def test_http_adapter_preserves_idempotency_key_and_never_falls_back():
    import requests
    session = Mock()
    client = ScheduleAPIClient('http://127.0.0.1:8765', 'token', session=session)
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {'status': 'booked'}
    assert client.book(idempotency_key='stable')['status'] == 'booked'
    assert session.post.call_args.kwargs['json']['idempotency_key'] == 'stable'
    session.post.side_effect = requests.Timeout()
    with pytest.raises(ValueError, match='retry the same request'):
        client.book(idempotency_key='stable')


def test_demo_seed_does_not_overwrite_existing_calendar(schedule):
    from scripts.seed_calendars import seed_calendars
    before = schedule.list_hours('d1')
    assert seed_calendars(str(schedule.store.database_path)) == 0
    assert schedule.list_hours('d1') == before
    schedule.set_day_off('d1', '2030-01-07')
    assert schedule.list_days_off('d1') == ['2030-01-07']


def test_plan_preferences_reach_slot_discovery(schedule):
    from src.agents.planner import Planner
    from src.agents.plan_execution import PlanExecution
    from src.agents.goal_execution import GoalExecution
    from tests.test_planning import sample
    payload = sample()
    payload['relationship'] = None
    payload['steps'][3]['preferences'] = dict(date_from='2030-01-07', date_to='2030-01-07',
        time_from='10:00', time_to='11:00', doctor_name='Second Doctor',
        location=None, consultation_type=None, reason=None)
    execution = GoalExecution(str(schedule.store.database_path))
    execution.schedule = schedule
    from tests.search_stub import OfflineMedicalSearch
    execution.medical_search = OfflineMedicalSearch()
    result = PlanExecution(execution, lambda e: 'Ready to confirm').run(Planner.validate(payload), patient_id='p1')
    slots = result['booking']['slots']
    assert [(s['doctor_id'], s['time']) for s in slots] == [('d2', '10:00'), ('d2', '10:30')]
    assert result['booking']['status'] == 'booked'
    assert result['booking']['slot']['doctor_id'] == 'd2'
    assert execution.get_patient_appointments('p1')[0]['appointment_datetime'] == '2030-01-07 10:00'


def test_inferred_today_business_hours_do_not_block_next_available_slot(schedule):
    with schedule.store._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'Cardiologist' WHERE doctor_id = 'd2'")
    schedule.now = lambda: datetime(2030, 1, 6, 8, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    from src.agents.planner import Planner
    from src.agents.plan_execution import PlanExecution
    from src.agents.goal_execution import GoalExecution
    payload = {'relationship': None, 'clarification': None, 'steps': [
        {'id': 'p', 'name': 'patient_lookup', 'query': 'Resolve patient', 'specialty': None, 'depends_on': []},
        {'id': 'd', 'name': 'specialist_discovery', 'query': 'Find a cardiologist', 'specialty': 'Cardiology', 'depends_on': ['p']},
        {'id': 'a', 'name': 'appointment', 'query': 'Prepare appointment with a cardiologist', 'specialty': 'Cardiology',
         'depends_on': ['p', 'd'], 'preferences': {'date_from': '2030-01-06', 'date_to': '2030-01-06',
                                                   'time_from': '09:00', 'time_to': '17:00', 'doctor_name': None,
                                                   'location': None, 'consultation_type': None, 'reason': None}},
        {'id': 'f', 'name': 'final_summary', 'query': 'Summarize appointment', 'specialty': None, 'depends_on': ['p', 'd', 'a']},
    ]}
    execution = GoalExecution(str(schedule.store.database_path))
    execution.schedule = schedule
    result = PlanExecution(execution, lambda evidence: 'Booked').run(Planner.validate(payload), patient_id='p1')
    assert result['booking']['slot']['date'] == '2030-01-07'
    assert result['booking']['slot']['time'] == '10:00'


def test_unspecified_booking_waits_four_hours_before_booking(schedule):
    with schedule.store._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'Cardiologist' WHERE doctor_id = 'd1'")
    schedule.now = lambda: datetime(2030, 1, 7, 8, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    from src.agents.planner import Planner
    from src.agents.plan_execution import PlanExecution
    from src.agents.goal_execution import GoalExecution
    payload = {'relationship': None, 'clarification': None, 'steps': [
        {'id': 'p', 'name': 'patient_lookup', 'query': 'Resolve patient', 'specialty': None, 'depends_on': []},
        {'id': 'd', 'name': 'specialist_discovery', 'query': 'Find a cardiologist', 'specialty': 'Cardiology', 'depends_on': ['p']},
        {'id': 'a', 'name': 'appointment', 'query': 'Book a cardiologist appointment', 'specialty': 'Cardiology',
         'depends_on': ['p', 'd'], 'preferences': {'date_from': None, 'date_to': None,
                                                   'time_from': None, 'time_to': None, 'doctor_name': None,
                                                   'location': None, 'consultation_type': None, 'reason': None}},
        {'id': 'f', 'name': 'final_summary', 'query': 'Summarize appointment', 'specialty': None, 'depends_on': ['p', 'd', 'a']},
    ]}
    execution = GoalExecution(str(schedule.store.database_path))
    execution.schedule = schedule
    result = PlanExecution(execution, lambda evidence: 'Booked').run(Planner.validate(payload), patient_id='p1')
    assert result['booking']['slot']['date'] == '2030-01-07'
    assert result['booking']['slot']['time'] == '14:00'


def test_location_and_consultation_type_reach_booking(schedule):
    with schedule.store._connect() as connection:
        connection.execute("UPDATE doctors SET speciality = 'Cardiologist', address = '42 Heart Road, Bengaluru' WHERE doctor_id = 'd1'")
    schedule.now = lambda: datetime(2030, 1, 6, 8, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    slots = schedule.discover(requester_patient_id='p1', patient_id='p1', specialty='Cardiology',
                              location='Bengaluru', date_from='2030-01-07', date_to='2030-01-07')
    assert slots and slots[0]['doctor_id'] == 'd1'
    result = schedule.book(requester_patient_id='p1', patient_id='p1', doctor_id='d1',
                           day=slots[0]['date'], slot=slots[0]['time'], reason='Annual review',
                           consultation_type='telehealth', idempotency_key='location-consultation')
    assert result['status'] == 'booked'
    with schedule.store._connect() as connection:
        assert connection.execute('select consultation_type from appointments').fetchone()[0] == 'telehealth'


def test_today_and_tomorrow_are_converted_in_clinic_timezone(schedule):
    schedule.now = lambda: datetime(2030, 1, 6, 8, tzinfo=ZoneInfo(SCHEDULE_TIMEZONE))
    from src.agents.planner import Planner
    from src.agents.plan_execution import PlanExecution
    from src.agents.goal_execution import GoalExecution
    payload = {'relationship': None, 'clarification': None, 'steps': [
        {'id': 'p', 'name': 'patient_lookup', 'query': 'Resolve patient', 'specialty': None, 'depends_on': []},
        {'id': 'd', 'name': 'specialist_discovery', 'query': 'Find a nephrologist', 'specialty': 'Nephrology', 'depends_on': ['p']},
        {'id': 'a', 'name': 'appointment', 'query': 'Book a nephrologist appointment tomorrow', 'specialty': 'Nephrology',
         'depends_on': ['p', 'd'], 'preferences': {'date_from': '2030-01-06', 'date_to': '2030-01-06',
            'time_from': None, 'time_to': None, 'doctor_name': None, 'location': None,
            'consultation_type': None, 'reason': None}},
        {'id': 'f', 'name': 'final_summary', 'query': 'Summarize appointment', 'specialty': None, 'depends_on': ['p', 'd', 'a']},
    ]}
    execution = GoalExecution(str(schedule.store.database_path))
    execution.schedule = schedule
    result = PlanExecution(execution, lambda evidence: 'Booked').run(Planner.validate(payload), patient_id='p1')
    assert result['booking']['slot']['date'] == '2030-01-07'
