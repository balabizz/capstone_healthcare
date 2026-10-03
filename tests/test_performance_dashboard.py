from datetime import date
from unittest.mock import Mock
import pytest
from tests.test_dependents import service
from src.repositories.performance_repository import PerformanceRepository


def event(service,request,status,day='2026-01-10',event_type='appointment_booking_completed',duration=100,details=None,tool='appointments.book'):
    id = service.events.log(request_id=request,patient_id='caller',event_type=event_type,tool_name=tool,status=status,duration_ms=duration,details=details)
    with service._connect() as c:
        c.execute('UPDATE agent_events SET created_at=? WHERE event_id=?',(day+' 12:00:00',id))


def report(service,start=date(2026,1,1),end=date(2026,1,31)):
    return PerformanceRepository(service).summarize(start,end)


def test_booking_rate_excludes_proposals_replays_and_duplicate_confirmations(service):
    event(service,'ok','success')
    event(service,'ok','success',day='2026-01-11')
    event(service,'ok','success',details={'replayed':True})
    event(service,'bad','slot_unavailable',duration=200)
    event(service,'denied','denied',duration=None)
    event(service,'proposal','awaiting_confirmation',event_type='goal_completed',tool='appointments.discover')
    r = report(service)
    assert r['booking']['attempts']==3
    assert r['booking']['confirmed']==1
    assert r['booking']['success_rate']==1/3
    assert r['booking']['replays_excluded']==1
    assert r['pending_proposals']==1
    assert r['booking']['mean_ms']==150 and r['booking']['p95_ms']==200
    assert 'caller' not in str(r) and 'request_id' not in str(r)


def test_filter_after_dedup_prevents_old_success_replay_inflation(service):
    event(service,'old','success',day='2025-12-01')
    event(service,'old','success',day='2026-01-10')
    assert report(service)['booking']['attempts']==0


def test_failed_request_then_success_counts_one_confirmation(service):
    event(service,'retry','slot_unavailable',day='2026-01-01')
    event(service,'retry','success',day='2026-01-02')
    event(service,'retry','invalid_request',day='2026-01-03')
    r = report(service)
    assert r['booking']['attempts']==r['booking']['confirmed']==1
    assert r['booking_daily'][0]['date']=='2026-01-02'


def test_empty_rates_are_unknown_and_bad_ranges_rejected(service):
    assert report(service)['booking']['success_rate'] is None
    with pytest.raises(ValueError):
        report(service,date(2026,2,1),date(2026,1,1))


def test_tool_rates_keep_failures_and_pending_in_denominator(service):
    for i,status in enumerate(['success','failed','blocked','awaiting_confirmation']):
        event(service,str(i),status,event_type='goal_completed',tool='example',duration=(i+1)*10)
    r = report(service)
    assert r['tool_metrics'][0]['success_rate']==.25
    assert r['tool_metrics'][0]['mean_ms']==25
    assert r['tool_metrics'][0]['p95_ms']==40
    assert len(r['tool_outcomes'])==4


@pytest.mark.parametrize('error,status',[(PermissionError(),'denied'),(ValueError(),'invalid_request'),(RuntimeError(),'failed')])
def test_booking_exceptions_are_logged_without_changing_exception(service,error,status):
    service.schedule.book = Mock(side_effect=error)
    with pytest.raises(type(error)):
        service.book_appointment('caller','doctor',date(2030,1,1),'09:00','private reason',request_id='attempt')
    rows = service.events.list_events(request_id='attempt')
    assert len(rows)==1 and rows[0]['status']==status
    assert rows[0]['event_type']=='appointment_booking_completed'
    assert 'private reason' not in str(rows)


def test_booking_without_request_id_and_replay_are_instrumented(service):
    service.schedule.book = Mock(return_value={'status':'booked','replayed':True})
    assert service.book_appointment('caller','doctor',date(2030,1,1),'09:00','')
    assert service.events.list_events()[0]['details']['replayed'] is True


def test_charts_render_and_patient_role_is_excluded(service,monkeypatch):
    import sys,importlib.util,json
    from pathlib import Path
    from types import SimpleNamespace
    from contextlib import nullcontext
    charts,metrics = [],[]
    ui=SimpleNamespace(expander=lambda *a,**k:nullcontext(),date_input=lambda label,**kw:date(2026,1,1) if 'start' in label else date(2026,1,31),
        metric=lambda *args:metrics.append(args),caption=lambda *a:None,write=lambda *a:None,
        info=lambda *a:None,warning=lambda *a:None,vega_lite_chart=lambda **kwargs:charts.append(kwargs['spec']),
        dataframe=lambda *a,**k:None,download_button=lambda *a,**k:None)
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec=importlib.util.spec_from_file_location('performance_view_test',Path('app/performance_view.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render_performance_dashboard('patient',service)
    assert not metrics and not charts
    event(service,'ok','success')
    event(service,'tool','success',event_type='goal_completed',tool='medical_rag.query')
    module.render_performance_dashboard('doctor',service)
    assert ('Booking success rate','100.0%') in metrics
    assert len(charts)==6
    module.render_quality_charts(json.loads(Path('reports/evaluation/latest.json').read_text()))
    assert len(charts)==10


def test_remote_timeout_is_unknown_not_invalid_request(service):
    from src.scheduling.client import ScheduleUnavailable
    service.schedule.book = Mock(side_effect=ScheduleUnavailable('Connection lost'))
    with pytest.raises(ScheduleUnavailable):
        service.book_appointment('caller','doctor',date(2030,1,1),'09:00','',request_id='uncertain')
    assert service.events.list_events(request_id='uncertain')[0]['status']=='outcome_unknown'


def test_telemetry_failure_does_not_report_successful_booking_as_failed(service):
    import sqlite3
    service.schedule.book = Mock(return_value={'status':'booked','replayed':False})
    service.events.log = Mock(side_effect=sqlite3.OperationalError('locked'))
    assert service.book_appointment('caller','doctor',date(2030,1,1),'09:00','')


def test_successful_remote_replay_resolves_previously_unknown_outcome(service):
    event(service,'remote','outcome_unknown',day='2026-01-01')
    event(service,'remote','success',day='2026-01-02',details={'replayed':True})
    r = report(service)
    assert r['booking']['attempts']==r['booking']['confirmed']==1
    assert r['booking']['replays_excluded']==0
    assert r['booking_daily'][0]['date']=='2026-01-02'
