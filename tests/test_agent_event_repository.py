import pytest

from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.repositories.agent_event_repository import AgentEventRepository


def test_logs_and_reads_patient_scoped_agent_event(tmp_path):
    database_path = str(tmp_path / "healthcare.db")
    store = SQLiteStore(database_path)
    store.save_patient(PatientVO(patient_id="p-1"))
    repository = AgentEventRepository(database_path)

    event_id = repository.log(
        request_id="request-1",
        patient_id="p-1",
        goal_id="medical_question",
        event_type="goal_completed",
        tool_name="medical_rag.query",
        status="success",
        duration_ms=12,
        details={"document_count": 3},
    )

    events = repository.list_events(patient_id="p-1")

    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["request_id"] == "request-1"
    assert events[0]["details"] == {"document_count": 3}


def test_event_queries_do_not_cross_patient_scope(tmp_path):
    database_path = str(tmp_path / "healthcare.db")
    store = SQLiteStore(database_path)
    store.save_patient(PatientVO(patient_id="p-1"))
    store.save_patient(PatientVO(patient_id="p-2"))
    repository = AgentEventRepository(database_path)
    repository.log(
        request_id="request-1", patient_id="p-1",
        event_type="plan_created", status="success",
    )
    repository.log(
        request_id="request-2", patient_id="p-2",
        event_type="plan_created", status="success",
    )

    events = repository.list_events(patient_id="p-1")

    assert [event["request_id"] for event in events] == ["request-1"]


def test_rejects_invalid_event_and_query_limits(tmp_path):
    database_path = str(tmp_path / "healthcare.db")
    SQLiteStore(database_path)
    repository = AgentEventRepository(database_path)

    with pytest.raises(ValueError):
        repository.log(request_id="", event_type="plan_created", status="success")
    with pytest.raises(ValueError):
        repository.list_events(limit=0)
