"""Persistent, privacy-conscious logging of agent execution events."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class AgentEventRepository:
    """Write and retrieve planner and tool execution traces from SQLite."""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def log(
        self,
        *,
        request_id: str,
        event_type: str,
        status: str,
        patient_id: Optional[str] = None,
        goal_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        duration_ms: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist one redacted execution event and return its identifier."""
        if not request_id.strip() or not event_type.strip() or not status.strip():
            raise ValueError("request_id, event_type, and status are required")

        event_id = f"event-{uuid4().hex}"
        details_json = json.dumps(details, sort_keys=True) if details else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_events (
                    event_id, request_id, patient_id, goal_id, event_type,
                    tool_name, status, duration_ms, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, request_id, patient_id, goal_id, event_type,
                    tool_name, status, duration_ms, details_json,
                ),
            )
        return event_id

    def list_events(
        self,
        *,
        patient_id: Optional[str] = None,
        request_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Return recent events restricted by patient or request when supplied."""
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")

        filters = []
        values: List[Any] = []
        if patient_id is not None:
            filters.append("patient_id = ?")
            values.append(patient_id)
        if request_id is not None:
            filters.append("request_id = ?")
            values.append(request_id)
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.append(limit)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT event_id, request_id, patient_id, goal_id, event_type,
                       tool_name, status, duration_ms, details_json, created_at
                FROM agent_events
                {where_clause}
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                values,
            ).fetchall()

        events = []
        for row in rows:
            event = dict(row)
            details_json = event.pop("details_json")
            event["details"] = json.loads(details_json) if details_json else None
            events.append(event)
        return events
