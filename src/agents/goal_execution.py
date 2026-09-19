"""Business operations for planned healthcare assistant goals."""

import sqlite3
from datetime import date, time
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.agents.planner import Goal
from src.repositories.agent_event_repository import AgentEventRepository


APPOINTMENT_SLOTS = [
    time(hour=hour, minute=minute).strftime("%H:%M")
    for hour, minute in [
        (9, 0), (9, 30), (10, 0), (10, 30), (11, 0), (11, 30),
        (12, 0), (12, 30), (15, 0), (15, 30), (16, 0), (16, 30),
        (17, 0), (17, 30), (18, 0), (18, 30), (19, 0), (19, 30),
    ]
]


class GoalExecution:
    """Execute planned goals and provide application business operations."""

    def __init__(self, database_path: str = SQLITE_DB_PATH):
        self.database_path = database_path
        SQLiteStore(database_path)
        self.events = AgentEventRepository(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def execute(
        self,
        goal: Goal,
        *,
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute one planned goal and return its result."""
        started_at = perf_counter()
        tool_name = {
            "medical_question": "medical_rag.query",
            "appointment": "appointments.propose_view",
        }.get(goal.name, "unsupported")
        if request_id:
            self.events.log(
                request_id=request_id, patient_id=patient_id, goal_id=goal.name,
                event_type="goal_started", tool_name=tool_name, status="started",
            )
        try:
            if goal.name == "medical_question":
                result = self.answer_question(goal.input_text)
            elif goal.name == "appointment":
                result = {
                    "goal": goal.name,
                    "message": "Use the appointment view to schedule.",
                }
            else:
                raise ValueError(f"Unsupported goal: {goal.name}")
        except Exception as error:
            if request_id:
                self.events.log(
                    request_id=request_id, patient_id=patient_id,
                    goal_id=goal.name, event_type="goal_completed",
                    tool_name=tool_name, status="failed",
                    duration_ms=round((perf_counter() - started_at) * 1000),
                    details={"error_type": type(error).__name__},
                )
            raise

        if request_id:
            self.events.log(
                request_id=request_id, patient_id=patient_id, goal_id=goal.name,
                event_type="goal_completed", tool_name=tool_name,
                status="success",
                duration_ms=round((perf_counter() - started_at) * 1000),
            )
        return result

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a medical question using the FAISS-backed RAG chain."""
        from src.chains.rag_chain import RAGChain
        from src.llm.llm_client import LLMClient
        from src.vector_store.faiss_store import FAISSStore

        LLMClient()
        vectorstore = FAISSStore().load_store()
        result = RAGChain(vectorstore).query(question)
        return result

    def authenticate_user(self, username: str, password: str, user_type: str) -> bool:
        """Return whether credentials match an active login record."""
        with self._connect() as connection:
            record = connection.execute(
                """
                SELECT 1 FROM login_details
                WHERE username = ? AND password_hash = ?
                  AND user_type = ? AND is_active = 1
                """,
                (username.strip().lower(), password, user_type),
            ).fetchone()
        return record is not None

    def get_user_profile(self, username: str, user_type: str) -> Optional[dict]:
        """Retrieve the profile linked to an authenticated login."""
        with self._connect() as connection:
            if user_type == "patient":
                row = connection.execute(
                    """
                    SELECT p.patient_id, p.first_name, p.last_name, p.gender,
                           p.date_of_birth
                    FROM login_details AS l JOIN patients AS p ON p.patient_id = l.patient_id
                    WHERE l.username = ? AND l.user_type = ? AND l.is_active = 1
                    """,
                    (username, user_type),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT d.doctor_id, d.first_name, d.last_name,
                           d.speciality, d.license_number
                    FROM login_details AS l JOIN doctors AS d ON d.doctor_id = l.doctor_id
                    WHERE l.username = ? AND l.user_type = ? AND l.is_active = 1
                    """,
                    (username, user_type),
                ).fetchone()
        if row is None:
            return None
        profile = dict(row)
        if user_type == "patient":
            profile["age"] = PatientVO.from_dict(profile).age()
        return profile

    def get_specialties(self) -> List[str]:
        """Return specialties currently offered by doctors."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT speciality FROM doctors WHERE speciality IS NOT NULL ORDER BY speciality"
            ).fetchall()
        return [row[0] for row in rows]

    def get_available_doctors(self, speciality: Optional[str], appointment_date: date, slot: str) -> List[dict]:
        """Return free doctors for a specialty and appointment slot."""
        appointment_datetime = f"{appointment_date.isoformat()} {slot}"
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.doctor_id, d.first_name, d.last_name, d.speciality
                FROM doctors AS d
                WHERE (? IS NULL OR d.speciality = ?)
                  AND NOT EXISTS (
                    SELECT 1 FROM appointments AS a
                    WHERE a.doctor_id = d.doctor_id
                      AND a.appointment_datetime = ? AND a.status = 'scheduled'
                  )
                ORDER BY d.last_name, d.first_name
                """,
                (speciality, speciality, appointment_datetime),
            ).fetchall()
        return [dict(row) for row in rows]

    def book_appointment(
        self,
        patient_id: str,
        doctor_id: str,
        appointment_date: date,
        slot: str,
        reason: str,
        *,
        request_id: Optional[str] = None,
    ) -> bool:
        """Book an available 30-minute appointment slot."""
        started_at = perf_counter()
        if slot not in APPOINTMENT_SLOTS:
            raise ValueError("Appointment time must be within the available hours")
        appointment_datetime = f"{appointment_date.isoformat()} {slot}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO appointments (appointment_id, patient_id, doctor_id, appointment_datetime, reason)
                SELECT ?, ?, ?, ?, ? WHERE NOT EXISTS (
                    SELECT 1 FROM appointments
                    WHERE doctor_id = ? AND appointment_datetime = ? AND status = 'scheduled'
                )
                """,
                (f"appointment-{uuid4().hex}", patient_id, doctor_id,
                 appointment_datetime, reason.strip() or None, doctor_id, appointment_datetime),
            )
            booked = connection.total_changes == 1
        if request_id:
            self.events.log(
                request_id=request_id,
                patient_id=patient_id,
                goal_id="book_appointment",
                event_type="appointment_booking_completed",
                tool_name="appointments.book",
                status="success" if booked else "slot_unavailable",
                duration_ms=round((perf_counter() - started_at) * 1000),
                details={"doctor_id": doctor_id, "appointment_date": appointment_date.isoformat()},
            )
        return booked

    def get_patient_appointments(self, patient_id: str) -> List[dict]:
        """Return a patient's scheduled appointments."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.appointment_datetime, a.status, a.reason,
                       d.first_name || ' ' || d.last_name AS doctor_name, d.speciality
                FROM appointments AS a JOIN doctors AS d ON d.doctor_id = a.doctor_id
                WHERE a.patient_id = ? ORDER BY a.appointment_datetime
                """,
                (patient_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_doctor_schedule(self, doctor_id: str, schedule_date: date) -> List[dict]:
        """Return a doctor's appointments for one day."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.appointment_datetime, a.status, a.reason,
                       p.first_name || ' ' || p.last_name AS patient_name, p.patient_id
                FROM appointments AS a JOIN patients AS p ON p.patient_id = a.patient_id
                WHERE a.doctor_id = ? AND date(a.appointment_datetime) = ?
                ORDER BY a.appointment_datetime
                """,
                (doctor_id, schedule_date.isoformat()),
            ).fetchall()
        return [dict(row) for row in rows]
