"""Business operations for planned healthcare assistant goals."""

import sqlite3
from datetime import date, time
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.config import SQLITE_DB_PATH, SCHEDULE_API_URL, SCHEDULE_API_TOKEN
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.agents.planner import Goal
from src.repositories.agent_event_repository import AgentEventRepository
from src.repositories.dependent_repository import DependentRepository


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
        self.dependents = DependentRepository(database_path)
        from src.repositories.medical_record_repository import MedicalRecordRepository
        self.medical_records = MedicalRecordRepository(database_path)
        from src.repositories.patient_history_repository import PatientHistoryRepository
        self.patient_history = PatientHistoryRepository(database_path)
        from src.tools.medical_search import MedicalSearch
        self.medical_search = MedicalSearch()
        from src.repositories.schedule_repository import ScheduleRepository
        self.calendars = ScheduleRepository(database_path)
        self.schedule = self.calendars
        if SCHEDULE_API_URL:
            from src.scheduling.client import ScheduleAPIClient
            self.schedule = ScheduleAPIClient(SCHEDULE_API_URL, SCHEDULE_API_TOKEN)

    @property
    def request_traces(self):
        from src.repositories.request_trace_repository import RequestTraceRepository
        return RequestTraceRepository(self.patient_history)

    @property
    def conversations(self):
        from src.repositories.conversation_repository import ConversationRepository
        if not hasattr(self, '_conversations'):
            self._conversations = ConversationRepository(self.patient_history)
        return self._conversations

    @property
    def patient_summaries(self):
        from src.vector_store.patient_summaries import PatientSummaryStore
        if not hasattr(self, '_patient_summaries'):
            self._patient_summaries = PatientSummaryStore(self.patient_history)
        return self._patient_summaries

    def list_staff_patients(self, staff_type, staff_id):
        if staff_type not in ('doctor', 'attendant') or not staff_id:
            raise PermissionError('An authenticated staff member is required.')
        with self._connect() as connection:
            if staff_type == 'doctor':
                if not connection.execute(
                    "SELECT 1 FROM login_details WHERE doctor_id=? AND user_type='doctor' AND is_active=1",
                    (staff_id,)).fetchone():
                    raise PermissionError('An active doctor account is required.')
                rows = connection.execute(
                    'SELECT patient_id, first_name, last_name, date_of_birth FROM patients '
                    'ORDER BY last_name, first_name, patient_id').fetchall()
            else:
                rows = connection.execute(
                    'SELECT p.patient_id, p.first_name, p.last_name, p.date_of_birth '
                    'FROM patients p JOIN attendant_patients a ON a.patient_id=p.patient_id '
                    'JOIN login_details l ON l.attendant_id=a.attendant_id '
                    "WHERE l.attendant_id=? AND l.user_type='attendant' AND l.is_active=1 "
                    'ORDER BY p.last_name, p.first_name, p.patient_id', (staff_id,)).fetchall()
        return [dict(row) for row in rows]

    def summarize_patient_for_staff(self, *, staff_type, staff_id, patient_id, request):
        bundle = self.patient_history.retrieve_for_staff(
            staff_type=staff_type, staff_id=staff_id, patient_id=patient_id)
        try:
            vector = self.patient_summaries.search_for_staff(
                staff_type=staff_type, staff_id=staff_id, patient_id=patient_id,
                query=request)
        except (ValueError, RuntimeError):
            vector = {'status': 'unavailable', 'matches': []}
        self.validate_staff_summary_snapshot(staff_type, staff_id, bundle)
        from src.llm.planning_client import PlanningClient
        answer = PlanningClient().summarize_staff_patient_context(bundle, vector['matches'], request)
        self.validate_staff_summary_snapshot(staff_type, staff_id, bundle)
        return {'answer': answer, 'history_snapshot': bundle, 'patient_summary': vector}

    def validate_staff_summary_snapshot(self, staff_type, staff_id, snapshot):
        """Recheck access and source freshness before generation or display."""
        current = self.patient_history.retrieve_for_staff(
            staff_type=staff_type, staff_id=staff_id, patient_id=snapshot['subject_patient_id'])
        if current['source_fingerprint'] != snapshot['source_fingerprint']:
            raise ValueError('Patient records changed. Generate a new patient health summary.')

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def execute(
        self,
        goal: Goal,
        *,
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        dependent_id: Optional[str] = None,
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
            subject_id = patient_id
            if goal.relationship or dependent_id:
                if not patient_id:
                    raise ValueError("Sign in as a patient to enquire about family members.")
                subject_id = self.dependents.resolve(patient_id, goal.relationship, dependent_id)
                self.dependents.require_access(patient_id, subject_id,
                    "book_appointment" if goal.name == "appointment" else "view_medical")
            if goal.name == "medical_question":
                result = self.answer_patient_question(goal.input_text, requester_patient_id=patient_id, patient_id=subject_id)
            elif goal.name == "appointment":
                result = {
                    "goal": goal.name,
                    "answer": "Use the appointment view to schedule for the selected patient.",
                    "source_documents": [],
                    "subject_patient_id": subject_id,
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

    def answer_patient_question(self, question, *, requester_patient_id=None, patient_id=None):
        """Resolve dialogue, retrieve current patient vectors and ground the chat answer."""
        if not requester_patient_id:
            return self.answer_question(question)
        patient_id = patient_id or requester_patient_id
        def check_access():
            self.patient_history.require_access(requester_patient_id, patient_id)
        check_access()
        history = self.conversations.retrieve(requester_patient_id, patient_id, question)
        standalone = question
        if history:
            from src.llm.conversation_context import resolve_question
            resolved = resolve_question(question, history, allow_patient_records=True)
            check_access()
            if resolved['clarification']:
                return {'question': question, 'answer': resolved['clarification'],
                        'source_documents': [], 'needs_clarification': True,
                        'context_subject_patient_id': patient_id}
            standalone = resolved['question']
        try:
            summary = self.patient_summaries.search(
                requester_patient_id=requester_patient_id, patient_id=patient_id,
                query=standalone)
        except PermissionError:
            raise
        except (ValueError, RuntimeError):
            summary = {'status': 'unavailable', 'matches': []}
        check_access()
        snapshot = self.patient_history.retrieve(
            requester_patient_id=requester_patient_id, patient_id=patient_id)
        if summary['status'] == 'ready' and summary['snapshot']['source_fingerprint'] != snapshot['source_fingerprint']:
            summary = {'status': 'stale', 'matches': []}
        has_summary = summary['status'] == 'ready' and bool(summary['matches'])
        has_records = any(snapshot.get(key) for key in ('records', 'prescriptions', 'alerts'))
        context = {'snapshot': snapshot, 'matches': summary['matches'] if has_summary else []} if has_summary or has_records else None
        result = (self.answer_question(standalone, patient_context=context, check_access=check_access)
                  if context else self.answer_question(standalone))
        check_access()
        if context:
            current = self.patient_history.retrieve(
                requester_patient_id=requester_patient_id, patient_id=patient_id)
            if current['source_fingerprint'] != context['snapshot']['source_fingerprint']:
                raise ValueError('Medical records changed while answering. Rebuild the patient summary and retry.')
        result['question'] = question
        if standalone != question:
            result['standalone_question'] = standalone
        result['patient_summary_status'] = summary['status']
        result['history_snapshot'] = snapshot
        if not has_summary:
            result['answer'] += ('\n\nPatient-summary context was not used (' + summary['status'] +
                                 '). Use Rebuild patient summary to refresh it.')
        # Even with no dialogue, a family-scoped query may contain personal information.
        from src.agents.memory_trace import memory_trace
        result['memory_trace'] = memory_trace('medical_question_context',history)
        result['context_subject_patient_id'] = patient_id
        return result

    def answer_question(self, question: str, *, chat_history=None, check_access=None,
                        patient_context=None) -> Dict[str, Any]:
        """Answer with local RAG, or live medical evidence when no index exists."""
        from src.chains.rag_chain import RAGChain
        from src.vector_store.faiss_store import FAISSStore

        try:
            vectorstore = FAISSStore().load_store()
        except ValueError as error:
            if not str(error).startswith('No reference index is available.'):
                raise
            if not patient_context:
                return self._answer_without_reference_index(question, chat_history, check_access)
            vectorstore = None
        if check_access:
            check_access()
        search_bundle = None
        search_attempted = False
        if patient_context:
            import re
            if re.search(r'\b(treatment|treatments|manage|management|advice|advise|what helps|what should|plan)\b', question, re.I):
                from src.llm.planning_client import PlanningClient
                search_attempted = True
                try:
                    topic = PlanningClient().medical_search_topic(question, patient_context['snapshot'])
                    if check_access:
                        check_access()
                    search_bundle = self.medical_search.search(topic)
                except (ValueError, RuntimeError):
                    search_bundle = None
                if check_access:
                    check_access()
            patient_context = {**patient_context, 'medical_search': search_bundle}
        chain = (RAGChain(vectorstore, patient_context=patient_context)
                 if patient_context else RAGChain(vectorstore))
        result = (chain.query_with_history(question, chat_history, check_access=check_access)
                  if chat_history else chain.query(question))
        if search_bundle is not None:
            result['medical_search'] = search_bundle
        if search_attempted:
            from src.llm.medical_search_summary import usable_sources
            if not search_bundle or not usable_sources(search_bundle):
                result['answer'] += '\n\nNo usable live medical evidence was obtained for this advice request.'
        return result

    def _answer_without_reference_index(self, question, chat_history=None, check_access=None):
        """Use live evidence or a clearly ungrounded OpenAI fallback without fake citations."""
        if check_access:
            check_access()
        standalone = question
        if chat_history:
            from src.llm.conversation_context import resolve_question
            resolved = resolve_question(question, chat_history)
            if resolved['clarification']:
                return {'question': question, 'answer': resolved['clarification'],
                        'source_documents': [], 'needs_clarification': True}
            standalone = resolved['question']
        if check_access:
            check_access()

        from src.llm.medical_search_summary import usable_sources
        from src.llm.planning_client import PlanningClient

        search_bundle = None
        try:
            search_bundle = self.medical_search.search(standalone)
        except (ValueError, RuntimeError):
            pass
        client = PlanningClient()
        if search_bundle and usable_sources(search_bundle):
            answer = client.summarize_medical_search(search_bundle)
            fallback_mode = 'live_medical_search'
        else:
            answer = client._complete(
                'You provide general educational health information when no local reference '
                'documents or live publication excerpts are available. Do not diagnose, '
                'prescribe, invent citations, or claim current clinical guidance. Explain '
                'uncertainty and advise consulting a qualified clinician for personal advice.',
                f'General medical question: {standalone}')
            fallback_mode = 'openai_general'
        if check_access:
            check_access()
        result = {'question': question, 'answer': answer, 'source_documents': [],
                  'fallback_mode': fallback_mode}
        if standalone != question:
            result['standalone_question'] = standalone
        if search_bundle:
            result['medical_search'] = search_bundle
        return result

    def authenticate_user(self, username: str, password: str, user_type: str) -> bool:
        """Return whether credentials match an active login record."""
        from src.utils.passwords import verify_password
        if user_type not in ('patient', 'doctor', 'attendant'):
            return False
        with self._connect() as connection:
            record = connection.execute(
                'SELECT password_hash FROM login_details WHERE username=? AND user_type=? AND is_active=1',
                (username.strip().lower(), user_type)).fetchone()
        return bool(record and verify_password(password, record['password_hash'], allow_legacy=user_type != 'attendant'))

    def get_user_profile(self, username: str, user_type: str) -> Optional[dict]:
        """Retrieve the profile linked to an authenticated login."""
        username = username.strip().lower()
        if user_type == 'attendant':
            with self._connect() as connection:
                row = connection.execute("SELECT a.* FROM attendants a JOIN login_details l ON l.attendant_id=a.attendant_id "
                    "WHERE l.username=? AND l.user_type='attendant' AND l.is_active=1", (username,)).fetchone()
            return dict(row) if row else None
        if user_type not in ('patient', 'doctor'):
            return None
        with self._connect() as connection:
            if user_type == "patient":
                row = connection.execute(
                    """
                    SELECT p.patient_id, p.first_name, p.last_name, p.gender,
                           p.date_of_birth, p.age
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
            calculated_age = PatientVO.from_dict(profile).calculate_age()
            if calculated_age is not None:
                profile["age"] = calculated_age
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
        # Legacy manual selector now respects each doctor's working calendar.
        doctors = self.schedule.specialists(speciality)
        with self._connect() as connection:
            return [doctor for doctor in doctors if self.calendars._allowed(
                connection, doctor["doctor_id"], None, appointment_date, slot)]

    def discover_appointments(self, patient_id, *, requester_patient_id=None, **preferences):
        """Discover earliest matching slots through the configured schedule backend."""
        return self.schedule.discover(patient_id=patient_id,
            requester_patient_id=requester_patient_id or patient_id, **preferences)

    def book_appointment(
        self,
        patient_id: str,
        doctor_id: str,
        appointment_date: date,
        slot: str,
        reason: str,
        *,
        request_id: Optional[str] = None,
        requester_patient_id: Optional[str] = None,
        consultation_type: Optional[str] = None,
    ) -> bool:
        """Book an available 30-minute appointment slot."""
        started_at = perf_counter()
        request_id = request_id or f"request-{uuid4().hex}"
        requester = requester_patient_id or patient_id
        status, replayed = 'failed', False
        try:
            self.dependents.require_access(requester, patient_id, "book_appointment")
            result = self.schedule.book(
                requester_patient_id=requester, patient_id=patient_id, doctor_id=doctor_id,
                day=appointment_date.isoformat(), slot=slot, reason=reason.strip(),
                consultation_type=consultation_type, idempotency_key=request_id)
            booked = result["status"] == "booked"
            replayed = bool(result.get('replayed',False))
            status = 'success' if booked else 'slot_unavailable'
            return booked
        except PermissionError:
            status = 'denied'
            raise
        except ValueError as error:
            from src.scheduling.client import ScheduleUnavailable
            status = 'outcome_unknown' if isinstance(error, ScheduleUnavailable) else 'invalid_request'
            raise
        finally:
            try:
                self.events.log(request_id=request_id, patient_id=patient_id,
                    goal_id="book_appointment", event_type="appointment_booking_completed",
                    tool_name="appointments.book", status=status,
                    duration_ms=round((perf_counter()-started_at)*1000),
                    details={'requester_patient_id':requester,'replayed':replayed})
            except sqlite3.Error:
                # Telemetry failure must not turn a confirmed booking into a reported failure.
                import logging
                logging.getLogger(__name__).warning('Booking telemetry could not be persisted.')

    def get_patient_appointments(self, patient_id: str, *, requester_patient_id: Optional[str] = None) -> List[dict]:
        """Return a patient's scheduled appointments."""
        self.dependents.require_access(requester_patient_id or patient_id, patient_id, "view_appointments")
        with self._connect() as connection:
            rows = connection.execute(
                """
                    SELECT a.appointment_datetime, a.status, a.reason, a.consultation_type,
                       d.first_name || ' ' || d.last_name AS doctor_name, d.speciality
                FROM appointments AS a JOIN doctors AS d ON d.doctor_id = a.doctor_id
                WHERE a.patient_id = ? ORDER BY a.appointment_datetime
                """,
                (patient_id,),
            ).fetchall()
        self.dependents.require_access(requester_patient_id or patient_id, patient_id, "view_appointments")
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
