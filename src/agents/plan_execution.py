"""Sequential execution of validated plans with isolated dependency failures."""

from time import perf_counter
from uuid import uuid4
from datetime import date, datetime, timedelta
import re
from src.agents.planner import Plan, Planner


class PlanExecution:
    def __init__(self, execution, summarizer=None):
        self.execution = execution
        self.summarizer = summarizer

    def run(self, plan: Plan, *, patient_id=None, dependent_id=None, request_id=None):
        # Revalidate at the execution boundary, including injected/test plans.
        plan = Planner.validate({'relationship': plan.relationship, 'clarification': plan.clarification,
            'steps': [{'id': g.goal_id, 'name': g.name, 'query': g.input_text,
                       'specialty': g.specialty, 'depends_on': list(g.depends_on), 'preferences': g.preferences} for g in plan.goals]})
        if plan.clarification:
            return {'answer': plan.clarification, 'steps': [], 'source_documents': [], 'booking': None}
        if dependent_id and plan.goals[0].name != "patient_lookup":
            raise ValueError("The plan must resolve the selected family member first.")
        results, subject_id, sources, booking = {}, None, [], None
        memory_traces = []
        context_subject_id = None
        summary_index_status = None
        history_bundle = None
        search_bundle = None
        for goal in plan.goals:
            start = perf_counter()
            if request_id:
                self.execution.events.log(request_id=request_id, patient_id=patient_id,
                    goal_id=goal.goal_id, event_type='goal_started', tool_name=goal.tool, status='started')
            try:
                failed_dependency = any(results[d]['status'] != 'success' for d in goal.depends_on)
                if goal.name != 'final_summary' and failed_dependency:
                    result = {'status': 'blocked', 'message': 'A required earlier step did not complete.'}
                elif goal.name == 'patient_lookup':
                    if not patient_id:
                        raise ValueError('Sign in as a patient to access patient-specific tasks.')
                    subject_id = (self.execution.dependents.resolve(patient_id, plan.relationship, dependent_id)
                                  if plan.relationship or dependent_id else patient_id)
                    if not self.execution.dependents.store.get_patient(subject_id):
                        raise ValueError('The patient record could not be found.')
                    result = {'status': 'success', 'verified': True,
                              'verified_patient_id': subject_id,
                              'message': 'Patient identity and access were verified.'}
                elif goal.name == 'conversation_recall':
                    turns = self.execution.conversations.retrieve(patient_id, subject_id, goal.input_text)
                    from src.agents.memory_trace import memory_trace
                    memory_traces.append(memory_trace('conversation_recall',turns))
                    result = {'status': 'success', 'message': 'Retrieved historical dialogue, not verified clinical facts.',
                              'conversation_memory': turns}
                elif goal.name == 'history_retrieval':
                    bundle = self.execution.patient_history.retrieve(requester_patient_id=patient_id, patient_id=subject_id)
                    result = {'status': 'success', 'message': 'Retrieved diagnoses, treatments, notes, prescriptions and alerts.',
                              'records': bundle['records'], 'history_snapshot': bundle}
                elif goal.name == 'specialist_discovery':
                    rows = self.execution.schedule.specialists(goal.specialty)
                    result = {'status': 'success' if rows else 'needs_input',
                              'message': 'Matching specialists found.' if rows else (
                                  f"I could not find a doctor for '{goal.specialty or 'that request'}'. "
                                  'Please provide a valid specialty or doctor name, and optionally a location. '
                                  'Examples: "book a cardiologist", "find a GP in Bengaluru", or "book with Dr. Sharma".'
                              ),
                              'specialty': rows[0]['speciality'] if rows and goal.specialty else None,
                              'doctors': rows}
                elif goal.name == 'appointment':
                    self.execution.dependents.require_access(patient_id, subject_id, 'book_appointment')
                    if not any(results[d].get('verified_patient_id') == subject_id for d in goal.depends_on):
                        raise PermissionError('Patient identity must be verified before booking.')
                    specialist = next(results[d] for d in goal.depends_on
                                      if results[d]['goal'] == 'specialist_discovery')
                    preferences = dict(goal.preferences or {})
                    schedule_text = goal.input_text.lower()
                    requested_day = None
                    if re.search(r'\btoday\b', schedule_text):
                        requested_day = self.execution.schedule.now().date()
                    elif re.search(r'\btomorrow\b', schedule_text):
                        requested_day = self.execution.schedule.now().date() + timedelta(days=1)
                    if requested_day:
                        preferences['date_from'] = requested_day.isoformat()
                        preferences['date_to'] = requested_day.isoformat()
                    has_explicit_schedule = bool(re.search(
                        r'\b(?:today|tomorrow|morning|afternoon|evening|monday|tuesday|wednesday|'
                        r'thursday|friday|saturday|sunday|on|at|between|from|after|before)\b|'
                        r'\b\d{1,4}[-/]\d{1,2}(?:[-/]\d{1,4})?\b|\b\d{1,2}:\d{2}\b',
                        schedule_text,
                    ))
                    same_day_default = (
                        preferences.get('date_from') == self.execution.schedule.now().date().isoformat()
                        and preferences.get('date_to') == self.execution.schedule.now().date().isoformat()
                        and (
                            (preferences.get('time_from') is None and preferences.get('time_to') is None)
                            or (preferences.get('time_from') == '09:00' and preferences.get('time_to') == '17:00')
                        )
                    )
                    if not has_explicit_schedule and same_day_default:
                        for key in ('date_from', 'date_to', 'time_from', 'time_to'):
                            preferences[key] = None
                    location = preferences.pop('location', None)
                    consultation_type = preferences.pop('consultation_type', None)
                    reason = preferences.pop('reason', None) or goal.input_text
                    automatic_window = not has_explicit_schedule and not any(preferences.values())
                    if automatic_window:
                        requested_at = self.execution.schedule.now()
                        earliest = requested_at + timedelta(hours=4)
                        first_day = earliest.date()
                        same_day = first_day == requested_at.date()
                        first_preferences = {
                            'date_from': first_day.isoformat(),
                            'date_to': first_day.isoformat(),
                            'time_from': earliest.strftime('%H:%M') if same_day else None,
                            'time_to': None,
                            'doctor_name': preferences.get('doctor_name'),
                        }
                        slots = self.execution.discover_appointments(
                            subject_id, requester_patient_id=patient_id,
                            specialty=specialist['specialty'], location=location, **first_preferences)
                        if not slots:
                            next_day = first_day + timedelta(days=1)
                            slots = self.execution.discover_appointments(
                                subject_id, requester_patient_id=patient_id,
                                specialty=specialist['specialty'],
                                date_from=next_day.isoformat(),
                                date_to=(next_day + timedelta(days=29)).isoformat(),
                                time_from=None, time_to=None, doctor_name=preferences.get('doctor_name'),
                                location=location)
                    else:
                        slots = self.execution.discover_appointments(
                            subject_id, requester_patient_id=patient_id,
                            specialty=specialist['specialty'], location=location, **preferences)
                    booked_slot = None
                    for slot in slots:
                        key_base = request_id or f'appointment-{uuid4().hex}'
                        booking_key = f'{key_base}:{goal.goal_id}:{slot["doctor_id"]}:{slot["date"]}:{slot["time"]}'[:128]
                        if self.execution.book_appointment(
                            subject_id, slot['doctor_id'],
                            date.fromisoformat(slot['date']), slot['time'],
                            reason + (f' [{consultation_type}]' if consultation_type else ''),
                            requester_patient_id=patient_id,
                            consultation_type=consultation_type,
                            request_id=booking_key,
                        ):
                            booked_slot = slot
                            break
                    if booked_slot:
                        booking = {'status': 'booked', 'subject_patient_id': subject_id,
                                   'specialty': specialist['specialty'], 'slot': booked_slot,
                                   'slots': slots}
                    result = {'status': 'success' if booked_slot else 'needs_input',
                              'message': (
                                  f"Appointment booked for {booked_slot['date']} at {booked_slot['time']} "
                                  f"({booked_slot['timezone']}) with Dr. {booked_slot['first_name']} "
                                  f"{booked_slot['last_name']} ({booked_slot['speciality']})."
                                  if booked_slot else
                                  'I could not find an available appointment matching those details. '
                                  'Please provide or adjust the specialty/doctor, location, date or date range, '
                                  'time or time range, consultation type, or reason. '
                                  'Example: "Book the next available cardiologist appointment tomorrow morning '
                                  'in Bengaluru for a follow-up."'
                              ), 'slots': slots, 'appointment': booked_slot}
                elif goal.name == 'medical_search':
                    search_bundle = self.execution.medical_search.search(goal.input_text)
                    result = {'status': search_bundle['status'], 'search_evidence': search_bundle,
                              'message': f"Live PubMed/WHO search returned {len(search_bundle['sources'])} sources."
                              if search_bundle['sources'] else 'No usable live medical evidence was retrieved; no current treatment claim can be verified.'}

                elif goal.name == 'medical_question':
                    if subject_id is None and patient_id:
                        subject_id = patient_id
                    if (plan.relationship or dependent_id) and subject_id is None:
                        raise ValueError('Resolve the family member before answering.')
                    if plan.relationship or dependent_id:
                        self.execution.dependents.require_access(patient_id, subject_id, 'view_medical')
                    question_history = None
                    if patient_id and (subject_id or patient_id):
                        question_history = self.execution.patient_history.retrieve(
                            requester_patient_id=patient_id,
                            patient_id=subject_id or patient_id,
                        )
                    answer = self.execution.answer_patient_question(goal.input_text,
                        requester_patient_id=patient_id, patient_id=subject_id or patient_id)
                    context_subject_id = answer.get('context_subject_patient_id')
                    if answer.get('memory_trace'):
                        memory_traces.append(answer['memory_trace'])
                    sources.extend(answer.get('source_documents', []))
                    if answer.get('medical_search'):
                        search_bundle = answer['medical_search']
                    result = {'status': 'needs_input' if answer.get('needs_clarification') else 'success',
                              'message': answer['answer'], 'context_subject_patient_id': context_subject_id,
                              'search_evidence': answer.get('medical_search')}
                    if question_history is not None:
                        result['history_snapshot'] = question_history
                else:
                    if context_subject_id:
                        self.execution.patient_history.require_access(patient_id, context_subject_id)
                    history_result = next((r for r in results.values() if 'history_snapshot' in r), None)
                    if history_result:
                        self.execution.patient_history.require_access(patient_id, subject_id)
                        history_bundle = history_result['history_snapshot']
                    if any('conversation_memory' in r for r in results.values()):
                        self.execution.patient_history.require_access(patient_id, subject_id)
                    evidence = list(results.values())
                    if self.summarizer is None:
                        from src.llm.planning_client import PlanningClient
                        self.summarizer = PlanningClient().summarize
                    result = {'status': 'success', 'message': self.summarizer(evidence)}
                    if context_subject_id:
                        self.execution.patient_history.require_access(patient_id, context_subject_id)
                    if any('conversation_memory' in r for r in results.values()):
                        self.execution.patient_history.require_access(patient_id, subject_id)
                    if history_result:
                        self.execution.patient_history.require_access(patient_id, subject_id)
                        generated = getattr(getattr(self.summarizer, '__self__', None), 'last_history_summary', None)
                        if generated:
                            try:
                                self.execution.patient_summaries.index(requester_patient_id=patient_id,
                                    bundle=history_bundle, summary=generated)
                                summary_index_status = 'indexed'
                            except PermissionError:
                                raise
                            except Exception:
                                summary_index_status = 'failed'
                                result['message'] += '\n\nSummary vector indexing failed. Use Rebuild patient summary to retry.'
            except (ValueError, PermissionError) as error:
                if isinstance(error, PermissionError) and goal.name == 'final_summary':
                    history_bundle = None
                    sources = []
                    memory_traces = []
                    for prior in results.values():
                        if prior.get('context_subject_patient_id'):
                            prior.update(status='denied', message='Medical conversation access was revoked.')
                        if 'conversation_memory' in prior:
                            prior.pop('conversation_memory')
                            prior.update(status='denied', message='Conversation access was revoked.')
                        if 'history_snapshot' in prior:
                            prior.pop('history_snapshot')
                            prior.pop('records', None)
                            prior.update(status='denied', message='Medical access was revoked before the summary completed.')
                if goal.name == 'appointment' and isinstance(error, PermissionError):
                    message = ('I could not verify booking access for this patient. Sign in as the patient or select '
                               'an authorized linked dependent before booking.')
                elif goal.name == 'appointment' and isinstance(error, ValueError):
                    message = (f'Booking could not be completed: {error} '
                               'Please provide a valid doctor or specialty, location, date/time preference, '
                               'consultation type, and reason so I can find the right slot.')
                else:
                    message = str(error)
                result = {'status': 'needs_input' if isinstance(error, ValueError) else 'denied', 'message': message}
            except Exception:
                result = {'status': 'failed', 'message': 'This step could not be completed. Please try again.'}
            result.update({'id': goal.goal_id, 'goal': goal.name, 'tool': goal.tool,
                           'duration_ms':round((perf_counter()-start)*1000),
                           'depends_on':list(goal.depends_on),
                           'blocked_by':[d for d in goal.depends_on if results[d]['status']!='success']
                               if result['status']=='blocked' else []})
            results[goal.goal_id] = result
            if request_id:
                self.execution.events.log(request_id=request_id, patient_id=patient_id,
                    goal_id=goal.goal_id, event_type='goal_completed', tool_name=goal.tool,
                    status=result['status'], duration_ms=round((perf_counter() - start) * 1000),
                    details={'depends_on': list(goal.depends_on),
                             'subject_patient_id': subject_id if goal.name in ('history_retrieval', 'final_summary') else None})
        final = results[plan.goals[-1].goal_id]
        # Explicit outcome lines remain visible even if generation omits a limitation.
        outcomes = '\n'.join(f"- {r['goal'].replace('_', ' ')}: {r['message']}" for r in results.values()
                             if r['goal'] != 'final_summary' and r['status'] != 'success')
        answer = final['message'] if final['status'] == 'success' else 'Summary generation failed. See step results below.'
        if outcomes:
            answer += '\n\n' + outcomes
        if history_bundle:
            from src.llm.history_summary import coverage_notice
            answer += '\n\n' + coverage_notice(history_bundle)
        if search_bundle:
            from src.llm.medical_search_summary import search_notice
            answer += '\n\n' + search_notice(search_bundle)
        from src.llm.reference_evidence import reference_evidence, reference_notice
        retrieved_evidence = reference_evidence(sources)
        if retrieved_evidence:
            answer += '\n\n' + reference_notice(retrieved_evidence)
        return {'memory_traces':memory_traces, 'reference_evidence': retrieved_evidence, 'answer': answer, 'steps': list(results.values()), 'source_documents': sources, 'booking': booking,
                'history_snapshot': history_bundle, 'medical_search': search_bundle, 'summary_index_status': summary_index_status, 'context_subject_patient_id': context_subject_id}

