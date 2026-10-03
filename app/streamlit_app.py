"""Streamlit web application for healthcare RAG system."""

import sys
import sqlite3
from datetime import date
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.goal_execution import GoalExecution
from app.input_layout import REQUEST_INPUT_HEIGHT
from app.medical_records_view import render_medical_records
from src.agents.planner import Planner
from src.agents.plan_execution import PlanExecution
from src.repositories.dependent_repository import RELATIONSHIPS, PERMISSIONS
from src.repositories.agent_event_repository import AgentEventRepository


def render_family(profile, execution):
    """Manage relationships without granting access by association."""
    owner = profile["patient_id"]
    with st.expander("Family & dependents"):
        relatives = execution.dependents.list_dependents(owner)
        choices = {"Myself": None}
        choices.update({f"{r['name']} ({r['relationship']}) [{r['dependent_id']}]": r['dependent_id']
                        for r in relatives})
        if "pending_family" in st.session_state:
            st.session_state.family_selection = st.session_state.pop("pending_family")
        selected = st.selectbox("Who is this enquiry or appointment for?", list(choices), key="family_selection")
        previous_selection = st.session_state.get("selected_dependent")
        st.session_state.selected_dependent = choices[selected]
        if previous_selection != choices[selected]:
            for key in ("latest_answer", "latest_sources", "latest_history", "latest_medical_search", "last_plan", "last_steps", "latest_context_subject"):
                st.session_state.pop(key, None)
            st.session_state.chat_history = []
        with st.form("add_family"):
            name = st.text_input("Family member name")
            relationship = st.selectbox("Relationship to you", RELATIONSHIPS)
            linked_id = st.text_input("Their patient ID (optional)")
            if st.form_submit_button("Add family member"):
                try:
                    execution.dependents.add_dependent(owner, name, relationship, linked_id.strip() or None)
                    st.rerun()
                except (ValueError, sqlite3.IntegrityError):
                    st.error("Check the name and patient ID; this relationship may already exist.")
        st.caption("Adding a relationship does not grant access. The family member must sign in and grant permission. Unregistered relatives can be saved, but cannot be booked yet.")
        with st.form("family_permission"):
            requester = st.text_input("Patient ID allowed to act for me")
            permission = st.selectbox("Permission for my record", PERMISSIONS)
            enabled = st.checkbox("Allow access", value=False)
            if st.form_submit_button("Save permission"):
                try:
                    execution.dependents.set_permission(owner, requester.strip(), permission, enabled)
                    st.success("Permission updated")
                except (ValueError, sqlite3.IntegrityError):
                    st.error("Enter a valid registered patient ID.")


def render_staff_patient_summary(profile, user_type, execution):
    staff_id = profile.get('doctor_id') if user_type == 'doctor' else profile.get('attendant_id')
    try:
        patients = execution.list_staff_patients(user_type, staff_id)
    except (ValueError, PermissionError) as error:
        st.session_state.pop('staff_patient_summary', None)
        st.error(str(error))
        return
    if not patients:
        st.session_state.pop('staff_patient_summary', None)
        st.info('No patients are available for this staff account.')
        return
    labels = {
        patient['patient_id']: f"{patient['first_name']} {patient['last_name']} [{patient['patient_id']}]"
        for patient in patients
    }
    selected = st.selectbox('Patient for health-record summary', list(labels),
                            format_func=labels.get, key='staff_summary_patient')
    scope = (user_type, staff_id, selected)
    if st.session_state.get('staff_summary_scope') != scope:
        st.session_state.pop('staff_patient_summary', None)
        st.session_state.staff_summary_scope = scope
    request = st.text_area('Summary request', value='Summarize the overall patient health records and condition.',
                            key='staff_summary_request', height=REQUEST_INPUT_HEIGHT)
    if st.button('Generate patient health summary', key='staff_summary_submit'):
        st.session_state.pop('staff_patient_summary', None)
        try:
            result = execution.summarize_patient_for_staff(
                staff_type=user_type, staff_id=staff_id, patient_id=selected, request=request)
            st.session_state.staff_patient_summary = result
        except (ValueError, PermissionError) as error:
            st.error(str(error))
    result = st.session_state.get('staff_patient_summary')
    if result and result['history_snapshot']['subject_patient_id'] == selected:
        try:
            execution.validate_staff_summary_snapshot(user_type, staff_id, result['history_snapshot'])
        except (ValueError, PermissionError) as error:
            st.session_state.pop('staff_patient_summary', None)
            st.error(str(error))
            return
        st.write(result['answer'])
        with st.expander('Structured patient records used'):
            snapshot = result['history_snapshot']
            for category in ('records', 'prescriptions', 'alerts'):
                st.write(category.capitalize())
                st.json(snapshot[category])
            st.json(snapshot['coverage'])
        with st.expander('Patient vector-summary matches used'):
            st.write(result['patient_summary']['status'])
            st.json(result['patient_summary']['matches'])


def main():
    """Run the Streamlit application."""
    goal_execution = GoalExecution()
    event_repository = AgentEventRepository(goal_execution.database_path)
    planner = Planner()
    st.set_page_config(
        page_title="Agentic Healthcare Assistant",
        page_icon="🏥",
        layout="wide",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    if st.session_state.authenticated_user is None:
        st.title("🏥 Agentic Healthcare Assistant")
        st.subheader("Log in to continue")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="firstname.lastname")
            password = st.text_input("Password", type="password")
            user_type = st.selectbox("User type", ("patient", "doctor", "attendant"))
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if user_type not in ("patient", "doctor", "attendant"):
                st.error("Please select a valid user type")
            elif goal_execution.authenticate_user(username, password, user_type):
                st.session_state.authenticated_user = {
                    "username": username.strip().lower(),
                    "user_type": user_type,
                }
                st.rerun()
            else:
                st.error("Invalid username, password, or user type")
        return

    authenticated_user = st.session_state.authenticated_user
    st.title(f"🏥 Medical Assistant for {authenticated_user['user_type'].title()}s")
    if authenticated_user["user_type"] != "attendant":
        st.markdown("Ask a medical question and receive an AI-assisted response")

    profile = goal_execution.get_user_profile(
        authenticated_user["username"], authenticated_user["user_type"]
    )
    if profile is None:
        st.error("The logged-in user profile could not be found")
        return

    with st.sidebar:
        st.header("User Details")
        if authenticated_user["user_type"] == "patient":
            st.write(f"**Name:** {profile['first_name']} {profile['last_name']}")
            st.write(f"**Age:** {profile['age']}")
            st.write(f"**Gender:** {profile['gender'] or 'Not provided'}")
            st.write(f"**Patient ID:** {profile['patient_id']}")
        elif authenticated_user["user_type"] == "doctor":
            st.write(f"**Name:** {profile['first_name']} {profile['last_name']}")
            st.write(f"**Speciality:** {profile['speciality'] or 'Not provided'}")
            st.write(f"**License Number:** {profile['license_number'] or 'Not provided'}")
            st.write(f"**Doctor ID:** {profile['doctor_id']}")

        else:
            st.write(f"**Name:** {profile['first_name']} {profile['last_name']}")
            st.write(f"**Attendant ID:** {profile['attendant_id']}")

        if st.button("Sign out", key="signout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if authenticated_user['user_type'] in ('doctor', 'attendant'):
        from app.document_ingestion_view import render_document_ingestion, render_patient_documents
        render_patient_documents(authenticated_user['username'], goal_execution.database_path)
        render_document_ingestion(authenticated_user['user_type'])
        from app.evaluation_view import render_model_evaluation
        render_model_evaluation(authenticated_user['user_type'], goal_execution)
        from app.performance_view import render_performance_dashboard
        render_performance_dashboard(authenticated_user['user_type'], goal_execution)
        with st.expander('Patient health-record summary', expanded=True):
            render_staff_patient_summary(profile, authenticated_user['user_type'], goal_execution)

    from app.request_trace_view import render_scenario_testing
    render_scenario_testing()

    if authenticated_user["user_type"] == "attendant":
        render_medical_records(profile, goal_execution)
        return

    if authenticated_user["user_type"] == "patient":
        render_family(profile, goal_execution)
        from app.patient_summary_view import render_patient_summary_search
        render_patient_summary_search(profile, goal_execution)

    memory_subject = None
    if authenticated_user['user_type'] == 'patient':
        from src.agents.conversation_flow import ConversationFlow
        try:
            memory_subject = ConversationFlow(goal_execution, planner).subject(
                profile['patient_id'], st.session_state.get('selected_dependent'))
            from app.request_trace_view import render_request_traces
            render_request_traces(goal_execution,profile['patient_id'],memory_subject)
            saved = goal_execution.conversations.retrieve(profile['patient_id'], memory_subject, limit=10)
            st.session_state.chat_history = [dict(turn, history_subject_patient_id=memory_subject) for turn in saved]
        except (ValueError, PermissionError):
            st.session_state.chat_history = []

    visible_chat = []
    for interaction in st.session_state.chat_history:
        scoped_patient = interaction.get("history_subject_patient_id")
        try:
            if scoped_patient:
                goal_execution.patient_history.require_access(profile.get("patient_id"), scoped_patient)
            visible_chat.append(interaction)
        except (ValueError, PermissionError):
            pass
    st.session_state.chat_history = visible_chat
    history = st.session_state.get("latest_history")
    context_subject = st.session_state.get('latest_context_subject')
    if history or context_subject:
        try:
            goal_execution.patient_history.require_access(profile.get("patient_id"), history["subject_patient_id"] if history else context_subject)
        except (ValueError, PermissionError):
            for key in ("latest_answer", "latest_sources", "latest_history", "latest_medical_search", "last_plan", "last_steps", "latest_context_subject"):
                st.session_state.pop(key, None)
            st.session_state.chat_history = []
            history = None
            st.warning("Medical access changed. Please submit a new request.")
    if history:
        st.caption(f"History for patient {history['subject_patient_id']} · retrieved {history['retrieved_at']}")
        active_alerts = [r for r in history['alerts'] if r['status'] == 'active']
        for alert in active_alerts:
            st.warning(f"Recorded active alert ({alert['severity']}): {alert['description']} [alert:{alert['alert_id']}]")
        with st.expander("Patient history source records"):
            for category in ("records", "prescriptions", "alerts"):
                st.write(category.capitalize())
                st.json(history[category])
            st.json(history["coverage"])

    if st.session_state.get('trace_save_warning'):
        st.warning(st.session_state.trace_save_warning)
    if st.session_state.get("last_plan"):
        with st.expander("Request plan and results", expanded=True):
            st.dataframe(st.session_state.last_plan, hide_index=True, use_container_width=True)
            for step in st.session_state.get("last_steps", []):
                st.write(f"{step['id']} — {step['goal']}: {step['status']}")
                st.write(step["message"])
    if st.session_state.get("latest_answer"):
        st.write(st.session_state.latest_answer)
    if st.session_state.get("latest_sources"):
        from app.reference_evidence_view import render_reference_evidence
        render_reference_evidence(st.session_state.latest_sources)
    search = st.session_state.get("latest_medical_search")
    if search:
        with st.expander("Live medical search sources", expanded=True):
            st.caption(f"Searched {search['searched_at']} · Publication window {search['date_from']} to {search['date_to']}")
            st.write("Search topic: " + search["query"])
            for provider in search["providers"]:
                st.write(f"{provider['provider']}: {provider['status']} ({provider['returned']} sources)")
            for source in search["sources"]:
                st.write(source["title"])
                st.markdown(f"[Open {source['provider']} source]({source['url']})")
                st.caption(f"Published: {source['published'] or 'not supplied'} · {source['evidence_type']}")
                st.write(source['excerpt'] or 'No abstract/overview supplied; metadata only.')
    proposal = st.session_state.get("booking_proposal")
    if proposal:
        appointment = proposal.get("slot")
        if proposal.get("status") == "booked" and appointment:
            st.success(
                f"Appointment booked for {appointment['date']} at {appointment['time']} "
                f"({appointment['timezone']}) with Dr. {appointment['first_name']} "
                f"{appointment['last_name']} ({appointment['speciality']})."
            )
            st.dataframe([{
                "Date": appointment["date"], "Time": appointment["time"],
                "Doctor": f"Dr. {appointment['first_name']} {appointment['last_name']}",
                "Specialty": appointment["speciality"], "Timezone": appointment["timezone"],
                "Duration": f"{appointment['duration_minutes']} minutes",
            }], hide_index=True, use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Medical Assistant")
        query = st.text_area(
            "Ask your medical question",
            placeholder="e.g., What are the symptoms of diabetes?",
            height=REQUEST_INPUT_HEIGHT,
        )

        if st.button("Submit", key="submit_btn"):
            if query:
                with st.spinner("Searching and generating response..."):
                    request_id = f"request-{uuid4().hex}"
                    planning_started = perf_counter()
                    try:
                        st.session_state.booking_proposal = None
                        from src.agents.conversation_flow import ConversationFlow
                        conversation = ConversationFlow(goal_execution, planner)
                        plan, conversation_subject = conversation.plan(query,
                            requester=profile.get('patient_id'), dependent_id=st.session_state.get('selected_dependent'))
                        event_repository.log(
                            request_id=request_id,
                            patient_id=profile.get("patient_id"),
                            event_type="plan_created",
                            tool_name="planner",
                            status="success",
                            duration_ms=round(
                                (perf_counter() - planning_started) * 1000
                            ),
                            details={"steps": plan.trace(), "needs_clarification": bool(plan.clarification)},
                        )
                        result = PlanExecution(goal_execution).run(
                            plan, request_id=request_id, patient_id=profile.get("patient_id"),
                            dependent_id=st.session_state.get("selected_dependent"),
                        )
                        try:
                            conversation.remember(query, result, requester=profile.get('patient_id'),
                                subject=conversation_subject, request_id=request_id)
                            st.session_state.pop('memory_save_warning', None)
                        except Exception:
                            st.session_state.memory_save_warning = 'This response could not be saved to long-term conversation memory.'
                        if profile.get('patient_id') and conversation_subject:
                            try:
                                goal_execution.request_traces.save(profile['patient_id'],conversation_subject,request_id,
                                    plan.details(),conversation.memory_traces+result.get('memory_traces',[]),result['steps'])
                                st.session_state.pop('trace_save_warning',None)
                            except Exception:
                                st.session_state.trace_save_warning = 'The request trace could not be saved or access is unavailable.'
                        st.session_state.last_plan = plan.trace()
                        st.session_state.last_steps = result["steps"]
                        st.session_state.latest_sources = result["source_documents"]
                        st.session_state.latest_history = result.get("history_snapshot")
                        st.session_state.latest_context_subject = result.get("context_subject_patient_id")
                        st.session_state.latest_medical_search = result.get("medical_search")
                        st.session_state.booking_proposal = result["booking"]
                    except Exception as error:
                        event_repository.log(
                            request_id=request_id,
                            patient_id=profile.get("patient_id"),
                            event_type="request_failed",
                            status="failed",
                            details={"error_type": type(error).__name__,
                                     "error_message": str(error)[:300] if isinstance(error, (ValueError, PermissionError)) else "request failed"},
                        )
                        st.error(str(error) if isinstance(error, (ValueError, PermissionError)) else "The request could not be completed. Please try again.")
                        return

                    st.session_state.chat_history.append(
                        {"query": query, "answer": result["answer"],
                         "history_subject_patient_id": (result.get("history_snapshot") or {}).get("subject_patient_id") or result.get("context_subject_patient_id")}
                    )
                    st.session_state.latest_answer = result["answer"]
                    st.rerun()
            else:
                st.warning("Please enter a question")

    with col2:
        st.header("Chat History")
        st.caption('Saved across sessions for this account and selected patient. Past dialogue is not verified medical history.')
        if st.session_state.get('memory_save_warning'):
            st.warning(st.session_state.memory_save_warning)
        if st.session_state.chat_history:
            for index, interaction in enumerate(
                st.session_state.chat_history[-5:], 1
            ):
                with st.expander(f"Q{index}: {interaction['query'][:50]}..."):
                    if interaction.get('created_at'):
                        st.caption(f"{interaction['created_at']} · turn {interaction['turn_id']}")
                    if interaction.get('truncated'):
                        st.caption('Long conversation shortened for display and context.')
                    st.write(interaction["answer"])

            if st.button("Delete saved conversations for selected patient"):
                if memory_subject:
                    goal_execution.conversations.clear(profile['patient_id'], memory_subject)
                st.session_state.chat_history = []
                for key in ('latest_answer', 'last_steps', 'last_plan'):
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            st.info("No chat history yet")

        if authenticated_user["user_type"] == "patient":
            with st.expander("Agent Execution & Tool Logs"):
                events = event_repository.list_events(
                    patient_id=profile["patient_id"], limit=20
                )
                if events:
                    st.dataframe(
                        [
                            {
                                "time": event["created_at"],
                                "event": event["event_type"],
                                "tool": event["tool_name"],
                                "status": event["status"],
                                "duration_ms": event["duration_ms"],
                            }
                            for event in events
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No agent execution events recorded yet")


if __name__ == "__main__":
    main()
