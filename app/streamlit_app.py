"""Streamlit web application for healthcare RAG system."""

import sys
from datetime import date
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.goal_execution import APPOINTMENT_SLOTS, GoalExecution
from src.agents.planner import Planner
from src.repositories.agent_event_repository import AgentEventRepository


def render_appointment_view(
    profile: dict, user_type: str, goal_execution: GoalExecution
) -> None:
    """Render patient booking or doctor daily schedule."""
    if st.button("Back to Medical Assistant", key="back_to_assistant"):
        st.session_state.appointment_view = False
        st.rerun()

    st.header("Appointment View")
    if user_type == "patient":
        st.subheader("Book an appointment")
        appointment_date = st.date_input(
            "Appointment date", min_value=date.today(), key="patient_appointment_date"
        )
        specialties = goal_execution.get_specialties()
        speciality_choice = st.selectbox(
            "Speciality preference", ["Any speciality"] + specialties,
            key="patient_speciality",
        )
        speciality = None if speciality_choice == "Any speciality" else speciality_choice
        slot = st.selectbox("Available time", APPOINTMENT_SLOTS, key="patient_slot")
        doctors = goal_execution.get_available_doctors(speciality, appointment_date, slot)
        if doctors:
            st.caption(f"{len(doctors)} doctor(s) available for this time slot")
            doctor_options = {
                f"Dr. {doctor['first_name']} {doctor['last_name']} ({doctor['speciality']})": doctor["doctor_id"]
                for doctor in doctors
            }
            selected_doctor = st.selectbox(
                "Available doctor", list(doctor_options), key="patient_doctor"
            )
            reason = st.text_input("Reason for visit", key="appointment_reason")
            if st.button("Book appointment", key="book_appointment"):
                booking_request_id = f"request-{uuid4().hex}"
                if goal_execution.book_appointment(
                    profile["patient_id"], doctor_options[selected_doctor],
                    appointment_date, slot, reason,
                    request_id=booking_request_id,
                ):
                    st.success("Appointment booked successfully")
                    st.rerun()
                else:
                    st.warning("That time was just booked. Please choose another slot.")
        else:
            st.info("No doctor is available for this speciality and time.")

        st.subheader("My appointments")
        appointments = goal_execution.get_patient_appointments(profile["patient_id"])
        if appointments:
            st.dataframe(appointments, use_container_width=True, hide_index=True)
        else:
            st.info("No appointments booked yet")
    else:
        schedule_date = st.date_input(
            "Schedule date", value=date.today(), key="doctor_schedule_date"
        )
        schedule = goal_execution.get_doctor_schedule(profile["doctor_id"], schedule_date)
        st.subheader(f"Schedule for {schedule_date.strftime('%d %B %Y')}")
        if schedule:
            st.dataframe(schedule, use_container_width=True, hide_index=True)
        else:
            st.info("No patients scheduled for this day")


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
    if "appointment_view" not in st.session_state:
        st.session_state.appointment_view = False

    if st.session_state.authenticated_user is None:
        st.title("🏥 Agentic Healthcare Assistant")
        st.subheader("Log in to continue")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="firstname.lastname")
            password = st.text_input("Password", type="password")
            user_type = st.selectbox("User type", ("patient", "doctor"))
            submitted = st.form_submit_button("Log in", use_container_width=True)

        if submitted:
            if user_type not in ("patient", "doctor"):
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
        else:
            st.write(f"**Name:** {profile['first_name']} {profile['last_name']}")
            st.write(f"**Speciality:** {profile['speciality'] or 'Not provided'}")
            st.write(f"**License Number:** {profile['license_number'] or 'Not provided'}")
            st.write(f"**Doctor ID:** {profile['doctor_id']}")

        st.divider()
        if st.button("Appointment View", key="appointment_view_button", use_container_width=True):
            st.session_state.appointment_view = True
            st.rerun()

        if st.button("Sign out", key="signout", use_container_width=True):
            st.session_state.authenticated_user = None
            st.session_state.appointment_view = False
            st.rerun()

    if st.session_state.appointment_view:
        render_appointment_view(profile, authenticated_user["user_type"], goal_execution)
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Medical Assistant")
        query = st.text_input(
            "Ask your medical question",
            placeholder="e.g., What are the symptoms of diabetes?",
        )

        if st.button("Submit", key="submit_btn"):
            if query:
                with st.spinner("Searching and generating response..."):
                    request_id = f"request-{uuid4().hex}"
                    planning_started = perf_counter()
                    try:
                        goals = planner.plan(query)
                        event_repository.log(
                            request_id=request_id,
                            patient_id=profile.get("patient_id"),
                            event_type="plan_created",
                            tool_name="planner",
                            status="success",
                            duration_ms=round(
                                (perf_counter() - planning_started) * 1000
                            ),
                            details={"goal_types": [goal.name for goal in goals]},
                        )
                        result = goal_execution.execute(
                            goals[0], request_id=request_id,
                            patient_id=profile.get("patient_id"),
                        )
                    except Exception as error:
                        event_repository.log(
                            request_id=request_id,
                            patient_id=profile.get("patient_id"),
                            event_type="request_failed",
                            status="failed",
                            details={"error_type": type(error).__name__},
                        )
                        st.error("The request could not be completed. Please try again.")
                        return

                    st.session_state.chat_history.append(
                        {"query": query, "answer": result["answer"]}
                    )
                    st.success("Response generated!")
                    st.markdown("### Response")
                    st.write(result["answer"])

                    if result["source_documents"]:
                        with st.expander("📚 Source Documents"):
                            for index, document in enumerate(
                                result["source_documents"], 1
                            ):
                                st.markdown(f"**Document {index}:**")
                                st.write(document.page_content[:500] + "...")
            else:
                st.warning("Please enter a question")

    with col2:
        st.header("Chat History")
        if st.session_state.chat_history:
            for index, interaction in enumerate(
                st.session_state.chat_history[-5:], 1
            ):
                with st.expander(f"Q{index}: {interaction['query'][:50]}..."):
                    st.write(interaction["answer"])

            if st.button("Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No chat history yet")

        if authenticated_user["user_type"] == "patient":
            with st.expander("Agent Execution, Memory Traces & Tool Logs"):
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
