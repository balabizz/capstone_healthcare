"""Streamlit web application for healthcare RAG system."""

import sqlite3
import sys
from datetime import date, time
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SQLITE_DB_PATH
from src.llm.llm_client import LLMClient
from src.chains.rag_chain import RAGChain
from src.models.patient_vo import PatientVO


def get_user_profile(username: str, user_type: str) -> dict | None:
    """Retrieve the profile linked to an authenticated login."""
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        if user_type == "patient":
            row = connection.execute(
                """
                SELECT p.patient_id, p.first_name, p.last_name, p.gender,
                       p.date_of_birth
                FROM login_details AS l
                JOIN patients AS p ON p.patient_id = l.patient_id
                WHERE l.username = ? AND l.user_type = ? AND l.is_active = 1
                """,
                (username, user_type),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT d.doctor_id, d.first_name, d.last_name,
                       d.speciality, d.license_number
                FROM login_details AS l
                JOIN doctors AS d ON d.doctor_id = l.doctor_id
                WHERE l.username = ? AND l.user_type = ? AND l.is_active = 1
                """,
                (username, user_type),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    profile = dict(row)
    if user_type == "patient":
        patient = PatientVO.from_dict(profile)
        profile["age"] = patient.age()
    return profile


APPOINTMENT_SLOTS = [
    time(hour=hour, minute=minute).strftime("%H:%M")
    for hour, minute in [
        (9, 0), (9, 30), (10, 0), (10, 30), (11, 0), (11, 30),
        (12, 0), (12, 30), (15, 0), (15, 30), (16, 0), (16, 30),
        (17, 0), (17, 30), (18, 0), (18, 30), (19, 0), (19, 30),
    ]
]


def get_specialties() -> list[str]:
    """Return the specialties currently offered by doctors."""
    connection = sqlite3.connect(SQLITE_DB_PATH)
    try:
        rows = connection.execute(
            "SELECT DISTINCT speciality FROM doctors "
            "WHERE speciality IS NOT NULL ORDER BY speciality"
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def get_available_doctors(speciality: str | None, appointment_date: date, slot: str) -> list[dict]:
    """Return free doctors, optionally limited to a specialty."""
    appointment_datetime = f"{appointment_date.isoformat()} {slot}"
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT d.doctor_id, d.first_name, d.last_name, d.speciality
            FROM doctors AS d
                        WHERE (? IS NULL OR d.speciality = ?)
              AND NOT EXISTS (
                  SELECT 1 FROM appointments AS a
                  WHERE a.doctor_id = d.doctor_id
                    AND a.appointment_datetime = ?
                    AND a.status = 'scheduled'
              )
            ORDER BY d.last_name, d.first_name
            """,
            (speciality, speciality, appointment_datetime),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def book_appointment(
    patient_id: str,
    doctor_id: str,
    appointment_date: date,
    slot: str,
    reason: str,
) -> bool:
    """Book an available 30-minute appointment slot for a patient."""
    if slot not in APPOINTMENT_SLOTS:
        raise ValueError("Appointment time must be within the available hours")

    appointment_datetime = f"{appointment_date.isoformat()} {slot}"
    connection = sqlite3.connect(SQLITE_DB_PATH)
    try:
        connection.execute(
            """
            INSERT INTO appointments (
                appointment_id, patient_id, doctor_id,
                appointment_datetime, reason
            )
            SELECT ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM appointments
                WHERE doctor_id = ?
                  AND appointment_datetime = ?
                  AND status = 'scheduled'
            )
            """,
            (
                f"appointment-{uuid4().hex}", patient_id, doctor_id,
                appointment_datetime, reason.strip() or None,
                doctor_id, appointment_datetime,
            ),
        )
        booked = connection.total_changes == 1
        connection.commit()
    finally:
        connection.close()
    return booked


def get_patient_appointments(patient_id: str) -> list[dict]:
    """Return a patient's scheduled appointments."""
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT a.appointment_datetime, a.status, a.reason,
                   d.first_name || ' ' || d.last_name AS doctor_name,
                   d.speciality
            FROM appointments AS a
            JOIN doctors AS d ON d.doctor_id = a.doctor_id
            WHERE a.patient_id = ?
            ORDER BY a.appointment_datetime
            """,
            (patient_id,),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def get_doctor_schedule(doctor_id: str, schedule_date: date) -> list[dict]:
    """Return a doctor's appointments for one day."""
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT a.appointment_datetime, a.status, a.reason,
                   p.first_name || ' ' || p.last_name AS patient_name,
                   p.patient_id
            FROM appointments AS a
            JOIN patients AS p ON p.patient_id = a.patient_id
            WHERE a.doctor_id = ?
              AND date(a.appointment_datetime) = ?
            ORDER BY a.appointment_datetime
            """,
            (doctor_id, schedule_date.isoformat()),
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


def render_appointment_view(profile: dict, user_type: str) -> None:
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
        specialties = get_specialties()
        speciality_choice = st.selectbox(
            "Speciality preference", ["Any speciality"] + specialties,
            key="patient_speciality",
        )
        speciality = None if speciality_choice == "Any speciality" else speciality_choice
        slot = st.selectbox("Available time", APPOINTMENT_SLOTS, key="patient_slot")
        doctors = get_available_doctors(speciality, appointment_date, slot)
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
                if book_appointment(
                    profile["patient_id"], doctor_options[selected_doctor],
                    appointment_date, slot, reason
                ):
                    st.success("Appointment booked successfully")
                    st.rerun()
                else:
                    st.warning("That time was just booked. Please choose another slot.")
        else:
            st.info("No doctor is available for this speciality and time.")

        st.subheader("My appointments")
        appointments = get_patient_appointments(profile["patient_id"])
        if appointments:
            st.dataframe(appointments, use_container_width=True, hide_index=True)
        else:
            st.info("No appointments booked yet")
    else:
        schedule_date = st.date_input(
            "Schedule date", value=date.today(), key="doctor_schedule_date"
        )
        schedule = get_doctor_schedule(profile["doctor_id"], schedule_date)
        st.subheader(f"Schedule for {schedule_date.strftime('%d %B %Y')}")
        if schedule:
            st.dataframe(schedule, use_container_width=True, hide_index=True)
        else:
            st.info("No patients scheduled for this day")


def authenticate_user(username: str, password: str, user_type: str) -> bool:
    """Return whether credentials match an active login record."""
    connection = sqlite3.connect(SQLITE_DB_PATH)
    try:
        record = connection.execute(
            """
            SELECT 1
            FROM login_details
            WHERE username = ?
              AND password_hash = ?
              AND user_type = ?
              AND is_active = 1
            """,
            (username.strip().lower(), password, user_type),
        ).fetchone()
    finally:
        connection.close()
    return record is not None


def main():
    """Run the Streamlit application."""
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
            elif authenticate_user(username, password, user_type):
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

    profile = get_user_profile(
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
        render_appointment_view(profile, authenticated_user["user_type"])
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
                    LLMClient()

                    from src.vector_store.chromadb_store import ChromaDBStore
                    vectorstore = ChromaDBStore().load_store()

                    rag_chain = RAGChain(vectorstore)
                    result = rag_chain.query(query)

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


if __name__ == "__main__":
    main()
