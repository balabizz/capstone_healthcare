"""Read-only timed appointment panels; forms and LLM workflows never run on ticks."""
from datetime import datetime, timezone
import streamlit as st
from src.config import APPOINTMENT_REFRESH_SECONDS


@st.fragment(run_every=APPOINTMENT_REFRESH_SECONDS)
def render_live_appointments(execution):
    # Read the current authenticated session on every tick rather than retaining a
    # patient/doctor ID captured when the fragment first rendered.
    if not st.session_state.get('appointment_view'):
        return
    user = st.session_state.get('authenticated_user')
    if not user or user.get('user_type') not in ('patient','doctor'):
        st.info('Sign in to view appointments.')
        return
    st.button('Refresh appointments now',key='refresh_live_appointments')
    try:
        profile = execution.get_user_profile(user['username'],user['user_type'])
        if profile is None:
            st.warning('Your account is no longer available. Sign in again.')
            return
        if user['user_type']=='patient':
            requester = profile['patient_id']
            selected = st.session_state.get('selected_dependent')
            patient = execution.dependents.resolve(requester,dependent_id=selected) if selected else requester
            rows = execution.get_patient_appointments(patient,requester_patient_id=requester)
            st.subheader('My appointments' if patient==requester else 'Selected family member appointments')
            empty = 'No appointments booked yet.'
        else:
            selected_date = st.session_state.get('doctor_schedule_date') or execution.calendars.now().date()
            rows = execution.get_doctor_schedule(profile['doctor_id'],selected_date)
            st.subheader(f"Schedule for {selected_date.strftime('%d %B %Y')}")
            empty = 'No patients scheduled for this day.'
        current_profile = execution.get_user_profile(user['username'],user['user_type'])
        identity_field = 'patient_id' if user['user_type']=='patient' else 'doctor_id'
        if current_profile is None or current_profile[identity_field] != profile[identity_field]:
            st.warning('Your account access changed. Sign in again.')
            return
        if rows:
            st.dataframe(rows,use_container_width=True,hide_index=True)
        else:
            st.info(empty)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        st.caption(f'Last refreshed: {now} · updates every {APPOINTMENT_REFRESH_SECONDS} seconds while this view is active.')
    except (PermissionError,ValueError) as error:
        st.warning(str(error))
    except Exception:
        st.warning('Appointments could not be refreshed. Retrying automatically; no stale rows are displayed.')
