"""Doctor-authored notes and previous visits for the selected patient."""
from uuid import uuid4
import sqlite3
import streamlit as st
from app.input_layout import REQUEST_INPUT_HEIGHT


def render_doctor_notes(profile, execution):
    doctor_id = profile['doctor_id']
    with st.expander('Notes by doctor', expanded=True):
        try:
            patients = execution.list_staff_patients('doctor', doctor_id)
            if not patients:
                st.info('No patients are available.')
                return
            labels = {p['patient_id']: f"{p['first_name'] or ''} {p['last_name'] or ''} [{p['patient_id']}]" for p in patients}
            patient_id = st.selectbox('Patient for doctor notes', list(labels),
                                      format_func=labels.get, key='doctor_notes_patient')
            st.caption('Previous notes from all doctors are available for this patient, including return visits. New notes are saved with your doctor identity and timestamp.')
            notes = execution.doctor_notes.list_notes(doctor_id, patient_id)
            scope = f'{doctor_id}_{patient_id}'
            message = st.session_state.pop('doctor_note_message_' + scope, None)
            if message:
                if message == 'indexed':
                    st.success('Doctor note saved and patient-summary embeddings updated.')
                else:
                    st.warning('Doctor note saved. Summary embeddings could not be updated; retry below. The original note remains available.')
            st.subheader('Past notes from doctors')
            if not notes:
                st.info('No previous doctor notes for this patient.')
            for note in notes:
                name = f"{note['first_name'] or ''} {note['last_name'] or ''}".strip()
                with st.expander(f"{note['recorded_at']} ? Dr {name} ? {note['history_id']}"):
                    st.text(note['notes'])
            request_key = 'doctor_note_request_' + scope
            request_id = st.session_state.setdefault(request_key, uuid4().hex)
            with st.form('doctor_note_form_' + scope + '_' + request_id):
                text = st.text_area('New visit note', height=REQUEST_INPUT_HEIGHT,
                                    max_chars=10000, key='doctor_note_text_' + scope + '_' + request_id)
                submitted = st.form_submit_button('Save doctor note and update summary')
            if submitted:
                with st.spinner('Saving note and updating patient summary?'):
                    result = execution.save_doctor_note(doctor_id=doctor_id, patient_id=patient_id,
                                                        notes=text, request_id=request_id)
                st.session_state['doctor_note_message_' + scope] = result['embedding_status']
                st.session_state.pop(request_key, None)
                st.rerun()
            if st.button('Refresh patient-summary embeddings', key='doctor_note_refresh_' + scope):
                with st.spinner('Summarizing the current patient record?'):
                    execution.patient_summaries.rebuild_for_staff(
                        staff_type='doctor', staff_id=doctor_id, patient_id=patient_id)
                st.success('Patient-summary embeddings updated from current records.')
        except (PermissionError, ValueError) as error:
            st.error(str(error))
        except sqlite3.Error:
            st.error('Patient notes could not be accessed. Please retry.')
        except Exception:
            st.error('The request could not complete. Saved notes remain in the patient record; retry the summary update.')
