"""Attendant medical-record editor; no LLM-generated clinical writes."""
from uuid import uuid4
import sqlite3
import streamlit as st


def render_medical_records(profile, execution):
    repository = execution.medical_records
    actor = profile['attendant_id']
    st.header('Medical records')
    st.caption('Add diagnoses and treatments or standalone clinical notes for your assigned patients.')
    try:
        patients = repository.list_patients(actor)
        if not patients:
            st.info('No patients are assigned to this attendant. Ask the application administrator to assign patients.')
            return
        patient = st.selectbox('Patient', patients, format_func=lambda p:
            f"{p['first_name'] or ''} {p['last_name'] or ''} — {p['patient_id']} — DOB {p['date_of_birth'] or 'unknown'}")
        pid = patient['patient_id']
        records = repository.list_records(actor, pid)
    except PermissionError as error:
        st.error(str(error))
        return
    st.write(f"Editing records for patient **{pid}**")
    if st.session_state.get('record_saved_message'):
        st.success(st.session_state.pop('record_saved_message'))
    if records:
        st.dataframe([{'record_id': r['history_id'], 'type': r['record_type'],
            'diagnosis': r['condition_name'], 'diagnosis_date': r['diagnosis_date'],
            'version': r['version'], 'updated': r['updated_at'] or r['recorded_at']} for r in records], hide_index=True)
    else:
        st.info('No medical records stored for this patient yet.')
    options = ['new_diagnosis', 'new_note'] + [r['history_id'] for r in records]
    labels = {'new_diagnosis': 'Add diagnosis / treatment', 'new_note': 'Add standalone clinical note'}
    labels.update({r['history_id']: f"Edit {r['record_type']}: {r['condition_name'] or 'Clinical note'} [{r['history_id']}]" for r in records})
    selected = st.selectbox('Action / record', options, format_func=lambda x: labels[x], key=f'record_choice_{pid}')
    identity = (actor, pid, selected)
    editor = st.session_state.get('record_editor')
    if not editor or editor['identity'] != identity:
        current = next((r for r in records if r['history_id'] == selected), None)
        editor = {'identity': identity, 'snapshot': current, 'request_id': uuid4().hex}
        st.session_state.record_editor = editor
    current = editor['snapshot']
    if st.button('Reload record / discard unsaved changes'):
        st.session_state.pop('record_editor', None)
        st.rerun()
    record_type = current['record_type'] if current else ('note' if selected == 'new_note' else 'diagnosis')
    with execution._connect() as c:
        doctors = [dict(r) for r in c.execute('SELECT doctor_id,first_name,last_name FROM doctors ORDER BY last_name,first_name')]
    doctor_ids = [None] + [d['doctor_id'] for d in doctors]
    doctor_labels = {None: 'Not specified', **{d['doctor_id']: f"{d['first_name']} {d['last_name']} ({d['doctor_id']})" for d in doctors}}
    snapshot = current or {}
    form_id = f"record_form_{editor['request_id']}"
    with st.form(form_id):
        if record_type == 'diagnosis':
            condition = st.text_input('Diagnosis / condition', value=snapshot.get('condition_name') or '', key=form_id+'condition')
            diagnosed = st.text_input('Diagnosis date (YYYY-MM-DD, optional)', value=snapshot.get('diagnosis_date') or '', key=form_id+'date')
            treatment = st.text_area('Treatment as documented by the care team', value=snapshot.get('treatment') or '', key=form_id+'treatment')
        else:
            condition, diagnosed, treatment = '', '', ''
        notes = st.text_area('Clinical notes', value=snapshot.get('notes') or '', height=180, key=form_id+'notes')
        original_doctor = snapshot.get('doctor_id')
        doctor = st.selectbox('Documenting / treating doctor (optional)', doctor_ids,
            index=doctor_ids.index(original_doctor) if original_doctor in doctor_ids else 0,
            format_func=lambda x: doctor_labels[x], key=form_id+'doctor')
        reason = st.text_input('Reason for this entry or correction', key=form_id+'reason')
        submitted = st.form_submit_button('Save medical record')
    if submitted:
        try:
            saved = repository.save(attendant_id=actor, patient_id=pid,
                record_type=record_type, condition_name=condition, diagnosis_date=diagnosed.strip() or None,
                treatment=treatment, notes=notes, doctor_id=doctor,
                history_id=current['history_id'] if current else None,
                expected_version=current['version'] if current else None,
                change_reason=reason, request_id=editor['request_id'])
            st.session_state.record_saved_message = f"Saved record {saved['history_id']} (version {saved['version']})."
            st.session_state.pop('record_editor', None)
            st.rerun()
        except (ValueError, PermissionError) as error:
            st.error(str(error))
        except sqlite3.Error:
            st.error("The record could not be saved. Your changes remain in the form; please retry.")
    if current:
        with st.expander('Record revision history'):
            try:
                revisions = repository.revisions(actor, pid, current['history_id'])
                for revision in revisions:
                    st.write(f"Version {revision['version']} · {revision['changed_at']} · attendant {revision['attendant_id']}")
                    st.write(revision['change_reason'])
                    st.json({'before': revision['before'], 'after': revision['after']})
                if not revisions:
                    st.info('This record predates revision tracking. Its first edit will preserve the previous content.')
            except PermissionError as error:
                st.error(str(error))
    render_patient_alerts(actor, pid, repository)


def render_patient_alerts(actor, patient_id, repository):
    with st.expander('Patient alerts / allergies'):
        try:
            alerts = repository.list_alerts(actor, patient_id)
            if alerts:
                st.dataframe(alerts, hide_index=True, use_container_width=True)
            else:
                st.info('No alerts recorded. This does not mean the patient has no allergies.')
            key = f'alert_save_{actor}_{patient_id}'
            request_id = st.session_state.setdefault(key, uuid4().hex)
            with st.form(f'add_alert_{patient_id}_{request_id}'):
                alert_type = st.selectbox('Alert type', ['allergy','clinical','other'])
                severity = st.selectbox('Recorded severity', ['low','moderate','high','critical'])
                description = st.text_area('Alert as documented by the care team')
                if st.form_submit_button('Add alert'):
                    repository.add_alert(attendant_id=actor,patient_id=patient_id,alert_type=alert_type,
                        severity=severity,description=description,request_id=request_id)
                    st.session_state.pop(key, None)
                    st.rerun()
            active = [r for r in alerts if r['status'] == 'active']
            if active:
                with st.form(f'resolve_alert_{patient_id}'):
                    chosen = st.selectbox('Active alert to resolve', active,
                        format_func=lambda r: f"{r['severity']}: {r['description']} [{r['alert_id']}]")
                    reason = st.text_input('Documented reason for resolving the alert')
                    if st.form_submit_button('Mark alert resolved'):
                        repository.resolve_alert(attendant_id=actor,patient_id=patient_id,alert_id=chosen['alert_id'],reason=reason)
                        st.rerun()
        except (ValueError, PermissionError) as error:
            st.error(str(error))
