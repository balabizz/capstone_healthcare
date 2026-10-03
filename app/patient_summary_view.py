"""Patient-scoped semantic summary search controls."""
import streamlit as st
from app.input_layout import REQUEST_INPUT_HEIGHT
from src.llm.history_summary import coverage_notice


def render_patient_summary_search(profile, execution):
    with st.expander('Search saved patient summary (FAISS)'):
        st.caption('Search applies only to the patient selected under Family & dependents. Results are summary excerpts, not complete medical history.')
        query = st.text_area('Search summary', key='patient_summary_query', height=REQUEST_INPUT_HEIGHT)
        rebuild = st.button('Rebuild patient summary')
        search = st.button('Search patient summary')
        if not (rebuild or search):
            return
        try:
            requester = profile['patient_id']
            dependent = st.session_state.get('selected_dependent')
            patient = execution.dependents.resolve(requester, dependent_id=dependent) if dependent else requester
            execution.patient_history.require_access(requester, patient)
            store = execution.patient_summaries
            if rebuild:
                with st.spinner('Summarizing and indexing medical history…'):
                    count = store.rebuild(requester_patient_id=requester, patient_id=patient)
                st.success(f'Patient summary indexed ({count} searchable excerpts).')
            if search:
                result = store.search(requester_patient_id=requester, patient_id=patient, query=query)
                if result['status'] != 'ready':
                    st.info('No current indexed summary is available. Click Rebuild patient summary.')
                else:
                    st.caption(f"Selected patient: {patient}")
                    for match in result['matches']:
                        st.write(match['text'])
                    st.caption(coverage_notice(result['snapshot']))
                    with st.expander('Indexed source snapshot'):
                        st.json(result['snapshot'])
        except (ValueError, PermissionError) as error:
            st.warning(str(error))
        except Exception:
            st.error('Patient summary indexing/search could not complete. Check configuration and retry.')
