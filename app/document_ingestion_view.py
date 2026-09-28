"""Staff-facing shared reference library upload and semantic-search preview."""
import streamlit as st
from src.data_processing.ingestion import DocumentIngestion
from src.config import REFERENCE_DOCUMENT_UPLOAD_ROLES


def render_patient_documents(username, database_path, repository=None):
    from src.repositories.patient_document_repository import PatientDocumentRepository
    repository = repository or PatientDocumentRepository(database_path)
    with st.expander('Patient documents — upload and download', expanded=True):
        st.caption('Select a patient to save PDFs in their record. Maximum 20 MB and 200 pages per PDF.')
        try:
            patients = repository.list_patients(username)
            if not patients:
                st.info('No patients are available for document upload.')
                return
            labels = {p['patient_id']: f"{p['first_name'] or ''} {p['last_name'] or ''} "
                      f"({p['patient_id']}) — DOB: {p['date_of_birth'] or 'Unknown'}" for p in patients}
            patient_id = st.selectbox('Patient', list(labels), index=None,
                                      format_func=labels.get, key=f'document_patient_{username}',
                                      placeholder='Select a patient')
            if patient_id is None:
                return
            files = st.file_uploader('Patient PDFs', type=['pdf'], accept_multiple_files=True,
                                     key=f'patient_pdfs_{username}_{patient_id}')
            if st.button('Save documents to selected patient', key='save_patient_documents'):
                if not files:
                    st.warning('Choose at least one PDF.')
                for uploaded in files or []:
                    try:
                        result = repository.save(username, patient_id, uploaded.name, uploaded.getvalue())
                        if result['status'] == 'duplicate':
                            st.info(f'{uploaded.name}: already saved for this patient.')
                        else:
                            st.success(f'{uploaded.name}: saved to {labels[patient_id]}.')
                        if result.get('history_notes_added'):
                            st.info(f"Added {result['history_notes_added']} source notes to patient history. Rebuild the patient summary to refresh vector search.")
                        if result.get('pages_without_text'):
                            st.warning('Some PDF pages have no extractable text and need OCR before they can be searched: ' +
                                       ', '.join(map(str, result['pages_without_text'])))
                    except ValueError as error:
                        st.error(f'{uploaded.name}: {error}')
            documents = repository.list_documents(username, patient_id)
            if not documents:
                st.info('No documents have been saved for this patient.')
            for document in documents:
                saved = repository.get_document(username, patient_id, document['document_id'])
                st.download_button(f"Download {document['filename']} ({document['uploaded_at']})",
                                   saved['content'], file_name=saved['filename'], mime='application/pdf',
                                   key=f"patient_document_{document['document_id']}")
        except (ValueError, PermissionError) as error:
            st.error(str(error))


def render_document_ingestion(user_type, ingestion=None):
    if user_type not in REFERENCE_DOCUMENT_UPLOAD_ROLES:
        return
    with st.expander('Reference documents — upload and index'):
        st.caption('Shared reference library available to every attendant and doctor for all patients. Upload general guidelines or educational PDFs, not patient records. Extracted text is sent to OpenAI for embeddings. Limit: 20 MB and 200 pages per PDF; scanned PDFs require OCR first.')
        ingestion = ingestion or DocumentIngestion()
        files = st.file_uploader('Medical reference PDFs', type=['pdf'], accept_multiple_files=True)
        if st.button('Index uploaded PDFs'):
            if not files:
                st.warning('Choose at least one PDF.')
            for uploaded in files or []:
                try:
                    with st.spinner(f'Indexing {uploaded.name}…'):
                        result = ingestion.ingest(uploaded.name, uploaded.getvalue())
                    if result['status'] == 'duplicate':
                        st.info(f"{uploaded.name}: already indexed; skipped.")
                    else:
                        st.success(f"{uploaded.name}: indexed {result['pages']} pages into {result['chunks']} chunks.")
                        if result['empty_pages']:
                            st.warning(f"{result['empty_pages']} pages had no extractable text and were skipped.")
                except ValueError as error:
                    st.error(f'{uploaded.name}: {error}')
                except Exception:
                    st.error(f'{uploaded.name}: indexing failed or the index is busy. Existing documents are unchanged; retry shortly.')
        try:
            documents = ingestion.store.list_documents()
            if documents:
                st.dataframe(documents, hide_index=True, use_container_width=True)
            else:
                st.info('No reference PDFs have been indexed yet.')
            query = st.text_input('Preview reference search')
            if st.button('Search indexed references'):
                matches = ingestion.store.search(query)
                if not matches:
                    st.info('No reference matches found.')
                for doc, score in matches:
                    st.caption(f"{doc.metadata.get('source', 'Reference')} · page {doc.metadata.get('page', '?')}")
                    st.write(doc.page_content)
        except ValueError as error:
            st.warning(str(error))
        except Exception:
            st.error('The reference index could not be read. Check configuration and retry.')
