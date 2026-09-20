"""Staff-facing shared reference library upload and semantic-search preview."""
import streamlit as st
from src.data_processing.ingestion import DocumentIngestion


def render_document_ingestion(user_type, ingestion=None):
    if user_type not in ('doctor', 'attendant'):
        return
    with st.expander('Reference documents — upload and index'):
        st.caption('Shared reference library used by all medical-reference answers. Upload general guidelines or educational PDFs, not patient records. Extracted text is sent to OpenAI for embeddings. Limit: 20 MB and 200 pages per PDF; scanned PDFs require OCR first.')
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
