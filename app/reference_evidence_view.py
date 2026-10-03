"""Render retrieved excerpts with the same labels used in the final answer."""
import streamlit as st
from src.llm.reference_evidence import reference_evidence


def render_reference_evidence(documents):
    rows = reference_evidence(documents)
    if not rows:
        st.info('No readable reference excerpts were returned.')
        return
    with st.expander('Source documents'):
        st.caption('Retrieved evidence for this answer. These excerpts show retrieval provenance; they do not independently verify every statement or establish that guidance is current.')
        for row in rows:
            st.text(f"[{row['id']}] {row['source']} · page {row['page'] if row['page'] is not None else 'not supplied'}")
            st.text(row['excerpt'])
