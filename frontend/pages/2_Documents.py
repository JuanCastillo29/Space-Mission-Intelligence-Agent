from __future__ import annotations

import streamlit as st

import api_client
from style import inject_css

st.set_page_config(
    page_title="Documents - Space Mission Intelligence",
    page_icon="\U0001F680",
    layout="wide",
)

inject_css()

st.header("\U0001F4DA Documents")

PAGE_SIZE = 20

if "doc_page" not in st.session_state:
    st.session_state.doc_page = 0


@st.cache_data(ttl=60)
def _fetch_documents(limit: int, offset: int) -> dict:
    return api_client.list_documents(limit=limit, offset=offset)


offset = st.session_state.doc_page * PAGE_SIZE

try:
    data = _fetch_documents(limit=PAGE_SIZE, offset=offset)
except Exception as exc:
    st.error(f"Failed to load documents: {exc}")
    st.stop()

total = data["total"]
docs = data["documents"]
total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

if docs:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", total)
    col2.metric("Source Types", len({d["source_type"] for d in docs}))
    col3.metric("Total Chunks", sum(d["chunk_count"] for d in docs))

    import pandas as pd

    df = pd.DataFrame(docs)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df = df[["title", "source_type", "mission_name", "chunk_count", "ingested_at"]]
    df.columns = ["Title", "Type", "Mission", "Chunks", "Ingested"]

    st.dataframe(df, use_container_width=True, hide_index=True)

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("Previous", disabled=st.session_state.doc_page == 0):
            st.session_state.doc_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f"Page **{st.session_state.doc_page + 1}** of **{total_pages}**"
        )
    with col_next:
        if st.button("Next", disabled=st.session_state.doc_page >= total_pages - 1):
            st.session_state.doc_page += 1
            st.rerun()
else:
    st.info("No documents ingested yet. Use the API to ingest PDFs first.")
