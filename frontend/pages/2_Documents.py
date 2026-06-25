from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client
from icons import svg_to_img_tag
from style import inject_css

st.set_page_config(
    page_title="Documents - Space Mission Intelligence",
    page_icon="\U0001F6F0️",
    layout="wide",
)

inject_css()

# ── Page header ──
docs_icon = svg_to_img_tag("icon_docs", 52, 52)
st.markdown(
    f"""
    <div class="page-header">
        <div class="page-header-icon">{docs_icon}</div>
        <div class="page-header-text">
            <h1>Document Library</h1>
            <p class="page-header-sub">Browse all ingested mission documents and research papers</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

    st.markdown("<br>", unsafe_allow_html=True)

    df = pd.DataFrame(docs)
    df["ingested_at"] = pd.to_datetime(df["ingested_at"]).dt.strftime("%Y-%m-%d %H:%M")
    df["title"] = df["title"].str.replace("_", " ").str.title()
    df["source_type"] = df["source_type"].str.upper()
    df["mission_name"] = df["mission_name"].fillna("—")
    df = df[["title", "source_type", "mission_name", "chunk_count", "ingested_at"]]
    df.columns = ["Title", "Type", "Mission", "Chunks", "Ingested"]

    st.dataframe(df, width="stretch", hide_index=True)

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Previous", disabled=st.session_state.doc_page == 0, width="stretch"):
            st.session_state.doc_page -= 1
            st.rerun()
    with col_info:
        st.markdown(
            f'<div style="text-align:center; padding:8px; color:#999; font-family:Orbitron,sans-serif; font-size:0.85em;">'
            f"Page {st.session_state.doc_page + 1} of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next →", disabled=st.session_state.doc_page >= total_pages - 1, width="stretch"):
            st.session_state.doc_page += 1
            st.rerun()
else:
    empty_icon = svg_to_img_tag("icon_docs", 64, 64)
    st.markdown(
        f"""
        <div style="text-align:center; padding:60px 20px; color:#888;">
            <div style="margin-bottom:16px;">{empty_icon}</div>
            <div style="font-size:1.1em;">No documents ingested yet</div>
            <div style="font-size:0.9em; margin-top:8px; color:#666;">Use the API to ingest PDFs first</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
