from __future__ import annotations

import streamlit as st

import api_client
from style import inject_css

st.set_page_config(
    page_title="Space Mission Intelligence",
    page_icon="\U0001F680",
    layout="wide",
)

inject_css()

with st.sidebar:
    st.title("\U0001F680 Space Mission Intelligence")
    st.caption("RAG-powered space mission Q&A")

    st.divider()

    health = api_client.check_health()
    status = health["status"]

    if status == "healthy":
        st.markdown(
            '<span class="status-dot green"></span> API: **healthy**',
            unsafe_allow_html=True,
        )
    elif status == "degraded":
        st.markdown(
            '<span class="status-dot amber"></span> API: **degraded**',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-dot red"></span> API: **unreachable**',
            unsafe_allow_html=True,
        )

    components = {
        "Database": health.get("database", False),
        "Embedder": health.get("embedder_loaded", False),
        "Reranker": health.get("reranker_loaded", False),
    }
    for name, ok in components.items():
        dot = "green" if ok else "red"
        st.markdown(
            f'<span class="status-dot {dot}"></span> {name}',
            unsafe_allow_html=True,
        )

    st.caption(f"Version: {health.get('version', 'unknown')}")

st.header("Welcome")
st.markdown(
    """
    Use the **Chat** page to ask questions about space missions.
    Answers are grounded in ingested documents with inline citations.

    Use the **Documents** page to browse all ingested documents.

    Select a page from the sidebar to get started.
    """
)
