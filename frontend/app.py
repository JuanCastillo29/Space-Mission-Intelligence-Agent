from __future__ import annotations

import streamlit as st

import api_client
from icons import svg_to_img_tag
from style import inject_css

st.set_page_config(
    page_title="Space Mission Intelligence",
    page_icon="\U0001F6F0️",
    layout="wide",
)

inject_css()

# ── Telemetry-style status formatting ──
_STATUS_MAP = {
    "green": {
        "api": "[STATUS: NOMINAL]",
        "on": "[LINK_ACTIVE]",
        "loaded": "[MODULE_LOADED]",
        "cls": "online",
    },
    "red": {
        "api": "[STATUS: OFFLINE]",
        "on": "[LINK_SEVERED]",
        "loaded": "[NOT_LOADED]",
        "cls": "offline",
    },
    "amber": {
        "api": "[STATUS: DEGRADED]",
        "on": "[LINK_UNSTABLE]",
        "loaded": "[PARTIAL]",
        "cls": "degraded",
    },
}


def _fmt(dot: str, kind: str) -> tuple[str, str]:
    m = _STATUS_MAP.get(dot, _STATUS_MAP["red"])
    return m[kind], m["cls"]


# ── Sidebar ──
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">
                <span class="sidebar-brand-acronym">SMI</span>
            </div>
            <div class="sidebar-brand-sub">Space Mission Intel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    health = api_client.check_health()
    status = health["status"]
    api_dot = "green" if status == "healthy" else ("amber" if status == "degraded" else "red")
    db_dot = "green" if health.get("database") else "red"
    emb_dot = "green" if health.get("embedder_loaded") else "red"
    rer_dot = "green" if health.get("reranker_loaded") else "red"

    components = [
        ("API", api_dot, "api"),
        ("Database", db_dot, "on"),
        ("Embedder", emb_dot, "loaded"),
        ("Reranker", rer_dot, "loaded"),
    ]

    rows_html = ""
    for label, dot, kind in components:
        value_text, value_cls = _fmt(dot, kind)
        rows_html += (
            f'<div class="status-row">'
            f'<span class="status-dot {dot}"></span>'
            f'<span class="status-label">{label}</span>'
            f'<span class="status-value {value_cls}">{value_text}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div class="status-panel">
            <div class="status-panel-title">&gt; System Status</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    version = health.get("version", "unknown")
    st.markdown(
        f'<div style="text-align:center; margin-top:8px;">'
        f'<span class="version-badge">BUILD {version}</span></div>',
        unsafe_allow_html=True,
    )

# ── Hero ──
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Space Mission Intelligence</div>
        <div class="hero-subtitle">// AI-Powered Mission Document Analysis //</div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_icon = svg_to_img_tag("icon_chat", 56, 56)
search_icon = svg_to_img_tag("icon_search", 56, 56)
docs_icon = svg_to_img_tag("icon_docs", 56, 56)

# ── Dashboard frame with corner accents ──
st.markdown(
    """
    <div class="dashboard-frame">
        <span class="dashboard-corner-tr"></span>
        <span class="dashboard-corner-bl"></span>
        <span class="coord-stamp top-left">COORD // 03.14.15</span>
        <span class="coord-stamp bot-right">SEC // ALPHA-7</span>
        <div class="frame-label">// Active Modules //</div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="feature-card">
            <span class="corner-tr"></span>
            <span class="corner-bl"></span>
            <span class="feature-card-serial">SMI-MOD-001</span>
            <div class="feature-icon">{chat_icon}</div>
            <div class="feature-title">Mission Chat</div>
            <div class="feature-desc">
                Ask questions about ESA & NASA missions.
                Get cited answers grounded in real documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="feature-card">
            <span class="corner-tr"></span>
            <span class="corner-bl"></span>
            <span class="feature-card-serial">SMI-MOD-002</span>
            <div class="feature-icon">{search_icon}</div>
            <div class="feature-title">Hybrid Search</div>
            <div class="feature-desc">
                Semantic + keyword search with cross-encoder
                reranking and MMR diversity filtering.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="feature-card">
            <span class="corner-tr"></span>
            <span class="corner-bl"></span>
            <span class="feature-card-serial">SMI-MOD-003</span>
            <div class="feature-icon">{docs_icon}</div>
            <div class="feature-title">Document Library</div>
            <div class="feature-desc">
                Browse 40+ ingested mission documents
                from ESA, NASA, and research archives.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Close dashboard frame
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

_, center, _ = st.columns([1, 2, 1])
with center:
    st.markdown(
        '<div style="text-align:center; color:#8899AA; font-size:0.82em; font-family:\'Share Tech Mono\',monospace; letter-spacing:1px;">'
        '[ Select <span style="color:#4FC3F7;">CHAT</span> or '
        '<span style="color:#4FC3F7;">DOCUMENTS</span> from the sidebar to initialize ]'
        "</div>",
        unsafe_allow_html=True,
    )
