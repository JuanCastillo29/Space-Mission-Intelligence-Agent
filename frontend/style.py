from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Citation badge */
.citation-badge {
    display: inline-block;
    background: #4FC3F7;
    color: #0E1117;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.8em;
    font-weight: 600;
    margin: 0 2px;
}

/* Status dots */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #66BB6A; }
.status-dot.red { background: #EF5350; }
.status-dot.amber { background: #FFA726; }

/* Compact metadata row */
.metadata-row {
    font-size: 0.78em;
    color: #9E9E9E;
    margin-top: 4px;
}

/* Sources expander tighter spacing */
div[data-testid="stExpander"] details summary {
    font-size: 0.9em;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
