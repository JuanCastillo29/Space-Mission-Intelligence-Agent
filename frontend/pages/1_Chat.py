from __future__ import annotations

from pathlib import Path

import streamlit as st

import api_client
from icons import svg_to_img_tag
from style import inject_chat_js, inject_css

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_USER_AVATAR = str(_ASSETS / "avatar_user.svg")
_BOT_AVATAR = str(_ASSETS / "avatar_bot.svg")

# Pre-render inline icons for metadata pills (16px for inline use)
_ICON_MODEL = svg_to_img_tag("icon_model", 14, 14)
_ICON_RADAR = svg_to_img_tag("icon_radar", 14, 14)
_ICON_BOLT = svg_to_img_tag("icon_bolt", 14, 14)
_ICON_TOKENS = svg_to_img_tag("icon_tokens", 14, 14)
_ICON_SOURCE = svg_to_img_tag("icon_source", 16, 16)
_ICON_FOLDER = svg_to_img_tag("icon_folder", 14, 14)

st.set_page_config(
    page_title="Chat - Space Mission Intelligence",
    page_icon="\U0001f6f0️",
    layout="wide",
)

inject_css()

# ── Page header ──
chat_icon = svg_to_img_tag("icon_chat", 52, 52)
st.markdown(
    f"""
    <div class="page-header">
        <div class="page-header-icon">{chat_icon}</div>
        <div class="page-header-text">
            <h1>Mission Chat</h1>
            <p class="page-header-sub">Ask anything about space missions — answers are grounded in real documents</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()


def _render_metadata(meta: dict) -> None:
    parts = []
    if meta.get("model_name"):
        parts.append(
            f'<span class="chat-meta">{_ICON_MODEL} {meta["model_name"]}</span>'
        )
    if meta.get("query_type"):
        parts.append(
            f'<span class="chat-meta">{_ICON_RADAR} {meta["query_type"]}</span>'
        )
    latency = meta.get("latency_ms")
    if latency:
        parts.append(f'<span class="chat-meta">{_ICON_BOLT} {latency:.0f}ms</span>')
    prompt_tok = meta.get("prompt_tokens")
    comp_tok = meta.get("completion_tokens")
    if prompt_tok and comp_tok:
        parts.append(
            f'<span class="chat-meta">{_ICON_TOKENS} {prompt_tok}+{comp_tok} tokens</span>'
        )
    if parts:
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _render_citations(citations: list[dict]) -> None:
    with st.expander(f"Sources ({len(citations)} references)", expanded=False):
        for cite in citations:
            title = cite["source_title"].replace("_", " ").title()
            section = cite.get("section_path")
            chip = f"{_ICON_SOURCE} **[{cite['ref_index']}]** {title}"
            st.markdown(chip, unsafe_allow_html=True)
            if section:
                st.markdown(
                    f'<div style="margin-left:22px; font-size:0.82em; color:#888;">{_ICON_FOLDER} {section}</div>',
                    unsafe_allow_html=True,
                )


# ── Render history ──
for msg in st.session_state.messages:
    avatar = _USER_AVATAR if msg["role"] == "user" else _BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("citations"):
                _render_citations(msg["citations"])
            if msg.get("metadata"):
                _render_metadata(msg["metadata"])

# ── Input ──
question = st.chat_input("Ask about space missions...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=_BOT_AVATAR):
        with st.spinner("Searching mission documents..."):
            try:
                resp = api_client.submit_query(question)
            except Exception as exc:
                st.error(f"Failed to get a response: {exc}")
                st.stop()

            st.markdown(resp["answer"])

            if resp.get("citations"):
                _render_citations(resp["citations"])

            meta = {
                "model_name": resp.get("model_name"),
                "query_type": resp.get("query_type"),
                "latency_ms": resp.get("latency_ms"),
                "prompt_tokens": resp.get("prompt_tokens"),
                "completion_tokens": resp.get("completion_tokens"),
            }
            _render_metadata(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp["answer"],
            "citations": resp.get("citations"),
            "sources_section": resp.get("sources_section"),
            "metadata": meta,
        }
    )

# Inject JS at the bottom so it runs after chat messages are rendered
inject_chat_js()
