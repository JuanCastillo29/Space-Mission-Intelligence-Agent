from __future__ import annotations

import streamlit as st

import api_client
from style import inject_css

st.set_page_config(
    page_title="Chat - Space Mission Intelligence",
    page_icon="\U0001F680",
    layout="wide",
)

inject_css()

st.header("\U0001F4AC Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(
                f"Sources ({len(msg['citations'])} references)", expanded=False
            ):
                for cite in msg["citations"]:
                    st.markdown(f"**[{cite['ref_index']}]** {cite['source_title']}")
                    if cite.get("section_path"):
                        st.caption(f"Section: {cite['section_path']}")

        if msg["role"] == "assistant" and msg.get("metadata"):
            meta = msg["metadata"]
            cols = st.columns(4)
            cols[0].caption(f"Model: {meta.get('model_name', '—')}")
            cols[1].caption(f"Type: {meta.get('query_type', '—')}")
            latency = meta.get("latency_ms")
            cols[2].caption(f"Latency: {latency:.0f}ms" if latency else "Latency: —")
            prompt_tok = meta.get("prompt_tokens")
            comp_tok = meta.get("completion_tokens")
            if prompt_tok and comp_tok:
                cols[3].caption(f"Tokens: {prompt_tok}+{comp_tok}")
            else:
                cols[3].caption("Tokens: —")

question = st.chat_input("Ask about space missions...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                resp = api_client.submit_query(question)
            except Exception as exc:
                st.error(f"Failed to get a response: {exc}")
                st.stop()

            st.markdown(resp["answer"])

            if resp.get("citations"):
                with st.expander(
                    f"Sources ({len(resp['citations'])} references)", expanded=False
                ):
                    for cite in resp["citations"]:
                        st.markdown(
                            f"**[{cite['ref_index']}]** {cite['source_title']}"
                        )
                        if cite.get("section_path"):
                            st.caption(f"Section: {cite['section_path']}")

            meta = {
                "model_name": resp.get("model_name"),
                "query_type": resp.get("query_type"),
                "latency_ms": resp.get("latency_ms"),
                "prompt_tokens": resp.get("prompt_tokens"),
                "completion_tokens": resp.get("completion_tokens"),
            }
            cols = st.columns(4)
            cols[0].caption(f"Model: {meta.get('model_name', '—')}")
            cols[1].caption(f"Type: {meta.get('query_type', '—')}")
            latency = meta.get("latency_ms")
            cols[2].caption(f"Latency: {latency:.0f}ms" if latency else "Latency: —")
            prompt_tok = meta.get("prompt_tokens")
            comp_tok = meta.get("completion_tokens")
            if prompt_tok and comp_tok:
                cols[3].caption(f"Tokens: {prompt_tok}+{comp_tok}")
            else:
                cols[3].caption("Tokens: —")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp["answer"],
            "citations": resp.get("citations"),
            "sources_section": resp.get("sources_section"),
            "metadata": meta,
        }
    )
