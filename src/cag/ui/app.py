"""
Streamlit UI for the generic CAG assistant.
"""
from __future__ import annotations

import json

import streamlit as st

from cag.config import settings

st.set_page_config(
    page_title="CAG — Cognitive Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin: 8px 0;
    }

    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
    }

    .main-header {
        background: linear-gradient(135deg, #1e40af, #0f766e);
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown("## 🧠 CAG v0.1")
    st.markdown("*Cognitive Augmented Generation*")
    st.divider()

    st.markdown("### Active Configuration")
    st.markdown(f"**LLM Provider:** `{settings.llm_provider.value}`")
    st.markdown(f"**Model:** `{settings.active_model_id}`")
    st.markdown(f"**Vector DB:** `{settings.vector_db.value}`")
    st.divider()

    st.markdown("### Pipeline Thresholds")
    rel_threshold = st.slider(
        "Relevance Threshold",
        0.0,
        1.0,
        settings.relevance_threshold,
        0.05,
        help="Minimum acceptable evidence relevance",
    )
    conf_threshold = st.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        settings.confidence_threshold,
        0.05,
        help="Minimum confidence required to avoid escalation",
    )
    hall_threshold = st.slider(
        "Hallucination Threshold",
        0.0,
        1.0,
        settings.hallucination_threshold,
        0.05,
        help="Maximum acceptable hallucination risk",
    )

    st.divider()

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

    if st.session_state.last_result:
        result = st.session_state.last_result
        st.divider()
        st.markdown("### Latest Response")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result.get('confidence', 0.0):.0%}")
        with col2:
            st.metric("Hallucination Risk", f"{result.get('hallucination_risk', 0.0):.0%}")

        st.metric("Query Type", result.get("query_type", "N/A"))
        st.metric("Relevance", f"{result.get('relevance_score', 0.0):.0%}")

        trace = result.get("node_trace", [])
        if trace:
            st.markdown("**Node Path:**")
            st.markdown(" -> ".join(f"`{node}`" for node in trace))

        citations = result.get("citations", [])
        if citations:
            st.markdown(f"**Sources ({len(citations)}):**")
            for index, citation in enumerate(citations[:5], 1):
                st.markdown(f"{index}. `{citation.get('source', 'N/A')}`")

        st.divider()
        st.download_button(
            "Export Session",
            data=json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
            file_name="cag_session.json",
            mime="application/json",
            use_container_width=True,
        )

st.markdown(
    """
<div class="main-header">
    <h1 style="margin:0; font-size:1.5rem;">🧠 CAG — Cognitive Assistant</h1>
    <p style="margin:4px 0 0; opacity:0.85; font-size:0.9rem;">
        A grounded workspace for documents, procedures, troubleshooting, and research
    </p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            """
Hello. I am your document intelligence assistant built on **CAG v0.1**.

I can help with:
- **Troubleshooting** — identify likely causes and next checks
- **Procedures** — explain supported steps in order
- **Configuration** — surface required fields, settings, and prerequisites
- **Knowledge retrieval** — summarize what the documentation actually says

**Example prompts:**
- *"What are the key responsibilities described in this guide?"*
- *"How do I complete this procedure step by step?"*
- *"Why might this workflow fail according to the documentation?"*
"""
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("meta"):
            meta = message["meta"]
            with st.expander("Technical Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{meta.get('confidence', 0):.0%}")
                col2.metric("Hallucination Risk", f"{meta.get('hallucination_risk', 0):.0%}")
                col3.metric("Type", meta.get("query_type", "N/A"))

                if meta.get("citations"):
                    st.markdown("**Citations:**")
                    for citation in meta["citations"]:
                        st.markdown(f"- `{citation.get('source', '')}` — {citation.get('text', '')[:100]}…")

                if meta.get("gaps"):
                    st.markdown("**Information Gaps:**")
                    for gap in meta["gaps"]:
                        st.markdown(f"- {gap}")


if prompt := st.chat_input("Ask a question about your documents or knowledge base..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                from cag.graph.graph import run_query

                settings.relevance_threshold = rel_threshold
                settings.confidence_threshold = conf_threshold
                settings.hallucination_threshold = hall_threshold

                result = run_query(
                    query=prompt,
                    conversation_history=st.session_state.history,
                )
                st.session_state.last_result = result
                st.session_state.history = result.get("conversation_history", [])

                answer = result.get("answer", "No answer generated.")

                if result.get("should_escalate"):
                    st.warning("This request should be reviewed by a human specialist.")

                st.markdown(answer)

                confidence = result.get("confidence", 0.0)
                hallucination = result.get("hallucination_risk", 0.0)

                if confidence > 0:
                    st.caption(
                        f"Confidence: {confidence:.0%} | "
                        f"Hallucination Risk: {hallucination:.0%} | "
                        f"Type: {result.get('query_type', 'N/A')}"
                    )

                meta = {
                    "confidence": confidence,
                    "hallucination_risk": hallucination,
                    "query_type": result.get("query_type", "N/A"),
                    "citations": result.get("citations", []),
                    "gaps": result.get("gaps", []),
                    "node_trace": result.get("node_trace", []),
                }
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )

            except Exception as exc:
                error_message = f"Processing error: {exc}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
