"""
app/streamlit_app.py
CampusGuide RAG — Streamlit web interface.

Run from the ElasticSearch/ root:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import types
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _load_cloud_secrets_into_environment() -> None:
    """Let the existing env-based configuration work with Streamlit Secrets."""
    try:
        for key in (
            "LLM_PROVIDER", "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL",
            "SEARCH_BACKEND", "ES_URL", "ES_USER", "ES_PASS", "ES_VERIFY_CERTS",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
        ):
            if key not in os.environ and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # Local runs normally have no st.secrets file; dotenv/env remains the source.
        pass


_load_cloud_secrets_into_environment()

# Streamlit Cloud can put the entrypoint directory (`app/`) on sys.path
# without also adding the repository root.  The application uses absolute
# `app.*` imports, so make the package root explicit for every launch mode.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.orchestrator import handle_turn

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CampusGuide RAG",
    page_icon="🎓",
    layout="centered",
)
st.title("CampusGuide RAG")
st.caption("Grounded institutional knowledge assistant · Policies · Tuition · Calendar · Contacts")

# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list = []
if "tuition_state" not in st.session_state:
    st.session_state.tuition_state: dict = {}
if "calendar_state" not in st.session_state:
    st.session_state.calendar_state: dict = {}

# ── Render existing conversation ──────────────────────────────────────────────

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Handle new input ──────────────────────────────────────────────────────────

if query := st.chat_input("Ask about policies, tuition, calendar, or contacts…"):

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        reply, domains, st.session_state.tuition_state, st.session_state.calendar_state, is_clar = handle_turn(
            query,
            st.session_state.history,
            st.session_state.tuition_state,
            st.session_state.calendar_state,
            stream=True,
        )

        if isinstance(reply, types.GeneratorType):
            reply = st.write_stream(reply)
        else:
            st.write(reply)

        if os.getenv("STREAMLIT_DEBUG") and domains:
            with st.expander("Debug", expanded=False):
                st.write(f"**Domains:** {domains}")
                st.write(f"**Reply preview:** {str(reply)[:200]}…")

    st.session_state.history.append({"role": "user",      "content": query})
    st.session_state.history.append({"role": "assistant", "content": reply})
