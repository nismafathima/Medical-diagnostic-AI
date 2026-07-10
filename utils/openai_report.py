# utils/openai_report.py
# Optional OpenAI (GPT) powered diagnostic summary generator.
#
# This module is fully OPTIONAL and additive: the app's original local
# RAG pipeline (FAISS + Flan-T5, see report_rag.py) keeps working exactly
# as before with zero API key required. This module simply gives the user
# a second, higher-quality option for generating the diagnostic summary
# when they choose to supply their own OpenAI API key — useful both as a
# genuinely nicer report writer and as a hands-on way to practice wiring
# a real LLM API key into a project (a common resume/interview talking
# point: "I integrated the OpenAI API end-to-end, including secret
# handling, error handling, and graceful fallback").
#
# SECURITY NOTES:
# - The API key is only ever held in memory (st.session_state) for the
#   duration of the browser session. It is never written to disk, logged,
#   or sent anywhere other than https://api.openai.com.
# - Prefer setting the key via environment variable OPENAI_API_KEY or
#   Streamlit secrets (.streamlit/secrets.toml -> OPENAI_API_KEY) for any
#   shared/deployed instance, rather than pasting it into the sidebar.

from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIReportError(Exception):
    """Raised for any recoverable OpenAI-related failure.

    The caller (app.py) is expected to catch this and fall back to the
    local Flan-T5 pipeline rather than crashing the app.
    """


def resolve_api_key(user_supplied_key: Optional[str] = None) -> Optional[str]:
    """Resolve an OpenAI API key from (in priority order):
    1. A key the user pasted into the UI this session.
    2. Streamlit secrets (``st.secrets["OPENAI_API_KEY"]``), if configured.
    3. The ``OPENAI_API_KEY`` environment variable.
    Returns None if no key is available from any source.
    """
    if user_supplied_key:
        return user_supplied_key.strip()

    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        # st.secrets raises if no secrets.toml exists at all — that's fine,
        # it just means this source isn't configured.
        pass

    env_key = os.environ.get("OPENAI_API_KEY")
    return env_key.strip() if env_key else None


def generate_diagnostic_report_openai(
    patient_data: dict,
    context: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Generate a diagnostic summary using the OpenAI Chat Completions API.

    Raises OpenAIReportError on any failure (missing/invalid key, network
    error, rate limit, etc.) so the caller can fall back to the local model.
    """
    if not api_key:
        raise OpenAIReportError(
            "No OpenAI API key configured. Add one in the sidebar, or set "
            "the OPENAI_API_KEY environment variable / Streamlit secret."
        )

    try:
        from openai import OpenAI
        from openai import (
            AuthenticationError,
            RateLimitError,
            APIConnectionError,
            APIError,
        )
    except ImportError as exc:
        raise OpenAIReportError(
            "The 'openai' package is not installed. Run: pip install openai"
        ) from exc

    xray_confidence = patient_data.get("xray_confidence")
    confidence_str = (
        f"{xray_confidence:.0%}" if isinstance(xray_confidence, (int, float)) else "N/A"
    )

    system_prompt = (
        "You are a careful clinical documentation assistant. Write concise, "
        "professional diagnostic summaries for a demo medical AI app. Always "
        "end with a note that this is AI-generated and requires physician "
        "confirmation. Never claim certainty; use hedged clinical language."
    )

    user_prompt = f"""Patient: {patient_data.get('name', 'Unknown')}, \
Age: {patient_data.get('age', 'N/A')}, Gender: {patient_data.get('gender', 'N/A')}
X-ray Finding: {patient_data.get('xray_result', 'N/A')} (Confidence: {confidence_str})
Symptom Analysis: {patient_data.get('top_symptom', 'N/A')}
Vitals Risk: {patient_data.get('vitals_risk', 'N/A')}

Relevant medical reference context:
{context[:1200]}

Write a 3-5 sentence clinical diagnostic summary covering the key findings
and a recommended next step. End with a physician-confirmation disclaimer."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=350,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except AuthenticationError as exc:
        raise OpenAIReportError(
            "OpenAI rejected the API key (invalid or revoked). Check the key "
            "in the sidebar and try again."
        ) from exc
    except RateLimitError as exc:
        raise OpenAIReportError(
            "OpenAI rate limit or quota exceeded. Wait a moment and retry, "
            "or check your account's usage limits."
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIReportError(
            "Could not reach the OpenAI API (network issue). Check your "
            "internet connection and try again."
        ) from exc
    except APIError as exc:
        raise OpenAIReportError(f"OpenAI API returned an error: {exc}") from exc
