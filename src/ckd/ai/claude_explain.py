"""
Claude AI integration — generates a plain-English clinical explanation
of a CKD prediction using SHAP feature impacts.

The Anthropic SDK is used with claude-sonnet-4-6.
Set ANTHROPIC_API_KEY in your environment (or .env file).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ckd.config import ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a clinical decision-support assistant that helps clinicians understand \
machine-learning predictions for Chronic Kidney Disease (CKD).

When given a prediction result and SHAP feature importance values, you will:
1. Summarise the overall risk level clearly (1–2 sentences).
2. Explain the top 5 contributing factors in plain English — what each means clinically \
   and WHY it pushes the risk up or down.
3. Suggest 2–3 follow-up clinical actions appropriate for the risk level.
4. Add a disclaimer that this is an ML tool for decision-support only, not a diagnosis.

Use clear, professional language that a GP can understand.
Do NOT use markdown headers. Use short numbered paragraphs.
Keep the full response under 280 words.
"""


def _build_user_message(
    prediction: str,
    prob_ckd: float,
    shap_top: dict[str, float],
    patient_context: dict[str, Any] | None = None,
) -> str:
    """Format the user turn sent to Claude."""
    risk_pct = f"{prob_ckd * 100:.1f}%"
    shap_lines = "\n".join(
        f"  • {feat}: {val:+.4f} ({'↑ toward CKD' if val > 0 else '↓ away from CKD'})"
        for feat, val in list(shap_top.items())[:8]
    )
    ctx_lines = ""
    if patient_context:
        ctx_lines = "\nKey patient values:\n" + "\n".join(
            f"  {k}: {v}" for k, v in list(patient_context.items())[:10]
        )

    return (
        f"Prediction: {prediction.upper()} — CKD probability {risk_pct}\n\n"
        f"Top SHAP feature contributions (XGBoost model):\n{shap_lines}"
        f"{ctx_lines}\n\n"
        "Please provide a clinical explanation of this result."
    )


def explain(
    prediction: str,
    prob_ckd: float,
    shap_top: dict[str, float],
    patient_context: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> str:
    """
    Call Claude to generate a clinical narrative explanation.

    Parameters
    ----------
    prediction      : 'ckd' or 'notckd'
    prob_ckd        : probability in [0, 1]
    shap_top        : ordered {feature: shap_value} dict (largest first)
    patient_context : optional raw patient values for extra context
    api_key         : override ANTHROPIC_API_KEY env var

    Returns
    -------
    str — the AI-generated clinical explanation text
    """
    try:
        import anthropic
    except ImportError:
        return (
            "AI explanation unavailable: 'anthropic' package not installed. "
            "Run: pip install anthropic"
        )

    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return (
            "AI explanation unavailable: ANTHROPIC_API_KEY not set. "
            "Add it to your .env file."
        )

    user_msg = _build_user_message(prediction, prob_ckd, shap_top, patient_context)
    logger.debug("Sending explanation request to Claude (%s)", ANTHROPIC_MODEL)

    try:
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        logger.info("Claude explanation received (%d chars)", len(text))
        return text

    except anthropic.AuthenticationError:
        logger.error("Invalid Anthropic API key")
        return "AI explanation failed: invalid API key."
    except anthropic.RateLimitError:
        logger.warning("Anthropic rate limit hit")
        return "AI explanation unavailable right now (rate limit). Please try again shortly."
    except Exception as exc:
        logger.exception("Unexpected error from Claude API: %s", exc)
        return f"AI explanation error: {exc}"


def explain_batch(
    records: list[dict[str, Any]],
    api_key: str | None = None,
) -> list[str]:
    """Explain multiple predictions (sequential — respect rate limits)."""
    return [
        explain(
            r["prediction"], r["prob_ckd"], r["shap_top"],
            r.get("patient_context"), api_key,
        )
        for r in records
    ]
