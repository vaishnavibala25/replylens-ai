"""
generation_service.py
----------------------
Builds a grounded, few-shot prompt from retrieved historical examples and
calls the LLM service to produce a suggested response.

Demo Mode: if no LLM API key is configured, we don't crash — we return a
clearly-labeled deterministic response composed from the incoming email and
the retrieved category guidance.
"""

from pathlib import Path
import re

from services.llm_service import LLMService, LLMResponse
from utils.helpers import extract_order_ids

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "generation_prompt.txt"

SYSTEM_PROMPT = (
    "You are a helpful, honest customer support agent. You never fabricate "
    "facts, policies, prices, or commitments that are not supported by the "
    "information you were given."
)


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(No closely matching historical examples were found.)"
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i} (category: {ex['category']}, similarity: {ex['similarity']}):\n"
            f"  Customer email: {ex['incoming_email']}\n"
            f"  Historical reply: {ex['historical_reply']}"
        )
    return "\n\n".join(blocks)


def generate_response(incoming_email: str, retrieved_examples: list[dict],
                       llm_service: LLMService) -> dict:
    """Returns dict with keys: response_text, is_demo, error (optional)."""
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = prompt_template.format(
        incoming_email=incoming_email.strip(),
        retrieved_examples=_format_examples(retrieved_examples),
    )

    if not llm_service.is_configured:
        return {
            "response_text": _demo_response(incoming_email, retrieved_examples),
            "is_demo": True,
            "error": None,
        }

    result: LLMResponse = llm_service.chat(SYSTEM_PROMPT, user_prompt, temperature=0.4, max_tokens=400)
    if not result.ok:
        # Graceful fallback rather than crashing the UI.
        return {
            "response_text": _demo_response(incoming_email, retrieved_examples),
            "is_demo": True,
            "error": result.error,
        }

    return {"response_text": result.text.strip(), "is_demo": False, "error": None}


def _demo_response(incoming_email: str, retrieved_examples: list[dict]) -> str:
    """Compose a deterministic, grounded response without copying a reply example."""
    category = retrieved_examples[0].get("category", "general_help") if retrieved_examples else "general_help"
    order_ids = sorted(extract_order_ids(incoming_email))
    order_context = f" for {order_ids[0]}" if order_ids else ""
    timeframe = _supported_timeframe(retrieved_examples)
    timeframe_context = f" The usual timeframe in similar cases is {timeframe}." if timeframe else ""

    templates = {
        "refund_request": f"I understand you are requesting a refund{order_context}. We will review the request and confirm the next steps.{timeframe_context}",
        "duplicate_charge": f"I understand you are reporting a duplicate charge{order_context}. We will review the duplicate transaction and confirm the refund status.{timeframe_context}",
        "subscription_cancellation": "I understand you want to cancel your subscription and avoid future billing. We will confirm the cancellation status and any remaining billing details.",
        "cancellation": f"I understand you want to cancel the order{order_context}. We will check whether it has shipped and confirm what can be done.",
        "password_reset": "I understand that the password reset message has not arrived. Please check your inbox and spam folder, then request the link again; if it still does not arrive, support can investigate further.",
        "account_access": "I understand that you cannot access your account. Please try signing in again and use the password-reset option if needed; support can help investigate if access remains blocked.",
        "address_change": f"I understand you want to update the shipping address{order_context}. We will check the shipment status and confirm whether the address can still be changed.",
        "exchange_request": f"I understand you want to exchange the item{order_context}. We will confirm the available exchange steps and replacement options.",
        "damaged_product": f"I am sorry the item arrived damaged{order_context}. We will review the issue and confirm the available replacement or refund options.",
        "delayed_delivery": f"I am sorry your delivery is delayed{order_context}. We will check the shipment status and follow up with the latest update.",
        "order_status": f"I understand you are asking for an update on your order{order_context}. We will check its current status and share the next available update.",
        "billing_issue": f"I understand you are asking about a billing issue{order_context}. We will review the charge and explain the discrepancy using the available account details.",
        "technical_support": "I understand you are having trouble using the product. Please retry the relevant setup or connection step, and share what happens next so support can narrow down the cause.",
        "product_information": "I understand you want more information before purchasing. Please share the specific details you need, such as sizing, materials, or warranty coverage, so support can answer accurately.",
        "complaint": f"I am sorry this experience fell short{order_context}. We will record the concern and review the next appropriate support step.",
        "general_help": "I understand you are looking for help with this request. We will review the details provided and confirm the next appropriate step.",
    }
    response = templates.get(category, templates["general_help"])
    return (
        "[DETERMINISTIC DEMO EVALUATION — no live LLM call made]\n\n"
        f"Hi there, thanks for reaching out. {response}\n\n"
        "This offline response is composed from the email and the retrieved support category; "
        "configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL for live generation."
    )


def _supported_timeframe(retrieved_examples: list[dict]) -> str:
    """Extract a short timeframe from grounding without copying a full reply."""
    timeframe_pattern = re.compile(
        r"\b\d+\s*(?:-|to)\s*\d+\s+(?:business|working|calendar)\s+days?\b",
        re.IGNORECASE,
    )
    for example in retrieved_examples:
        match = timeframe_pattern.search(example.get("historical_reply", ""))
        if match:
            return match.group(0)
    return ""
