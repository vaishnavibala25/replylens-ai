"""
evaluation_service.py
----------------------
The ReplyLens Quality Score.

This module intentionally keeps two evaluation mechanisms SEPARATE and
clearly labeled, rather than blending them into one opaque number:

1. Objective checks — cheap, deterministic, rule-based signals that do not
   depend on any LLM call (empty response, excessive length, missing
   greeting/sign-off, unsupported numeric claims, repeated phrases, TF-IDF
   relevance to the source email, order-ID/entity consistency). These are
   fast, free, and 100% reproducible, but shallow — they can't tell whether
   a reply is *actually* correct.

2. LLM-as-judge assessment — an LLM scores six weighted dimensions
   (intent alignment, resolution/actionability, factual consistency,
   completeness, tone, reference alignment) with structured JSON reasoning.
   This captures nuance the objective checks can't, but is itself an LLM
   opinion and can be inconsistent or biased — hence dimension F) reference
   alignment anchors it against real historical examples, and hence the
   separate human-calibration step in evaluation/metrics.py + benchmark.py.

The final weighted overall score is ALWAYS computed here in Python from the
six dimension scores, never trusted directly from the LLM's own output, per
the challenge's explicit requirement.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

from services.llm_service import LLMService
from utils.helpers import normalize_text, safe_json_parse, extract_numbers, extract_order_ids

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluation_prompt.txt"

# Weights per the challenge specification (must sum to 1.0)
WEIGHTS = {
    "intent_alignment": 0.20,
    "resolution_actionability": 0.25,
    "factual_consistency": 0.20,
    "completeness": 0.15,
    "tone_professionalism": 0.10,
    "reference_alignment": 0.10,
}

DIMENSION_LABELS = {
    "intent_alignment": "Intent Alignment",
    "resolution_actionability": "Resolution / Actionability",
    "factual_consistency": "Factual Consistency",
    "completeness": "Completeness",
    "tone_professionalism": "Tone & Professionalism",
    "reference_alignment": "Reference Alignment",
}

GREETING_PATTERNS = [r"\bhi\b", r"\bhello\b", r"\bdear\b", r"\bhey\b"]
SIGNOFF_PATTERNS = [r"\bregards\b", r"\bbest\b", r"\bthank you\b", r"\bthanks\b", r"\bsincerely\b"]


# ---------------------------------------------------------------------------
# 1) OBJECTIVE / DETERMINISTIC CHECKS
# ---------------------------------------------------------------------------

def run_objective_checks(incoming_email: str, generated_response: str,
                         retrieved_examples: list[dict] | None = None) -> dict:
    """Returns a dict of individually-labeled deterministic checks, no LLM involved."""
    checks = {}
    resp = generated_response or ""
    resp_norm = normalize_text(resp)
    word_count = len(resp_norm.split())

    checks["not_empty"] = {
        "passed": word_count > 0,
        "detail": "Response is empty." if word_count == 0 else f"{word_count} words.",
    }

    checks["reasonable_length"] = {
        "passed": 5 <= word_count <= 350,
        "detail": f"{word_count} words (expected roughly 5-350).",
    }

    has_greeting = any(re.search(p, resp_norm, re.IGNORECASE) for p in GREETING_PATTERNS)
    checks["has_greeting"] = {
        "passed": has_greeting,
        "detail": "Greeting detected." if has_greeting else "No clear greeting found.",
    }

    has_signoff = any(re.search(p, resp_norm, re.IGNORECASE) for p in SIGNOFF_PATTERNS)
    checks["has_signoff"] = {
        "passed": has_signoff,
        "detail": "Sign-off/closing detected." if has_signoff else "No clear sign-off found.",
    }

    # Repeated phrase detection: look for any 4+ word sequence repeated verbatim.
    words = resp_norm.split()
    seen_ngrams = set()
    repeated = False
    n = 4
    for i in range(len(words) - n):
        gram = " ".join(words[i:i + n])
        if gram in seen_ngrams:
            repeated = True
            break
        seen_ngrams.add(gram)
    checks["no_repeated_phrases"] = {
        "passed": not repeated,
        "detail": "Repeated phrase detected." if repeated else "No significant repetition found.",
    }

    # Unsupported numeric claims: numbers in the response not present anywhere
    # in the incoming email. A rough but useful hallucination signal for
    # amounts/dates/IDs the model should not be inventing.
    reference_text = " ".join(
        f"{example.get('incoming_email', '')} {example.get('historical_reply', '')}"
        for example in (retrieved_examples or [])
    )
    supported_text = f"{incoming_email} {reference_text}"
    email_numbers = extract_numbers(supported_text)
    response_numbers = extract_numbers(resp)
    unsupported_numbers = response_numbers - email_numbers
    checks["no_unsupported_numbers"] = {
        "passed": len(unsupported_numbers) == 0,
        "detail": (
            "No numeric values in the response beyond what's in the email."
            if not unsupported_numbers
            else f"Numbers in response not found in the email: {sorted(unsupported_numbers)}"
        ),
    }

    # Order-ID / entity consistency: if the email mentions an order ID, the
    # response should either reference the same one or none at all — never
    # a different, invented order ID.
    email_orders = extract_order_ids(incoming_email)
    response_orders = extract_order_ids(resp)
    supported_orders = extract_order_ids(supported_text)
    invented_orders = response_orders - supported_orders
    checks["order_id_consistency"] = {
        "passed": len(invented_orders) == 0,
        "detail": (
            "No mismatched order IDs."
            if not invented_orders
            else f"Response references order ID(s) not in the email: {sorted(invented_orders)}"
        ),
    }

    response_dates = set(re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b|\b(?:today|tomorrow|yesterday)\b",
        resp, re.IGNORECASE,
    ))
    supported_dates = set(re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b|\b(?:today|tomorrow|yesterday)\b",
        supported_text, re.IGNORECASE,
    ))
    unsupported_dates = {date.lower() for date in response_dates} - {
        date.lower() for date in supported_dates
    }
    checks["no_unsupported_dates"] = {
        "passed": not unsupported_dates,
        "detail": (
            "No unsupported dates or relative time claims."
            if not unsupported_dates
            else f"Dates or time claims not supported by the email/examples: {sorted(unsupported_dates)}"
        ),
    }

    passed_count = sum(1 for c in checks.values() if c["passed"])
    checks["_summary"] = {"passed": passed_count, "total": len(checks)}
    return checks


def semantic_relevance_score(incoming_email: str, generated_response: str) -> float:
    """
    TF-IDF cosine similarity between the incoming email and the response, as
    a cheap objective proxy for topical relevance (not quality). Returns 0-1.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    a, b = normalize_text(incoming_email), normalize_text(generated_response)
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english").fit([a, b])
        matrix = vec.transform([a, b])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        # can happen if both texts are entirely stopwords / too short
        return 0.0


# ---------------------------------------------------------------------------
# 2) LLM-AS-JUDGE
# ---------------------------------------------------------------------------

def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "(No historical examples were retrieved.)"
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i} (similarity: {ex['similarity']}): "
            f"Email: {ex['incoming_email']} | Reply: {ex['historical_reply']}"
        )
    return "\n".join(blocks)


DEFAULT_DIMENSION = {"score": 50, "reason": "Judge output could not be parsed; default applied.", "issues": []}


def run_llm_judge(incoming_email: str, retrieved_examples: list[dict],
                   generated_response: str, llm_service: LLMService) -> dict:
    """
    Returns a dict with per-dimension {score, reason, issues}, plus summary/
    strengths/improvements, plus a `judge_ok` flag and optional `judge_error`.
    Never raises — falls back to neutral defaults on any failure so the UI
    stays usable.
    """
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = prompt_template.format(
        incoming_email=incoming_email.strip(),
        retrieved_examples=_format_examples(retrieved_examples),
        generated_response=generated_response.strip(),
    )
    system_prompt = (
        "You are a strict, impartial evaluator. Always respond with valid JSON "
        "only, matching exactly the schema you were given."
    )

    if not llm_service.is_configured:
        objective = run_objective_checks(incoming_email, generated_response, retrieved_examples)
        return _deterministic_demo_judge(incoming_email, retrieved_examples, generated_response, objective)

    result = llm_service.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=900)
    if not result.ok:
        objective = run_objective_checks(incoming_email, generated_response, retrieved_examples)
        out = _deterministic_demo_judge(incoming_email, retrieved_examples, generated_response, objective)
        out["judge_ok"] = False
        out["judge_error"] = result.error
        return out

    parsed = safe_json_parse(result.text)
    if parsed is None:
        objective = run_objective_checks(incoming_email, generated_response, retrieved_examples)
        out = _deterministic_demo_judge(incoming_email, retrieved_examples, generated_response, objective)
        out["judge_ok"] = False
        out["judge_error"] = "Could not parse JSON from LLM judge output."
        return out

    # Validate & fill in any missing/malformed dimension with a safe default.
    for dim in WEIGHTS:
        val = parsed.get(dim)
        if not isinstance(val, dict) or "score" not in val:
            parsed[dim] = dict(DEFAULT_DIMENSION)
        else:
            try:
                parsed[dim]["score"] = max(0, min(100, int(val["score"])))
            except (TypeError, ValueError):
                parsed[dim]["score"] = 50
            parsed[dim].setdefault("reason", "")
            parsed[dim].setdefault("issues", [])

    parsed.setdefault("summary", "")
    parsed.setdefault("strengths", [])
    parsed.setdefault("improvements", [])
    parsed["judge_ok"] = True
    parsed["judge_error"] = None
    return parsed


INTENT_PROFILES = {
    "refund": ("refund", "refund request", "return", "money back"),
    "duplicate_charge": ("charged twice", "duplicate charge", "billed twice", "twice"),
    "subscription_cancellation": ("subscription", "recurring", "billed again"),
    "cancellation": ("cancel order", "cancel my order", "no longer need"),
    "password_reset": ("password", "reset link", "reset email", "forgot my password"),
    "account_access": ("log into", "login", "locked out", "credentials"),
    "address_change": ("shipping address", "delivery address", "wrong address", "change address"),
    "exchange": ("exchange", "different size", "swap"),
    "damaged_product": ("damaged", "broken", "tear", "arrived damaged"),
    "delayed_delivery": ("delayed", "hasn't shown up", "late", "past its estimated"),
    "order_status": ("order status", "order update", "checking in", "on track"),
    "billing_issue": ("billing error", "charge", "invoice", "statement", "billing"),
    "technical_support": ("isn't connecting", "setup", "instructions", "app keeps", "trouble"),
    "product_information": ("warranty", "questions", "sizing", "materials", "before i order"),
    "complaint": ("unhappy", "complain", "frustrating", "experience", "not satisfied"),
}

ACTION_PROFILES = {
    "refund": ("refund", "refunded", "processed", "initiated", "money back"),
    "duplicate_charge": ("refund", "refunded", "duplicate", "billed twice"),
    "subscription_cancellation": ("cancelled", "canceled", "no further charges", "confirmation"),
    "cancellation": ("cancelled", "canceled", "confirmation", "no charges"),
    "password_reset": ("reset", "resent", "check", "spam", "inbox"),
    "account_access": ("unlocked", "sign in", "log in", "forgot password"),
    "address_change": ("updated", "change", "correct", "shipping address"),
    "exchange": ("exchange", "replacement", "return label", "swap"),
    "damaged_product": ("replacement", "refund", "options", "damaged"),
    "delayed_delivery": ("checked", "escalated", "follow up", "update"),
    "order_status": ("processed", "ship", "tracking", "status", "update"),
    "billing_issue": ("reviewed", "corrected", "clarify", "breakdown", "billing"),
    "technical_support": ("try", "reset", "update", "reconnect", "help"),
    "product_information": ("warranty", "details", "specs", "questions", "information"),
    "complaint": ("sorry", "escalated", "logged", "follow up", "feedback"),
}

RESPONSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "that", "the",
    "this", "to", "we", "will", "with", "you", "your", "hi", "hello", "thanks", "thank",
}


def _text_contains(text: str, phrases: tuple[str, ...]) -> int:
    normalized = normalize_text(text)
    return sum(phrase in normalized for phrase in phrases)


def _infer_intent(text: str) -> str:
    scores = {intent: _text_contains(text, phrases) for intent, phrases in INTENT_PROFILES.items()}
    best = max(scores.values(), default=0)
    return max(scores, key=scores.get) if best else "unknown"


def _intent_matches(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    related = {
        "duplicate_charge": {"refund", "billing_issue"},
        "refund": {"duplicate_charge", "damaged_product"},
        "cancellation": {"subscription_cancellation"},
        "subscription_cancellation": {"cancellation"},
        "damaged_product": {"refund", "exchange"},
        "exchange": {"damaged_product"},
        "account_access": {"password_reset"},
        "password_reset": {"account_access"},
        "order_status": {"delayed_delivery"},
        "delayed_delivery": {"order_status"},
    }
    return actual in related.get(expected, set())


def _reference_similarity(response: str, references: list[str]) -> float:
    if not references:
        return 0.0
    response_words = {word for word in re.findall(r"[a-z0-9]+", normalize_text(response))
                      if word not in RESPONSE_STOPWORDS and len(word) > 2}
    if not response_words:
        return 0.0
    scores = []
    for reference in references:
        reference_words = {word for word in re.findall(r"[a-z0-9]+", normalize_text(reference))
                           if word not in RESPONSE_STOPWORDS and len(word) > 2}
        overlap = len(response_words & reference_words) / max(1, len(response_words | reference_words))
        sequence = SequenceMatcher(None, normalize_text(response), normalize_text(reference)).ratio()
        scores.append(max(overlap, sequence * 0.75))
    return max(scores)


def _deterministic_demo_judge(incoming_email: str, retrieved_examples: list[dict],
                              generated_response: str, objective: dict) -> dict:
    """Score Demo Mode from reproducible evidence instead of a fixed placeholder."""
    response = normalize_text(generated_response)
    intent = _infer_intent(incoming_email)
    response_intent = _infer_intent(response)
    references = [example.get("historical_reply", "") for example in retrieved_examples]
    source_words = set(re.findall(r"[a-z0-9]+", normalize_text(incoming_email)))
    response_words = set(re.findall(r"[a-z0-9]+", response))
    topical_overlap = len(source_words & response_words) / max(1, len(source_words))
    action_hits = _text_contains(response, ACTION_PROFILES.get(intent, ()))
    request_terms = [phrase for phrase in INTENT_PROFILES.get(intent, ())
                     if phrase in normalize_text(incoming_email)]
    covered_terms = sum(term in response for term in request_terms)
    requested_concepts = []
    if re.search(r"\brefund|money back", incoming_email, re.IGNORECASE):
        requested_concepts.append(("refund", ("refund", "refunded", "money back")))
    if re.search(r"\bcancel|cancellation", incoming_email, re.IGNORECASE):
        requested_concepts.append(("cancellation", ("cancel", "cancelled", "canceled")))
    if re.search(r"\bnot billed|billed again|future charge|charge", incoming_email, re.IGNORECASE):
        requested_concepts.append(("billing outcome", ("charge", "billed", "billing")))
    covered_concepts = sum(
        any(term in response for term in terms) for _, terms in requested_concepts
    )
    has_tone = any(re.search(pattern, response) for pattern in GREETING_PATTERNS) and any(
        re.search(pattern, response) for pattern in SIGNOFF_PATTERNS
    )
    tone_score = 85 if has_tone else 65
    if any(word in response for word in ("sorry", "happy to help", "appreciate", "thanks")):
        tone_score = min(100, tone_score + 10)
    intent_score = 92 if intent != "unknown" and _intent_matches(intent, response_intent) else 25
    if intent == "unknown":
        intent_score = min(80, 45 + int(topical_overlap * 40))
    action_score = min(100, 25 + action_hits * 18 + int(topical_overlap * 25)) if action_hits else 15
    factual_score = max(0, 100
                        - 35 * (not objective["no_unsupported_numbers"]["passed"])
                        - 30 * (not objective["no_unsupported_dates"]["passed"])
                        - 30 * (not objective["order_id_consistency"]["passed"]))
    completeness_score = min(100, 40 + covered_terms * 12 + covered_concepts * 18 + (15 if action_hits else 0))
    if intent == "unknown":
        completeness_score = 45
    reference_score = min(100, 35 + int(_reference_similarity(response, references) * 80)) if references else 50
    if intent != "unknown" and not _intent_matches(intent, response_intent):
        intent_score, action_score, completeness_score = 15, min(action_score, 20), min(completeness_score, 30)
    if not response:
        intent_score = action_score = completeness_score = tone_score = reference_score = 0
    dimensions = {
        "intent_alignment": (intent_score, "Response intent matches the customer request."),
        "resolution_actionability": (action_score, "Response includes concrete resolution or next steps."),
        "factual_consistency": (factual_score, "Claims, numbers, dates, and identifiers are checked against available evidence."),
        "completeness": (completeness_score, "Response covers the request details detected in the email."),
        "tone_professionalism": (tone_score, "Response uses a professional and appropriately empathetic tone."),
        "reference_alignment": (reference_score, "Response is compared with retrieved examples by substance and wording."),
    }
    out = {dim: {"score": int(score), "reason": f"[Deterministic Demo Evaluation] {reason}", "issues": []}
           for dim, (score, reason) in dimensions.items()}
    for name, check in objective.items():
        if name == "_summary" or check["passed"]:
            continue
        target = "factual_consistency" if name in {"no_unsupported_numbers", "no_unsupported_dates", "order_id_consistency"} else "completeness"
        out[target]["issues"].append(check["detail"])
    if intent != "unknown" and not _intent_matches(intent, response_intent):
        out["intent_alignment"]["issues"].append("The response appears to address a different intent.")
    out["summary"] = "[Deterministic Demo Evaluation] Scores are computed from the email, response, objective checks, and retrieved examples; they are not an LLM judge."
    out["strengths"] = [dim.replace("_", " ") for dim, (score, _) in dimensions.items() if score >= 80]
    out["improvements"] = [check["detail"] for name, check in objective.items()
                           if name != "_summary" and not check["passed"]]
    out["judge_ok"] = False
    out["judge_error"] = None
    return out


# ---------------------------------------------------------------------------
# 3) WEIGHTED OVERALL SCORE (computed here, never trusted from the LLM)
# ---------------------------------------------------------------------------

def compute_weighted_score(judge_result: dict) -> float:
    total = 0.0
    for dim, weight in WEIGHTS.items():
        score = judge_result.get(dim, {}).get("score", 0)
        total += score * weight
    return round(total, 1)


def evaluate_response(incoming_email: str, retrieved_examples: list[dict],
                       generated_response: str, llm_service: LLMService) -> dict:
    """Full ReplyLens evaluation: objective checks + LLM judge + weighted score."""
    objective = run_objective_checks(incoming_email, generated_response, retrieved_examples)
    relevance = semantic_relevance_score(incoming_email, generated_response)
    judge = run_llm_judge(incoming_email, retrieved_examples, generated_response, llm_service)
    overall = compute_weighted_score(judge)

    return {
        "overall_score": overall,
        "dimensions": {dim: judge[dim] for dim in WEIGHTS},
        "weights": WEIGHTS,
        "dimension_labels": DIMENSION_LABELS,
        "summary": judge.get("summary", ""),
        "strengths": judge.get("strengths", []),
        "improvements": judge.get("improvements", []),
        "objective_checks": objective,
        "semantic_relevance": round(relevance, 3),
        "judge_ok": judge.get("judge_ok", False),
        "judge_error": judge.get("judge_error"),
    }
