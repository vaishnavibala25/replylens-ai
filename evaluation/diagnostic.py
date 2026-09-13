"""Runtime audit for deterministic Demo Evaluation and benchmark grounding."""

import pandas as pd

from evaluation.benchmark import run_benchmark
from services.evaluation_service import WEIGHTS, evaluate_response
from services.generation_service import generate_response
from services.llm_service import LLMService
from services.retrieval_service import RetrievalService


class OfflineLLM:
    is_configured = False


def print_case(name, email, response, retrieved, llm):
    result = evaluate_response(email, retrieved, response, llm)
    scores = {dimension: result["dimensions"][dimension]["score"] for dimension in WEIGHTS}
    print(f"{name}: {scores} overall={result['overall_score']}")
    if name in {"C", "D", "E"}:
        checks = result["objective_checks"]
        print(
            f"  factual checks: numbers={checks['no_unsupported_numbers']['passed']} "
            f"dates={checks['no_unsupported_dates']['passed']} "
            f"order_ids={checks['order_id_consistency']['passed']}"
        )


def main():
    llm = OfflineLLM()
    dataframe = pd.read_csv("data/emails.csv")
    retrieval = RetrievalService(dataframe)

    audit_email = "Hi, I was charged twice for order ORD-88221. Can you refund the duplicate charge?"
    audit_retrieved = retrieval.retrieve(audit_email, top_k=3)
    cases = {
        "A": (audit_email, "Hi there, I am sorry for the duplicate charge on order ORD-88221. I have refunded the extra charge; it should appear in 5-7 business days. Thanks."),
        "B": (audit_email, "Hi, please use the Forgot Password link to reset your password."),
        "C": (audit_email, "Hi, I refunded ₹15,999 and the money will arrive soon. Thanks."),
        "D": (audit_email, "Hi, I refunded the duplicate charge and the money will arrive tomorrow. Thanks."),
        "E": (audit_email, "Hi, I refunded the duplicate charge for order ORD-99999. Thanks."),
        "F": (audit_email, "Hi, thanks for your message. We are sorry to hear that."),
        "G": ("Hi, please cancel my subscription, refund the latest payment, and confirm I will not be charged again.", "Hi, your subscription has been cancelled."),
        "H": ("Hi, I need a refund for order ORD-48213.", "We have initiated your refund. It should be credited within approximately 5-7 working days."),
    }
    print("DIAGNOSTIC CASES")
    for name, (email, response) in cases.items():
        retrieved = audit_retrieved if name not in {"H"} else [{
            "similarity": 0.95,
            "incoming_email": email,
            "historical_reply": "Your refund has been processed and should arrive within 5-7 business days.",
        }]
        print_case(name, email, response, retrieved, llm)

    evaluation_rows = dataframe[dataframe["split"] == "evaluation"]
    reference_ids = set(dataframe.loc[dataframe["split"] == "reference", "id"])
    leaked = [item for item in audit_retrieved if item["id"] not in reference_ids]
    generated = generate_response(evaluation_rows.iloc[0]["incoming_email"], retrieval.retrieve(evaluation_rows.iloc[0]["incoming_email"]), llm)
    top_reference = retrieval.retrieve(evaluation_rows.iloc[0]["incoming_email"], top_k=1)[0]["historical_reply"]
    print("PIPELINE AUDIT")
    print(f"evaluation_rows={len(evaluation_rows)} reference_rows={len(reference_ids)} retrieval_leaks={len(leaked)}")
    print(f"demo_generation_is_demo={generated['is_demo']} copies_top_reference={top_reference in generated['response_text']}")
    print(f"demo_generated_response={generated['response_text'][:240]!r}")

    benchmark = run_benchmark(llm)
    factual_scores = [item["dimensions"]["factual_consistency"]["score"] for item in benchmark["results"]]
    unsupported_counts = [
        sum(not item["objective_checks"][name]["passed"] for name in ("no_unsupported_numbers", "no_unsupported_dates", "order_id_consistency"))
        for item in benchmark["results"]
    ]
    print(f"benchmark_factual_scores={sorted(set(factual_scores))}")
    print(f"benchmark_responses_with_factual_check_failures={sum(count > 0 for count in unsupported_counts)}")


if __name__ == "__main__":
    main()