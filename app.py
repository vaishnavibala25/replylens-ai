"""
app.py
------
Streamlit UI for ReplyLens AI: "Explainable AI suggested-response generation
and evaluation."

Sections:
  1. Incoming email input -> generate suggested reply
  2. Retrieved historical examples (with similarity scores)
  3. Suggested response (editable)
  4. ReplyLens evaluation scorecard (six weighted dimensions)
  5. "Why this score?" — strengths / issues / improvements + objective checks
  6. System evaluation benchmark over the held-out evaluation split
  7. Optional: compare two responses head-to-head
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from services.llm_service import LLMService
from services.retrieval_service import RetrievalService
from services.generation_service import generate_response
from services.evaluation_service import evaluate_response
from evaluation.benchmark import run_benchmark, run_calibration

DATA_PATH = Path(__file__).parent / "data" / "emails.csv"

st.set_page_config(page_title="ReplyLens AI", page_icon="🔎", layout="wide")


@st.cache_resource
def load_data():
    df = pd.read_csv(DATA_PATH)
    retrieval = RetrievalService(df)
    return df, retrieval


@st.cache_resource
def get_llm_service():
    return LLMService()


df, retrieval_service = load_data()
llm_service = get_llm_service()

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("ReplyLens AI")
st.caption("AI Email Response Quality, Explained")

if not llm_service.is_configured:
    st.info(
        "**Deterministic Demo Evaluation** — no `LLM_API_KEY` configured. The app runs end-to-end "
        "with clearly-labeled placeholder generations and response-dependent deterministic scores so you can see "
        "the full flow. Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` "
        "(see `.env.example`) for live results.",
        icon="ℹ️",
    )

for key, default in [
    ("retrieved", []), ("generated", ""), ("evaluation", None),
    ("compare_evaluation", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------------
# SECTION 1: Incoming Email
# --------------------------------------------------------------------------
st.header("1. Incoming Email")
incoming_email = st.text_area(
    "Paste or write a customer email",
    height=140,
    placeholder="Hi, I'd like to request a refund for order ORD-48213. "
                "The headphones I received don't work at all.",
)

if st.button("Generate Suggested Reply", type="primary"):
    if not incoming_email.strip():
        st.warning("Please enter an incoming email first.")
    else:
        with st.spinner("Retrieving similar examples and generating a response..."):
            retrieved = retrieval_service.retrieve(incoming_email, top_k=3)
            gen = generate_response(incoming_email, retrieved, llm_service)
        st.session_state["retrieved"] = retrieved
        st.session_state["generated"] = gen["response_text"]
        st.session_state["evaluation"] = None
        if gen.get("error"):
            st.warning(f"Falling back to Demo Mode: {gen['error']}")

# --------------------------------------------------------------------------
# SECTION 2: Retrieved Historical Examples
# --------------------------------------------------------------------------
if st.session_state["retrieved"]:
    st.header("2. Retrieved Historical Examples")
    st.caption("Top matches from the reference split, used as style/context grounding.")
    cols = st.columns(len(st.session_state["retrieved"]))
    for col, ex in zip(cols, st.session_state["retrieved"]):
        with col:
            st.metric("Similarity", ex["similarity"])
            st.caption(f"Category: {ex['category']}")
            with st.expander("Historical email"):
                st.write(ex["incoming_email"])
            with st.expander("Historical reply"):
                st.write(ex["historical_reply"])

# --------------------------------------------------------------------------
# SECTION 3: Suggested Response
# --------------------------------------------------------------------------
if st.session_state["generated"]:
    st.header("3. Suggested Response")
    edited_response = st.text_area(
        "Generated response (editable before evaluation)",
        value=st.session_state["generated"],
        height=180,
        key="editable_response",
    )

    if st.button("Evaluate Response"):
        with st.spinner("Running objective checks and LLM-as-judge evaluation..."):
            evaluation = evaluate_response(
                incoming_email, st.session_state["retrieved"], edited_response, llm_service
            )
        st.session_state["evaluation"] = evaluation
        if not evaluation["judge_ok"] and evaluation.get("judge_error"):
            st.warning(f"LLM judge fell back to Demo Mode: {evaluation['judge_error']}")

# --------------------------------------------------------------------------
# SECTION 4 & 5: ReplyLens Evaluation + Why This Score
# --------------------------------------------------------------------------
def render_scorecard(evaluation: dict, key_prefix: str = ""):
    st.subheader(f"Overall Quality: {evaluation['overall_score']:.0f} / 100")
    st.progress(min(1.0, max(0.0, evaluation["overall_score"] / 100)))

    dim_cols = st.columns(3)
    dims = list(evaluation["dimensions"].items())
    for i, (dim_key, dim_val) in enumerate(dims):
        with dim_cols[i % 3]:
            label = evaluation["dimension_labels"][dim_key]
            weight_pct = int(evaluation["weights"][dim_key] * 100)
            st.metric(f"{label} ({weight_pct}%)", f"{dim_val['score']}")
            st.caption(dim_val["reason"])
            if dim_val["issues"]:
                for issue in dim_val["issues"]:
                    st.caption(f"⚠️ {issue}")

    st.markdown("#### Why This Score?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Strengths**")
        if evaluation["strengths"]:
            for s in evaluation["strengths"]:
                st.write(f"- {s}")
        else:
            st.write("_None listed_")
    with c2:
        st.markdown("**Issues**")
        any_issue = False
        for dim_val in evaluation["dimensions"].values():
            for issue in dim_val["issues"]:
                st.write(f"- {issue}")
                any_issue = True
        if not any_issue:
            st.write("_None flagged_")
    with c3:
        st.markdown("**Suggested Improvements**")
        if evaluation["improvements"]:
            for imp in evaluation["improvements"]:
                st.write(f"- {imp}")
        else:
            st.write("_None listed_")

    with st.expander("Objective checks (deterministic, no LLM involved)"):
        st.caption(
            f"Semantic relevance to source email (TF-IDF cosine): "
            f"{evaluation['semantic_relevance']}"
        )
        objective = evaluation["objective_checks"]
        summary = objective.get("_summary", {})
        st.write(f"Passed {summary.get('passed', 0)} / {summary.get('total', 0)} checks")
        for check_name, check_val in objective.items():
            if check_name == "_summary":
                continue
            icon = "✅" if check_val["passed"] else "❌"
            st.write(f"{icon} **{check_name.replace('_', ' ')}** — {check_val['detail']}")


if st.session_state["evaluation"]:
    st.header("4. ReplyLens Evaluation")
    render_scorecard(st.session_state["evaluation"])

# --------------------------------------------------------------------------
# SECTION 6: System Evaluation Benchmark
# --------------------------------------------------------------------------
st.header("5. System Evaluation")
st.caption(
    "Runs the full pipeline (retrieve → generate → evaluate) over every email "
    "in the held-out evaluation split, with retrieval restricted to the "
    "reference split only (no leakage)."
)

eval_count = int((df["split"] == "evaluation").sum())
limit = st.slider(
    "Number of evaluation emails to run (full set for a real report; a smaller "
    "number for a quick check)",
    min_value=5, max_value=eval_count, value=min(15, eval_count), step=5,
)

if st.button("Run Evaluation Benchmark"):
    progress_bar = st.progress(0.0, text="Starting benchmark...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Evaluated {done}/{total} emails")

    with st.spinner("Running benchmark..."):
        bench = run_benchmark(llm_service, limit=limit, progress_callback=_progress)
    progress_bar.empty()

    summary = bench["summary"]
    st.subheader(f"Overall system score: {summary['mean_overall']:.1f} / 100")
    st.caption(
        f"Evaluation emails: {summary['count']}  |  "
        f"median: {summary['median_overall']}  |  "
        f"stdev: {summary['stdev_overall']}  |  "
        f"min/max: {summary['min_overall']}/{summary['max_overall']}"
    )
    if summary.get("is_demo"):
        st.warning("Deterministic Demo Evaluation scores are offline objective estimates, not an LLM judge. Configure an LLM API key for a live benchmark.")

    st.markdown("**Dimension averages**")
    dim_cols = st.columns(6)
    for col, (dim, val) in zip(dim_cols, summary["dimension_means"].items()):
        col.metric(dim.replace("_", " ").title(), val)

    st.markdown("**Category-level performance**")
    cat_df = pd.DataFrame({
        "category": list(summary["category_means"].keys()),
        "mean_score": list(summary["category_means"].values()),
        "n": [summary["category_counts"][c] for c in summary["category_means"].keys()],
    }).sort_values("mean_score", ascending=False)
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

with st.expander("Human-calibration correlation (metric validation)"):
    st.caption(
        "Compares the automated overall_score against a small (n≈24), single-"
        "reviewer set of 1-5 human quality ratings. This is NOT a statistically "
        "sufficient sample to prove general evaluator validity — see README."
    )
    if st.button("Run Calibration Check"):
        with st.spinner("Scoring calibration set..."):
            cal = run_calibration(llm_service)
        st.write(f"n = {cal['n_cases']}, Spearman ρ = {cal['spearman_rho']}")
        if cal["is_demo"]:
            st.warning("Deterministic Demo Evaluation: correlation is an offline objective sanity check, not validation of a live LLM judge.")
        cal_df = pd.DataFrame(cal["per_case"])
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# SECTION 7 (optional): Compare two responses
# --------------------------------------------------------------------------
st.header("6. Compare (optional)")
st.caption("Evaluate an alternative response against the same email using the same framework.")

alt_response = st.text_area("Alternative response", height=140, key="alt_response")
if st.button("Compare Responses"):
    if not incoming_email.strip() or not st.session_state.get("generated"):
        st.warning("Generate a suggested response first (Section 1) before comparing.")
    elif not alt_response.strip():
        st.warning("Enter an alternative response to compare.")
    else:
        with st.spinner("Evaluating both responses..."):
            eval_a = st.session_state["evaluation"] or evaluate_response(
                incoming_email, st.session_state["retrieved"],
                st.session_state.get("editable_response", st.session_state["generated"]),
                llm_service,
            )
            eval_b = evaluate_response(
                incoming_email, st.session_state["retrieved"], alt_response, llm_service
            )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Response A: {eval_a['overall_score']:.0f}")
            render_scorecard(eval_a, key_prefix="a")
        with c2:
            st.markdown(f"### Response B: {eval_b['overall_score']:.0f}")
            render_scorecard(eval_b, key_prefix="b")

st.divider()
st.caption(
    "ReplyLens AI — synthetic dataset, LLM-as-judge + deterministic objective "
    "checks, weighted overall score always computed server-side. "
    "See README.md for full methodology, validation, and limitations."
)
