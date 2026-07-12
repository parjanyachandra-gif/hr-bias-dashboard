"""Streamlit review app with the original overview page and a simplified fairness audit tab."""

from __future__ import annotations

import json
from urllib import error, request

import pandas as pd
import streamlit as st

from alp_project import (
    AI_DECISION_COLUMN,
    APP_TITLE,
    FAIRNESS_DIMENSIONS,
    FAVORABLE_LABEL,
    apply_filters,
    compute_kpis,
    inject_page_styles,
    load_dataset,
    render_bias_context,
    render_business_cuts,
    render_dataset_profile,
    render_decision_summary,
    render_header,
    render_population_distribution,
    render_score_landscape,
    render_section_intro,
)

from agent3_recruitment_insights import render_recruitment_insights_tab

DEMOGRAPHIC_PARITY_GAP_THRESHOLD = 0.15
EQUALIZED_ODDS_GAP_THRESHOLD = 0.15
DISPARATE_IMPACT_RATIO_THRESHOLD = 0.80
OBSERVED_OUTCOME_COLUMN = "final_hiring_decision"
LOW_CONFIDENCE_SAMPLE_SIZE = 30
MEDIUM_CONFIDENCE_SAMPLE_SIZE = 75

def format_percent(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.1%}"

def format_score(value: float) -> str:

    return f"{value:.0f}%"

def classify_severity(gap_value: float, threshold: float) -> str:
    if pd.isna(gap_value):
        return "Unavailable"
    absolute_gap = abs(gap_value)
    if absolute_gap < threshold * 0.5:
        return "Low"
    if absolute_gap < threshold:
        return "Moderate"
    if absolute_gap < threshold * 1.5:
        return "High"
    return "Critical"

def classify_confidence(candidate_count: int) -> str:
    if candidate_count < LOW_CONFIDENCE_SAMPLE_SIZE:
        return "Low confidence"
    if candidate_count < MEDIUM_CONFIDENCE_SAMPLE_SIZE:
        return "Medium confidence"
    return "High confidence"

def classify_demographic_parity_risk_type(gap_value: float, threshold: float) -> str:
    if pd.isna(gap_value):
        return "Unavailable"
    if abs(gap_value) < threshold * 0.5:
        return "Near parity"
    if gap_value < 0:
        return "Under-selection"
    return "Over-selection"

def classify_equalized_odds_risk_type(tpr_gap: float, fpr_gap: float) -> str:
    if pd.isna(tpr_gap) and pd.isna(fpr_gap):
        return "Unavailable"
    if pd.isna(fpr_gap) or abs(tpr_gap) >= abs(fpr_gap):
        return "Missed qualified candidates" if tpr_gap < 0 else "Higher approval capture"
    return "Excess false approvals" if fpr_gap > 0 else "Stricter rejection pattern"

def classify_disparate_impact_severity(ratio_value: float) -> str:
    if pd.isna(ratio_value):
        return "Unavailable"
    if ratio_value >= 0.90:
        return "Low"
    if ratio_value >= DISPARATE_IMPACT_RATIO_THRESHOLD:
        return "Moderate"
    if ratio_value >= 0.65:
        return "High"
    return "Critical"

def classify_disparate_impact_risk_type(ratio_value: float, is_reference_group: bool) -> str:
    if is_reference_group:
        return "Reference group"
    if pd.isna(ratio_value):
        return "Unavailable"
    if ratio_value >= 0.90:
        return "Near reference"
    if ratio_value >= DISPARATE_IMPACT_RATIO_THRESHOLD:
        return "Near-threshold adverse impact"
    if ratio_value >= 0.65:
        return "Adverse impact risk"
    return "Severe adverse impact"

def classify_bias_score_band(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"

def compute_scaled_bias_score(values: pd.Series, threshold: float) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty or threshold <= 0:
        return 0.0
    scaled_score = (numeric_values.abs() / threshold).mean() * 100
    return float(min(100.0, scaled_score))

def compute_ratio_bias_score(values: pd.Series, threshold: float) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty or threshold <= 0:
        return 0.0
    deficit = ((threshold - numeric_values).clip(lower=0)) / threshold
    scaled_score = deficit.mean() * 100
    return float(min(100.0, scaled_score))

def render_bias_score_card(title: str, score: float, description: str) -> None:
    band = classify_bias_score_band(score)
    st.markdown(
        f"""
        <section class="alp-card-grid">
            <div class="alp-stat-card">
                <div class="alp-stat-label">{title}</div>
                <div class="alp-stat-value">{format_score(score)}</div>
                <div class="alp-stat-caption">Severity band: {band}. {description}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_filter_form(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    default_filters = {
        "departments": [],
        "job_roles": [],
        "months": [],
        "geographies": [],
    }
    applied_filters = st.session_state.setdefault("applied_dashboard_filters", default_filters.copy())

    st.sidebar.header("Dashboard Filters")
    with st.sidebar.form("dashboard_filters_form"):
        departments = st.multiselect(
            "Department",
            options=sorted(dataframe["department"].dropna().unique().tolist()),
            default=applied_filters["departments"],
        )
        job_roles = st.multiselect(
            "Job Role",
            options=sorted(dataframe["job_role"].dropna().unique().tolist()),
            default=applied_filters["job_roles"],
        )
        months = st.multiselect(
            "Application Month",
            options=sorted(dataframe["application_month"].dropna().unique().tolist()),
            default=applied_filters["months"],
        )
        geographies = st.multiselect(
            "Geography",
            options=sorted(dataframe["geography"].dropna().unique().tolist()),
            default=applied_filters["geographies"],
        )
        apply_filters_clicked = st.form_submit_button("Apply Filters")

    if apply_filters_clicked:
        applied_filters = {
            "departments": departments,
            "job_roles": job_roles,
            "months": months,
            "geographies": geographies,
        }
        st.session_state["applied_dashboard_filters"] = applied_filters
        st.session_state["dashboard_refresh_phase"] = "show_loading"

    return st.session_state["applied_dashboard_filters"]

def compute_overall_bias_score(dataframe: pd.DataFrame) -> float:
    demographic_parity_bias_score = compute_demographic_parity_bias_score(dataframe)
    equalized_odds_bias_score = compute_equalized_odds_bias_score(dataframe)
    disparate_impact_bias_score = compute_disparate_impact_bias_score(dataframe)
    return float(
        (demographic_parity_bias_score + equalized_odds_bias_score + disparate_impact_bias_score) / 3
    )

def render_overview_kpis_with_bias(dataframe: pd.DataFrame) -> None:
    metrics = compute_kpis(dataframe)
    overall_bias_score = compute_overall_bias_score(dataframe)
    st.markdown(
        f"""
        <section class="alp-card-grid" style="grid-template-columns: repeat(4, minmax(0, 1fr));">
            <div class="alp-stat-card">
                <div class="alp-stat-label">Total Candidates</div>
                <div class="alp-stat-value">{int(metrics['records']):,}</div>
                <div class="alp-stat-caption">Filtered candidate records currently included in the audit view.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Hire Rate</div>
                <div class="alp-stat-value">{format_percent(float(metrics['ai_hire_rate']))}</div>
                <div class="alp-stat-caption">Share of candidates the AI system recommended for hiring.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Recommended Hires</div>
                <div class="alp-stat-value">{int(metrics['ai_hire_count']):,}</div>
                <div class="alp-stat-caption">Absolute count of candidates recommended for hiring by the AI system.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Recommended Rejects</div>
                <div class="alp-stat-value">{int(metrics['ai_reject_count']):,}</div>
                <div class="alp-stat-caption">Candidates the AI system did not recommend for hiring.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">Departments</div>
                <div class="alp-stat-value">{metrics['departments']}</div>
                <div class="alp-stat-caption">Distinct business units represented in the filtered dataset.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">Job Roles</div>
                <div class="alp-stat-value">{metrics['job_roles']}</div>
                <div class="alp-stat-caption">Distinct role families currently visible in the overview.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">Overall Bias</div>
                <div class="alp-stat-value">{format_score(overall_bias_score)}</div>
                <div class="alp-stat-caption">Combined fairness-risk score across Demographic Parity, Equalized Odds, and Disparate Impact Ratio.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def compute_demographic_parity_bias_score(dataframe: pd.DataFrame) -> float:
    scores: list[float] = []
    for sensitive_column, _ in FAIRNESS_DIMENSIONS:
        metrics, _ = compute_demographic_parity_table(dataframe, sensitive_column)
        if not metrics.empty:
            scores.append(
                compute_scaled_bias_score(metrics["abs_demographic_parity_gap"], DEMOGRAPHIC_PARITY_GAP_THRESHOLD)
            )
    return float(sum(scores) / len(scores)) if scores else 0.0

def compute_equalized_odds_bias_score(dataframe: pd.DataFrame) -> float:
    scores: list[float] = []
    for sensitive_column, _ in FAIRNESS_DIMENSIONS:
        metrics, _ = compute_equalized_odds_table(dataframe, sensitive_column)
        if not metrics.empty:
            scores.append(compute_scaled_bias_score(metrics["max_gap"], EQUALIZED_ODDS_GAP_THRESHOLD))
    return float(sum(scores) / len(scores)) if scores else 0.0

def compute_disparate_impact_bias_score(dataframe: pd.DataFrame) -> float:
    scores: list[float] = []
    for sensitive_column, _ in FAIRNESS_DIMENSIONS:
        metrics, _ = compute_disparate_impact_ratio_table(dataframe, sensitive_column)
        if not metrics.empty:
            scores.append(compute_ratio_bias_score(metrics["disparate_impact_ratio"], DISPARATE_IMPACT_RATIO_THRESHOLD))
    return float(sum(scores) / len(scores)) if scores else 0.0

def compute_demographic_parity_table(
    dataframe: pd.DataFrame,
    sensitive_column: str,
) -> tuple[pd.DataFrame, float]:
    audit_frame = dataframe[[sensitive_column, AI_DECISION_COLUMN]].dropna().copy()
    if audit_frame.empty:
        return pd.DataFrame(), 0.0

    audit_frame["selected"] = (audit_frame[AI_DECISION_COLUMN] == FAVORABLE_LABEL).astype(int)
    overall_selection_rate = audit_frame["selected"].mean()

    metrics = (
        audit_frame.groupby(sensitive_column, dropna=False)
        .agg(candidate_count=("selected", "size"), selected=("selected", "sum"))
        .reset_index()
    )
    metrics["population_share"] = metrics["candidate_count"] / len(audit_frame)
    metrics["selection_rate"] = metrics["selected"] / metrics["candidate_count"]
    metrics["demographic_parity_gap"] = metrics["selection_rate"] - overall_selection_rate
    metrics["abs_demographic_parity_gap"] = metrics["demographic_parity_gap"].abs()
    metrics["severity"] = metrics["abs_demographic_parity_gap"].apply(
        lambda value: classify_severity(value, DEMOGRAPHIC_PARITY_GAP_THRESHOLD)
    )
    metrics["risk_type"] = metrics["demographic_parity_gap"].apply(
        lambda value: classify_demographic_parity_risk_type(value, DEMOGRAPHIC_PARITY_GAP_THRESHOLD)
    )
    metrics["confidence"] = metrics["candidate_count"].apply(classify_confidence)
    metrics["risk_level"] = metrics.apply(
        lambda row: f"{row['severity']} | {row['risk_type']} | {row['confidence']}",
        axis=1,
    )
    return metrics.sort_values("abs_demographic_parity_gap", ascending=False), float(overall_selection_rate)

def build_demographic_parity_prompt(
    parameter_label: str,
    group_column: str,
    overall_selection_rate: float,
    metrics: pd.DataFrame,
) -> str:
    rows = "\n".join(
        f"- {parameter_label}: group={row[group_column]}, candidates={int(row['candidate_count'])}, "
        f"population_share={row['population_share']:.3f}, selection_rate={row['selection_rate']:.3f}, "
        f"demographic_parity_gap={row['demographic_parity_gap']:.3f}, severity={row['severity']}, "
        f"risk_type={row['risk_type']}, confidence={row['confidence']}"
        for _, row in metrics.iterrows()
    )
    return f"""
You are Fairness Audit Agent.
You evaluate systemic applicant selection trends across groups, monitor disparity patterns, and flag high-risk statistical deviations across protected variables.

Current protected variable: {parameter_label}
Metric: Demographic Parity
High-risk threshold: absolute demographic parity gap >= {DEMOGRAPHIC_PARITY_GAP_THRESHOLD:.2f}
Overall AI selection rate: {overall_selection_rate:.3f}

Use the following group data:
{rows}

Write an AI insight card with exactly 3 short bullet points and 1 concluding sentence.
Focus only on the demographic parity findings for {parameter_label}.
Call out the highest-risk group, the direction of the gap, and the overall selection pattern.
Do not mention code, prompts, tools, missing data, or implementation details.
""".strip()

def compute_equalized_odds_table(
    dataframe: pd.DataFrame,
    sensitive_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    audit_frame = dataframe[[sensitive_column, AI_DECISION_COLUMN, OBSERVED_OUTCOME_COLUMN]].dropna().copy()
    if audit_frame.empty:
        return pd.DataFrame(), {"overall_tpr": 0.0, "overall_fpr": 0.0}

    audit_frame[AI_DECISION_COLUMN] = audit_frame[AI_DECISION_COLUMN].astype(str).str.strip()
    audit_frame[OBSERVED_OUTCOME_COLUMN] = audit_frame[OBSERVED_OUTCOME_COLUMN].astype(str).str.strip()
    audit_frame["prediction_positive"] = (audit_frame[AI_DECISION_COLUMN] == FAVORABLE_LABEL).astype(int)
    audit_frame["actual_positive"] = (audit_frame[OBSERVED_OUTCOME_COLUMN] == FAVORABLE_LABEL).astype(int)

    rows: list[dict[str, str | int | float]] = []
    overall_tp = 0
    overall_fp = 0
    overall_tn = 0
    overall_fn = 0

    for group_name, group_frame in audit_frame.groupby(sensitive_column, dropna=False):
        tp = int(((group_frame["prediction_positive"] == 1) & (group_frame["actual_positive"] == 1)).sum())
        fp = int(((group_frame["prediction_positive"] == 1) & (group_frame["actual_positive"] == 0)).sum())
        tn = int(((group_frame["prediction_positive"] == 0) & (group_frame["actual_positive"] == 0)).sum())
        fn = int(((group_frame["prediction_positive"] == 0) & (group_frame["actual_positive"] == 1)).sum())

        overall_tp += tp
        overall_fp += fp
        overall_tn += tn
        overall_fn += fn

        tpr = tp / (tp + fn) if (tp + fn) else pd.NA
        fpr = fp / (fp + tn) if (fp + tn) else pd.NA

        rows.append(
            {
                sensitive_column: group_name,
                "candidate_count": int(len(group_frame)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
            }
        )

    overall_tpr = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0.0
    overall_fpr = overall_fp / (overall_fp + overall_tn) if (overall_fp + overall_tn) else 0.0

    metrics = pd.DataFrame(rows)
    metrics["tpr_gap"] = metrics["true_positive_rate"] - overall_tpr
    metrics["fpr_gap"] = metrics["false_positive_rate"] - overall_fpr
    metrics["max_gap"] = metrics[["tpr_gap", "fpr_gap"]].abs().max(axis=1)
    metrics["severity"] = metrics["max_gap"].apply(
        lambda value: classify_severity(value, EQUALIZED_ODDS_GAP_THRESHOLD)
    )
    metrics["risk_type"] = metrics.apply(
        lambda row: classify_equalized_odds_risk_type(row["tpr_gap"], row["fpr_gap"]),
        axis=1,
    )
    metrics["confidence"] = metrics["candidate_count"].apply(classify_confidence)
    metrics["risk_level"] = metrics.apply(
        lambda row: f"{row['severity']} | {row['risk_type']} | {row['confidence']}",
        axis=1,
    )
    return metrics.sort_values("max_gap", ascending=False), {
        "overall_tpr": float(overall_tpr),
        "overall_fpr": float(overall_fpr),
    }

def build_equalized_odds_prompt(
    parameter_label: str,
    group_column: str,
    overall_metrics: dict[str, float],
    metrics: pd.DataFrame,
) -> str:
    rows = "\n".join(
        f"- {parameter_label}: group={row[group_column]}, candidates={int(row['candidate_count'])}, "
        f"tp={int(row['tp'])}, tn={int(row['tn'])}, fp={int(row['fp'])}, fn={int(row['fn'])}, "
        f"true_positive_rate={float(row['true_positive_rate']):.3f}, false_positive_rate={float(row['false_positive_rate']):.3f}, "
        f"tpr_gap={float(row['tpr_gap']):.3f}, fpr_gap={float(row['fpr_gap']):.3f}, severity={row['severity']}, "
        f"risk_type={row['risk_type']}, confidence={row['confidence']}"
        for _, row in metrics.fillna(0).iterrows()
    )
    return f"""
You are Fairness Audit Agent.
You evaluate systemic applicant selection trends across groups, monitor disparity patterns, and flag high-risk statistical deviations across protected variables.

Current protected variable: {parameter_label}
Metric: Equalized Odds
High-risk threshold: max absolute TPR or FPR gap >= {EQUALIZED_ODDS_GAP_THRESHOLD:.2f}
Overall true positive rate: {overall_metrics['overall_tpr']:.3f}
Overall false positive rate: {overall_metrics['overall_fpr']:.3f}

Use the following group data:
{rows}

Write an AI insight card with exactly 3 short bullet points and 1 concluding sentence.
Focus only on the equalized odds findings for {parameter_label}.
Call out the highest-risk group, whether the main issue is missed qualified candidates or false approvals, and the overall pattern.
Do not mention code, prompts, tools, missing data, or implementation details.
""".strip()

def compute_disparate_impact_ratio_table(
    dataframe: pd.DataFrame,
    sensitive_column: str,
) -> tuple[pd.DataFrame, float]:
    audit_frame = dataframe[[sensitive_column, AI_DECISION_COLUMN]].dropna().copy()
    if audit_frame.empty:
        return pd.DataFrame(), 0.0

    audit_frame["selected"] = (audit_frame[AI_DECISION_COLUMN] == FAVORABLE_LABEL).astype(int)
    metrics = (
        audit_frame.groupby(sensitive_column, dropna=False)
        .agg(candidate_count=("selected", "size"), selected=("selected", "sum"))
        .reset_index()
    )
    metrics["population_share"] = metrics["candidate_count"] / len(audit_frame)
    metrics["selection_rate"] = metrics["selected"] / metrics["candidate_count"]
    reference_selection_rate = float(metrics["selection_rate"].max()) if not metrics.empty else 0.0
    if reference_selection_rate > 0:
        metrics["disparate_impact_ratio"] = metrics["selection_rate"] / reference_selection_rate
    else:
        metrics["disparate_impact_ratio"] = pd.NA
    metrics["reference_selection_rate"] = reference_selection_rate
    metrics["is_reference_group"] = metrics["selection_rate"] == reference_selection_rate
    metrics["severity"] = metrics["disparate_impact_ratio"].apply(classify_disparate_impact_severity)
    metrics["risk_type"] = metrics.apply(
        lambda row: classify_disparate_impact_risk_type(row["disparate_impact_ratio"], bool(row["is_reference_group"])),
        axis=1,
    )
    return metrics.sort_values(["disparate_impact_ratio", "candidate_count"], ascending=[True, False]), reference_selection_rate

def build_disparate_impact_prompt(
    parameter_label: str,
    group_column: str,
    reference_selection_rate: float,
    metrics: pd.DataFrame,
) -> str:
    rows = "\n".join(
        f"- {parameter_label}: group={row[group_column]}, candidates={int(row['candidate_count'])}, "
        f"population_share={row['population_share']:.3f}, selection_rate={row['selection_rate']:.3f}, "
        f"reference_selection_rate={float(row['reference_selection_rate']):.3f}, disparate_impact_ratio={float(row['disparate_impact_ratio']):.3f}, "
        f"severity={row['severity']}, risk_type={row['risk_type']}"
        for _, row in metrics.fillna(0).iterrows()
    )
    return f"""
You are Fairness Audit Agent.
You evaluate systemic applicant selection trends across groups, monitor disparity patterns, and flag high-risk statistical deviations across protected variables.

Current protected variable: {parameter_label}
Metric: Disparate Impact Ratio
Reference selection rate: {reference_selection_rate:.3f}
Threshold: ratios below {DISPARATE_IMPACT_RATIO_THRESHOLD:.2f} indicate adverse impact risk under the 80 percent rule.

Use the following group data:
{rows}

Write an AI insight card with exactly 3 short bullet points and 1 concluding sentence.
Focus only on the disparate impact ratio findings for {parameter_label}.
Call out the reference group, which groups fall below the threshold, and where the strongest adverse impact signal exists.
Do not mention code, prompts, tools, missing data, or implementation details.
""".strip()

def query_ollama(prompt: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    req = request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=45) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("response", "")).strip()

def generate_ai_insight(prompt: str, model: str, parameter_label: str) -> tuple[str | None, str | None]:
    try:
        with st.spinner(f"Generating {parameter_label.lower()} insight with Ollama model '{model}'..."):
            return query_ollama(prompt, model), None
    except error.URLError:
        return None, "Unable to reach Ollama at http://localhost:11434. Start Ollama and retry."
    except Exception as exc:
        return None, f"Ollama insight generation failed: {exc}"

def normalize_insight_state(state: object) -> dict[str, str | None]:
    if isinstance(state, dict):
        return {
            "prompt": state.get("prompt"),
            "model": state.get("model"),
            "response": state.get("response"),
            "error": state.get("error"),
        }
    if isinstance(state, str):
        return {
            "prompt": None,
            "model": None,
            "response": state,
            "error": None,
        }
    return {
        "prompt": None,
        "model": None,
        "response": None,
        "error": None,
    }

def format_insight_response(response: str) -> str:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    formatted_lines: list[str] = []
    for line in lines:
        if line.startswith(("- ", "* ", "• ")):
            formatted_lines.append(line)
        else:
            formatted_lines.append(f"- {line}")
    return "\n".join(formatted_lines)

def build_insight_html(response: str) -> str:
    formatted_response = format_insight_response(response)
    lines = [line for line in formatted_response.splitlines() if line.strip()]
    body = "".join(f"<p class='alp-stat-caption'>{line}</p>" for line in lines)
    return body

def build_summary_prompt(insight_records: list[dict[str, str]]) -> str:
    grouped_lines = []
    for record in insight_records:
        grouped_lines.append(
            f"- dimension={record['dimension']}; metric={record['metric']}; insight={record['insight']}"
        )
    return f"""
You are Fairness Audit Agent.
You are reviewing 12 fairness insights across three metrics: Demographic Parity, Equalized Odds, and Disparate Impact Ratio.

Protected variables covered: Gender, Age, Region, Religion.

Use the following generated insights:
{chr(10).join(grouped_lines)}

Write exactly 4 bullet points.
Write one bullet for Gender, one for Age, one for Region, and one for Religion.
Each bullet should synthesize what all three metrics together suggest for that protected variable.
Focus on the main fairness pattern, the strongest risk signal, and whether the variable needs immediate attention.
Do not mention code, prompts, tools, missing data, or implementation details.
""".strip()

def collect_metric_insights() -> list[dict[str, str]]:
    metric_names = ["Demographic Parity", "Equalized Odds", "Disparate Impact Ratio"]
    insight_records: list[dict[str, str]] = []
    for sensitive_column, label in FAIRNESS_DIMENSIONS:
        for metric_name in metric_names:
            session_key = f"ollama_insight_{metric_name}_{sensitive_column}"
            insight_state = normalize_insight_state(st.session_state.get(session_key))
            response = insight_state.get("response")
            if response:
                insight_records.append(
                    {
                        "dimension": label,
                        "metric": metric_name,
                        "insight": str(response).replace("\n", " ").strip(),
                    }
                )
    return insight_records

def render_summary_tab(ollama_model: str) -> None:
    render_section_intro(
        "Summary",
        "This tab synthesizes all generated fairness insights across Demographic Parity, Equalized Odds, and Disparate Impact Ratio into one concise cross-metric view.",
    )
    insight_records = collect_metric_insights()
    expected_insight_count = len(FAIRNESS_DIMENSIONS) * 3

    if len(insight_records) < expected_insight_count:
        st.info("Summary will appear after the metric-specific insights are generated for all four protected variables across the three fairness metrics.")
        return

    prompt = build_summary_prompt(insight_records)
    session_key = "ollama_insight_summary_all_metrics"
    insight_state = normalize_insight_state(st.session_state.get(session_key))
    prompt_changed = (
        insight_state.get("prompt") != prompt or insight_state.get("model") != ollama_model
    )

    if prompt_changed or insight_state.get("response") is None:
        response, error_message = generate_ai_insight(prompt, ollama_model, "summary")
        insight_state = {
            "prompt": prompt,
            "model": ollama_model,
            "response": response,
            "error": error_message,
        }
        st.session_state[session_key] = insight_state

    if insight_state.get("error"):
        st.warning(insight_state["error"])

    if insight_state.get("response"):
        insight_html = build_insight_html(str(insight_state["response"]))
        st.markdown(
            f"""
            <div class='alp-stat-card'>
                <div class='alp-stat-label'>Cross-Metric Summary Insight</div>
                {insight_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Show Summary Ollama prompt"):
        st.code(prompt)

def render_ai_insight_card(
    parameter_key: str,
    parameter_label: str,
    metric_name: str,
    prompt: str,
    metrics: pd.DataFrame,
    ollama_model: str,
) -> None:
    session_key = f"ollama_insight_{metric_name}_{parameter_key}"

    insight_state = normalize_insight_state(st.session_state.get(session_key))
    prompt_changed = (
        insight_state.get("prompt") != prompt or insight_state.get("model") != ollama_model
    )

    if prompt_changed or insight_state.get("response") is None:
        response, error_message = generate_ai_insight(prompt, ollama_model, parameter_label)
        insight_state = {
            "prompt": prompt,
            "model": ollama_model,
            "response": response,
            "error": error_message,
        }
        st.session_state[session_key] = insight_state

    if insight_state.get("error"):
        st.warning(insight_state["error"])

    if insight_state.get("response"):
        insight_html = build_insight_html(str(insight_state["response"]))
        st.markdown(
            f"""
            <div class='alp-stat-card'>
                <div class='alp-stat-label'>{parameter_label} {metric_name} Insight</div>
                {insight_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f"Show {parameter_label} Ollama prompt"):
        st.code(prompt)

def cache_metric_insight(
    session_key: str,
    prompt: str,
    ollama_model: str,
    parameter_label: str,
) -> None:
    insight_state = normalize_insight_state(st.session_state.get(session_key))
    prompt_changed = (
        insight_state.get("prompt") != prompt or insight_state.get("model") != ollama_model
    )
    if prompt_changed or insight_state.get("response") is None:
        response, error_message = generate_ai_insight(prompt, ollama_model, parameter_label)
        st.session_state[session_key] = {
            "prompt": prompt,
            "model": ollama_model,
            "response": response,
            "error": error_message,
        }

def precompute_metric_insights(dataframe: pd.DataFrame, ollama_model: str) -> None:
    for sensitive_column, label in FAIRNESS_DIMENSIONS:
        demographic_metrics, overall_selection_rate = compute_demographic_parity_table(dataframe, sensitive_column)
        demographic_prompt = build_demographic_parity_prompt(label, sensitive_column, overall_selection_rate, demographic_metrics)
        cache_metric_insight(
            session_key=f"ollama_insight_Demographic Parity_{sensitive_column}",
            prompt=demographic_prompt,
            ollama_model=ollama_model,
            parameter_label=label,
        )

        equalized_odds_metrics, overall_metrics = compute_equalized_odds_table(dataframe, sensitive_column)
        equalized_odds_prompt = build_equalized_odds_prompt(label, sensitive_column, overall_metrics, equalized_odds_metrics)
        cache_metric_insight(
            session_key=f"ollama_insight_Equalized Odds_{sensitive_column}",
            prompt=equalized_odds_prompt,
            ollama_model=ollama_model,
            parameter_label=label,
        )

        disparate_impact_metrics, reference_selection_rate = compute_disparate_impact_ratio_table(dataframe, sensitive_column)
        disparate_impact_prompt = build_disparate_impact_prompt(label, sensitive_column, reference_selection_rate, disparate_impact_metrics)
        cache_metric_insight(
            session_key=f"ollama_insight_Disparate Impact Ratio_{sensitive_column}",
            prompt=disparate_impact_prompt,
            ollama_model=ollama_model,
            parameter_label=label,
        )

    summary_prompt = build_summary_prompt(collect_metric_insights())
    cache_metric_insight(
        session_key="ollama_insight_summary_all_metrics",
        prompt=summary_prompt,
        ollama_model=ollama_model,
        parameter_label="summary",
    )

def render_demographic_parity_section(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    label: str,
    ollama_model: str,
) -> None:
    metrics, overall_selection_rate = compute_demographic_parity_table(dataframe, sensitive_column)
    render_section_intro(
        f"{label}: Demographic Parity",
        f"Overall AI selection rate is {format_percent(overall_selection_rate)}. This table compares each {label.lower()} group's selection rate with the overall rate.",
    )

    if metrics.empty:
        st.info(f"No records are available for {label.lower()} after filtering.")
        return

    formatted_metrics = metrics.assign(
        population_share=metrics["population_share"].map(format_percent),
        selection_rate=metrics["selection_rate"].map(format_percent),
        demographic_parity_gap=metrics["demographic_parity_gap"].map(format_percent),
    )[[
        sensitive_column,
        "candidate_count",
        "population_share",
        "selection_rate",
        "demographic_parity_gap",
        "severity",
        "risk_type",
    ]]
    st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)
    prompt = build_demographic_parity_prompt(label, sensitive_column, overall_selection_rate, metrics)
    render_ai_insight_card(
        parameter_key=sensitive_column,
        parameter_label=label,
        metric_name="Demographic Parity",
        prompt=prompt,
        metrics=metrics,
        ollama_model=ollama_model,
    )

def render_equalized_odds_section(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    label: str,
    ollama_model: str,
) -> None:
    metrics, overall_metrics = compute_equalized_odds_table(dataframe, sensitive_column)
    render_section_intro(
        f"{label}: Equalized Odds",
        f"Overall AI true positive rate is {format_percent(overall_metrics['overall_tpr'])} and overall false positive rate is {format_percent(overall_metrics['overall_fpr'])}. This table compares error-rate gaps for {label.lower()} groups.",
    )

    if metrics.empty:
        st.info(f"No records are available for {label.lower()} after filtering.")
        return

    formatted_metrics = metrics.assign(
        true_positive_rate=metrics["true_positive_rate"].map(format_percent),
        false_positive_rate=metrics["false_positive_rate"].map(format_percent),
        tpr_gap=metrics["tpr_gap"].map(format_percent),
        fpr_gap=metrics["fpr_gap"].map(format_percent),
        max_gap=metrics["max_gap"].map(format_percent),
    )[[
        sensitive_column,
        "candidate_count",
        "true_positive_rate",
        "false_positive_rate",
        "tpr_gap",
        "fpr_gap",
        "max_gap",
        "severity",
        "risk_type",
    ]]
    st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)
    prompt = build_equalized_odds_prompt(label, sensitive_column, overall_metrics, metrics)
    render_ai_insight_card(
        parameter_key=sensitive_column,
        parameter_label=label,
        metric_name="Equalized Odds",
        prompt=prompt,
        metrics=metrics,
        ollama_model=ollama_model,
    )

def render_disparate_impact_ratio_section(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    label: str,
    ollama_model: str,
) -> None:
    metrics, reference_selection_rate = compute_disparate_impact_ratio_table(dataframe, sensitive_column)
    render_section_intro(
        f"{label}: Disparate Impact Ratio",
        f"Reference selection rate is {format_percent(reference_selection_rate)}. This table compares each {label.lower()} group's selection rate against the reference group using the 80 percent rule.",
    )

    if metrics.empty:
        st.info(f"No records are available for {label.lower()} after filtering.")
        return

    formatted_metrics = metrics.assign(
        population_share=metrics["population_share"].map(format_percent),
        selection_rate=metrics["selection_rate"].map(format_percent),
        reference_selection_rate=metrics["reference_selection_rate"].map(format_percent),
        disparate_impact_ratio=metrics["disparate_impact_ratio"].map(lambda value: "NA" if pd.isna(value) else f"{value:.3f}"),
    )[[
        sensitive_column,
        "candidate_count",
        "population_share",
        "selection_rate",
        "reference_selection_rate",
        "disparate_impact_ratio",
        "severity",
        "risk_type",
    ]]
    st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)
    prompt = build_disparate_impact_prompt(label, sensitive_column, reference_selection_rate, metrics)
    render_ai_insight_card(
        parameter_key=sensitive_column,
        parameter_label=label,
        metric_name="Disparate Impact Ratio",
        prompt=prompt,
        metrics=metrics,
        ollama_model=ollama_model,
    )

def render_fairness_audit_tab(dataframe: pd.DataFrame, ollama_model: str) -> None:
    demographic_parity_bias_score = compute_demographic_parity_bias_score(dataframe)
    equalized_odds_bias_score = compute_equalized_odds_bias_score(dataframe)
    disparate_impact_bias_score = compute_disparate_impact_bias_score(dataframe)
    overall_bias_score = compute_overall_bias_score(dataframe)

    render_section_intro(
        "Fairness Audit Agent",
        "Evaluates systemic applicant selection trends across Age, Gender, Region, and Religion using Demographic Parity, Equalized Odds, Disparate Impact Ratio, and a consolidated summary.",
    )
    demographic_parity_tab, equalized_odds_tab, disparate_impact_tab, summary_tab = st.tabs([
        "Demographic Parity",
        "Equalized Odds",
        "Disparate Impact Ratio",
        "Summary",
    ])

    with demographic_parity_tab:
        st.markdown(
            f"<div class='alp-note'><strong>Metric definition:</strong> Demographic Parity compares each group's AI selection rate with the overall AI selection rate. Severity is based on the size of the parity gap, risk type shows under-selection versus over-selection, and confidence depends on group sample size.</div>",
            unsafe_allow_html=True,
        )
        render_bias_score_card(
            "Demographic Parity Bias Score",
            demographic_parity_bias_score,
            "Higher scores indicate larger average selection-rate gaps across the four protected variables.",
        )
        for sensitive_column, label in FAIRNESS_DIMENSIONS:
            render_demographic_parity_section(dataframe, sensitive_column, label, ollama_model)

    with equalized_odds_tab:
        st.markdown(
            f"<div class='alp-note'><strong>Metric definition:</strong> Equalized Odds compares whether the AI has similar true positive and false positive rates across groups using final hiring decision as the observed outcome. Severity is based on the larger of the TPR or FPR gaps, risk type names the dominant issue, and confidence depends on group sample size.</div>",
            unsafe_allow_html=True,
        )
        render_bias_score_card(
            "Equalized Odds Bias Score",
            equalized_odds_bias_score,
            "Higher scores indicate larger average error-rate gaps across the four protected variables.",
        )
        for sensitive_column, label in FAIRNESS_DIMENSIONS:
            render_equalized_odds_section(dataframe, sensitive_column, label, ollama_model)

    with disparate_impact_tab:
        st.markdown(
            f"<div class='alp-note'><strong>Metric definition:</strong> Disparate Impact Ratio compares each group's selection rate against the reference group with the highest selection rate. Ratios below {DISPARATE_IMPACT_RATIO_THRESHOLD:.2f} indicate adverse impact risk under the 80 percent rule, while severity and risk type describe how far each group falls from the reference.</div>",
            unsafe_allow_html=True,
        )
        render_bias_score_card(
            "Disparate Impact Bias Score",
            disparate_impact_bias_score,
            "Higher scores indicate larger average shortfalls below the 80 percent rule across the four protected variables.",
        )
        for sensitive_column, label in FAIRNESS_DIMENSIONS:
            render_disparate_impact_ratio_section(dataframe, sensitive_column, label, ollama_model)

    with summary_tab:
        st.markdown(
            "<div class='alp-note'><strong>Summary logic:</strong> This view combines the 12 generated insights from the three fairness metrics and synthesizes them into one bullet point each for Gender, Age, Region, and Religion.</div>",
            unsafe_allow_html=True,
        )
        render_bias_score_card(
            "Overall Bias Score",
            overall_bias_score,
            "This is the average of the Demographic Parity, Equalized Odds, and Disparate Impact Bias Scores.",
        )
        render_summary_tab(ollama_model)

def render_dashboard_view(current_view: str, filtered_dataset: pd.DataFrame, ollama_model: str) -> None:
    if current_view == "Overview":
        render_header()
        render_overview_kpis_with_bias(filtered_dataset)
        render_dataset_profile(filtered_dataset)
        render_score_landscape(filtered_dataset)
        render_population_distribution(filtered_dataset)
        render_bias_context(filtered_dataset)
        render_decision_summary(filtered_dataset)
        render_business_cuts(filtered_dataset)

    if current_view == "Fairness Audit Agent":
        render_fairness_audit_tab(filtered_dataset, ollama_model)

    if current_view == "Recruitment Insights Agent":
        render_recruitment_insights_tab(filtered_dataset, ollama_model)

def render_dashboard_loading_state(current_view: str) -> None:
    st.markdown(
        f"""
        <section class="alp-hero">
            <div class="alp-hero-kicker">Dashboard Status</div>
            <h1>{current_view} is refreshing</h1>
            <p>
                Applying the selected filters and rebuilding the full dashboard view. Updated data, charts,
                tables, and insights will appear once the refresh is complete.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    inject_page_styles()
    st.sidebar.title("ISB ALP Overview")
    uploaded_file = st.sidebar.file_uploader("Upload Excel dataset", type=["xlsx"])
    ollama_model = st.sidebar.text_input("Ollama model", value="llama3.2")

    try:
        file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        dataset = load_dataset(file_bytes)
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")
        st.stop()

    filters = render_sidebar_filter_form(dataset)
    current_view = st.session_state.get("dashboard_current_view", "Overview")

    refresh_phase = st.session_state.get("dashboard_refresh_phase")

    if refresh_phase == "show_loading":
        render_dashboard_loading_state(current_view)
        st.info(f"{current_view} dashboard loading. Applying selected filters and refreshing all views...")
        st.session_state["dashboard_refresh_phase"] = "apply_filters"
        st.rerun()
    elif refresh_phase == "apply_filters":
        render_dashboard_loading_state(current_view)
        filtered_dataset = apply_filters(
            dataset,
            departments=filters["departments"],
            job_roles=filters["job_roles"],
            months=filters["months"],
            geographies=filters["geographies"],
        )

        if filtered_dataset.empty:
            st.warning("No records match the selected filters. Adjust the sidebar selections and try again.")
            st.session_state["dashboard_refresh_phase"] = None
            st.stop()

        status = st.status("Running Status", expanded=True)
        status.write("Applying global filters to the dataset.")
        status.write("Generating Demographic Parity insights across Gender, Age, Region, and Religion.")
        status.write("Generating Equalized Odds insights across Gender, Age, Region, and Religion.")
        status.write("Generating Disparate Impact Ratio insights across Gender, Age, Region, and Religion.")
        status.write("Generating the cross-metric summary insight.")
        precompute_metric_insights(filtered_dataset, ollama_model)
        status.update(label="Running Status: completed", state="complete", expanded=False)

        st.session_state["dashboard_refresh_phase"] = None
        st.rerun()
    else:
        filtered_dataset = apply_filters(
            dataset,
            departments=filters["departments"],
            job_roles=filters["job_roles"],
            months=filters["months"],
            geographies=filters["geographies"],
        )

        if filtered_dataset.empty:
            st.warning("No records match the selected filters. Adjust the sidebar selections and try again.")
            st.stop()

        st.markdown("<div class='alp-nav-shell'>", unsafe_allow_html=True)
        current_view = st.radio(
            "Navigation",
            options=["Overview", "Fairness Audit Agent", "Recruitment Insights Agent"],
            horizontal=True,
            label_visibility="collapsed",
            key="dashboard_current_view",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        render_dashboard_view(current_view, filtered_dataset, ollama_model)

if __name__ == "__main__":
    main()
