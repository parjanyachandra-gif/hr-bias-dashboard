"""Streamlit overview page for the ISB Action Learning HR bias audit project."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib import error, request

import pandas as pd
import streamlit as st

APP_TITLE = "ISB Action Learning: HR Bias Audit Overview"
DEFAULT_DATASET = Path(__file__).with_name("AI_Hiring_Bias_Dataset_200K_v2.xlsx")
FAVORABLE_LABEL = "Hire"
AI_DECISION_COLUMN = "ai_recommendation"
THEME_TEXT = "Responsible AI"
DEMOGRAPHIC_COLUMNS = ["gender", "age_group", "religion", "geography"]
BUSINESS_COLUMNS = ["department", "job_role", "application_month"]
FAIRNESS_DIMENSIONS = [
    ("gender", "Gender"),
    ("age_group", "Age"),
    ("geography", "Region"),
    ("religion", "Religion"),
]
DISPARATE_IMPACT_THRESHOLD = 0.80
SCORE_COLUMNS = [
    "experience_years",
    "skill_score",
    "interview_score",
    "communication_score",
    "certification_count",
    "employment_gap_months",
]
PROFILE_GROUPS = {
    "Demographic columns": DEMOGRAPHIC_COLUMNS,
    "Qualification columns": ["education", "university_tier", "experience_years", "certification_count"],
    "Assessment columns": ["skill_score", "interview_score", "communication_score"],
    "AI decision column": [AI_DECISION_COLUMN],
}

def inject_page_styles() -> None:

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(11, 92, 122, 0.14), transparent 32%),
                linear-gradient(180deg, #f6f7f4 0%, #eef2ec 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(246, 247, 244, 0) !important;
            height: 2.25rem;
        }
        [data-testid="stToolbar"] {
            top: 0.2rem;
            right: 0.75rem;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(11, 92, 122, 0.14), transparent 32%),
                linear-gradient(180deg, #f6f7f4 0%, #eef2ec 100%);
        }
        .block-container {
            max-width: 1240px;
            padding-top: 0.35rem;
            padding-bottom: 2rem;
        }
        .alp-hero {
            background: linear-gradient(135deg, #0b3c49 0%, #17646f 55%, #d98c3f 130%);
            color: #f9faf7;
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            box-shadow: 0 24px 48px rgba(11, 60, 73, 0.16);
            margin-bottom: 1.1rem;
        }
        .alp-hero-kicker {
            display: inline-block;
            background: rgba(255, 255, 255, 0.16);
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }
        .alp-hero h1 {
            margin: 0;
            font-size: 2.35rem;
            line-height: 1.08;
            font-weight: 700;
        }
        .alp-hero p {
            margin: 0.7rem 0 0 0;
            max-width: 900px;
            font-size: 1rem;
            line-height: 1.6;
            color: rgba(249, 250, 247, 0.92);
        }
        .alp-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.25rem 0;
        }
        .alp-stat-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(12, 61, 72, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 30px rgba(12, 61, 72, 0.08);
        }
        .alp-stat-label {
            color: #526067;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .alp-stat-value {
            color: #0d2f39;
            font-size: 1.85rem;
            line-height: 1.1;
            font-weight: 700;
        }
        .alp-stat-caption {
            color: #5e6a70;
            font-size: 0.92rem;
            margin-top: 0.35rem;
            line-height: 1.45;
        }
        .alp-section-title {
            color: #0d2f39;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.2rem 0 0.25rem 0;
        }
        .alp-section-text {
            color: #5a666c;
            font-size: 0.96rem;
            margin-bottom: 0.8rem;
        }
        .alp-note {
            background: rgba(217, 140, 63, 0.10);
            border-left: 4px solid #d98c3f;
            padding: 0.9rem 1rem;
            border-radius: 0 16px 16px 0;
            color: #4b3c2e;
            margin: 0.7rem 0 1rem 0;
        }
        .alp-nav-shell {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(12, 61, 72, 0.10);
            border-radius: 22px;
            padding: 0.5rem 0.8rem 0.25rem 0.8rem;
            box-shadow: 0 10px 22px rgba(12, 61, 72, 0.08);
            margin: 0.3rem 0 1rem 0;
        }
        div[role="radiogroup"] {
            gap: 0.75rem;
        }
        div[role="radiogroup"] label {
            flex: 1 1 0;
            min-width: 220px;
            background: rgba(245, 248, 247, 0.95);
            border: 1px solid rgba(12, 61, 72, 0.12);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin-right: 0.2rem;
        }
        div[role="radiogroup"] label p {
            color: #24434d;
            font-weight: 700;
            font-size: 1rem;
        }
        div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #0f4e5b 0%, #17646f 100%);
            border-color: #0f4e5b;
            box-shadow: 0 12px 22px rgba(15, 78, 91, 0.18);
        }
        div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff;
        }
        @media (max-width: 900px) {
            .alp-card-grid {
                grid-template-columns: 1fr;
            }
            .alp-hero h1 {
                font-size: 1.85rem;
            }
            div[role="radiogroup"] label {
                min-width: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def format_count(value: int) -> str:
    return f"{value:,}"

def render_section_intro(title: str, description: str) -> None:
    st.markdown(
        f"<div class='alp-section-title'>{title}</div><div class='alp-section-text'>{description}</div>",
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes | None) -> pd.DataFrame:
    if file_bytes is None:
        if not DEFAULT_DATASET.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DEFAULT_DATASET}. Upload the Excel file from the sidebar."
            )
        dataframe = pd.read_excel(DEFAULT_DATASET)
    else:
        dataframe = pd.read_excel(io.BytesIO(file_bytes))

    dataframe.columns = [column.strip() for column in dataframe.columns]

    for column in SCORE_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    string_columns = [
        "candidate_id",
        *DEMOGRAPHIC_COLUMNS,
        "education",
        "university_tier",
        *BUSINESS_COLUMNS,
        AI_DECISION_COLUMN,
    ]
    for column in string_columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    dataframe["candidate_id"] = dataframe["candidate_id"].str.replace(r"\.0$", "", regex=True)
    return dataframe

def apply_filters(
    dataframe: pd.DataFrame,
    departments: list[str],
    job_roles: list[str],
    months: list[str],
    geographies: list[str],
) -> pd.DataFrame:
    filtered = dataframe.copy()
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if job_roles:
        filtered = filtered[filtered["job_role"].isin(job_roles)]
    if months:
        filtered = filtered[filtered["application_month"].isin(months)]
    if geographies:
        filtered = filtered[filtered["geography"].isin(geographies)]
    return filtered

def format_percent(value: float) -> str:
    return f"{value:.1%}"

def compute_kpis(dataframe: pd.DataFrame) -> dict[str, float | int]:

    ai_hire_rate = (dataframe[AI_DECISION_COLUMN] == FAVORABLE_LABEL).mean()
    ai_hire_count = int((dataframe[AI_DECISION_COLUMN] == FAVORABLE_LABEL).sum())
    ai_reject_count = int(len(dataframe) - ai_hire_count)
    return {
        "records": int(len(dataframe)),
        "ai_hire_rate": ai_hire_rate,
        "ai_hire_count": ai_hire_count,
        "ai_reject_count": ai_reject_count,
        "departments": int(dataframe["department"].nunique()),
        "job_roles": int(dataframe["job_role"].nunique()),
    }

def summarize_distribution(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        dataframe[column]
        .value_counts(dropna=False)
        .rename_axis(column)
        .to_frame("candidate_count")
        .reset_index()
    )
    summary["population_share"] = summary["candidate_count"] / len(dataframe)
    return summary

def summarize_group_rates(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:

    summary = (
        dataframe.groupby(column, dropna=False)
        .agg(
            candidate_count=("candidate_id", "size"),
            ai_hire_rate=(AI_DECISION_COLUMN, lambda values: (values == FAVORABLE_LABEL).mean()),
            avg_skill_score=("skill_score", "mean"),
            avg_interview_score=("interview_score", "mean"),
            avg_communication_score=("communication_score", "mean"),
        )
        .reset_index()
        .sort_values("candidate_count", ascending=False)
    )
    return summary

def build_funnel(dataframe: pd.DataFrame) -> pd.DataFrame:

    total_applicants = len(dataframe)
    ai_hires = int((dataframe[AI_DECISION_COLUMN] == FAVORABLE_LABEL).sum())
    ai_rejects = total_applicants - ai_hires
    return pd.DataFrame(
        {
            "Stage": [
                "Total Applicants",
                "AI Recommended Hire",
                "AI Rejections",
            ],
            "Count": [total_applicants, ai_hires, ai_rejects],
        }
    )

def build_score_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SCORE_COLUMNS:
        rows.append(
            {
                "metric": column,
                "mean": dataframe[column].mean(),
                "median": dataframe[column].median(),
                "min": dataframe[column].min(),
                "max": dataframe[column].max(),
            }
        )
    return pd.DataFrame(rows)

def style_summary_table(dataframe: pd.DataFrame, percent_columns: list[str]) -> pd.DataFrame:
    formatted = dataframe.copy()
    for column in percent_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_percent)
    numeric_round_columns = [
        "avg_skill_score",
        "avg_interview_score",
        "avg_communication_score",
        "mean",
        "median",
        "min",
        "max",
    ]
    for column in numeric_round_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].round(1)
    return formatted

def format_ratio(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"

def compute_fairness_metrics(
    dataframe: pd.DataFrame,
    sensitive_column: str,
    threshold: float = DISPARATE_IMPACT_THRESHOLD,
) -> tuple[pd.DataFrame, float]:
    audit_frame = dataframe[[sensitive_column, AI_DECISION_COLUMN]].dropna().copy()
    audit_frame["selected"] = (audit_frame[AI_DECISION_COLUMN] == FAVORABLE_LABEL).astype(int)

    metrics = (
        audit_frame.groupby(sensitive_column, dropna=False)
        .agg(candidate_count=("selected", "size"), selected=("selected", "sum"))
        .reset_index()
    )

    overall_selection_rate = audit_frame["selected"].mean()
    metrics["selection_rate"] = metrics["selected"] / metrics["candidate_count"]
    reference_rate = metrics["selection_rate"].max()
    metrics["demographic_parity_gap"] = metrics["selection_rate"] - overall_selection_rate
    if pd.notna(reference_rate) and reference_rate > 0:
        metrics["disparate_impact_ratio"] = metrics["selection_rate"] / reference_rate
    else:
        metrics["disparate_impact_ratio"] = pd.NA
    metrics["flagged"] = metrics["disparate_impact_ratio"].fillna(1.0) < threshold
    metrics["diversity_share"] = metrics["candidate_count"] / len(audit_frame)
    return metrics.sort_values("selection_rate", ascending=False), overall_selection_rate

def build_fairness_flag_summary(
    dataframe: pd.DataFrame,
    dimensions: list[tuple[str, str]],
    threshold: float = DISPARATE_IMPACT_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for column, label in dimensions:
        metrics, overall_selection_rate = compute_fairness_metrics(dataframe, column, threshold)
        if metrics.empty:
            continue
        worst_group = metrics.sort_values("disparate_impact_ratio", ascending=True).iloc[0]
        rows.append(
            {
                "dimension": label,
                "groups": int(len(metrics)),
                "overall_selection_rate": overall_selection_rate,
                "worst_group": worst_group[column],
                "worst_disparate_impact": float(worst_group["disparate_impact_ratio"]),
                "largest_parity_gap": float(metrics["demographic_parity_gap"].abs().max()),
                "flagged_groups": int(metrics["flagged"].sum()),
            }
        )
    return pd.DataFrame(rows)

def build_ollama_prompt(summary_frame: pd.DataFrame, threshold: float) -> str:
    lines = [
        "You are a fairness audit assistant for an AI hiring dashboard.",
        "Summarize bias indicators, disparities, and diversity insights in 4 short bullet points.",
        f"Use the disparate impact threshold of {threshold:.2f}.",
        "Do not mention missing data or implementation details.",
        "Audit summary:",
    ]
    for row in summary_frame.to_dict("records"):
        lines.append(
            f"- {row['dimension']}: groups={row['groups']}, overall_selection_rate={row['overall_selection_rate']:.3f}, "
            f"worst_group={row['worst_group']}, worst_disparate_impact={row['worst_disparate_impact']:.3f}, "
            f"largest_parity_gap={row['largest_parity_gap']:.3f}, flagged_groups={row['flagged_groups']}"
        )
    return "\n".join(lines)

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

def render_ollama_audit_summary(summary_frame: pd.DataFrame, model: str) -> None:
    render_section_intro(
        "Ollama narrative",
        "Use a local Ollama model to convert the fairness signals into a short audit narrative for faculty or stakeholders.",
    )
    if summary_frame.empty:
        st.info("No fairness summary is available for Ollama because the filtered dataset is empty.")
        return

    prompt = build_ollama_prompt(summary_frame, DISPARATE_IMPACT_THRESHOLD)
    if st.button("Generate Ollama fairness summary", key="generate_ollama_fairness"):
        try:
            with st.spinner(f"Generating fairness summary with Ollama model '{model}'..."):
                response = query_ollama(prompt, model)
            if response:
                st.success("Ollama summary generated.")
                st.markdown(response)
            else:
                st.warning("Ollama returned an empty summary.")
        except error.URLError:
            st.warning("Unable to reach Ollama at http://localhost:11434. Start Ollama and retry.")
        except Exception as exc:
            st.warning(f"Ollama summary failed: {exc}")

    with st.expander("Show Ollama prompt"):
        st.code(prompt)

def render_sidebar(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    st.sidebar.header("Dashboard Filters")
    departments = st.sidebar.multiselect(
        "Department",
        options=sorted(dataframe["department"].dropna().unique().tolist()),
    )
    job_roles = st.sidebar.multiselect(
        "Job Role",
        options=sorted(dataframe["job_role"].dropna().unique().tolist()),
    )
    months = st.sidebar.multiselect(
        "Application Month",
        options=sorted(dataframe["application_month"].dropna().unique().tolist()),
    )
    geographies = st.sidebar.multiselect(
        "Geography",
        options=sorted(dataframe["geography"].dropna().unique().tolist()),
    )
    return {
        "departments": departments,
        "job_roles": job_roles,
        "months": months,
        "geographies": geographies,
    }

def render_header() -> None:

    st.markdown(
        f"""
        <section class="alp-hero">
            <div class="alp-hero-kicker">{THEME_TEXT}</div>
            <h1>{APP_TITLE}</h1>
            <p>
                This first page is a presentation-ready overview of the synthetic hiring dataset. It frames the
                candidate pool, AI recommendations, and operational slices before moving into fairness
                diagnostics.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='alp-note'><strong>Problem statement:</strong> Design an audit tool that evaluates AI-assisted hiring decisions for demographic bias, visualises disparities across groups, and supports structured remediation.</div>",
        unsafe_allow_html=True,
    )

def render_kpis(dataframe: pd.DataFrame) -> None:

    metrics = compute_kpis(dataframe)
    st.markdown(
        f"""
        <section class="alp-card-grid">
            <div class="alp-stat-card">
                <div class="alp-stat-label">Total Candidates</div>
                <div class="alp-stat-value">{format_count(int(metrics['records']))}</div>
                <div class="alp-stat-caption">Filtered candidate records currently included in the audit view.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Hire Rate</div>
                <div class="alp-stat-value">{format_percent(float(metrics['ai_hire_rate']))}</div>
                <div class="alp-stat-caption">Share of candidates the AI system recommended for hiring.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Recommended Hires</div>
                <div class="alp-stat-value">{format_count(int(metrics['ai_hire_count']))}</div>
                <div class="alp-stat-caption">Absolute count of candidates recommended for hiring by the AI system.</div>
            </div>
            <div class="alp-stat-card">
                <div class="alp-stat-label">AI Recommended Rejects</div>
                <div class="alp-stat-value">{format_count(int(metrics['ai_reject_count']))}</div>
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
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_dataset_profile(dataframe: pd.DataFrame) -> None:

    render_section_intro(
        "Dataset Profile",
        "A quick inventory of what this hiring dataset contains and how complete each field is before any bias metrics are calculated.",
    )
    visible_dataframe = dataframe.drop(columns=["final_hiring_decision"], errors="ignore")
    left, right = st.columns((1.1, 1))

    with left:
        profile_rows = []
        for label, columns in PROFILE_GROUPS.items():
            profile_rows.append({"Section": label, "Columns": ", ".join(columns)})
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Data Coverage")
        coverage = pd.DataFrame(
            {
                "Column": visible_dataframe.columns,
                "Missing %": (visible_dataframe.isna().mean() * 100).round(2),
                "Unique Values": [visible_dataframe[column].nunique(dropna=False) for column in visible_dataframe.columns],
            }
        )
        st.dataframe(coverage, use_container_width=True, hide_index=True)

def render_population_distribution(dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "Population Distribution",
        "Use this cut to understand representation across protected and geographic groups. This matters because very small groups can distort later fairness metrics.",
    )
    selected_dimension = st.selectbox(
        "Choose demographic distribution",
        options=DEMOGRAPHIC_COLUMNS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    summary = summarize_distribution(dataframe, selected_dimension)
    chart_frame = summary.set_index(selected_dimension)[["candidate_count", "population_share"]]
    st.bar_chart(chart_frame)
    st.dataframe(
        style_summary_table(summary, percent_columns=["population_share"]),
        use_container_width=True,
        hide_index=True,
    )

def render_decision_summary(dataframe: pd.DataFrame) -> None:

    render_section_intro(
        "AI Recommendation Flow",
        "This section summarises how the AI system split the candidate pool into recommended hires and recommended rejects.",
    )
    left, right = st.columns(2)

    with left:
        funnel = build_funnel(dataframe).set_index("Stage")
        st.markdown("#### Hiring Funnel Snapshot")
        st.bar_chart(funnel)
        st.dataframe(build_funnel(dataframe), use_container_width=True, hide_index=True)

    with right:
        decision_comparison = pd.DataFrame(
            {
                "Decision": ["AI Hire Rate", "AI Reject Rate"],
                "Rate": [
                    (dataframe[AI_DECISION_COLUMN] == FAVORABLE_LABEL).mean(),
                    (dataframe[AI_DECISION_COLUMN] != FAVORABLE_LABEL).mean(),
                ],
            }
        ).set_index("Decision")
        st.markdown("#### Recommendation Split")
        st.bar_chart(decision_comparison)
        st.caption("This shows the overall balance between positive and negative AI recommendations.")

def render_business_cuts(dataframe: pd.DataFrame) -> None:

    render_section_intro(
        "Business Cuts",
        "Operational slices often explain where disparities originate. Review departments, roles, and seasonality before making fairness claims in isolation.",
    )
    selected_cut = st.selectbox(
        "Choose business cut",
        options=BUSINESS_COLUMNS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    summary = summarize_group_rates(dataframe, selected_cut)
    chart_frame = summary.set_index(selected_cut)[["candidate_count", "ai_hire_rate"]]
    st.bar_chart(chart_frame)
    st.dataframe(
        style_summary_table(summary, percent_columns=["ai_hire_rate"]),
        use_container_width=True,
        hide_index=True,
    )

def render_score_landscape(dataframe: pd.DataFrame) -> None:
    render_section_intro(
        "Score Landscape",
        "These summary statistics provide context on candidate quality, assessment spread, and the overall scale of experience and gaps in the filtered dataset.",
    )
    score_summary = build_score_summary(dataframe)
    st.dataframe(style_summary_table(score_summary, percent_columns=[]), use_container_width=True, hide_index=True)

def render_bias_context(dataframe: pd.DataFrame) -> None:

    render_section_intro(
        "Potential Bias-Relevant Context",
        "This is an early warning view, not a verdict. It highlights groups where AI recommendation rates visibly diverge and may warrant deeper audit analysis later.",
    )
    selected_dimension = st.selectbox(
        "Choose demographic cut for context",
        options=["gender", "age_group", "religion"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    summary = summarize_group_rates(dataframe, selected_dimension)
    chart_frame = summary.set_index(selected_dimension)[["ai_hire_rate"]]
    st.bar_chart(chart_frame)
    st.dataframe(
        style_summary_table(summary, percent_columns=["ai_hire_rate"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("This section does not claim bias. It highlights where later fairness analysis should inspect differences more carefully.")

def render_fairness_audit_tab(dataframe: pd.DataFrame, ollama_model: str) -> None:

    render_section_intro(
        "Fairness Audit",
        "Agent 2 evaluates hiring patterns across gender, age, region, and religion to surface bias indicators, disparity flags, and diversity insights from AI recommendations.",
    )

    summary_frame = build_fairness_flag_summary(dataframe, FAIRNESS_DIMENSIONS)
    if not summary_frame.empty:
        min_disparate_impact = float(summary_frame["worst_disparate_impact"].min())
        max_parity_gap = float(summary_frame["largest_parity_gap"].max())
        total_flags = int(summary_frame["flagged_groups"].sum())
        top_dimension = summary_frame.sort_values(
            ["flagged_groups", "worst_disparate_impact", "largest_parity_gap"],
            ascending=[False, True, False],
        ).iloc[0]

        st.markdown(
            f"""
            <section class="alp-card-grid">
                <div class="alp-stat-card">
                    <div class="alp-stat-label">Dimensions Audited</div>
                    <div class="alp-stat-value">{len(summary_frame)}</div>
                    <div class="alp-stat-caption">Gender, age, region, and religion are scanned on the current filtered slice.</div>
                </div>
                <div class="alp-stat-card">
                    <div class="alp-stat-label">Flagged Groups</div>
                    <div class="alp-stat-value">{total_flags}</div>
                    <div class="alp-stat-caption">Groups below the disparate impact threshold of {DISPARATE_IMPACT_THRESHOLD:.2f}.</div>
                </div>
                <div class="alp-stat-card">
                    <div class="alp-stat-label">Worst Disparate Impact</div>
                    <div class="alp-stat-value">{format_ratio(min_disparate_impact)}</div>
                    <div class="alp-stat-caption">Lowest ratio currently appears in {top_dimension['dimension']}.</div>
                </div>
                <div class="alp-stat-card">
                    <div class="alp-stat-label">Largest Parity Gap</div>
                    <div class="alp-stat-value">{format_percent(max_parity_gap)}</div>
                    <div class="alp-stat-caption">Biggest gap from the overall AI selection rate across the four audited dimensions.</div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        top_left, top_right = st.columns((1.2, 1))
        with top_left:
            st.dataframe(
                summary_frame.assign(
                    overall_selection_rate=summary_frame["overall_selection_rate"].map(format_percent),
                    worst_disparate_impact=summary_frame["worst_disparate_impact"].map(format_ratio),
                    largest_parity_gap=summary_frame["largest_parity_gap"].map(format_percent),
                ),
                use_container_width=True,
                hide_index=True,
            )
        with top_right:
            render_section_intro(
                "Bias indicators",
                f"This view combines severity and flag counts so low-visibility risk still shows up even when no group crosses the hard threshold of {DISPARATE_IMPACT_THRESHOLD:.2f}.",
            )
            bias_chart = summary_frame.set_index("dimension")[["flagged_groups", "worst_disparate_impact", "largest_parity_gap"]].copy()
            bias_chart["disparate_impact_risk"] = 1 - bias_chart["worst_disparate_impact"]
            st.bar_chart(bias_chart[["flagged_groups", "disparate_impact_risk", "largest_parity_gap"]])
            st.caption("`Disparate impact risk` is shown as `1 - worst disparate impact`, so higher bars indicate more disparity.")

    for column, label in FAIRNESS_DIMENSIONS:
        metrics, overall_selection_rate = compute_fairness_metrics(dataframe, column)
        render_section_intro(
            f"{label} audit",
            f"Overall AI selection rate is {format_percent(float(overall_selection_rate))}. This view shows representation, selection disparity, and flagged groups for {label.lower()}.",
        )
        left, right = st.columns((1, 1.15))
        with left:
            chart_frame = metrics.set_index(column)[["selection_rate", "disparate_impact_ratio"]]
            st.bar_chart(chart_frame)
        with right:
            st.dataframe(
                metrics.assign(
                    diversity_share=metrics["diversity_share"].map(format_percent),
                    selection_rate=metrics["selection_rate"].map(format_percent),
                    demographic_parity_gap=metrics["demographic_parity_gap"].map(format_percent),
                    disparate_impact_ratio=metrics["disparate_impact_ratio"].map(format_ratio),
                ),
                use_container_width=True,
                hide_index=True,
            )

    render_ollama_audit_summary(summary_frame, ollama_model)

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

    filters = render_sidebar(dataset)
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
        options=["Overview", "Fairness Audit"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if current_view == "Overview":
        render_header()
        render_kpis(filtered_dataset)
        render_dataset_profile(filtered_dataset)
        render_score_landscape(filtered_dataset)
        render_population_distribution(filtered_dataset)
        render_bias_context(filtered_dataset)
        render_decision_summary(filtered_dataset)
        render_business_cuts(filtered_dataset)

    if current_view == "Fairness Audit":
        render_fairness_audit_tab(filtered_dataset, ollama_model)

if __name__ == "__main__":
    main()