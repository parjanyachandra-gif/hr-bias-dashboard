"""Agent 3 - Recruitment Insights Agent.

A RAG-based conversational agent for the AI Bias Audit Dashboard.

Design
------
The dataset is structured/tabular, so naive "embed each row" RAG cannot answer
aggregation questions ("what % of women were selected in Sales?"). Instead this
module builds a RAG knowledge base out of *precomputed aggregation chunks*:
selection rates by every dimension, cross-cuts, score profiles, fairness
summaries, plus schema and metric definitions. Each chunk is a short natural
language paragraph with the real numbers embedded.

Pipeline:
  1. build_knowledge_base(df)  -> list of insight documents (with numbers)
  2. InsightRetriever          -> TF-IDF retrieval (offline, no extra services)
  3. query_ollama              -> grounded generation over the retrieved chunks

An optional "precise mode" uses the LLM to emit a single safe pandas expression
that is validated and executed for arbitrary numeric drilldowns.

Wire into alp_project_review2.py by importing render_recruitment_insights_tab
and adding "Recruitment Insights Agent" to the navigation radio (see the
docstring at the bottom of this file).
"""

from __future__ import annotations

import re
from urllib import error

import pandas as pd
import streamlit as st

from alp_project import (
    AI_DECISION_COLUMN,
    BUSINESS_COLUMNS,
    DEMOGRAPHIC_COLUMNS,
    FAVORABLE_LABEL,
    query_ollama,
    render_section_intro,
)

OBSERVED_OUTCOME_COLUMN = "final_hiring_decision"
TOP_K_CHUNKS = 6
MAX_GROUP_ROWS = 40  # guard against huge category explosions in a chunk

# Columns that make sense to slice by for recruitment insight questions.
CATEGORICAL_DIMENSIONS = [
    ("gender", "Gender"),
    ("age_group", "Age group"),
    ("religion", "Religion"),
    ("geography", "Geography"),
    ("education", "Education"),
    ("university_tier", "University tier"),
    ("department", "Department"),
    ("job_role", "Job role"),
    ("application_month", "Application month"),
]
SCORE_COLUMNS = [
    "experience_years",
    "skill_score",
    "interview_score",
    "communication_score",
    "certification_count",
    "employment_gap_months",
]
# Cross-cuts worth precomputing (protected variable x business dimension).
CROSS_CUTS = [
    ("gender", "department"),
    ("gender", "job_role"),
    ("age_group", "department"),
    ("geography", "department"),
    ("religion", "department"),
    ("gender", "application_month"),
]


# --------------------------------------------------------------------------- #
# 1. Knowledge base construction (precomputed aggregation chunks)             #
# --------------------------------------------------------------------------- #
def _selected(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.strip() == FAVORABLE_LABEL).astype(int)


def _pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value * 100:.1f}%"


def _schema_chunks(df: pd.DataFrame) -> list[dict]:
    schema_text = (
        "Dataset schema and columns. This hiring dataset has "
        f"{len(df):,} candidate records with the following columns: "
        + ", ".join(df.columns)
        + ". Demographic / protected columns: gender, age_group, religion, geography. "
        "Qualification columns: education, university_tier, experience_years, "
        "certification_count. Assessment scores: skill_score, interview_score, "
        "communication_score (0-100 scale). Business columns: department, job_role, "
        "application_month. ai_recommendation is the AI system's hire/reject "
        "recommendation. final_hiring_decision is the observed real-world outcome."
    )
    definitions_text = (
        "Metric definitions. Selection rate / AI hire rate = share of candidates "
        "the AI recommended for hire (ai_recommendation == 'Hire'). Observed hire "
        "rate = share with final_hiring_decision == 'Hire'. Demographic Parity "
        "compares each group's selection rate with the overall rate. Disparate "
        "Impact Ratio = group selection rate / highest group's selection rate; "
        "below 0.80 signals adverse impact under the 80% rule. Equalized Odds "
        "compares true positive and false positive rates across groups."
    )
    return [
        {"id": "schema", "title": "Dataset schema", "text": schema_text},
        {"id": "definitions", "title": "Metric definitions", "text": definitions_text},
    ]


def _overall_chunk(df: pd.DataFrame) -> dict:
    ai_rate = _selected(df[AI_DECISION_COLUMN]).mean()
    obs_rate = _selected(df[OBSERVED_OUTCOME_COLUMN]).mean()
    text = (
        f"Overall recruitment summary. Total candidates: {len(df):,}. "
        f"Overall AI hire rate (selection rate): {_pct(ai_rate)}. "
        f"Overall observed hire rate: {_pct(obs_rate)}. "
        f"Distinct departments: {df['department'].nunique()}. "
        f"Distinct job roles: {df['job_role'].nunique()}. "
        f"Average skill score: {df['skill_score'].mean():.1f}, "
        f"average interview score: {df['interview_score'].mean():.1f}, "
        f"average communication score: {df['communication_score'].mean():.1f}, "
        f"average experience years: {df['experience_years'].mean():.1f}."
    )
    return {"id": "overall", "title": "Overall summary", "text": text}


def _dimension_chunk(df: pd.DataFrame, col: str, label: str) -> dict | None:
    if col not in df.columns:
        return None
    frame = df[[col, AI_DECISION_COLUMN, OBSERVED_OUTCOME_COLUMN]].copy()
    frame["ai_sel"] = _selected(frame[AI_DECISION_COLUMN])
    frame["obs_sel"] = _selected(frame[OBSERVED_OUTCOME_COLUMN])
    grp = (
        frame.groupby(col, dropna=False)
        .agg(n=("ai_sel", "size"), ai=("ai_sel", "mean"), obs=("obs_sel", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(MAX_GROUP_ROWS)
    )
    overall = frame["ai_sel"].mean()
    lines = []
    for _, r in grp.iterrows():
        gap = r["ai"] - overall
        lines.append(
            f"{r[col]}: {int(r['n']):,} candidates, AI hire rate {_pct(r['ai'])}, "
            f"observed hire rate {_pct(r['obs'])}, gap vs overall {gap * 100:+.1f} pts"
        )
    text = (
        f"AI hire rate by {label.lower()}. Overall AI hire rate is {_pct(overall)}. "
        + " | ".join(lines)
    )
    return {"id": f"dim_{col}", "title": f"Hire rate by {label}", "text": text}


def _cross_chunk(df: pd.DataFrame, c1: str, c2: str) -> dict | None:
    if c1 not in df.columns or c2 not in df.columns:
        return None
    frame = df[[c1, c2, AI_DECISION_COLUMN]].copy()
    frame["ai_sel"] = _selected(frame[AI_DECISION_COLUMN])
    grp = (
        frame.groupby([c1, c2], dropna=False)
        .agg(n=("ai_sel", "size"), ai=("ai_sel", "mean"))
        .reset_index()
        .sort_values("n", ascending=False)
        .head(MAX_GROUP_ROWS)
    )
    lines = [
        f"{r[c1]} in {r[c2]}: {int(r['n']):,} candidates, AI hire rate {_pct(r['ai'])}"
        for _, r in grp.iterrows()
    ]
    text = f"AI hire rate by {c1} and {c2}. " + " | ".join(lines)
    return {"id": f"cross_{c1}_{c2}", "title": f"Hire rate by {c1} x {c2}", "text": text}


def _score_by_dim_chunk(df: pd.DataFrame, col: str, label: str) -> dict | None:
    if col not in df.columns:
        return None
    grp = (
        df.groupby(col, dropna=False)[
            ["skill_score", "interview_score", "communication_score", "experience_years"]
        ]
        .mean()
        .reset_index()
        .head(MAX_GROUP_ROWS)
    )
    lines = [
        f"{r[col]}: avg skill {r['skill_score']:.1f}, interview {r['interview_score']:.1f}, "
        f"communication {r['communication_score']:.1f}, experience {r['experience_years']:.1f} yrs"
        for _, r in grp.iterrows()
    ]
    text = f"Average candidate scores by {label.lower()}. " + " | ".join(lines)
    return {"id": f"score_{col}", "title": f"Scores by {label}", "text": text}


def _agreement_chunk(df: pd.DataFrame) -> dict:
    ai = _selected(df[AI_DECISION_COLUMN])
    obs = _selected(df[OBSERVED_OUTCOME_COLUMN])
    tp = int(((ai == 1) & (obs == 1)).sum())
    fp = int(((ai == 1) & (obs == 0)).sum())
    tn = int(((ai == 0) & (obs == 0)).sum())
    fn = int(((ai == 0) & (obs == 1)).sum())
    agree = (tp + tn) / len(df) if len(df) else 0
    text = (
        "Agreement between AI recommendation and observed final hiring decision. "
        f"AI recommended hire AND actually hired (true positive): {tp:,}. "
        f"AI recommended hire but not hired (false positive): {fp:,}. "
        f"AI recommended reject and not hired (true negative): {tn:,}. "
        f"AI recommended reject but actually hired (false negative): {fn:,}. "
        f"Overall agreement rate: {_pct(agree)}."
    )
    return {"id": "agreement", "title": "AI vs observed agreement", "text": text}


@st.cache_data(show_spinner=False)
def build_knowledge_base(df: pd.DataFrame) -> list[dict]:
    """Precompute the RAG knowledge base of insight chunks from the dataframe."""
    chunks: list[dict] = []
    chunks.extend(_schema_chunks(df))
    chunks.append(_overall_chunk(df))
    chunks.append(_agreement_chunk(df))
    for col, label in CATEGORICAL_DIMENSIONS:
        chunk = _dimension_chunk(df, col, label)
        if chunk:
            chunks.append(chunk)
    for col, label in [
        ("gender", "Gender"),
        ("age_group", "Age group"),
        ("geography", "Geography"),
        ("department", "Department"),
        ("job_role", "Job role"),
    ]:
        chunk = _score_by_dim_chunk(df, col, label)
        if chunk:
            chunks.append(chunk)
    for c1, c2 in CROSS_CUTS:
        chunk = _cross_chunk(df, c1, c2)
        if chunk:
            chunks.append(chunk)
    return chunks


# --------------------------------------------------------------------------- #
# 2. Retrieval (TF-IDF, offline-safe, sklearn with a pure-python fallback)    #
# --------------------------------------------------------------------------- #
class InsightRetriever:
    """TF-IDF retriever over the knowledge base chunks."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.corpus = [f"{d['title']}. {d['text']}" for d in docs]
        self._backend = "tfidf"
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._cosine_similarity = cosine_similarity
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(self.corpus)
        except Exception:
            # Fallback: simple token-overlap scoring, no external dependency.
            self._backend = "overlap"
            self._tokenized = [set(_tokenize(text)) for text in self.corpus]

    def search(self, query: str, k: int = TOP_K_CHUNKS) -> list[dict]:
        if self._backend == "tfidf":
            q_vec = self._vectorizer.transform([query])
            scores = self._cosine_similarity(q_vec, self._matrix)[0]
        else:
            q_tokens = set(_tokenize(query))
            scores = [
                len(q_tokens & doc_tokens) / (len(q_tokens) + 1)
                for doc_tokens in self._tokenized
            ]
        ranked = sorted(
            zip(self.docs, scores), key=lambda pair: pair[1], reverse=True
        )
        results = []
        for doc, score in ranked[:k]:
            if score <= 0:
                continue
            results.append({**doc, "score": float(score)})
        # Always keep schema/definitions available if nothing scored.
        if not results:
            results = [{**doc, "score": 0.0} for doc in self.docs[:k]]
        return results


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource(show_spinner=False)
def get_retriever(_docs: list[dict], signature: str) -> InsightRetriever:
    """Cache the retriever; `signature` invalidates it when data changes."""
    return InsightRetriever(_docs)


# --------------------------------------------------------------------------- #
# 3. Generation (grounded answer over retrieved chunks)                       #
# --------------------------------------------------------------------------- #
def build_rag_prompt(question: str, retrieved: list[dict]) -> str:
    context = "\n".join(f"[{i + 1}] {d['title']}: {d['text']}" for i, d in enumerate(retrieved))
    return f"""
You are Recruitment Insights Agent, a helpful analyst for an HR hiring dashboard.
You help non-technical HR users understand recruitment outcomes by answering
their questions in plain language.

Answer the user's question using ONLY the data context below. The context
contains precomputed statistics from the actual hiring dataset. Quote the exact
numbers from the context. If the specific cut the user asks about is not present
in the context, say you do not have that breakdown rather than guessing. Never
invent numbers.

Data context:
{context}

User question: {question}

Write a concise, direct answer (2-5 sentences or a short bullet list). Lead with
the key number. Do not mention code, prompts, retrieval, or implementation
details.
""".strip()


# --------------------------------------------------------------------------- #
# 4. Optional precise mode (LLM -> safe pandas expression)                    #
# --------------------------------------------------------------------------- #
_ALLOWED_PANDAS = re.compile(r"^[\w\s\.\(\)\[\]\"'=<>!&|+\-*/,:%]+$")
_FORBIDDEN = ("import", "__", "exec", "eval", "open", "os.", "sys.", "to_", "read_")


def build_pandas_prompt(question: str, df: pd.DataFrame) -> str:
    cols = ", ".join(df.columns)
    return f"""
You translate an HR analyst's question into ONE line of pandas code operating on
a DataFrame named df. Return ONLY the expression, no explanation, no markdown.

Rules:
- The favorable label for ai_recommendation and final_hiring_decision is "Hire".
- Use only df and its columns: {cols}
- No imports, no assignments, no file or system access. One expression only.
- Prefer returning a Series or small DataFrame.

Question: {question}
pandas expression:
""".strip()


def run_safe_pandas(expression: str, df: pd.DataFrame):
    expression = _clean_expression(expression)
    if not expression:
        raise ValueError("Empty expression.")
    if any(tok in expression for tok in _FORBIDDEN):
        raise ValueError("Expression contains a disallowed operation.")
    if not _ALLOWED_PANDAS.match(expression):
        raise ValueError("Expression contains disallowed characters.")
    return expression, eval(expression, {"__builtins__": {}}, {"df": df, "pd": pd})  # noqa: S307


def build_explain_prompt(question: str, computed: str) -> str:
    return f"""
You are Recruitment Insights Agent for an HR hiring dashboard. A precise
calculation has already been run on the real dataset to answer the user's
question. Explain the result in clear, plain language for a non-technical HR
user.

User question: {question}

Exact computed result (ground truth - use these numbers, do not change them):
{computed}

Write a concise, direct answer (2-4 sentences). Lead with the key number. If the
result is a table, call out the highest and lowest values. Do not mention code,
pandas, queries, or implementation details.
""".strip()


def _clean_expression(expression: str) -> str:
    """Strip markdown fences, language tags, and any leading assignment."""
    text = (expression or "").strip()
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[a-zA-Z_][\w]*\s*=\s*", "", line)
        return line.strip().rstrip(";")
    return ""


def _render_result(result) -> str:
    """Turn a pandas result into a compact text block for display / narration."""
    def _table(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown()
        except Exception:  # tabulate not installed - plain text fallback
            return frame.to_string()

    if isinstance(result, pd.DataFrame):
        return _table(result.head(40))
    if isinstance(result, pd.Series):
        return _table(result.head(40).to_frame())
    if isinstance(result, float):
        return f"{result:.4f}"
    return str(result)


# --------------------------------------------------------------------------- #
# 5. Streamlit UI                                                             #
# --------------------------------------------------------------------------- #
SUGGESTED_QUESTIONS = [
    "Which department has the lowest AI hire rate for women?",
    "How does the AI hire rate differ by geography?",
    "Do older age groups get selected less often?",
    "Which group has the highest interview scores?",
    "How often does the AI recommendation match the final hiring decision?",
]


def render_recruitment_insights_tab(dataframe: pd.DataFrame, ollama_model: str) -> None:
    render_section_intro(
        "Recruitment Insights Agent",
        "Ask questions about recruitment outcomes in plain language. Agent 3 "
        "retrieves precomputed statistics from the current filtered dataset and "
        "uses a local Ollama model to answer, grounded strictly in the data.",
    )

    docs = build_knowledge_base(dataframe)
    signature = f"{len(dataframe)}-{'-'.join(dataframe.columns)}"
    retriever = get_retriever(docs, signature)

    st.markdown(
        f"<div class='alp-note'><strong>How it works:</strong> The agent builds a "
        f"knowledge base of {len(docs)} precomputed insight summaries from the "
        "filtered data, retrieves the most relevant ones for your question, and "
        "answers using only those numbers so figures are always accurate.</div>",
        unsafe_allow_html=True,
    )

    precise_mode = st.toggle(
        "Precise mode - compute an exact answer for any question (recommended)",
        value=True,
        help="Type any question in your own words. The agent computes the exact "
        "answer on the real data, then explains it in plain language. Turn off "
        "for faster, conversational answers from precomputed summaries.",
    )

    st.caption("Type your own question below, or try one of these:")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[i].button(question, key=f"agent3_suggest_{i}"):
            st.session_state["agent3_pending"] = question

    history = st.session_state.setdefault("agent3_chat", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources used"):
                    for src in turn["sources"]:
                        st.markdown(f"**{src['title']}** — {src['text']}")

    typed = st.chat_input("Ask about recruitment outcomes...")
    question = typed or st.session_state.pop("agent3_pending", None)

    if not question:
        return

    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if precise_mode:
                answer, sources = _answer_precise(question, dataframe, ollama_model)
            else:
                answer, sources = _answer_rag(question, retriever, ollama_model)
            st.markdown(answer)
            if sources:
                with st.expander("Sources used"):
                    for src in sources:
                        st.markdown(f"**{src['title']}** — {src['text']}")
            history.append({"role": "assistant", "content": answer, "sources": sources})
        except error.URLError:
            msg = (
                "Unable to reach Ollama at http://localhost:11434. "
                "Start Ollama and retry."
            )
            st.warning(msg)
            history.append({"role": "assistant", "content": msg, "sources": []})
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not generate an answer: {exc}"
            st.warning(msg)
            history.append({"role": "assistant", "content": msg, "sources": []})


def _answer_rag(question: str, retriever: InsightRetriever, model: str) -> tuple[str, list[dict]]:
    retrieved = retriever.search(question, k=TOP_K_CHUNKS)
    prompt = build_rag_prompt(question, retrieved)
    with st.spinner(f"Thinking with Ollama model '{model}'..."):
        answer = query_ollama(prompt, model)
    sources = [{"title": d["title"], "text": d["text"]} for d in retrieved]
    return (answer or "The model returned an empty answer."), sources


def _answer_precise(question: str, df: pd.DataFrame, model: str) -> tuple[str, list[dict]]:
    """Compute an exact answer, then have the model explain it in plain English.

    Retries once if the first generated expression fails to run, so arbitrary
    typed questions are handled robustly.
    """
    pandas_prompt = build_pandas_prompt(question, df)
    last_error = None
    for _ in range(2):  # one retry on failure
        with st.spinner(f"Computing an exact answer with '{model}'..."):
            raw = query_ollama(pandas_prompt, model)
        try:
            expression, result = run_safe_pandas(raw, df)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            pandas_prompt = (
                build_pandas_prompt(question, df)
                + f"\n\nYour previous attempt failed with: {exc}. "
                "Return a corrected single-line pandas expression."
            )
    else:
        raise ValueError(
            f"Could not build a valid calculation for that question ({last_error}). "
            "Try rephrasing it a bit."
        )

    rendered = _render_result(result)
    with st.spinner("Explaining the result..."):
        narration = query_ollama(build_explain_prompt(question, rendered), model)
    answer = (narration or "").strip()
    # Always show the exact computed figures beneath the narration for trust.
    answer += f"\n\n<details><summary>Exact figures</summary>\n\n{rendered}\n\n</details>"
    sources = [{"title": "Computed from the dataset", "text": f"Query: {expression}"}]
    return answer, sources


# --------------------------------------------------------------------------- #
# Wiring instructions (for alp_project_review2.py)                            #
# --------------------------------------------------------------------------- #
"""
To add Agent 3 to the existing multi-page app in alp_project_review2.py:

1. At the top, add the import:

       from agent3_recruitment_insights import render_recruitment_insights_tab

2. In render_dashboard_view(), add a branch:

       if current_view == "Recruitment Insights Agent":
           render_recruitment_insights_tab(filtered_dataset, ollama_model)

3. In main(), extend the navigation radio options:

       options=["Overview", "Fairness Audit Agent", "Recruitment Insights Agent"],

That is the only change needed - it reuses load_dataset, apply_filters,
inject_page_styles, query_ollama, and the alp-* CSS from alp_project.py.
"""
