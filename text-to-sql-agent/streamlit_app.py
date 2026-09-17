from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from text2sql_agent.config import Settings
from text2sql_agent.contracts import AgentResponse
from text2sql_agent.ollama_models import discover_chat_models
from text2sql_agent.service import AnalyticsAgent

EXAMPLE_QUESTIONS = {
    "Delivery performance": "How does late delivery affect review scores?",
    "Repeat customers": "What percentage of customers placed more than one order?",
    "Payment mix": "Compare total payment value by payment type.",
}

st.set_page_config(
    page_title="Olist analytics agent",
    page_icon=":material/query_stats:",
    layout="wide",
)


@st.cache_resource(max_entries=4)
def get_agent(model_name: str) -> AnalyticsAgent:
    settings = replace(Settings.from_env(), chat_model=model_name)
    return AnalyticsAgent(settings)


@st.cache_data(ttl="30s", max_entries=8)
def available_chat_models(
    base_url: str, configured_model: str, embedding_model: str
) -> tuple[tuple[str, ...], str | None]:
    return discover_chat_models(
        base_url,
        configured_model=configured_model,
        embedding_model=embedding_model,
    )


@st.cache_data(ttl="30s", max_entries=8)
def service_status(model_name: str) -> dict[str, Any]:
    agent = get_agent(model_name)
    status: dict[str, Any] = {"model": agent.settings.chat_model}
    try:
        status["database"] = agent.database.health(read_only=True)["database_name"]
    except Exception as exc:  # noqa: BLE001 - health check must not crash the UI
        status["database_error"] = str(exc)
    try:
        status["knowledge_chunks"] = agent.knowledge.count()
    except Exception as exc:  # noqa: BLE001 - health check must not crash the UI
        status["knowledge_error"] = str(exc)
    try:
        agent.settings.assert_separate_database_identities()
        status["write_ready"] = True
    except RuntimeError:
        status["write_ready"] = False
    return status


def clear_analysis() -> None:
    for key in ("response", "activity_steps", "activity_state"):
        st.session_state.pop(key, None)


def revise_question(question: str) -> None:
    clear_analysis()
    st.session_state.question_input = question


def retry_question(question: str) -> None:
    clear_analysis()
    st.session_state.pending_question = question


def reject_write() -> None:
    clear_analysis()
    st.session_state.write_notice = "The proposed change was rejected. Nothing was written."


def select_example() -> None:
    selected = st.session_state.get("example_question")
    if selected:
        st.session_state.question_input = EXAMPLE_QUESTIONS[selected]


def render_chart(response: AgentResponse, frame: pd.DataFrame) -> bool:
    proposal = response.proposal
    if not proposal or proposal.chart_type == "none" or frame.empty:
        return False
    if not proposal.chart_x or proposal.chart_x not in frame.columns:
        return False
    y_columns = [column for column in proposal.chart_y if column in frame.columns]
    if not y_columns:
        return False

    chart_data = frame.set_index(proposal.chart_x)[y_columns]
    if proposal.chart_type == "bar":
        st.bar_chart(chart_data)
    elif proposal.chart_type == "line":
        st.line_chart(chart_data)
    elif proposal.chart_type == "area":
        st.area_chart(chart_data)
    elif proposal.chart_type == "scatter":
        st.scatter_chart(frame, x=proposal.chart_x, y=y_columns)
    else:
        return False
    return True


def render_activity(steps: list[str], state: str) -> None:
    if not steps:
        return
    label = "Analysis complete" if state == "complete" else "Analysis stopped with an error"
    with st.status(label, state=state, expanded=state == "error"):
        completed = steps if state == "complete" else steps[:-1]
        for step in completed:
            st.markdown(f":material/check_circle: {step}")
        if state == "error":
            st.markdown(f":material/error: {steps[-1]}")


def render_evidence(response: AgentResponse) -> None:
    with st.container(border=True):
        st.subheader("Evidence and details", icon=":material/fact_check:")

        if response.proposal:
            with st.expander("Validated SQL", icon=":material/code:"):
                st.code(response.proposal.sql, language="sql")
                st.caption(response.proposal.explanation)

        if response.retrieved_sources:
            with st.expander(
                f"Business sources ({len(response.retrieved_sources)})",
                icon=":material/library_books:",
            ):
                for source in response.retrieved_sources:
                    st.markdown(f"- {source}")

        assumptions = response.proposal.assumptions if response.proposal else []
        limitations = response.summary.limitations if response.summary else []
        if assumptions or limitations:
            with st.expander("Assumptions and limitations", icon=":material/info:"):
                for assumption in assumptions:
                    st.markdown(f"**Assumption:** {assumption}")
                for limitation in limitations:
                    st.markdown(f"**Limitation:** {limitation}")


def render_response_actions(response: AgentResponse, frame: pd.DataFrame | None = None) -> None:
    with st.container(horizontal=True, wrap=True):
        st.button(
            "Retry analysis",
            icon=":material/refresh:",
            on_click=retry_question,
            args=(response.question,),
        )
        st.button(
            "Revise question",
            icon=":material/edit:",
            on_click=revise_question,
            args=(response.question,),
        )
        if frame is not None:
            st.download_button(
                "Download CSV",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="olist-analysis.csv",
                mime="text/csv",
                icon=":material/download:",
            )


def render_analysis(response: AgentResponse) -> None:
    if not response.summary or not response.result or not response.proposal:
        return

    with st.container(border=True):
        st.subheader("Business answer", icon=":material/analytics:")
        st.markdown(response.summary.answer)
        for insight in response.summary.insights:
            st.markdown(f"- {insight}")

    with st.container(horizontal=True):
        st.metric("Returned rows", response.result.row_count, border=True)
        st.metric("Query execution", f"{response.result.execution_ms:.0f} ms", border=True)
        st.metric(
            "Result completeness",
            "Truncated" if response.result.truncated else "Complete",
            border=True,
        )
        st.metric("Sources retrieved", len(response.retrieved_sources), border=True)

    frame = pd.DataFrame(response.result.rows, columns=response.result.columns)
    has_chart = (
        response.proposal.chart_type != "none"
        and response.proposal.chart_x in frame.columns
        and any(column in frame.columns for column in response.proposal.chart_y)
    )
    if has_chart:
        chart_column, table_column = st.columns(2, vertical_alignment="top")
        with chart_column.container(border=True, height="stretch"):
            st.subheader("Visual summary", icon=":material/bar_chart:")
            render_chart(response, frame)
        with table_column.container(border=True, height="stretch"):
            st.subheader("Result data", icon=":material/table_chart:")
            st.dataframe(frame, hide_index=True, key="analysis_result_table")
    else:
        with st.container(border=True):
            st.subheader("Result data", icon=":material/table_chart:")
            st.dataframe(frame, hide_index=True, key="analysis_result_table")

    render_evidence(response)
    render_response_actions(response, frame)


def render_write_preview(response: AgentResponse, model_name: str) -> None:
    preview = response.write_preview
    if not preview:
        return

    with st.container(border=True):
        st.subheader("Review proposed database change", icon=":material/edit_note:")
        st.warning(
            "This operation can modify stored data. Nothing has executed yet.",
            icon=":material/warning:",
        )
        st.markdown(preview.explanation)
        st.metric(
            "Estimated rows examined",
            preview.estimated_rows if preview.estimated_rows is not None else "Unknown",
            border=True,
        )
        st.code(preview.sql, language="sql")
        for warning in preview.warnings:
            st.caption(warning)

        approval_suffix = preview.approval_code.split()[-1]
        reviewed = st.checkbox(
            "I reviewed the exact SQL and predicted impact.",
            key=f"reviewed_{approval_suffix}",
        )
        confirmation = st.text_input(
            f'Type "{preview.approval_code}" to execute:',
            key=f"confirmation_{approval_suffix}",
        )
        with st.container(horizontal=True, wrap=True):
            if st.button(
                "Execute approved edit",
                type="primary",
                icon=":material/check_circle:",
                disabled=not reviewed or confirmation != preview.approval_code,
            ):
                try:
                    result = get_agent(model_name).execute_approved(preview, confirmation)
                    clear_analysis()
                    st.session_state.write_success = (
                        f"{result.operation} committed. Affected rows: {result.affected_rows}."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - present a safe UI error
                    st.error(f"The edit was not committed: {exc}")
            st.button(
                "Reject change",
                icon=":material/cancel:",
                on_click=reject_write,
            )
            st.button(
                "Revise request",
                icon=":material/edit:",
                on_click=revise_question,
                args=(response.question,),
            )

    render_evidence(response)


def render_non_result(response: AgentResponse) -> None:
    if response.kind == "clarification":
        st.info(response.error, icon=":material/help:")
    elif response.kind == "unsupported":
        st.warning(response.error, icon=":material/block:")
    else:
        st.error(response.error or "The request failed.", icon=":material/error:")
    render_evidence(response)
    render_response_actions(response)


st.session_state.setdefault("question_input", "")
st.session_state.setdefault("response", None)
configured_settings = Settings.from_env()
model_options, model_discovery_error = available_chat_models(
    configured_settings.ollama_base_url,
    configured_settings.chat_model,
    configured_settings.embedding_model,
)

with st.sidebar:
    st.header("System status", icon=":material/monitor_heart:")
    selected_model = st.selectbox(
        "Chat model",
        model_options,
        index=model_options.index(configured_settings.chat_model),
        key="selected_chat_model",
        help="Choose an installed Ollama chat model for this browser session.",
        on_change=clear_analysis,
    )
    if model_discovery_error:
        st.warning(
            "The installed model list is unavailable. Using the model configured in `.env`.",
            icon=":material/warning:",
        )
    else:
        st.caption("The `.env` model is the startup default; this choice applies to this session.")
    status = service_status(selected_model)
    if "database_error" in status:
        st.error("Database unavailable", icon=":material/database_off:")
    else:
        st.success(f"Database: {status['database']}", icon=":material/database:")
    if "knowledge_error" in status:
        st.warning("Knowledge index unavailable", icon=":material/warning:")
    else:
        st.markdown(f"**Knowledge index**  \n{status['knowledge_chunks']} chunks")
    if status["write_ready"]:
        st.caption("Write workflow is available and approval-gated.")
    else:
        st.caption("Write workflow is unavailable until separate database identities are set.")
    st.button(
        "Clear analysis",
        icon=":material/ink_eraser:",
        on_click=clear_analysis,
        disabled=st.session_state.response is None,
    )
    st.caption("Queries are validated and limited to 10 seconds before execution.")

st.title("Olist analytics agent", icon=":material/query_stats:")
st.caption(
    "Ask a business question. The agent retrieves context, validates its SQL, "
    "and executes approved database operations."
)

if notice := st.session_state.pop("write_notice", None):
    st.toast(notice, icon=":material/cancel:")
if success := st.session_state.pop("write_success", None):
    st.toast(success, icon=":material/check_circle:")

with st.container(border=True):
    st.markdown("**Try an example**")
    st.pills(
        "Example questions",
        list(EXAMPLE_QUESTIONS),
        key="example_question",
        label_visibility="collapsed",
        on_change=select_example,
    )
    with st.form("question_form", border=False):
        question = st.text_area(
            "Business question",
            key="question_input",
            placeholder="Ask about customers, orders, delivery, reviews, or payments...",
            height=100,
        )
        submitted = st.form_submit_button(
            "Analyze question",
            type="primary",
            icon=":material/play_arrow:",
        )

question_to_run = st.session_state.pop("pending_question", None)
if submitted:
    question_to_run = question

ran_analysis = False
if question_to_run is not None:
    ran_analysis = True
    activity_steps: list[str] = []
    activity = st.status("Starting analysis", expanded=True)

    def report_progress(label: str) -> None:
        if activity_steps:
            activity.markdown(f":material/check_circle: {activity_steps[-1]}")
        activity_steps.append(label)
        activity.update(label=label, state="running", expanded=True)

    response = get_agent(selected_model).ask(question_to_run, progress=report_progress)
    if activity_steps:
        if response.kind == "error":
            activity.markdown(f":material/error: {activity_steps[-1]}")
        else:
            activity.markdown(f":material/check_circle: {activity_steps[-1]}")
    activity_state = "error" if response.kind == "error" else "complete"
    activity.update(
        label=(
            "Analysis stopped with an error"
            if activity_state == "error"
            else "Analysis complete"
        ),
        state=activity_state,
        expanded=activity_state == "error",
    )
    st.session_state.response = response
    st.session_state.activity_steps = activity_steps
    st.session_state.activity_state = activity_state

response = st.session_state.get("response")
if response:
    if not ran_analysis:
        render_activity(
            st.session_state.get("activity_steps", []),
            st.session_state.get("activity_state", "complete"),
        )
    if response.kind == "analysis":
        render_analysis(response)
    elif response.kind == "write_preview":
        render_write_preview(response, selected_model)
    else:
        render_non_result(response)
