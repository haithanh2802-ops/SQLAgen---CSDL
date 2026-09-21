from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from text2sql_agent.config import Settings
from text2sql_agent.contracts import AgentResponse
from text2sql_agent.ollama_models import discover_chat_models
from text2sql_agent.ollama_runtime import model_runtime
from text2sql_agent.service import AnalyticsAgent
from text2sql_agent.ui_charts import ChartSpec, infer_chart_spec
from text2sql_agent.ui_questions import EXAMPLE_QUESTIONS
from text2sql_agent.ui_state import clear_chat_session

AGENT_CACHE_VERSION = "sql-policy-v5"

st.set_page_config(
    page_title="Olist Insights",
    page_icon=":material/analytics:",
    layout="wide",
)


@st.cache_resource(ttl="5m", max_entries=4)
def get_agent(model_name: str, implementation_version: str) -> AnalyticsAgent:
    if implementation_version != AGENT_CACHE_VERSION:
        raise RuntimeError("The cached agent implementation is out of date.")
    configured_settings = Settings.from_env()
    settings = replace(configured_settings, chat_model=model_name)
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
    agent = get_agent(model_name, AGENT_CACHE_VERSION)
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


def queue_question(question: str) -> None:
    st.session_state.pending_question = question


def select_example() -> None:
    selected = st.session_state.get("example_question")
    if selected:
        queue_question(EXAMPLE_QUESTIONS[selected])


def find_turn(turn_id: str) -> dict[str, Any] | None:
    for turn in st.session_state.get("conversation", []):
        if turn["id"] == turn_id:
            return turn
    return None


def reject_write(turn_id: str) -> None:
    turn = find_turn(turn_id)
    if turn is not None:
        turn["write_result"] = "The proposed change was rejected. Nothing was written."


def end_chat_session() -> None:
    clear_chat_session(st.session_state)
    st.session_state.session_notice = "Session ended and chat history deleted."


@st.dialog("End this session?", icon=":material/delete_sweep:")
def confirm_end_session() -> None:
    st.write(
        "This permanently removes every question and answer from this browser session. "
        "Database changes that were already approved are not reversed."
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Keep session", key="keep_chat_session"):
            st.rerun()
        st.button(
            "End and delete",
            type="primary",
            icon=":material/delete_forever:",
            key="confirm_end_chat_session",
            on_click=end_chat_session,
        )


def _chart_label(column: str) -> str:
    return column.replace("_", " ").strip().title()


def _chart_title(spec: ChartSpec) -> str:
    metrics = ", ".join(_chart_label(column) for column in spec.y)
    return f"{metrics} by {_chart_label(spec.x)}"


def _style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(
            gridColor="#E5E7EB",
            gridOpacity=0.55,
            labelFontSize=12,
            titleFontSize=13,
            titlePadding=12,
        )
        .configure_legend(
            labelFontSize=12,
            orient="top",
            title=None,
        )
        .configure_view(strokeOpacity=0)
    )


def render_chart(spec: ChartSpec, frame: pd.DataFrame) -> None:
    columns = [spec.x, *spec.y]
    chart_frame = frame.loc[:, columns].copy()
    for column in spec.y:
        chart_frame[column] = pd.to_numeric(chart_frame[column], errors="coerce")

    st.markdown(f"**{_chart_title(spec)}**")
    st.caption("Hover over the chart to see exact values.")

    if spec.kind == "scatter":
        chart_frame[spec.x] = pd.to_numeric(chart_frame[spec.x], errors="coerce")
        chart_frame = chart_frame.dropna(subset=[spec.x, *spec.y])
        y_column = spec.y[0]
        chart = (
            alt.Chart(chart_frame)
            .mark_circle(size=90, opacity=0.78, color="#2563EB")
            .encode(
                x=alt.X(
                    f"{spec.x}:Q",
                    title=_chart_label(spec.x),
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    f"{y_column}:Q",
                    title=_chart_label(y_column),
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip(f"{spec.x}:Q", title=_chart_label(spec.x), format=",.3~g"),
                    alt.Tooltip(
                        f"{y_column}:Q", title=_chart_label(y_column), format=",.3~g"
                    ),
                ],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(_style_chart(chart), width="stretch")
        return

    temporal_x = False
    if spec.kind in {"line", "area"}:
        x_name = spec.x.casefold()
        if any(token in x_name for token in ("date", "month", "year", "week", "day")):
            parsed_x = pd.to_datetime(chart_frame[spec.x], errors="coerce")
            if parsed_x.notna().mean() >= 0.8:
                chart_frame[spec.x] = parsed_x
                temporal_x = True

    tidy = chart_frame.melt(
        id_vars=[spec.x],
        value_vars=list(spec.y),
        var_name="Metric",
        value_name="Value",
    ).dropna(subset=[spec.x, "Value"])
    tidy["Metric"] = tidy["Metric"].map(_chart_label)
    x_type = "T" if temporal_x else "N"
    x_tooltip = (
        alt.Tooltip(f"{spec.x}:T", title=_chart_label(spec.x), format="%b %d, %Y")
        if temporal_x
        else alt.Tooltip(f"{spec.x}:N", title=_chart_label(spec.x))
    )
    tooltip = [
        x_tooltip,
        alt.Tooltip("Metric:N", title="Metric"),
        alt.Tooltip("Value:Q", title="Value", format=",.3~g"),
    ]
    color = alt.Color(
        "Metric:N",
        scale=alt.Scale(scheme="tableau10"),
        legend=None if len(spec.y) == 1 else alt.Legend(),
    )

    if spec.kind == "bar":
        metric = _chart_label(spec.y[0]) if len(spec.y) == 1 else None
        if metric is not None:
            if tidy[spec.x].nunique() > 30:
                st.caption(
                    "Showing the 30 highest values for readability; "
                    "the Data tab contains the complete result."
                )
            tidy = tidy.sort_values("Value", ascending=False).head(30)
        long_categories = tidy[spec.x].astype(str).str.len().max() > 14
        horizontal = long_categories or tidy[spec.x].nunique() > 8
        if horizontal:
            chart = (
                alt.Chart(tidy)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Value:Q", title=metric or "Value"),
                    y=alt.Y(
                        f"{spec.x}:N",
                        title=_chart_label(spec.x),
                        sort=alt.EncodingSortField(field="Value", order="descending"),
                    ),
                    color=color,
                    tooltip=tooltip,
                )
                .properties(height=max(300, min(650, tidy[spec.x].nunique() * 30)))
            )
        else:
            bar_encoding: dict[str, Any] = {
                "x": alt.X(
                    f"{spec.x}:N",
                    title=_chart_label(spec.x),
                    sort=alt.EncodingSortField(field="Value", order="descending"),
                    axis=alt.Axis(labelAngle=-25),
                ),
                "y": alt.Y("Value:Q", title=metric or "Value"),
                "color": color,
                "tooltip": tooltip,
            }
            if len(spec.y) > 1:
                bar_encoding["xOffset"] = alt.XOffset("Metric:N")
            chart = (
                alt.Chart(tidy)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(**bar_encoding)
                .properties(height=380)
            )
    else:
        x_encoding = alt.X(
            f"{spec.x}:{x_type}",
            title=_chart_label(spec.x),
            sort=None,
            axis=alt.Axis(labelAngle=0 if temporal_x else -25),
        )
        encoding = {
            "x": x_encoding,
            "y": alt.Y("Value:Q", title="Value" if len(spec.y) > 1 else _chart_label(spec.y[0])),
            "color": color,
            "tooltip": tooltip,
        }
        if spec.kind == "area":
            chart = (
                alt.Chart(tidy)
                .mark_area(opacity=0.65, line=True)
                .encode(**encoding)
                .properties(height=380)
            )
        else:
            chart = (
                alt.Chart(tidy)
                .mark_line(point=alt.OverlayMarkDef(size=70))
                .encode(**encoding)
                .properties(height=380)
            )

    st.altair_chart(_style_chart(chart), width="stretch")


def render_activity(steps: list[str], state: str) -> None:
    if not steps:
        return
    label = "Analysis details" if state == "complete" else "Analysis stopped with an error"
    with st.expander(
        label,
        icon=":material/check_circle:" if state == "complete" else ":material/error:",
    ):
        completed = steps if state == "complete" else steps[:-1]
        for step in completed:
            st.markdown(f":material/check_circle: {step}")
        if state == "error":
            st.markdown(f":material/error: {steps[-1]}")


def render_evidence(response: AgentResponse) -> None:
    if response.proposal:
        st.markdown("**Validated SQL**")
        st.code(response.proposal.sql, language="sql")
        st.caption(response.proposal.explanation)

    if response.retrieved_sources:
        st.markdown("**Business sources**")
        for source in response.retrieved_sources:
            st.markdown(f"- {source}")

    assumptions = response.proposal.assumptions if response.proposal else []
    limitations = response.summary.limitations if response.summary else []
    if assumptions or limitations:
        st.markdown("**Assumptions and limitations**")
        for assumption in assumptions:
            st.markdown(f"- Assumption: {assumption}")
        for limitation in limitations:
            st.markdown(f"- Limitation: {limitation}")


def render_response_actions(
    response: AgentResponse,
    turn_id: str,
    frame: pd.DataFrame | None = None,
) -> None:
    with st.container(horizontal=True, wrap=True):
        st.button(
            "Retry",
            icon=":material/refresh:",
            on_click=queue_question,
            args=(response.question,),
            key=f"retry_{turn_id}",
        )
        if frame is not None:
            st.download_button(
                "Download CSV",
                frame.to_csv(index=False).encode("utf-8"),
                file_name="olist-analysis.csv",
                mime="text/csv",
                icon=":material/download:",
                key=f"download_{turn_id}",
            )


def render_analysis(response: AgentResponse, turn_id: str) -> None:
    if not response.summary or not response.result or not response.proposal:
        return

    st.markdown("### Business answer")
    st.markdown(response.summary.answer)
    if response.summary.insights:
        st.markdown("**Key insights**")
        for insight in response.summary.insights:
            st.markdown(f"- {insight}")

    metric_columns = st.columns(3, vertical_alignment="center")
    metric_columns[0].metric("Returned rows", response.result.row_count, border=True)
    metric_columns[1].metric(
        "Query execution", f"{response.result.execution_ms:.0f} ms", border=True
    )
    metric_columns[2].metric(
        "Sources used", len(response.retrieved_sources), border=True
    )
    if response.result.truncated:
        st.warning("The displayed result reached the configured row limit.")
    else:
        st.caption(":green-badge[Complete result]")

    frame = pd.DataFrame(response.result.rows, columns=response.result.columns)
    chart_spec = infer_chart_spec(
        frame,
        requested_kind=response.proposal.chart_type,
        requested_x=response.proposal.chart_x,
        requested_y=response.proposal.chart_y,
    )

    if chart_spec:
        chart_tab, data_tab, evidence_tab = st.tabs(
            [
                ":material/bar_chart: Visualization",
                ":material/table_chart: Data",
                ":material/fact_check: Evidence",
            ]
        )
        with chart_tab:
            render_chart(chart_spec, frame)
        with data_tab:
            st.dataframe(frame, hide_index=True, key=f"analysis_table_{turn_id}")
        with evidence_tab:
            render_evidence(response)
    else:
        data_tab, evidence_tab = st.tabs(
            [":material/table_chart: Data", ":material/fact_check: Evidence"]
        )
        with data_tab:
            st.dataframe(frame, hide_index=True, key=f"analysis_table_{turn_id}")
        with evidence_tab:
            render_evidence(response)

    render_response_actions(response, turn_id, frame)


def render_write_preview(
    response: AgentResponse,
    model_name: str,
    turn_id: str,
    write_result: str | None,
) -> None:
    preview = response.write_preview
    if not preview:
        return

    st.markdown("### Review proposed database change")
    if write_result:
        st.info(write_result, icon=":material/info:")
        render_evidence(response)
        return

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
        key=f"reviewed_{turn_id}_{approval_suffix}",
    )
    confirmation = st.text_input(
        f'Type "{preview.approval_code}" to execute:',
        key=f"confirmation_{turn_id}_{approval_suffix}",
    )
    with st.container(horizontal=True, wrap=True):
        if st.button(
            "Execute approved edit",
            type="primary",
            icon=":material/check_circle:",
            disabled=(
                not preview.execution_available
                or not reviewed
                or confirmation != preview.approval_code
            ),
            key=f"execute_{turn_id}",
        ):
            try:
                result = get_agent(model_name, AGENT_CACHE_VERSION).execute_approved(
                    preview, confirmation
                )
                turn = find_turn(turn_id)
                if turn is not None:
                    turn["write_result"] = (
                        f"{result.operation} committed. Affected rows: {result.affected_rows}."
                    )
                st.toast("Database change committed.", icon=":material/check_circle:")
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - present a safe UI error
                st.error(f"The edit was not committed: {exc}")
        st.button(
            "Reject change",
            icon=":material/cancel:",
            on_click=reject_write,
            args=(turn_id,),
            key=f"reject_{turn_id}",
        )

    with st.expander("Methodology and evidence", icon=":material/fact_check:"):
        render_evidence(response)


def render_non_result(response: AgentResponse, turn_id: str) -> None:
    if response.kind == "clarification":
        st.info(response.error, icon=":material/help:")
    elif response.kind == "unsupported":
        st.warning(response.error, icon=":material/block:")
    else:
        st.error(response.error or "The request failed.", icon=":material/error:")
    with st.expander("Methodology and evidence", icon=":material/fact_check:"):
        render_evidence(response)
    render_response_actions(response, turn_id)


def render_turn(turn: dict[str, Any]) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant", avatar=":material/analytics:"):
        response = turn.get("response")
        if response is None:
            st.warning(
                "This analysis was stopped before it finished. "
                "Use Retry when you want to run it again."
            )
            st.button(
                "Retry",
                icon=":material/refresh:",
                on_click=queue_question,
                args=(turn["question"],),
                key=f"retry_stopped_{turn['id']}",
            )
            return
        render_activity(turn.get("steps", []), turn.get("state", "complete"))
        if response.kind == "analysis":
            render_analysis(response, turn["id"])
        elif response.kind == "write_preview":
            render_write_preview(
                response,
                turn["model"],
                turn["id"],
                turn.get("write_result"),
            )
        else:
            render_non_result(response, turn["id"])
        st.caption(f"Analyzed with {turn['model']}")


st.session_state.setdefault("conversation", [])
st.session_state.setdefault("pending_question", None)

# Preserve a result from the pre-chat UI across the first hot reload.
legacy_response = st.session_state.pop("response", None)
if legacy_response is not None and not st.session_state.conversation:
    st.session_state.conversation.append(
        {
            "id": uuid.uuid4().hex,
            "question": legacy_response.question,
            "response": legacy_response,
            "steps": st.session_state.pop("activity_steps", []),
            "state": st.session_state.pop("activity_state", "complete"),
            "model": st.session_state.get("selected_chat_model", "Unknown model"),
        }
    )

configured_settings = Settings.from_env()
model_options, model_discovery_error = available_chat_models(
    configured_settings.ollama_base_url,
    configured_settings.chat_model,
    configured_settings.embedding_model,
)

with st.sidebar:
    st.header("Olist Insights", icon=":material/analytics:")
    st.caption("Local business intelligence for the Olist marketplace")

    with st.expander("Settings", icon=":material/tune:", expanded=True):
        selected_model = st.selectbox(
            "Analysis model",
            model_options,
            index=model_options.index(configured_settings.chat_model),
            key="selected_chat_model",
            help="This choice applies to new questions in the current browser session.",
        )
        runtime = model_runtime(selected_model, configured_settings.ollama_num_gpu)
        if runtime.profile == "GPU-first with CPU fallback":
            st.caption(":green-badge[GPU + CPU fallback] Small model profile.")
        elif runtime.profile == "CPU-focused":
            st.caption(":orange-badge[CPU mode] Large model profile.")
        else:
            st.caption(f":blue-badge[{runtime.profile}]")
        if model_discovery_error:
            st.warning("Could not refresh the installed model list.")

    status = service_status(selected_model)
    service_errors = "database_error" in status or "knowledge_error" in status
    if service_errors:
        st.badge("Attention required", icon=":material/warning:", color="orange")
    else:
        st.badge("Systems operational", icon=":material/check_circle:", color="green")

    with st.expander("Diagnostics", icon=":material/monitor_heart:"):
        if "database_error" in status:
            st.error("Database unavailable")
        else:
            st.caption(f"Database: {status['database']}")
        if "knowledge_error" in status:
            st.warning("Knowledge index unavailable")
        else:
            st.caption(f"Knowledge index: {status['knowledge_chunks']:,} chunks")
        write_label = "Ready" if status["write_ready"] else "Read only"
        st.caption(f"Protected write workflow: {write_label}")
        st.caption("SQL execution limit: 10 seconds")

    turn_count = len(st.session_state.conversation)
    st.caption(f"Current session · {turn_count} {'turn' if turn_count == 1 else 'turns'}")
    if st.button(
        "End session",
        icon=":material/delete_sweep:",
        disabled=not st.session_state.conversation,
        key="end_chat_session",
    ):
        confirm_end_session()
    st.caption("Chat history is temporary and is deleted when this session ends.")

st.title("Olist Insights", icon=":material/query_stats:")
st.caption(
    "Ask questions about customers, revenue, delivery, reviews, and payments. "
    "Every database query is validated before execution."
)

if notice := st.session_state.pop("session_notice", None):
    st.toast(notice, icon=":material/delete_sweep:")

question_to_run = st.session_state.pop("pending_question", None)

for saved_turn in st.session_state.conversation:
    render_turn(saved_turn)

if not st.session_state.conversation and question_to_run is None:
    with st.container(border=True):
        st.subheader("Start with a business question", icon=":material/lightbulb:")
        st.caption("Choose an example or write your own question below.")
        st.pills(
            "Example questions",
            list(EXAMPLE_QUESTIONS),
            key="example_question",
            label_visibility="collapsed",
            on_change=select_example,
        )

submitted_question = st.chat_input(
    "Ask Olist Insights…",
    key="chat_question",
    submit_mode="stop",
)
if submitted_question:
    question_to_run = submitted_question

if question_to_run:
    turn = {
        "id": uuid.uuid4().hex,
        "question": question_to_run.strip(),
        "response": None,
        "steps": [],
        "state": "running",
        "model": selected_model,
    }
    st.session_state.conversation.append(turn)

    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant", avatar=":material/analytics:"):
        activity_steps: list[str] = []
        activity = st.status(":shimmer[Analyzing your data]", type="compact", expanded=True)

        def report_progress(label: str) -> None:
            if activity_steps:
                activity.markdown(f":material/check_circle: {activity_steps[-1]}")
            activity_steps.append(label)
            activity.update(label=f":shimmer[{label}]", state="running", expanded=True)

        response = get_agent(selected_model, AGENT_CACHE_VERSION).ask(
            question_to_run, progress=report_progress
        )
        if activity_steps:
            final_icon = (
                ":material/error:" if response.kind == "error" else ":material/check_circle:"
            )
            activity.markdown(f"{final_icon} {activity_steps[-1]}")
        activity_state = "error" if response.kind == "error" else "complete"
        activity.update(
            label="Analysis failed" if activity_state == "error" else "Analysis complete",
            state=activity_state,
            expanded=activity_state == "error",
        )

        turn["response"] = response
        turn["steps"] = activity_steps
        turn["state"] = activity_state

        if response.kind == "analysis":
            render_analysis(response, turn["id"])
        elif response.kind == "write_preview":
            render_write_preview(response, selected_model, turn["id"], None)
        else:
            render_non_result(response, turn["id"])
        st.caption(f"Analyzed with {selected_model}")
