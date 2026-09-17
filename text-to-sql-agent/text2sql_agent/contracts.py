from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IntentDecision(BaseModel):
    action: Literal["read", "write", "unsupported"]
    reason: str
    clarification_needed: bool = False
    clarification_question: str | None = None


class SqlProposal(BaseModel):
    intent: str
    sql: str = Field(description="Exactly one MySQL statement.")
    explanation: str
    assumptions: list[str] = Field(default_factory=list)
    chart_type: Literal["none", "bar", "line", "area", "scatter"] = "none"
    chart_x: str | None = None
    chart_y: list[str] = Field(default_factory=list)


class BusinessSummary(BaseModel):
    answer: str
    insights: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False
    execution_ms: float


class WritePreview(BaseModel):
    sql: str
    normalized_sql: str
    operation: Literal["INSERT", "UPDATE", "DELETE"]
    explanation: str
    estimated_rows: int | None = None
    approval_code: str
    warnings: list[str] = Field(default_factory=list)


class WriteResult(BaseModel):
    operation: Literal["INSERT", "UPDATE", "DELETE"]
    affected_rows: int
    execution_ms: float


class AgentResponse(BaseModel):
    kind: Literal["analysis", "write_preview", "clarification", "unsupported", "error"]
    question: str
    proposal: SqlProposal | None = None
    summary: BusinessSummary | None = None
    result: QueryResult | None = None
    write_preview: WritePreview | None = None
    retrieved_sources: list[str] = Field(default_factory=list)
    error: str | None = None
