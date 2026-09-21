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
    execution_available: bool = True
    warnings: list[str] = Field(default_factory=list)


class WriteResult(BaseModel):
    operation: Literal["INSERT", "UPDATE", "DELETE"]
    affected_rows: int
    execution_ms: float


class AgentResponse(BaseModel):
    kind: Literal[
        "analysis", "write_preview", "memory", "clarification", "unsupported", "error"
    ]
    question: str
    proposal: SqlProposal | None = None
    summary: BusinessSummary | None = None
    result: QueryResult | None = None
    write_preview: WritePreview | None = None
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_memories: list[str] = Field(default_factory=list)
    memory_actions: list[str] = Field(default_factory=list)
    error: str | None = None


class MemoryCandidate(BaseModel):
    action: Literal["store", "forget"] = "store"
    text: str = Field(min_length=1, max_length=500)
    canonical_key: str = Field(min_length=1, max_length=120)
    memory_type: Literal[
        "preference",
        "decision",
        "constraint",
        "project_fact",
        "task_state",
        "episodic",
    ]
    scope: Literal["user", "project", "task", "session"]
    source: Literal["user_explicit", "user_implied", "assistant_inference"]
    explicitness: float = Field(ge=0, le=1)
    future_utility: float = Field(ge=0, le=1)
    durability: float = Field(ge=0, le=1)
    decision_impact: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    sensitivity: float = Field(ge=0, le=1)
    temporary: float = Field(ge=0, le=1)
    evidence: str = Field(default="", max_length=500)
    suggested_ttl_days: int | None = Field(default=None, ge=1, le=90)


class MemoryExtraction(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=8)
