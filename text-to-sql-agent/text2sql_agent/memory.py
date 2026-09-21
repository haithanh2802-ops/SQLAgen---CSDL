from __future__ import annotations

import atexit
import math
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models

from text2sql_agent.audit import AuditLogger
from text2sql_agent.config import Settings
from text2sql_agent.contracts import MemoryCandidate, MemoryExtraction

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:password|passcode|api[_ -]?key|access[_ -]?token|secret)\b\s*"
        r"(?:is|=|:)\s*\S+"
    ),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)
_REMEMBER_PATTERN = re.compile(r"(?i)\b(?:remember|save this|keep this in mind)\b")
_CLEAR_ALL_PATTERN = re.compile(
    r"(?i)\b(?:forget|delete|clear)\s+(?:all|everything|every memory|all memories)\b"
)


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    score: float
    payload: dict[str, Any]
    vector: list[float] | None = None

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))


@dataclass(frozen=True)
class MemoryContext:
    text: str
    records: tuple[MemoryRecord, ...] = ()

    @property
    def memories(self) -> list[str]:
        return [record.text for record in self.records if record.text]


class QdrantMemoryRepository:
    """Small, synchronized repository around an embedded Qdrant collection."""

    def __init__(
        self,
        path: Path | str,
        collection_name: str,
        *,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._lock = threading.RLock()
        if client is not None:
            self.client = client
        elif str(path) == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            resolved = Path(path)
            resolved.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(
                path=str(resolved),
                force_disable_check_same_thread=True,
            )
        atexit.register(self.close)

    def close(self) -> None:
        with self._lock:
            try:
                self.client.close()
            except (ImportError, TypeError):
                return

    def exists(self) -> bool:
        with self._lock:
            return self.client.collection_exists(self.collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection_name):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def query(
        self,
        vector: list[float],
        *,
        user_id: str,
        limit: int,
    ) -> list[MemoryRecord]:
        if not self.exists():
            return []
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                ),
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                ),
            ]
        )
        with self._lock:
            points = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=True,
                limit=limit,
            ).points
        return [
            MemoryRecord(
                memory_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
                vector=list(point.vector) if isinstance(point.vector, list) else None,
            )
            for point in points
        ]

    def find_active_by_key(
        self,
        *,
        user_id: str,
        project_id: str | None,
        canonical_key: str,
    ) -> MemoryRecord | None:
        if not self.exists():
            return None
        conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(
                key="canonical_key", match=models.MatchValue(value=canonical_key)
            ),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
        ]
        if project_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="project_id", match=models.MatchValue(value=project_id)
                )
            )
        query_filter = models.Filter(must=conditions)
        with self._lock:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1,
                with_payload=True,
                with_vectors=True,
            )
        if not points:
            return None
        point = points[0]
        return MemoryRecord(
            memory_id=str(point.id),
            score=1.0,
            payload=dict(point.payload or {}),
            vector=list(point.vector) if isinstance(point.vector, list) else None,
        )

    def upsert(self, memory_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.ensure_collection(len(vector))
        with self._lock:
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(id=memory_id, vector=vector, payload=payload)
                ],
            )

    def delete_by_key(
        self, *, user_id: str, project_id: str | None, canonical_key: str
    ) -> int:
        conditions = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(
                key="canonical_key", match=models.MatchValue(value=canonical_key)
            ),
        ]
        if project_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="project_id", match=models.MatchValue(value=project_id)
                )
            )
        return self._delete_filter(models.Filter(must=conditions))

    def delete_user(self, user_id: str) -> int:
        return self._delete_filter(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        )

    def _delete_filter(self, query_filter: models.Filter) -> int:
        if not self.exists():
            return 0
        with self._lock:
            count = int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=query_filter,
                    exact=True,
                ).count
            )
            if count:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(filter=query_filter),
                    wait=True,
                )
        return count

    def count_user(self, user_id: str) -> int:
        if not self.exists():
            return 0
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                ),
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                ),
            ]
        )
        with self._lock:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=query_filter,
                    exact=True,
                ).count
            )


@lru_cache(maxsize=8)
def _shared_repository(path: str, collection_name: str) -> QdrantMemoryRepository:
    return QdrantMemoryRepository(Path(path), collection_name)


class MemoryManager:
    """Application policy for extracting, scoring, resolving, and retrieving memory."""

    def __init__(
        self,
        settings: Settings,
        model: Any,
        audit: AuditLogger,
        *,
        repository: QdrantMemoryRepository | None = None,
        embeddings: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.audit = audit
        self.repository = repository or _shared_repository(
            str(settings.memory_directory), settings.memory_collection
        )
        self.embeddings = embeddings or OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.ollama_timeout_seconds},
        )

    def retrieve(
        self,
        query: str,
        *,
        user_id: str,
        project_id: str,
        session_id: str | None,
    ) -> MemoryContext:
        if not self.settings.memory_enabled or not self.repository.exists():
            return MemoryContext(text="")
        vector = list(self.embeddings.embed_query(query))
        candidates = self.repository.query(
            vector,
            user_id=user_id,
            limit=max(self.settings.memory_retrieval_limit * 4, 12),
        )
        now = datetime.now(UTC)
        ranked: list[tuple[float, MemoryRecord]] = []
        for record in candidates:
            payload = record.payload
            if not _is_in_scope(payload, project_id=project_id, session_id=session_id):
                continue
            if _is_expired(payload, now):
                continue
            similarity = max(0.0, min(1.0, record.score))
            if similarity < self.settings.memory_min_similarity:
                continue
            importance = _unit_float(payload.get("importance"))
            confidence = _unit_float(payload.get("confidence"))
            recency = _recency_score(payload.get("updated_at"), now)
            retrieval_score = (
                0.60 * similarity
                + 0.20 * importance
                + 0.10 * confidence
                + 0.10 * recency
            )
            ranked.append((retrieval_score, record))
        selected = tuple(
            record
            for _, record in sorted(ranked, key=lambda item: item[0], reverse=True)[
                : self.settings.memory_retrieval_limit
            ]
        )
        blocks = [
            (
                f'<memory id="{record.memory_id}" '
                f'type="{record.payload.get("memory_type", "unknown")}">\n'
                f"{record.text}\n</memory>"
            )
            for record in selected
        ]
        return MemoryContext(text="\n\n".join(blocks), records=selected)

    def consider_turn(
        self,
        *,
        user_message: str,
        assistant_response: str,
        response_kind: str,
        user_id: str,
        project_id: str,
        session_id: str | None,
    ) -> list[str]:
        if not self.settings.memory_enabled or response_kind not in {"analysis", "memory"}:
            return []
        if _CLEAR_ALL_PATTERN.search(user_message):
            removed = self.repository.delete_user(user_id)
            self.audit.try_append("memory_cleared", user_id=user_id, removed=removed)
            return [f"Forgot {removed} saved memories."]

        structured = self.model.with_structured_output(MemoryExtraction, method="json_schema")
        extraction = structured.invoke(
            [
                (
                    "system",
                    (
                        "Extract at most eight atomic memory candidates from an Olist analytics "
                        "conversation. Store only stable user preferences, confirmed project "
                        "decisions, durable constraints, reusable metric definitions, or explicit "
                        "remember/forget requests. Do not store ordinary analytical questions, "
                        "query results, SQL text, customer/order identifiers, credentials, personal "
                        "data, assistant conclusions, or unconfirmed inferences. The assistant text "
                        "is context only and is never evidence that a fact is true. Use a stable "
                        "dotted canonical_key such as user.analysis_style or "
                        "project.default_order_scope. Set source to assistant_inference whenever "
                        "the user did not supply the fact. Return no candidates when nothing is "
                        "worth preserving. Scores must be between zero and one."
                    ),
                ),
                (
                    "human",
                    (
                        f"USER MESSAGE:\n{user_message}\n\n"
                        f"ASSISTANT RESPONSE (context only):\n{assistant_response[:3000]}"
                    ),
                ),
            ]
        )

        actions: list[str] = []
        explicit_remember = bool(_REMEMBER_PATTERN.search(user_message))
        for candidate in extraction.candidates:
            action = self._apply_candidate(
                candidate,
                explicit_remember=explicit_remember,
                user_id=user_id,
                project_id=project_id,
                session_id=session_id,
            )
            if action:
                actions.append(action)
        return actions

    def count(self, *, user_id: str) -> int:
        if not self.settings.memory_enabled:
            return 0
        return self.repository.count_user(user_id)

    def clear_user(self, *, user_id: str) -> int:
        removed = self.repository.delete_user(user_id)
        self.audit.try_append("memory_cleared", user_id=user_id, removed=removed)
        return removed

    def _apply_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        explicit_remember: bool,
        user_id: str,
        project_id: str,
        session_id: str | None,
    ) -> str | None:
        canonical_key = _canonical_key(candidate.canonical_key)
        if candidate.action == "forget":
            removed = self.repository.delete_by_key(
                user_id=user_id,
                project_id=None if candidate.scope == "user" else project_id,
                canonical_key=canonical_key,
            )
            if removed:
                self.audit.try_append(
                    "memory_forgotten", canonical_key=canonical_key, removed=removed
                )
                return f"Forgot {canonical_key}."
            return None

        if candidate.source == "assistant_inference":
            return None
        if candidate.sensitivity >= 0.5 or _contains_secret(candidate.text):
            self.audit.try_append(
                "memory_rejected", reason="sensitive", canonical_key=canonical_key
            )
            return None

        optimistic_score = memory_score(candidate, novelty=1.0)
        if explicit_remember and candidate.source == "user_explicit":
            optimistic_score = max(optimistic_score, 0.85)
        if optimistic_score < self.settings.memory_persistent_threshold:
            return None

        vector = list(self.embeddings.embed_query(candidate.text))
        existing = self.repository.find_active_by_key(
            user_id=user_id,
            project_id=None if candidate.scope == "user" else project_id,
            canonical_key=canonical_key,
        )
        related = self.repository.query(vector, user_id=user_id, limit=1)
        novelty = (
            1.0
            if existing is not None
            else 1.0 - max(0.0, min(1.0, related[0].score))
            if related
            else 1.0
        )
        score = memory_score(candidate, novelty=novelty)
        if explicit_remember and candidate.source == "user_explicit":
            score = max(score, 0.85)
        if score < self.settings.memory_persistent_threshold:
            return None
        if existing is not None:
            if _normalized_text(candidate.text) == _normalized_text(existing.text):
                payload = {
                    **existing.payload,
                    "text": candidate.text.strip(),
                    "importance": max(score, _unit_float(existing.payload.get("importance"))),
                    "confidence": max(
                        candidate.confidence,
                        _unit_float(existing.payload.get("confidence")),
                    ),
                    "last_seen_at": _timestamp(),
                    "updated_at": _timestamp(),
                }
                self.repository.upsert(existing.memory_id, vector, payload)
                self.audit.try_append(
                    "memory_refreshed", memory_id=existing.memory_id, canonical_key=canonical_key
                )
                return f"Refreshed {canonical_key}."
            if existing.vector is not None:
                superseded = {
                    **existing.payload,
                    "status": "superseded",
                    "updated_at": _timestamp(),
                }
                self.repository.upsert(existing.memory_id, existing.vector, superseded)

        now = datetime.now(UTC)
        expires_at: str | None = None
        if candidate.scope in {"task", "session"}:
            ttl_days = candidate.suggested_ttl_days or 7
            expires_at = (now + timedelta(days=min(ttl_days, 30))).isoformat()
        memory_id = str(uuid.uuid4())
        payload = {
            "text": candidate.text.strip(),
            "canonical_key": canonical_key,
            "memory_type": candidate.memory_type,
            "scope": candidate.scope,
            "source": candidate.source,
            "evidence": candidate.evidence.strip(),
            "user_id": user_id,
            "project_id": project_id,
            "session_id": session_id,
            "importance": round(score, 4),
            "confidence": candidate.confidence,
            "status": "active",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
            "expires_at": expires_at,
            "embedding_model": self.settings.embedding_model,
            "schema_version": 1,
        }
        self.repository.upsert(memory_id, vector, payload)
        self.audit.try_append(
            "memory_stored",
            memory_id=memory_id,
            canonical_key=canonical_key,
            memory_type=candidate.memory_type,
            scope=candidate.scope,
            importance=round(score, 4),
        )
        return f"Remembered {canonical_key}."


def memory_score(candidate: MemoryCandidate, *, novelty: float) -> float:
    score = (
        0.25 * candidate.explicitness
        + 0.20 * candidate.future_utility
        + 0.15 * candidate.durability
        + 0.15 * candidate.decision_impact
        + 0.15 * candidate.confidence
        + 0.10 * max(0.0, min(1.0, novelty))
        - 0.35 * candidate.sensitivity
        - 0.25 * candidate.temporary
    )
    return max(0.0, min(1.0, score))


def _is_in_scope(
    payload: dict[str, Any], *, project_id: str, session_id: str | None
) -> bool:
    scope = payload.get("scope")
    if scope == "user":
        return True
    if payload.get("project_id") != project_id:
        return False
    if scope == "session":
        return bool(session_id and payload.get("session_id") == session_id)
    return True


def _is_expired(payload: dict[str, Any], now: datetime) -> bool:
    value = payload.get("expires_at")
    if not value:
        return False
    try:
        return datetime.fromisoformat(str(value)) <= now
    except ValueError:
        return True


def _recency_score(value: Any, now: datetime) -> float:
    try:
        updated = datetime.fromisoformat(str(value))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return 0.0
    age_days = max(0.0, (now - updated).total_seconds() / 86400)
    return math.exp(-age_days / 90)


def _unit_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _canonical_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.]+", "_", value.strip().lower())
    normalized = re.sub(r"[._]{2,}", ".", normalized).strip("._")
    return normalized[:120] or "memory.uncategorized"


def _normalized_text(value: str) -> str:
    return " ".join(value.lower().split())


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
