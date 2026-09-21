from pathlib import Path

from qdrant_client import QdrantClient

from text2sql_agent.audit import AuditLogger
from text2sql_agent.config import Settings
from text2sql_agent.contracts import MemoryCandidate, MemoryExtraction
from text2sql_agent.memory import MemoryManager, QdrantMemoryRepository, memory_score


class StructuredRunner:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)

    def with_structured_output(self, _schema, **_kwargs):
        return StructuredRunner(self.responses.pop(0))


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        if "qdrant" in lowered or "memory database" in lowered:
            return [1.0, 0.0, 0.0]
        if "postgres" in lowered:
            return [0.9, 0.1, 0.0]
        return [0.0, 1.0, 0.0]


def configured_settings(monkeypatch, tmp_path: Path) -> Settings:
    monkeypatch.setenv("MEMORY_DIRECTORY", str(tmp_path / "memory"))
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    return Settings.from_env(tmp_path)


def candidate(**overrides) -> MemoryCandidate:
    values = {
        "text": "The project uses Qdrant for conversational memory.",
        "canonical_key": "project.memory_database",
        "memory_type": "decision",
        "scope": "project",
        "source": "user_explicit",
        "explicitness": 1.0,
        "future_utility": 1.0,
        "durability": 0.9,
        "decision_impact": 0.9,
        "confidence": 1.0,
        "sensitivity": 0.0,
        "temporary": 0.0,
        "evidence": "Use Qdrant for memory.",
    }
    values.update(overrides)
    return MemoryCandidate(**values)


def manager(monkeypatch, tmp_path: Path, responses) -> MemoryManager:
    settings = configured_settings(monkeypatch, tmp_path)
    repository = QdrantMemoryRepository(
        ":memory:",
        settings.memory_collection,
        client=QdrantClient(":memory:"),
    )
    return MemoryManager(
        settings,
        FakeModel(responses),
        AuditLogger(settings.audit_log),
        repository=repository,
        embeddings=FakeEmbeddings(),
    )


def test_high_value_explicit_memory_is_stored_and_retrieved(monkeypatch, tmp_path):
    memory = manager(
        monkeypatch,
        tmp_path,
        [MemoryExtraction(candidates=[candidate()])],
    )

    actions = memory.consider_turn(
        user_message="Remember that we use Qdrant for memory.",
        assistant_response="Understood.",
        response_kind="analysis",
        user_id="user-1",
        project_id="project-1",
        session_id="session-1",
    )
    context = memory.retrieve(
        "Which memory database do we use?",
        user_id="user-1",
        project_id="project-1",
        session_id="session-2",
    )

    assert actions == ["Remembered project.memory_database."]
    assert context.memories == ["The project uses Qdrant for conversational memory."]


def test_same_canonical_key_supersedes_conflicting_value(monkeypatch, tmp_path):
    memory = manager(
        monkeypatch,
        tmp_path,
        [
            MemoryExtraction(
                candidates=[
                    candidate(
                        text="The project uses PostgreSQL for conversational memory."
                    )
                ]
            ),
            MemoryExtraction(candidates=[candidate()]),
        ],
    )
    common = {
        "assistant_response": "Understood.",
        "response_kind": "analysis",
        "user_id": "user-1",
        "project_id": "project-1",
        "session_id": "session-1",
    }

    memory.consider_turn(user_message="Use PostgreSQL for memory.", **common)
    actions = memory.consider_turn(user_message="Switch memory to Qdrant.", **common)
    active = memory.repository.find_active_by_key(
        user_id="user-1",
        project_id="project-1",
        canonical_key="project.memory_database",
    )

    assert actions == ["Remembered project.memory_database."]
    assert active is not None
    assert active.text == "The project uses Qdrant for conversational memory."
    assert memory.count(user_id="user-1") == 1


def test_assistant_inference_and_secret_are_not_stored(monkeypatch, tmp_path):
    memory = manager(
        monkeypatch,
        tmp_path,
        [
            MemoryExtraction(
                candidates=[
                    candidate(source="assistant_inference"),
                    candidate(
                        text="The API key is abc123.",
                        canonical_key="project.api_key",
                        memory_type="project_fact",
                        sensitivity=1.0,
                    ),
                ]
            )
        ],
    )

    actions = memory.consider_turn(
        user_message="Run the analysis.",
        assistant_response="I inferred a configuration.",
        response_kind="analysis",
        user_id="user-1",
        project_id="project-1",
        session_id="session-1",
    )

    assert actions == []
    assert memory.count(user_id="user-1") == 0


def test_user_scoped_preference_updates_across_projects(monkeypatch, tmp_path):
    memory = manager(
        monkeypatch,
        tmp_path,
        [
            MemoryExtraction(
                candidates=[
                    candidate(
                        text="The user prefers concise business summaries.",
                        canonical_key="user.analysis_style",
                        memory_type="preference",
                        scope="user",
                    )
                ]
            ),
            MemoryExtraction(
                candidates=[
                    candidate(
                        text="The user prefers detailed business summaries.",
                        canonical_key="user.analysis_style",
                        memory_type="preference",
                        scope="user",
                    )
                ]
            ),
        ],
    )
    common = {
        "assistant_response": "Understood.",
        "response_kind": "analysis",
        "user_id": "user-1",
        "session_id": "session-1",
    }

    memory.consider_turn(
        user_message="I prefer concise summaries.", project_id="project-1", **common
    )
    memory.consider_turn(
        user_message="I now prefer detailed summaries.", project_id="project-2", **common
    )
    active = memory.repository.find_active_by_key(
        user_id="user-1",
        project_id=None,
        canonical_key="user.analysis_style",
    )

    assert active is not None
    assert active.text == "The user prefers detailed business summaries."
    assert memory.count(user_id="user-1") == 1


def test_memory_score_penalizes_temporary_sensitive_candidates():
    durable = candidate()
    temporary = candidate(sensitivity=0.8, temporary=1.0)

    assert memory_score(durable, novelty=1.0) > 0.75
    assert memory_score(temporary, novelty=1.0) < 0.50
