from app import api_profile
from app.llm import EMBEDDING_DIMENSIONS, LLMError, LLMNotConfigured


class FakeResult:
    def __init__(self, row=None, rowcount=1):
        self.row = row or {}
        self.rowcount = rowcount

    def mappings(self):
        return self

    def one(self):
        return self.row


class FakeConnection:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, params):
        sql = str(statement)
        self.statements.append((sql, params))
        if "INSERT INTO profiles" in sql:
            return FakeResult({
                "id": 4,
                "name": params["name"],
                "skills": params["skills"],
                "titles": params["titles"],
                "years_experience": params["years"],
                "locations": params["locations"],
                "remote_ok": params["remote_ok"],
                "seniority": params["seniority"],
                "has_embedding": False,
            })
        return FakeResult(rowcount=1)


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def begin(self):
        return self.connection


class FakeEmbedder:
    dimensions = EMBEDDING_DIMENSIONS
    model = "test-embedding-model"

    def __init__(self):
        self.texts = []
        self.task_types = []

    def embed(self, texts, *, task_type):
        self.texts.extend(texts)
        self.task_types.append(task_type)
        return [[0.25] * self.dimensions]


def profile_input():
    return api_profile.ProfileIn(
        name="me",
        skills=["python"],
        titles=["Backend Engineer"],
    )


def test_saving_profile_refreshes_embedding_and_returns_ready_status(monkeypatch):
    engine = FakeEngine()
    embedder = FakeEmbedder()
    monkeypatch.setattr(api_profile, "get_engine", lambda: engine)
    monkeypatch.setattr(api_profile, "get_embedder", lambda: embedder)

    result = api_profile.save_profile(profile_input())

    assert result.embedding_status == "ready"
    assert embedder.task_types == ["RETRIEVAL_QUERY"]
    assert "Python" in embedder.texts[0]
    assert any("UPDATE profiles" in sql for sql, _ in engine.connection.statements)


def test_profile_is_saved_and_reports_pending_when_embedding_provider_is_unavailable(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(api_profile, "get_engine", lambda: engine)
    monkeypatch.setattr(
        api_profile,
        "get_embedder",
        lambda: (_ for _ in ()).throw(LLMNotConfigured("quota unavailable")),
    )

    result = api_profile.save_profile(profile_input())

    assert result.embedding_status == "pending"
    assert result.embedding_message == "quota unavailable"
    assert "INSERT INTO profiles" in engine.connection.statements[0][0]


def test_profile_is_saved_and_reports_pending_when_embedding_call_fails(monkeypatch):
    engine = FakeEngine()

    class FailingEmbedder(FakeEmbedder):
        def embed(self, texts, *, task_type):
            raise LLMError("429 quota exhausted")

    monkeypatch.setattr(api_profile, "get_engine", lambda: engine)
    monkeypatch.setattr(api_profile, "get_embedder", FailingEmbedder)

    result = api_profile.save_profile(profile_input())

    assert result.embedding_status == "pending"
    assert "429" in result.embedding_message
