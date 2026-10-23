from app import main


class FakeResult:
    def mappings(self):
        return self

    def all(self):
        return [{"id": 1, "title": "Engineer"}]


class FakeConnection:
    def __init__(self):
        self.statement = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, params):
        self.statement = str(statement)
        self.params = params
        return FakeResult()


class FakeEngine:
    def __init__(self):
        self.connection = FakeConnection()

    def connect(self):
        return self.connection


def test_job_endpoint_applies_location_and_remote_filters_before_returning_results(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main, "get_engine", lambda: engine)

    result = main.list_jobs(location=" Seattle ", remote=True)

    assert result == [{"id": 1, "title": "Engineer"}]
    assert "j.is_remote = :remote" in engine.connection.statement
    assert "(j.location ILIKE :location OR j.location IS NULL)" in engine.connection.statement
    assert engine.connection.params["remote"] is True
    assert engine.connection.params["location"] == "%Seattle%"


def test_job_endpoint_does_not_apply_location_filter_for_blank_location(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main, "get_engine", lambda: engine)

    main.list_jobs(location="   ")

    assert "j.location ILIKE" not in engine.connection.statement
    assert "experience" not in engine.connection.statement
    assert "j.seniority = :seniority" not in engine.connection.statement


def test_recommended_job_sort_uses_profile_and_job_embedding_similarity(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(main, "get_engine", lambda: engine)

    main.list_jobs(q=None, remote=None, location=None, seniority=None, status=None, sort="recommended", profile="me", limit=30, offset=0)

    assert "j.embedding <=> p.embedding" in engine.connection.statement
    assert "ORDER BY match_score DESC NULLS LAST" in engine.connection.statement
    assert engine.connection.params["profile"] == "me"
