import pytest

from app.skills import extract_skills_from_text, normalize_skill, normalize_skills


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ReactJS", "React"),
        ("React.js", "React"),
        ("react", "React"),
        ("k8s", "Kubernetes"),
        ("JS", "JavaScript"),
        ("node", "Node.js"),
        ("Postgres", "PostgreSQL"),
        ("c#", "C#"),
        ("golang", "Go"),
        ("Go", "Go"),
        ("  Python (5 yrs) ", "Python"),
        ("Node.js.", "Node.js"),
        (".net core", ".NET"),
        ("Data Modeling", "Data Modeling"),  # unknown skills are kept, not dropped
    ],
)
def test_normalize_skill(raw, expected):
    assert normalize_skill(raw) == expected


def test_normalize_skill_rejects_junk():
    assert normalize_skill("") is None
    assert normalize_skill(None) is None
    assert normalize_skill("x" * 80) is None


def test_normalize_skills_dedupes_case_insensitively_and_keeps_order():
    assert normalize_skills(["ReactJS", "python", "React", "Python", "k8s"]) == ["React", "Python", "Kubernetes"]


def test_scan_finds_common_skills_in_order():
    text = "We use Python, React.js and Postgres on AWS, deployed with k8s."
    assert extract_skills_from_text(text) == ["Python", "React", "PostgreSQL", "AWS", "Kubernetes"]


def test_scan_does_not_match_inside_other_words():
    skills = extract_skills_from_text("JavaScript and NoSQL and GitHub and Rustic")
    assert skills == ["JavaScript"]  # not Java, SQL, Git or Rust


def test_scan_handles_symbols():
    assert set(extract_skills_from_text("C++ and C#, plus .NET and Node.js.")) == {"C++", "C#", ".NET", "Node.js"}


def test_scan_skips_ambiguous_words():
    text = "Go to market. The rest of the team will take a swift look; a node in the cluster."
    assert extract_skills_from_text(text) == []


def test_scan_ambiguous_names_need_their_proper_form():
    assert extract_skills_from_text("Experience with Golang, Rust and Swift.") == ["Go", "Rust", "Swift"]
    assert extract_skills_from_text("We use RAG pipelines") == ["RAG"]
    assert extract_skills_from_text("a rag and a swift reply") == []


def test_scan_prefers_longest_match_and_flexible_whitespace():
    assert extract_skills_from_text("React Native apps") == ["React Native"]
    assert extract_skills_from_text("machine\n learning and RESTful APIs") == ["Machine Learning", "REST APIs"]


def test_scan_empty():
    assert extract_skills_from_text("") == []
