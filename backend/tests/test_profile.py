import json
from datetime import date

import pytest

from app.llm import LLMClient
from app.profile import (
    MAX_PDF_BYTES,
    ProfileError,
    extract_profile,
    extract_text_from_pdf,
    heuristic_extract,
    parse_json_object,
    validate_extraction,
)
from tests.pdf_utils import make_pdf

RESUME = (
    "Karthik K - Senior Software Engineer, Hyderabad, India. "
    "Acme Corp 2019 - Present: built backend services with Python, Django and Postgres, "
    "deployed on AWS with k8s. Earlier: Initech 2016-2019, Java developer."
)

GOOD = {
    "skills": ["ReactJS", "python", "Kubernetes"],
    "years_experience": 7,
    "titles": ["Senior Software Engineer", "Java Developer"],
    "locations": ["Hyderabad, India"],
    "remote_ok": None,
    "seniority": "senior",
}


class FakeLLM(LLMClient):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, system, user, max_tokens=1500):
        self.prompts.append((system, user))
        return self.replies.pop(0)


# ------------------------------------------------------------------ PDF


def test_extract_text_from_pdf_reads_text():
    pdf = make_pdf(
        [
            "Karthik K - Senior Software Engineer",
            "Skills: Python, Django, PostgreSQL, Kubernetes",
            "Experience: 7 years building backend services in Hyderabad",
        ]
    )
    text = extract_text_from_pdf(pdf)
    assert "Python, Django" in text
    assert "7 years" in text


def test_pdf_without_text_is_rejected_with_a_helpful_message():
    with pytest.raises(ProfileError, match="Scanned"):
        extract_text_from_pdf(make_pdf([]))


def test_non_pdf_is_rejected():
    with pytest.raises(ProfileError, match="PDF"):
        extract_text_from_pdf(b"just some text, not a pdf")


def test_oversized_pdf_is_rejected():
    with pytest.raises(ProfileError, match="too large"):
        extract_text_from_pdf(b"%PDF" + b"0" * MAX_PDF_BYTES)


# ------------------------------------------------------------------ JSON parsing / validation


def test_parse_json_object_handles_fences_and_chatter():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no json here")
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


def test_validate_extraction_is_strict_on_structure_lenient_on_scalars():
    p = validate_extraction({"skills": ["a", 5, "b"], "years_experience": "5+", "seniority": " Senior ", "remote_ok": "yes"})
    assert p.skills == ["a", "b"]
    assert p.years_experience == 5.0
    assert p.seniority == "senior"
    assert p.remote_ok is None
    with pytest.raises(ValueError):
        validate_extraction({"skills": "python, java"})


def test_validate_extraction_drops_implausible_values():
    p = validate_extraction({"years_experience": 400, "seniority": "wizard"})
    assert p.years_experience is None
    assert p.seniority is None


# ------------------------------------------------------------------ extract_profile


def test_extract_profile_normalizes_and_merges_scanned_skills():
    llm = FakeLLM([json.dumps(GOOD)])
    result = extract_profile(RESUME, llm, today=date(2026, 10, 20))

    skills = result.profile.skills
    assert skills[:3] == ["React", "Python", "Kubernetes"]  # model's skills first, normalized
    assert "PostgreSQL" in skills and "AWS" in skills and "Django" in skills  # found in the text
    assert len(skills) == len({s.lower() for s in skills})  # no duplicates
    assert result.profile.years_experience == 7.0
    assert result.profile.seniority == "senior"
    assert result.warnings == []
    assert result.used_llm is True
    # the resume goes in the user message, wrapped in tags, with today's date for "Present" roles
    system, user = llm.prompts[0]
    assert "<resume>" in user and "2026-10-20" in user
    assert "untrusted" in system


def test_extract_profile_accepts_fenced_json():
    llm = FakeLLM(["```json\n" + json.dumps(GOOD) + "\n```"])
    assert extract_profile(RESUME, llm).profile.titles[0] == "Senior Software Engineer"


def test_extract_profile_retries_once_on_bad_output():
    llm = FakeLLM(["I could not do that", json.dumps(GOOD)])
    result = extract_profile(RESUME, llm)
    assert len(llm.prompts) == 2
    assert "previous reply was invalid" in llm.prompts[1][1]
    assert result.profile.seniority == "senior"


def test_extract_profile_gives_up_after_two_bad_outputs():
    llm = FakeLLM(["nope", '{"skills": "not a list"}'])
    with pytest.raises(ProfileError):
        extract_profile(RESUME, llm)


def test_missing_seniority_is_estimated_from_years_with_a_warning():
    data = {**GOOD, "seniority": None, "years_experience": 3}
    result = extract_profile(RESUME, FakeLLM([json.dumps(data)]))
    assert result.profile.seniority == "mid"
    assert any("Seniority was estimated" in w for w in result.warnings)


def test_unknown_years_produces_a_warning():
    data = {**GOOD, "years_experience": None, "seniority": None}
    result = extract_profile(RESUME, FakeLLM([json.dumps(data)]))
    assert result.profile.seniority is None
    assert any("years of experience" in w for w in result.warnings)


# ------------------------------------------------------------------ no-LLM fallback


def test_heuristic_extract_only_finds_skills_and_says_so():
    result = heuristic_extract(RESUME)
    assert result.used_llm is False
    assert {"Python", "Django", "PostgreSQL", "AWS", "Kubernetes", "Java"} <= set(result.profile.skills)
    assert result.profile.titles == []
    assert "No LLM key" in result.warnings[0]
