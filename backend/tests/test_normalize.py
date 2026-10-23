import pytest

from app.normalize import content_hash, html_to_text, infer_seniority, looks_remote


def test_html_to_text_unescapes_and_strips_tags():
    raw = "&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt; &amp;amp; friends&lt;/p&gt;&lt;ul&gt;&lt;li&gt;One&lt;/li&gt;&lt;li&gt;Two&lt;/li&gt;&lt;/ul&gt;"
    text = html_to_text(raw)
    assert "<" not in text
    assert "Hello world & friends" in text
    assert "- One\n- Two" in text


def test_html_to_text_drops_script_and_style():
    assert html_to_text("<style>p{color:red}</style><p>Keep</p><script>alert(1)</script>") == "Keep"


def test_html_to_text_handles_empty():
    assert html_to_text("") == ""
    assert html_to_text(None) == ""


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Senior Backend Engineer", "senior"),
        ("Sr. Software Engineer", "senior"),
        ("Software Engineer Intern", "intern"),
        ("Staff Software Engineer", "staff"),
        ("Engineering Manager", "manager"),
        ("Senior Engineering Manager", "manager"),
        ("Junior Data Analyst", "junior"),
        ("Software Engineer", None),
    ],
)
def test_infer_seniority(title, expected):
    assert infer_seniority(title) == expected


def test_looks_remote():
    assert looks_remote("Remote - India")
    assert looks_remote("Bengaluru", "Backend Engineer (Remote)")
    assert not looks_remote("Bengaluru", "Backend Engineer")
    assert not looks_remote(None)


def test_content_hash_ignores_case_and_punctuation():
    a = content_hash("Acme Inc.", "Senior Engineer", "Remote, India")
    b = content_hash("acme inc", "senior  engineer", "remote india")
    assert a == b
    assert a != content_hash("Acme Inc.", "Senior Engineer", "Bengaluru")
