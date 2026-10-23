"""Shared helpers that turn messy ATS data into clean, comparable values."""
from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "br", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "section", "article", "blockquote", "hr",
}
_SKIP_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")
        else:
            self.handle_starttag(tag, attrs)
            self.handle_endtag(tag)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    """Greenhouse returns entity-escaped HTML ('&lt;p&gt;...'), so unescape first, then strip tags."""
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(raw))
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def content_hash(company: str, title: str, location: str | None) -> str:
    """Cross-source dedupe key: company + normalized title + normalized location."""
    raw = "|".join((_key(company), _key(title), _key(location)))
    return hashlib.sha256(raw.encode()).hexdigest()


# Order matters: the first matching rule wins ("Senior Engineering Manager" -> manager).
_SENIORITY_RULES = [
    ("intern", r"\bintern(ship)?\b"),
    ("junior", r"\b(junior|jr|entry[- ]level|new grad|graduate)\b"),
    ("principal", r"\b(principal|distinguished|fellow)\b"),
    ("staff", r"\bstaff\b"),
    ("manager", r"\b(manager|director|head of|vp|vice president)\b"),
    ("senior", r"\b(senior|sr)\b"),
    ("lead", r"\blead\b"),
]


def infer_seniority(title: str) -> str | None:
    """Cheap title heuristic. Returns None when unsure; later filters should treat None as 'pass'."""
    lowered = (title or "").lower()
    for label, pattern in _SENIORITY_RULES:
        if re.search(pattern, lowered):
            return label
    return None


def looks_remote(location: str | None, title: str = "") -> bool:
    return "remote" in f"{location or ''} {title}".lower()
