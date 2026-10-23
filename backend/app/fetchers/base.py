from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class NormalizedJob:
    """The one job shape every fetcher must return, whatever the ATS looks like."""

    source: str  # "greenhouse", "lever", "ashby"
    external_id: str  # the ATS's own id, as a string
    title: str
    description: str  # plain text, tags already stripped
    location: str | None
    is_remote: bool
    apply_url: str
    posted_at: datetime | None = None
    seniority: str | None = None


class Fetcher(ABC):
    source: str

    @abstractmethod
    def fetch(self, slug: str) -> list[NormalizedJob]:
        """Return every currently-open job for one company board.

        Raise FetchError if the fetch fails. Never return a partial list on failure, because
        ingestion treats "not in the list" as "job closed".
        """
