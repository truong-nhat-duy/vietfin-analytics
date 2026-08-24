"""Base contract every source connector must implement.

Design goals (see project data policy, Section II of the spec):
  * Every collected record must carry provenance: source_name, source_url,
    retrieved_at (and document_id / document_hash where applicable).
  * No collector may bypass CAPTCHA, authentication, paywalls, anti-bot
    protection, rate limits, or other access controls.
  * Business logic (universe building, normalization, etc.) must never
    hard-code scraping logic; it only depends on this interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterator

logger = logging.getLogger("vietfin.ingestion")


@dataclass(frozen=True)
class CollectedCompany:
    """One company record as returned by a source connector.

    This is intentionally source-agnostic: connectors normalize whatever
    shape their upstream API/page returns into this structure before
    handing it to the universe-building orchestrator.
    """

    ticker: str
    company_name: str
    exchange: str  # HOSE | HNX | UPCOM
    isin: str | None = None
    listing_date: date | None = None
    delisting_date: date | None = None
    status: str = "active"  # active | delisted | suspended
    sector: str | None = None
    industry: str | None = None
    website: str | None = None

    # --- mandatory provenance fields (Section II of the spec) ---
    source_name: str = ""
    source_url: str | None = None
    retrieved_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    document_id: str | None = None
    document_hash: str | None = None
    source_page: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name:
            raise ValueError(
                f"CollectedCompany for ticker={self.ticker!r} is missing "
                "source_name; every observation must carry provenance."
            )
        if not self.ticker:
            raise ValueError("CollectedCompany.ticker must not be empty")
        if not self.exchange:
            raise ValueError("CollectedCompany.exchange must not be empty")


class RateLimiter:
    """Enforces a minimum delay between requests to a single source.

    This exists to be polite and predictable, and explicitly NOT to
    evade a server's own rate limiting -- collectors must still respect
    whatever limits/backoff a source communicates (HTTP 429, Retry-After,
    documented quotas, etc.) on top of this floor.
    """

    def __init__(self, min_interval_seconds: float = 1.0) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


class SourceAccessError(RuntimeError):
    """Raised when a source cannot be legitimately accessed.

    Collectors should raise this (rather than silently returning partial
    data) when, e.g., an endpoint now requires login, returns a CAPTCHA
    challenge, or otherwise signals that automated access is not
    permitted. This is a signal to fall back to another permitted source,
    per requirement #18 -- never a signal to bypass the obstacle.
    """


class BaseCollector(ABC):
    """Abstract base class for all source connectors.

    Subclasses implement `fetch_companies`, which must be a generator
    (or return an iterator) so callers can support pagination and large
    universes without loading everything into memory at once.
    """

    #: Human-readable name used to stamp CollectedCompany.source_name
    source_name: str = "unset"

    def __init__(
        self,
        min_request_interval_seconds: float = 1.0,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> None:
        self._rate_limiter = RateLimiter(min_request_interval_seconds)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"vietfin.ingestion.{self.source_name}")

    @abstractmethod
    def fetch_companies(self, exchange: str | None = None) -> Iterator[CollectedCompany]:
        """Yield CollectedCompany records.

        Args:
            exchange: optionally restrict to one exchange (HOSE/HNX/UPCOM).
                If None, the connector should yield for all exchanges it
                covers.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Cheap check that the source is reachable and returning data.

        Default implementation tries to pull a single record. Subclasses
        may override with something cheaper (e.g. a HEAD request).
        """
        try:
            next(iter(self.fetch_companies()))
            return True
        except StopIteration:
            return True
        except SourceAccessError:
            return False
        except Exception:  # noqa: BLE001 - health check must never raise
            self.logger.exception("health_check failed for %s", self.source_name)
            return False
