# -*- coding: utf-8 -*-
"""Phase 3 / Wave 2.5 — PDF fetcher for the unified ingestion runner.

Why this module exists
----------------------
The Phase 3 plan §"Block 3 -- PDF/OCR family" calls for a fetcher that
either reads a local PDF (the design-partner upload path) or downloads
one from an HTTP endpoint with an explicit "archival fetch" user-agent
(so the publisher can audit the request as a deliberate factors-data
ingestion, not a casual browse).

Per the Phase 3 / Wave 2.5 contract:

  * For ``file://`` URIs and bare local paths: read the bytes directly.
    No archival headers needed (the bytes never leave the runtime).

  * For ``http://`` / ``https://`` URIs: issue a GET with a
    ``User-Agent`` header matching the canonical archival pattern
    ``GreenLang-Factors-PdfArchival/<version> (+contact)``. The header is
    deliberately verbose so the publisher's ops team can identify the
    request in their log archives.

The fetcher returns the raw bytes; the runner is responsible for storing
the artifact via :class:`LocalArtifactStore` (which produces a
:class:`StoredArtifact` carrying the ``content_type='application/pdf'``
hint that the parser later asserts on).

Design notes
------------
* No third-party HTTP library required — :mod:`urllib.request` is in the
  stdlib and the request shape is trivial. Tests stub the fetch via the
  unified runner's ``fetcher`` parameter so the network never resolves.

* The fetcher does NOT parse the PDF. Decoding is the parser's job; this
  is purely the byte-acquisition stage (Phase 3 stage 1).

References
----------
- ``docs/factors/PHASE_3_PLAN.md`` §"Block 3 -- PDF/OCR family"
- ``docs/factors/PHASE_3_EXIT_CHECKLIST.md`` Block 3 (PDF/OCR family).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

from greenlang.factors.ingestion.fetchers import BaseFetcher

logger = logging.getLogger(__name__)


__all__ = [
    "PDF_ARCHIVAL_USER_AGENT",
    "PDF_CONTENT_TYPE",
    "PdfFetcher",
]


#: Canonical user-agent for archival PDF fetches. Used as the literal
#: ``User-Agent`` header so the source publisher's logs distinguish a
#: GreenLang factors-archival GET from any other client.
PDF_ARCHIVAL_USER_AGENT: str = (
    "GreenLang-Factors-PdfArchival/1.0 (+factors-ops@greenlang.io)"
)

#: MIME type the parser asserts on. Stored via
#: :class:`StoredArtifact` once the runner persists the bytes.
PDF_CONTENT_TYPE: str = "application/pdf"


class PdfFetcher(BaseFetcher):
    """Fetch a PDF artifact (local file or HTTP) for the Phase 3 runner.

    Resolution rules:

      * ``file://`` URIs are unwrapped via :func:`urllib.parse.urlparse`,
        the path is :func:`urllib.parse.unquote`-decoded, and (on
        Windows) a leading ``/`` immediately followed by a drive letter
        is stripped.

      * Bare paths (``C:\\path\\to\\file.pdf`` or ``/usr/share/x.pdf``)
        are read directly via :class:`pathlib.Path`.

      * ``http://`` / ``https://`` URIs are fetched via
        :func:`urllib.request.urlopen` with the archival user-agent
        attached. The connection inherits :attr:`timeout_s`.

    Attributes:
        timeout_s: Per-request timeout, in seconds.
        user_agent: The literal ``User-Agent`` header value to send on
            HTTP fetches. Defaults to :data:`PDF_ARCHIVAL_USER_AGENT`.
        max_bytes: Optional safety cap for archival downloads. ``None``
            means "no cap". When set, an HTTP fetch that returns more
            bytes is truncated and the read is logged as a warning.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 60.0,
        user_agent: str = PDF_ARCHIVAL_USER_AGENT,
        max_bytes: Optional[int] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.user_agent = str(user_agent)
        self.max_bytes = max_bytes

    # -- BaseFetcher contract -------------------------------------------------

    def fetch(self, url: str) -> bytes:
        """Return the PDF bytes addressed by *url*.

        Args:
            url: A ``file://`` URI, a bare local path, or an
                ``http(s)://`` URL pointing at a PDF document.

        Returns:
            The raw PDF byte string.

        Raises:
            FileNotFoundError: If the local path does not exist.
            urllib.error.URLError: On HTTP failures.
        """
        if not url:
            raise ValueError("PdfFetcher.fetch: url must be non-empty")

        scheme = urlparse(url).scheme.lower()

        if scheme in ("http", "https"):
            return self._fetch_http(url)
        if scheme == "file":
            return self._fetch_file_uri(url)
        # Bare path (no scheme) or arbitrary single-letter Windows drive
        # treated as a path. ``urlparse`` returns scheme='c' for
        # ``C:\foo`` on some Windows configs, so we take the path branch
        # for any single-letter scheme too.
        if len(scheme) <= 1:
            return self._fetch_local_path(url)
        # Unknown scheme — try to read as a local path; if that fails,
        # surface a clear error.
        return self._fetch_local_path(url)

    # -- internal helpers -----------------------------------------------------

    def _fetch_http(self, url: str) -> bytes:
        """Download *url* with the archival user-agent attached."""
        req = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                # Hint to the source server we expect a PDF. Servers that
                # negotiate content-type respect this.
                "Accept": PDF_CONTENT_TYPE + ";q=1.0, */*;q=0.1",
            },
            method="GET",
        )
        logger.info(
            "PdfFetcher: archival GET url=%s ua=%s timeout=%.1fs",
            url, self.user_agent, self.timeout_s,
        )
        with urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310 — registry-controlled URLs
            if self.max_bytes is not None:
                data = resp.read(int(self.max_bytes))
                # Probe one extra byte to see if the source over-shot the cap.
                tail = resp.read(1)
                if tail:
                    logger.warning(
                        "PdfFetcher: archival GET truncated at max_bytes=%d "
                        "(server returned more); url=%s",
                        self.max_bytes, url,
                    )
                return data
            return resp.read()

    def _fetch_file_uri(self, url: str) -> bytes:
        """Resolve a ``file://`` URI and read its bytes."""
        parsed = urlparse(url)
        raw_path = unquote(parsed.path)
        # Windows: ``urlparse`` yields '/C:/Users/...'; strip the leading
        # slash so :class:`Path` resolves against the local filesystem.
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        return self._fetch_local_path(raw_path)

    @staticmethod
    def _fetch_local_path(path: str) -> bytes:
        """Read a local path from disk, raising FileNotFoundError if absent."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        return p.read_bytes()
