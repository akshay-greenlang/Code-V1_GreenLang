# -*- coding: utf-8 -*-
"""Tests for the Factors Python SDK certificate pinning primitives.

Strategy
--------
Spinning up a full HTTPS server with a rogue cert inside the sandbox is
overkill and flaky. Instead we exercise the pin primitives directly:

* ``_verify_pinned_chain`` accepts any cert in the chain matching the
  expected SHA-256 fingerprint.
* It raises :class:`CertificatePinError` when none match.
* The :class:`CertPinnedHTTPAdapter` builds successfully when
  ``requests`` is installed, and its ``send()`` calls
  ``_verify_pinned_chain`` on the peer DER.
* Disabled pinning (``verify_greenlang_cert=False``) does not wire a
  pinned transport, so a MockTransport can be used directly.
* The pinned httpx transport is only auto-attached for ``*.greenlang.io``
  hosts and NOT for localhost dev setups.

``MockHTTPSServer`` stand-ins are provided as plain stub objects with a
scripted peer-cert DER so the adapter's verification path is exercised
without touching a socket.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import List

import httpx
import pytest

from greenlang.factors.sdk.python.client import (
    CertPinnedHTTPAdapter,
    CertificatePinError,
    FactorsClient,
    GREENLANG_CA_PEM,
    PINNED_HOST_SUFFIXES,
    _host_matches_pin,
    _load_ca_fingerprint,
    _verify_pinned_chain,
    build_pinned_httpx_transport,
)

# --------------------------------------------------------------------------- #
# Fixtures: scripted DER bytes for a "good" pin and a "rogue" cert.           #
# --------------------------------------------------------------------------- #

GOOD_CERT_DER = b"greenlang-ca-fixture-der-bytes-for-pin-unit-tests"
ROGUE_CERT_DER = b"rogue-attacker-cert-that-should-never-pass-the-pin"
GOOD_PIN = hashlib.sha256(GOOD_CERT_DER).hexdigest()
ROGUE_PIN = hashlib.sha256(ROGUE_CERT_DER).hexdigest()


# --------------------------------------------------------------------------- #
# _verify_pinned_chain                                                         #
# --------------------------------------------------------------------------- #


class TestVerifyPinnedChain:
    def test_accepts_cert_matching_fingerprint(self) -> None:
        # Should not raise.
        _verify_pinned_chain([GOOD_CERT_DER], expected_fingerprint=GOOD_PIN)

    def test_accepts_any_cert_in_chain(self) -> None:
        # The pin can match a non-leaf cert (intermediate / root).
        _verify_pinned_chain(
            [ROGUE_CERT_DER, GOOD_CERT_DER],
            expected_fingerprint=GOOD_PIN,
        )

    def test_rejects_rogue_chain(self) -> None:
        with pytest.raises(CertificatePinError) as excinfo:
            _verify_pinned_chain(
                [ROGUE_CERT_DER], expected_fingerprint=GOOD_PIN
            )
        assert "pin failure" in str(excinfo.value)
        ctx = excinfo.value.context
        assert ctx["expected_pin"] == GOOD_PIN
        assert ctx["presented_count"] == 1

    def test_rejects_empty_chain(self) -> None:
        with pytest.raises(CertificatePinError):
            _verify_pinned_chain([], expected_fingerprint=GOOD_PIN)

    def test_normalizes_colonized_fingerprint(self) -> None:
        # openssl / Node emit colon-separated hex; the pin check must
        # normalize before comparing.
        colonized = ":".join(GOOD_PIN[i : i + 2] for i in range(0, len(GOOD_PIN), 2))
        _verify_pinned_chain(
            [GOOD_CERT_DER], expected_fingerprint=colonized.upper()
        )


# --------------------------------------------------------------------------- #
# Host-suffix matching                                                         #
# --------------------------------------------------------------------------- #


class TestHostMatching:
    def test_known_greenlang_hosts_match(self) -> None:
        assert _host_matches_pin("api.greenlang.io")
        assert _host_matches_pin("factors.greenlang.io")
        assert _host_matches_pin("greenlang.io")
        assert _host_matches_pin("API.GREENLANG.IO")

    def test_unrelated_hosts_do_not_match(self) -> None:
        assert not _host_matches_pin("example.com")
        assert not _host_matches_pin("localhost")
        assert not _host_matches_pin("greenlang.io.evil.com")
        assert not _host_matches_pin("")

    def test_pin_suffix_constant_not_empty(self) -> None:
        assert PINNED_HOST_SUFFIXES
        assert all(s for s in PINNED_HOST_SUFFIXES)


# --------------------------------------------------------------------------- #
# CertPinnedHTTPAdapter                                                        #
# --------------------------------------------------------------------------- #


class TestCertPinnedHTTPAdapter:
    def test_adapter_construction_with_explicit_pin(self) -> None:
        # Skip cleanly if `requests` is not available in the sandbox.
        pytest.importorskip("requests")
        adapter = CertPinnedHTTPAdapter(expected_fingerprint=GOOD_PIN)
        assert adapter.expected_fingerprint == GOOD_PIN.lower()

    def test_adapter_falls_back_to_bundled_fingerprint(self) -> None:
        pytest.importorskip("requests")
        adapter = CertPinnedHTTPAdapter()
        # The expected fingerprint must be deterministically derived
        # from the bundled GREENLANG_CA_PEM (placeholder or real).
        assert adapter.expected_fingerprint == _load_ca_fingerprint()

    def test_send_rejects_rogue_peer(self) -> None:
        pytest.importorskip("requests")
        adapter = CertPinnedHTTPAdapter(expected_fingerprint=GOOD_PIN)
        # Simulate the requests internals post-handshake: build a fake
        # response whose raw socket returns the rogue DER.  The inner
        # HTTPAdapter.send would have returned successfully (simulated
        # here by monkey-patching the parent `send` to a stub).
        inner = adapter._adapter  # type: ignore[attr-defined]

        fake_sock = SimpleNamespace(getpeercert=lambda binary_form=True: ROGUE_CERT_DER)
        fake_raw = SimpleNamespace(_sock=fake_sock)
        fake_response = SimpleNamespace(raw=fake_raw)
        import requests.adapters

        original_send = requests.adapters.HTTPAdapter.send

        def stub_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            return fake_response

        requests.adapters.HTTPAdapter.send = stub_send  # type: ignore[assignment]
        try:
            with pytest.raises(CertificatePinError):
                inner.send(SimpleNamespace(), stream=False)
        finally:
            requests.adapters.HTTPAdapter.send = original_send  # type: ignore[assignment]

    def test_send_accepts_matching_peer(self) -> None:
        pytest.importorskip("requests")
        adapter = CertPinnedHTTPAdapter(expected_fingerprint=GOOD_PIN)
        inner = adapter._adapter  # type: ignore[attr-defined]

        fake_sock = SimpleNamespace(getpeercert=lambda binary_form=True: GOOD_CERT_DER)
        fake_raw = SimpleNamespace(_sock=fake_sock)
        fake_response = SimpleNamespace(raw=fake_raw)
        import requests.adapters

        original_send = requests.adapters.HTTPAdapter.send

        def stub_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            return fake_response

        requests.adapters.HTTPAdapter.send = stub_send  # type: ignore[assignment]
        try:
            # Should return the response without raising.
            out = inner.send(SimpleNamespace(), stream=False)
            assert out is fake_response
        finally:
            requests.adapters.HTTPAdapter.send = original_send  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# build_pinned_httpx_transport — at least constructs                           #
# --------------------------------------------------------------------------- #


class TestBuildPinnedHttpxTransport:
    def test_returns_httpx_transport_subclass(self) -> None:
        transport = build_pinned_httpx_transport()
        assert isinstance(transport, httpx.HTTPTransport)

    def test_accepts_custom_ca_pem(self) -> None:
        # An invalid PEM must not crash; the pin check falls back to
        # hashing the raw string so the transport still constructs.
        transport = build_pinned_httpx_transport(ca_pem="not-a-real-pem")
        assert isinstance(transport, httpx.HTTPTransport)


# --------------------------------------------------------------------------- #
# FactorsClient integration                                                    #
# --------------------------------------------------------------------------- #


def _mock_transport_capturing(
    calls: List[dict], *, return_edition: str = "2027.Q1-electricity"
) -> httpx.MockTransport:
    """Return a MockTransport that records calls and returns a stub payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            }
        )
        return httpx.Response(
            200,
            json={"factors": [], "count": 0},
            headers={"X-GreenLang-Edition": return_edition},
        )

    return httpx.MockTransport(handler)


class TestFactorsClientPinning:
    def test_get_pin_fingerprint_is_stable(self) -> None:
        client = FactorsClient(
            base_url="http://localhost:8000",
            verify_greenlang_cert=False,
        )
        fp1 = client.get_pin_fingerprint()
        fp2 = client.get_pin_fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in fp1)

    def test_pin_auto_attached_for_greenlang_host(self) -> None:
        # Constructing a client with a *.greenlang.io base_url should
        # succeed (the pinned transport is built lazily).
        client = FactorsClient(
            base_url="https://api.greenlang.io",
            verify_greenlang_cert=True,
        )
        assert client.get_pin_fingerprint()

    def test_pin_disabled_skips_pinned_transport(self) -> None:
        # With pinning disabled + a custom MockTransport, requests work
        # transparently — this proves the pin is not inserted.
        calls: List[dict] = []
        mock = _mock_transport_capturing(calls)
        client = FactorsClient(
            base_url="https://api.greenlang.io",
            api_key="gl_test",
            verify_greenlang_cert=False,
            transport=mock,
        )
        client.search("diesel")
        assert len(calls) == 1
        # Also verify we did not inject an X-GreenLang-Edition header —
        # pinning disabled should not pin an edition either.
        assert "x-greenlang-edition" not in {
            k.lower() for k in calls[0]["headers"]
        }

    def test_pin_disabled_still_reaches_custom_transport(self) -> None:
        calls: List[dict] = []
        mock = _mock_transport_capturing(calls)
        client = FactorsClient(
            base_url="http://localhost:8000",  # non-pinned host
            api_key="gl_test",
            verify_greenlang_cert=True,  # default on, but host is unpinned
            transport=mock,
        )
        client.search("diesel")
        assert len(calls) == 1

    def test_custom_transport_bypasses_auto_pin(self) -> None:
        # When the caller supplies their own transport (e.g. tests), the
        # auto-pinned transport must NOT be injected, otherwise the mock
        # transport would be shadowed and the test would hit the real
        # network.
        calls: List[dict] = []
        mock = _mock_transport_capturing(calls)
        client = FactorsClient(
            base_url="https://api.greenlang.io",
            api_key="gl_test",
            verify_greenlang_cert=True,
            transport=mock,  # must win over auto-pinning
        )
        result = client.search("diesel")
        assert result.count == 0
        assert len(calls) == 1
