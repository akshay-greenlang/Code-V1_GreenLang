# -*- coding: utf-8 -*-
"""Phase F6 — Provenance hardening tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from greenlang.factors.quality.impact_simulator import (
    ImpactReport,
    ImpactSimulator,
    load_ledger_entries_from_sqlite,
)
from greenlang.factors.quality.versioning import (
    FactorVersionChain,
    VersioningError,
    compute_chain_hash,
)
from greenlang.factors.signing import (
    Receipt,
    SigningError,
    sign_sha256_hmac,
    verify_sha256_hmac,
)
from greenlang.factors.webhooks import (
    WebhookEventType,
    WebhookRegistry,
    sign_webhook_payload,
)


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------


class TestVersionChain:
    def test_append_single_version(self, tmp_path: Path):
        chain = FactorVersionChain(tmp_path / "chain.sqlite")
        try:
            entry = chain.append(
                factor_id="EF:US:diesel:2024",
                factor_version="1.0.0",
                content_hash="a" * 64,
                changed_by="methodology_lead",
                change_reason="initial import",
            )
            assert entry.previous_version is None
            assert entry.previous_chain_hash is None
            assert entry.chain_hash
        finally:
            chain.close()

    def test_chain_hash_is_deterministic(self):
        h1 = compute_chain_hash(
            factor_id="A", factor_version="1", content_hash="x", previous_chain_hash=None
        )
        h2 = compute_chain_hash(
            factor_id="A", factor_version="1", content_hash="x", previous_chain_hash=None
        )
        assert h1 == h2

    def test_chain_hash_changes_with_content(self):
        h1 = compute_chain_hash(
            factor_id="A", factor_version="1", content_hash="x", previous_chain_hash=None
        )
        h2 = compute_chain_hash(
            factor_id="A", factor_version="1", content_hash="y", previous_chain_hash=None
        )
        assert h1 != h2

    def test_append_linked_chain(self, tmp_path: Path):
        chain = FactorVersionChain(tmp_path / "chain.sqlite")
        try:
            v1 = chain.append(
                factor_id="F1", factor_version="1.0.0", content_hash="aaa",
                changed_by="lead", change_reason="init",
            )
            v2 = chain.append(
                factor_id="F1", factor_version="1.1.0", content_hash="bbb",
                changed_by="lead", change_reason="refresh 2025",
            )
            assert v2.previous_version == "1.0.0"
            assert v2.previous_chain_hash == v1.chain_hash
            assert chain.verify_chain("F1") is True
        finally:
            chain.close()

    def test_duplicate_version_rejected(self, tmp_path: Path):
        chain = FactorVersionChain(tmp_path / "chain.sqlite")
        try:
            chain.append(
                factor_id="F1", factor_version="1.0", content_hash="x",
                changed_by="lead", change_reason="init",
            )
            with pytest.raises(VersioningError):
                chain.append(
                    factor_id="F1", factor_version="1.0", content_hash="y",
                    changed_by="lead", change_reason="dup",
                )
        finally:
            chain.close()

    def test_append_only_update_rejected(self, tmp_path: Path):
        chain = FactorVersionChain(tmp_path / "chain.sqlite")
        try:
            chain.append(
                factor_id="F1", factor_version="1.0", content_hash="x",
                changed_by="lead", change_reason="init",
            )
            with pytest.raises(sqlite3.IntegrityError):
                chain._conn.execute(
                    "UPDATE factor_version_chain SET content_hash='TAMPER'"
                )
        finally:
            chain.close()

    def test_verify_detects_tamper(self, tmp_path: Path):
        """Force-corrupt a chain_hash directly in SQLite and ensure verify fails."""
        chain = FactorVersionChain(tmp_path / "chain.sqlite")
        try:
            chain.append(
                factor_id="F1", factor_version="1.0", content_hash="x",
                changed_by="lead", change_reason="init",
            )
            # Tamper via DELETE trigger bypass? no — test via malformed insert.
            # Insert a row with a bad chain_hash by dropping + recreating triggers.
            chain._conn.execute("DROP TRIGGER IF EXISTS trg_fvc_no_update")
            chain._conn.execute(
                "UPDATE factor_version_chain SET chain_hash='tampered' "
                "WHERE factor_id = 'F1'"
            )
            assert chain.verify_chain("F1") is False
        finally:
            chain.close()


# --------------------------------------------------------------------------
# Impact simulator
# --------------------------------------------------------------------------


class TestImpactSimulator:
    def test_no_matches_returns_empty(self):
        sim = ImpactSimulator(ledger_entries=[], evidence_records=[])
        report = sim.simulate_replacement(replaced_factor_ids={"F-NONE"})
        assert report.computations == []
        assert report.tenants == []

    def test_finds_matching_ledger_entry(self):
        ledger = [
            {
                "id": 1,
                "entity_id": "run-001",
                "operation": "compute:CBAM",
                "content_hash": "h1",
                "chain_hash": "c1",
                "tenant_id": "acme",
                "metadata": {"factor_id": "F-CBAM-STEEL-2024"},
            }
        ]
        evidence = [
            {"evidence_id": "ev-001", "content_hash": "h1", "tenant_id": "acme"}
        ]
        sim = ImpactSimulator(ledger_entries=ledger, evidence_records=evidence)
        report = sim.simulate_replacement(replaced_factor_ids={"F-CBAM-STEEL-2024"})
        assert len(report.computations) == 1
        assert report.tenants == ["acme"]
        assert report.computations[0].evidence_bundle == "ev-001"

    def test_delta_populated_from_value_map(self):
        ledger = [
            {
                "id": 1, "entity_id": "run-001", "tenant_id": "acme",
                "content_hash": "h1", "chain_hash": "c1",
                "metadata": {"factor_id": "F-A"},
            }
        ]
        sim = ImpactSimulator(ledger_entries=ledger)
        report = sim.simulate_replacement(
            replaced_factor_ids={"F-A"},
            value_map={"F-A": {"old": 100.0, "new": 110.0}},
        )
        c = report.computations[0]
        assert c.delta_abs == pytest.approx(10.0)
        assert c.delta_pct == pytest.approx(10.0)

    def test_summary_counts(self):
        ledger = [
            {"id": i, "entity_id": f"r{i}", "tenant_id": "t1",
             "content_hash": f"h{i}", "chain_hash": f"c{i}",
             "metadata": {"factor_id": "F-X"}}
            for i in range(3)
        ]
        sim = ImpactSimulator(ledger_entries=ledger)
        report = sim.simulate_replacement(replaced_factor_ids={"F-X"})
        assert report.summary["affected_computations"] == 3
        assert report.summary["affected_tenants"] == 1

    def test_to_dict_round_trip(self):
        sim = ImpactSimulator(
            ledger_entries=[
                {"id": 1, "entity_id": "r1", "tenant_id": "t1",
                 "content_hash": "h1", "chain_hash": "c1",
                 "metadata": {"factor_id": "F-X"}}
            ]
        )
        report = sim.simulate_replacement(replaced_factor_ids={"F-X"})
        d = report.to_dict()
        assert d["computation_count"] == 1
        assert isinstance(d["computations"], list)

    def test_loader_handles_missing_file(self, tmp_path: Path):
        assert load_ledger_entries_from_sqlite(tmp_path / "missing.sqlite") == []


# --------------------------------------------------------------------------
# Signed receipts
# --------------------------------------------------------------------------


class TestSigning:
    def test_hmac_sign_and_verify(self):
        payload = {"chosen": "F-1", "co2e_total_kg": 42.0}
        receipt = sign_sha256_hmac(payload, secret="s3cret")
        assert receipt.algorithm == "sha256-hmac"
        assert receipt.signature
        assert verify_sha256_hmac(payload, receipt.to_dict(), secret="s3cret")

    def test_hmac_detects_tamper(self):
        receipt = sign_sha256_hmac({"x": 1}, secret="s3cret")
        assert not verify_sha256_hmac({"x": 2}, receipt.to_dict(), secret="s3cret")

    def test_hmac_wrong_secret_fails(self):
        receipt = sign_sha256_hmac({"x": 1}, secret="s3cret")
        assert not verify_sha256_hmac({"x": 1}, receipt.to_dict(), secret="other")

    def test_missing_secret_raises(self, monkeypatch):
        monkeypatch.delenv("GL_FACTORS_SIGNING_SECRET", raising=False)
        with pytest.raises(SigningError):
            sign_sha256_hmac({"x": 1})

    def test_canonical_ordering(self):
        """Field order must not affect the signature."""
        r1 = sign_sha256_hmac({"a": 1, "b": 2}, secret="k")
        r2 = sign_sha256_hmac({"b": 2, "a": 1}, secret="k")
        assert r1.payload_hash == r2.payload_hash


# --------------------------------------------------------------------------
# Webhook registry
# --------------------------------------------------------------------------


class TestWebhookRegistry:
    def test_register_and_list(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "webhooks.sqlite")
        try:
            sub = reg.register(
                tenant_id="t1",
                target_url="https://customer.example.com/hook",
                event_types=[WebhookEventType.FACTOR_DEPRECATED],
            )
            assert sub.subscription_id.startswith("whs_")
            assert sub.secret
            listed = reg.list_for_tenant("t1")
            assert len(listed) == 1
            assert listed[0].subscription_id == sub.subscription_id
        finally:
            reg.close()

    def test_rejects_non_https(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "webhooks.sqlite")
        try:
            with pytest.raises(ValueError):
                reg.register(
                    tenant_id="t1",
                    target_url="http://insecure.example.com/hook",
                    event_types=[WebhookEventType.FACTOR_DEPRECATED],
                )
        finally:
            reg.close()

    def test_rejects_unknown_event_type(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "webhooks.sqlite")
        try:
            with pytest.raises(ValueError):
                reg.register(
                    tenant_id="t1",
                    target_url="https://x/hook",
                    event_types=["bogus.event"],
                )
        finally:
            reg.close()

    def test_subscribers_for_event(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "webhooks.sqlite")
        try:
            reg.register(
                tenant_id="t1", target_url="https://x/hook",
                event_types=[WebhookEventType.FACTOR_DEPRECATED],
            )
            reg.register(
                tenant_id="t2", target_url="https://y/hook",
                event_types=[WebhookEventType.FACTOR_UPDATED],
            )
            subs = reg.subscribers_for_event(WebhookEventType.FACTOR_DEPRECATED)
            assert len(subs) == 1
            assert subs[0].tenant_id == "t1"
        finally:
            reg.close()

    def test_deactivate_removes_from_active(self, tmp_path: Path):
        reg = WebhookRegistry(tmp_path / "webhooks.sqlite")
        try:
            sub = reg.register(
                tenant_id="t1", target_url="https://x/hook",
                event_types=[WebhookEventType.FACTOR_DEPRECATED],
            )
            assert reg.deactivate(sub.subscription_id) is True
            assert reg.subscribers_for_event(WebhookEventType.FACTOR_DEPRECATED) == []
        finally:
            reg.close()

    def test_sign_webhook_payload_deterministic(self):
        s1 = sign_webhook_payload({"a": 1}, "secret")
        s2 = sign_webhook_payload({"a": 1}, "secret")
        assert s1 == s2

    def test_sign_webhook_payload_depends_on_secret(self):
        s1 = sign_webhook_payload({"a": 1}, "k1")
        s2 = sign_webhook_payload({"a": 1}, "k2")
        assert s1 != s2
