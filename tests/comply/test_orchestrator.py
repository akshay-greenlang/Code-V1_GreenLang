# -*- coding: utf-8 -*-
"""Phase 3.1 Comply orchestrator tests."""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from greenlang.comply import (
    ComplianceRunRequest,
    ComplianceRunResult,
    ComplyOrchestrator,
    EntityProfile,
    FrameworkResult,
)
from greenlang.scope_engine import ScopeEngineService
from greenlang.scope_engine.models import (
    ComputationRequest,
    ComputationResponse,
    Framework,
    FrameworkView,
    GWPBasis,
    ScopeBreakdown,
    ScopeComputation,
)


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


class _StubScopeEngine:
    """Scope-Engine stand-in that returns a canned ComputationResponse."""

    def compute(self, request: ComputationRequest) -> ComputationResponse:
        comp = ScopeComputation(
            computation_id="test-computation",
            entity_id=request.entity_id,
            reporting_period_start=request.reporting_period_start,
            reporting_period_end=request.reporting_period_end,
            gwp_basis=request.gwp_basis,
            consolidation=request.consolidation,
            breakdown=ScopeBreakdown(
                scope_1_co2e_kg=Decimal("100"),
                scope_2_location_co2e_kg=Decimal("50"),
                scope_3_co2e_kg=Decimal("25"),
            ),
            results=[],
            total_co2e_kg=Decimal("175"),
            computation_hash="abc" * 21 + "a",
        )
        views: dict[Framework, FrameworkView] = {
            fw: FrameworkView(
                framework=fw,
                rows=[
                    {
                        "framework": fw.value,
                        "total_co2e_kg": "175",
                        "scope_1_co2e_kg": "100",
                        "scope_2_co2e_kg": "50",
                        "scope_3_co2e_kg": "25",
                    }
                ],
                metadata={"computation_hash": "abc" * 21 + "a"},
            )
            for fw in request.frameworks
        }
        return ComputationResponse(computation=comp, framework_views=views)


def _request(
    *,
    tmp_path: Path,
    with_activities: bool = True,
    with_ledger: bool = True,
    with_vault: bool = True,
    force: list[str] | None = None,
) -> ComplianceRunRequest:
    activities = []
    if with_activities:
        activities = [
            {
                "activity_id": "a1",
                "activity_type": "stationary_combustion",
                "fuel_type": "natural_gas",
                "quantity": "1000",
                "unit": "kWh",
                "year": 2026,
            }
        ]
    return ComplianceRunRequest(
        entity=EntityProfile(
            entity_id="acme-001",
            legal_name="Acme Steel Ltd",
            hq_country="IN",
            operates_in=["EU", "US-CA"],
            employees=500,
            turnover_m_eur=120.0,
            revenue_usd=2_000_000_000,
        ),
        reporting_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2026, 6, 30, tzinfo=timezone.utc),
        jurisdiction="EU",
        activities=activities,
        vault_id="test-vault",
        ledger_sqlite=str(tmp_path / "ledger.sqlite") if with_ledger else None,
        vault_sqlite=str(tmp_path / "vault.sqlite") if with_vault else None,
        force_frameworks=force,
    )


# --------------------------------------------------------------------------
# Applicability plumbing
# --------------------------------------------------------------------------


class TestApplicability:
    def test_policy_graph_applied_for_eu_cbam_profile(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(_request(tmp_path=tmp_path, with_activities=False))
        assert "CBAM" in result.applicable_regulations or "CSRD" in result.applicable_regulations
        assert "GHG-Protocol" in result.applicable_regulations

    def test_force_frameworks_bypasses_policy_graph(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(
            _request(
                tmp_path=tmp_path,
                with_activities=False,
                force=["CBAM", "CSRD"],
            )
        )
        assert result.applicable_regulations == ["CBAM", "CSRD"]
        assert all(
            r.startswith("forced") for r in result.applicability_rationale.values()
        )


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------


class TestResultShape:
    def test_returns_compliance_run_result(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(_request(tmp_path=tmp_path))
        assert isinstance(result, ComplianceRunResult)
        assert result.case_id.startswith("comply-")
        assert result.entity_id == "acme-001"
        assert all(isinstance(fr, FrameworkResult) for fr in result.framework_results)

    def test_framework_result_has_totals_when_scope_engine_matches(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(
            _request(tmp_path=tmp_path, force=["CBAM", "CSRD", "GHG-Protocol"])
        )
        cbam = next(fr for fr in result.framework_results if fr.regulation == "CBAM")
        assert cbam.total_co2e_kg == 175.0
        assert cbam.scope_1_co2e_kg == 100.0
        assert cbam.computation_hash is not None

    def test_sb253_has_no_scope_engine_view(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(
            _request(tmp_path=tmp_path, force=["SB-253"])
        )
        sb = next(fr for fr in result.framework_results if fr.regulation == "SB-253")
        assert sb.framework_id is None
        assert sb.total_co2e_kg is None


# --------------------------------------------------------------------------
# Evidence Vault / Ledger side effects
# --------------------------------------------------------------------------


class TestEvidenceAndLedger:
    def test_evidence_records_written_for_each_framework(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(
            _request(tmp_path=tmp_path, force=["CBAM", "CSRD"])
        )

        # Reopen the vault and check case records persisted.
        from greenlang.evidence_vault import EvidenceVault

        vault = EvidenceVault(
            "test-vault",
            storage="sqlite",
            sqlite_path=str(tmp_path / "vault.sqlite"),
        )
        try:
            records = vault.list_evidence(case_id=result.case_id)
        finally:
            vault.close()

        types = {r["evidence_type"] for r in records}
        assert "applicability_verdict" in types
        assert "scope_computation" in types
        assert len(records) >= 1 + len(
            [fr for fr in result.framework_results if fr.framework_id]
        )

    def test_ledger_chain_head_written(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        result = orch.run(
            _request(tmp_path=tmp_path, force=["CBAM"])
        )
        assert result.ledger_global_chain_head is not None

        from greenlang.climate_ledger import ClimateLedger

        ledger = ClimateLedger(
            agent_name="comply-orchestrator",
            storage_backend="sqlite",
            sqlite_path=str(tmp_path / "ledger.sqlite"),
        )
        try:
            entries = ledger.sqlite_backend.read_entity(result.case_id)
        finally:
            ledger.close()
        ops = [e["operation"] for e in entries]
        assert "applicability" in ops
        assert any(op.startswith("compute:") for op in ops)

    def test_bundle_export_when_requested(self, tmp_path: Path):
        orch = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        bundle_path = tmp_path / "evidence.zip"
        result = orch.run(
            _request(tmp_path=tmp_path, force=["CBAM"]),
            bundle_output=str(bundle_path),
        )
        assert result.evidence_bundle_path == str(bundle_path)
        assert bundle_path.exists()
        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            assert "manifest.json" in names
            assert "signature.json" in names
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["case_id"] == result.case_id


# --------------------------------------------------------------------------
# Deterministic behaviour
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_give_same_regulations(self, tmp_path: Path):
        orch1 = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        orch2 = ComplyOrchestrator(scope_engine=_StubScopeEngine())
        r1 = orch1.run(_request(tmp_path=tmp_path / "a", force=["CBAM", "CSRD"]))
        r2 = orch2.run(_request(tmp_path=tmp_path / "b", force=["CBAM", "CSRD"]))
        assert r1.applicable_regulations == r2.applicable_regulations
