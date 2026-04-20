# -*- coding: utf-8 -*-
"""ComplyOrchestrator: chain Policy Graph → Scope Engine → Evidence Vault → Climate Ledger.

This is the Phase 3.1 product layer.  It is deliberately thin: every capability
it exposes is delegated to an existing substrate module, not reimplemented.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from greenlang.climate_ledger import ClimateLedger
from greenlang.comply.models import (
    ComplianceRunRequest,
    ComplianceRunResult,
    FrameworkResult,
)
from greenlang.evidence_vault import EvidenceVault
from greenlang.policy_graph import PolicyGraph
from greenlang.scope_engine import (
    ActivityRecord,
    ComputationRequest,
    ScopeEngineService,
)
from greenlang.scope_engine.models import (
    ConsolidationApproach,
    Framework,
    GWPBasis,
)

logger = logging.getLogger(__name__)


# Map Policy-Graph regulation names to Scope-Engine Framework enum values.
# Regulations with no direct Scope-Engine adapter (e.g. GHG-Protocol alone,
# TCFD) map to None — the orchestrator still records them in the result so
# callers see they applied, but skips the scope computation step.
_POLICY_TO_FRAMEWORK: dict[str, Optional[Framework]] = {
    "CBAM": Framework.CBAM,
    "CSRD": Framework.CSRD_E1,
    "SB-253": None,             # Uses GHG Protocol methodology; falls through to GHG.
    "TCFD": None,               # Qualitative scenario framework; no scope calc.
    "SBTi": Framework.SBTI,
    "ISO-14064": Framework.ISO_14064,
    "GHG-Protocol": Framework.GHG_PROTOCOL,
}


class ComplyOrchestrator:
    """Bundle Policy Graph + Scope Engine + Evidence Vault + Climate Ledger."""

    def __init__(
        self,
        policy_graph: Optional[PolicyGraph] = None,
        scope_engine: Optional[ScopeEngineService] = None,
    ) -> None:
        self.policy_graph = policy_graph or PolicyGraph()
        self.scope_engine = scope_engine or ScopeEngineService()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        request: ComplianceRunRequest,
        *,
        bundle_output: Optional[str] = None,
    ) -> ComplianceRunResult:
        """Execute a Comply run end-to-end.

        Args:
            request: The compliance run request.
            bundle_output: When provided, write a signed Evidence Vault ZIP to
                this path at the end of the run.
        """
        case_id = request.case_id or f"comply-{uuid.uuid4().hex[:12]}"

        # ---- Step 1: determine applicability ----
        if request.force_frameworks:
            applicable_names = list(request.force_frameworks)
            rationale = {n: "forced via request.force_frameworks" for n in applicable_names}
            deadlines: dict[str, Optional[str]] = {n: None for n in applicable_names}
            jurisdictions: dict[str, str] = {n: request.jurisdiction for n in applicable_names}
            required_factor_classes: dict[str, list[str]] = {n: [] for n in applicable_names}
        else:
            verdict = self.policy_graph.applies_to(
                entity=self._entity_to_dict(request),
                activity={"category": "comply_run"},
                jurisdiction=request.jurisdiction,
                date=request.reporting_period_end.date(),
            )
            applicable_names = [reg.name for reg in verdict.applicable_regulations]
            rationale = {
                reg.name: reg.rationale for reg in verdict.applicable_regulations
            }
            deadlines = {
                reg.name: reg.deadline for reg in verdict.applicable_regulations
            }
            jurisdictions = {
                reg.name: reg.jurisdiction for reg in verdict.applicable_regulations
            }
            required_factor_classes = {
                reg.name: list(reg.required_factor_classes)
                for reg in verdict.applicable_regulations
            }

        logger.info("Comply run %s: applicable regulations %s", case_id, applicable_names)

        # ---- Step 2: build Scope-Engine request (once for all frameworks) ----
        frameworks = [
            _POLICY_TO_FRAMEWORK.get(name)
            for name in applicable_names
        ]
        scope_frameworks = sorted(
            {fw for fw in frameworks if fw is not None},
            key=lambda f: f.value,
        )

        scope_response = None
        if scope_frameworks and request.activities:
            activities = [self._to_activity_record(a) for a in request.activities]
            comp_req = ComputationRequest(
                entity_id=request.entity.entity_id,
                reporting_period_start=request.reporting_period_start,
                reporting_period_end=request.reporting_period_end,
                gwp_basis=GWPBasis.AR6_100YR,
                consolidation=ConsolidationApproach.OPERATIONAL_CONTROL,
                activities=activities,
                frameworks=scope_frameworks,
            )
            scope_response = self.scope_engine.compute(comp_req)
            logger.info(
                "Comply run %s: scope computation %s total=%s",
                case_id,
                scope_response.computation.computation_id,
                scope_response.computation.total_co2e_kg,
            )

        # ---- Step 3: evidence + ledger wiring ----
        vault = EvidenceVault(
            vault_id=request.vault_id,
            storage="sqlite" if request.vault_sqlite else "memory",
            sqlite_path=request.vault_sqlite,
        )
        ledger = ClimateLedger(
            agent_name="comply-orchestrator",
            storage_backend="sqlite" if request.ledger_sqlite else "memory",
            sqlite_path=request.ledger_sqlite,
        )

        framework_results: list[FrameworkResult] = []
        evidence_count_total = 0
        global_chain_head: Optional[str] = None

        try:
            # One evidence record for the applicability verdict itself
            # (lets auditors retrieve why a regulation was deemed applicable).
            applicability_eid = vault.collect(
                evidence_type="applicability_verdict",
                source="policy_graph.applies_to",
                data={
                    "applicable_regulations": applicable_names,
                    "rationale": rationale,
                    "deadlines": deadlines,
                    "jurisdictions": jurisdictions,
                },
                metadata={"entity_id": request.entity.entity_id},
                case_id=case_id,
            )
            evidence_count_total += 1
            ledger.record_entry(
                entity_type="comply_run",
                entity_id=case_id,
                operation="applicability",
                content_hash=applicability_eid,
                metadata={"regulations": applicable_names},
            )

            for name in applicable_names:
                framework_enum = _POLICY_TO_FRAMEWORK.get(name)
                fr = FrameworkResult(
                    regulation=name,
                    framework_id=framework_enum.value if framework_enum else None,
                    jurisdiction=jurisdictions.get(name, request.jurisdiction),
                    deadline=deadlines.get(name),
                    required_factor_classes=required_factor_classes.get(name, []),
                    evidence_count=0,
                )

                if scope_response is not None and framework_enum is not None:
                    view = scope_response.framework_views.get(framework_enum)
                    comp = scope_response.computation
                    fr.total_co2e_kg = float(comp.total_co2e_kg)
                    fr.scope_1_co2e_kg = float(comp.breakdown.scope_1_co2e_kg)
                    fr.scope_2_co2e_kg = float(comp.breakdown.scope_2_location_co2e_kg)
                    fr.scope_3_co2e_kg = float(comp.breakdown.scope_3_co2e_kg)
                    fr.computation_id = comp.computation_id
                    fr.computation_hash = comp.computation_hash

                    # Persist the scope-engine emission result as evidence.
                    framework_eid = vault.collect(
                        evidence_type="scope_computation",
                        source="scope_engine.service",
                        data={
                            "regulation": name,
                            "framework_id": framework_enum.value,
                            "computation_id": comp.computation_id,
                            "total_co2e_kg": float(comp.total_co2e_kg),
                            "view": view.model_dump() if view is not None else None,
                        },
                        metadata={
                            "entity_id": request.entity.entity_id,
                            "computation_hash": comp.computation_hash,
                        },
                        case_id=case_id,
                    )
                    fr.evidence_count += 1
                    evidence_count_total += 1
                    fr.ledger_chain_hash = ledger.record_entry(
                        entity_type="comply_run",
                        entity_id=case_id,
                        operation=f"compute:{name}",
                        content_hash=comp.computation_hash,
                        metadata={"evidence_id": framework_eid},
                    )
                    global_chain_head = fr.ledger_chain_hash

                framework_results.append(fr)

            evidence_bundle_path: Optional[str] = None
            if bundle_output:
                bundle_path = vault.bundle(
                    output_path=Path(bundle_output),
                    case_id=case_id,
                )
                evidence_bundle_path = str(bundle_path)
                ledger.record_entry(
                    entity_type="comply_run",
                    entity_id=case_id,
                    operation="bundle",
                    content_hash=str(bundle_path),
                    metadata={"records": evidence_count_total},
                )

        finally:
            vault.close()
            ledger.close()

        return ComplianceRunResult(
            entity_id=request.entity.entity_id,
            reporting_period_start=request.reporting_period_start,
            reporting_period_end=request.reporting_period_end,
            jurisdiction=request.jurisdiction,
            applicable_regulations=applicable_names,
            framework_results=framework_results,
            case_id=case_id,
            evidence_bundle_path=evidence_bundle_path,
            ledger_global_chain_head=global_chain_head,
            applicability_rationale=rationale,
            evaluated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_to_dict(request: ComplianceRunRequest) -> dict[str, Any]:
        e = request.entity
        return {
            "type": e.type,
            "hq_country": e.hq_country,
            "operates_in": e.operates_in,
            "employees": e.employees,
            "turnover_m_eur": e.turnover_m_eur,
            "balance_sheet_m_eur": e.balance_sheet_m_eur,
            "revenue_usd": e.revenue_usd,
        }

    @staticmethod
    def _to_activity_record(payload: dict[str, Any]) -> ActivityRecord:
        # Defer validation to Pydantic — raises ValidationError on bad input.
        return ActivityRecord(**payload)
