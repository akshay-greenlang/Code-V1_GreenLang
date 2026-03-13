# -*- coding: utf-8 -*-
"""
Compliance Checker Engine - AGENT-EUDR-039

Verifies EUDR Article 4 compliance before customs declaration submission.
Performs comprehensive checks including DDS reference number presence,
deforestation-free verification, legality verification, geolocation
data availability, supply chain traceability completeness, risk
assessment results, and country benchmarking status.

Algorithm:
    1. Accept declaration identifier and compliance parameters
    2. Check DDS reference number presence (Article 4(2) mandatory)
    3. Verify deforestation-free status (cutoff date 31 Dec 2020)
    4. Verify legality of production (Article 3(b))
    5. Check geolocation data availability (Article 9(1)(d))
    6. Verify supply chain traceability completeness
    7. Check risk assessment results (Article 10)
    8. Verify country benchmarking status (Article 29)
    9. Aggregate results and return pass/fail determination

Zero-Hallucination Guarantees:
    - All compliance checks against codified EUDR requirements
    - No LLM involvement in compliance determination
    - Binary pass/fail based on deterministic rule evaluation
    - Complete provenance trail for every check

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-039 Customs Declaration Support (GL-EUDR-CDS-039)
Regulation: EU 2023/1115 (EUDR) Articles 3, 4, 5, 9, 10, 29
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import CustomsDeclarationSupportConfig, get_config
from .models import (
    AGENT_ID,
    ComplianceCheck,
    ComplianceCheckType,
    ComplianceStatus,
    EUDR_COMMODITY_CN_CODES,
    ValidationResult,
    VerificationStatus,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compliance check definitions
# ---------------------------------------------------------------------------

_COMPLIANCE_CHECK_DEFINITIONS: Dict[ComplianceCheckType, Dict[str, str]] = {
    ComplianceCheckType.DDS_REFERENCE: {
        "article": "Article 4(2)",
        "description": (
            "Due Diligence Statement reference number must be provided "
            "for all EUDR-regulated commodities entering the EU market"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.DEFORESTATION_FREE: {
        "article": "Article 3(a)",
        "description": (
            "Products must be deforestation-free, meaning produced on "
            "land not subject to deforestation after 31 December 2020"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.LEGALITY: {
        "article": "Article 3(b)",
        "description": (
            "Products must be produced in accordance with the relevant "
            "legislation of the country of production"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.GEOLOCATION: {
        "article": "Article 9(1)(d)",
        "description": (
            "Geolocation coordinates of all plots of land where the "
            "relevant commodities were produced must be available"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.SUPPLY_CHAIN: {
        "article": "Article 9(1)(a-c)",
        "description": (
            "Complete supply chain traceability from production to "
            "import including supplier identification and quantities"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.RISK_ASSESSMENT: {
        "article": "Article 10",
        "description": (
            "Risk assessment must be completed and the risk level "
            "must be non-negligible or adequately mitigated"
        ),
        "severity": "mandatory",
    },
    ComplianceCheckType.COUNTRY_BENCHMARKING: {
        "article": "Article 29",
        "description": (
            "Country of production risk benchmarking status must be "
            "assessed (low/standard/high) with appropriate scrutiny"
        ),
        "severity": "advisory",
    },
}


class ComplianceChecker:
    """EUDR compliance verification engine for customs declarations.

    Performs all required EUDR compliance checks before a customs
    declaration can be submitted to authorities.

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.
        _checks: In-memory compliance check store.
    """

    def __init__(
        self, config: Optional[CustomsDeclarationSupportConfig] = None,
    ) -> None:
        """Initialize Compliance Checker.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._checks: Dict[str, List[ComplianceCheck]] = {}
        self._checks_count: int = 0
        logger.info("ComplianceChecker initialized")

    async def run_compliance_checks(
        self,
        declaration_id: str,
        compliance_data: Dict[str, Any],
    ) -> List[ComplianceCheck]:
        """Run all EUDR compliance checks for a declaration.

        Args:
            declaration_id: Declaration identifier.
            compliance_data: Data required for compliance verification.

        Returns:
            List of ComplianceCheck results.
        """
        start = time.monotonic()
        logger.info(
            "Running compliance checks for declaration '%s'",
            declaration_id,
        )

        checks: List[ComplianceCheck] = []

        # 1. DDS Reference Check (Article 4(2))
        dds_check = await self._check_dds_reference(
            declaration_id, compliance_data,
        )
        checks.append(dds_check)

        # 2. Deforestation-Free Check (Article 3(a))
        deforestation_check = await self._check_deforestation_free(
            declaration_id, compliance_data,
        )
        checks.append(deforestation_check)

        # 3. Legality Check (Article 3(b))
        legality_check = await self._check_legality(
            declaration_id, compliance_data,
        )
        checks.append(legality_check)

        # 4. Geolocation Check (Article 9(1)(d))
        geolocation_check = await self._check_geolocation(
            declaration_id, compliance_data,
        )
        checks.append(geolocation_check)

        # 5. Supply Chain Check (Article 9(1)(a-c))
        supply_chain_check = await self._check_supply_chain(
            declaration_id, compliance_data,
        )
        checks.append(supply_chain_check)

        # 6. Risk Assessment Check (Article 10)
        risk_check = await self._check_risk_assessment(
            declaration_id, compliance_data,
        )
        checks.append(risk_check)

        # 7. Country Benchmarking Check (Article 29)
        country_check = await self._check_country_benchmarking(
            declaration_id, compliance_data,
        )
        checks.append(country_check)

        # Store checks
        self._checks[declaration_id] = checks

        # Provenance chain entry
        passed_count = sum(
            1 for c in checks if c.status == VerificationStatus.PASSED
        )
        total = len(checks)

        self._provenance.record(
            entity_type="compliance_check",
            action="run_all",
            entity_id=declaration_id,
            actor=AGENT_ID,
            metadata={
                "total_checks": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "overall_pass": passed_count == total,
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
            },
        )

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Compliance checks for '%s': %d/%d passed (%.1f ms)",
            declaration_id, passed_count, total, elapsed,
        )
        return checks

    async def check_overall_compliance(
        self,
        declaration_id: str,
    ) -> Dict[str, Any]:
        """Get overall compliance status for a declaration.

        Args:
            declaration_id: Declaration identifier.

        Returns:
            Dictionary with overall compliance summary.
        """
        checks = self._checks.get(declaration_id, [])

        mandatory_checks = [
            c for c in checks
            if _COMPLIANCE_CHECK_DEFINITIONS.get(
                c.check_type, {}
            ).get("severity") == "mandatory"
        ]

        mandatory_passed = all(
            c.status == VerificationStatus.PASSED
            for c in mandatory_checks
        )

        all_passed = all(
            c.status == VerificationStatus.PASSED
            for c in checks
        )

        return {
            "declaration_id": declaration_id,
            "overall_status": "passed" if mandatory_passed else "failed",
            "mandatory_checks_passed": mandatory_passed,
            "all_checks_passed": all_passed,
            "total_checks": len(checks),
            "passed_count": sum(
                1 for c in checks
                if c.status == VerificationStatus.PASSED
            ),
            "failed_count": sum(
                1 for c in checks
                if c.status == VerificationStatus.FAILED
            ),
            "warning_count": sum(
                1 for c in checks
                if c.status == VerificationStatus.WARNING
            ),
            "checks": [
                {
                    "check_type": c.check_type.value,
                    "status": c.status.value,
                    "article": c.article_reference,
                    "details": c.details,
                }
                for c in checks
            ],
        }

    async def get_checks_for_declaration(
        self, declaration_id: str,
    ) -> List[ComplianceCheck]:
        """Get compliance checks for a declaration.

        Args:
            declaration_id: Declaration identifier.

        Returns:
            List of ComplianceCheck results.
        """
        return self._checks.get(declaration_id, [])

    # ------------------------------------------------------------------
    # Public Check Methods (keyword-based interface for tests)
    # ------------------------------------------------------------------

    async def check_dds_reference(
        self,
        declaration_id: str,
        dds_reference: str,
    ) -> ComplianceCheck:
        """Check DDS reference validity.

        Args:
            declaration_id: Declaration identifier.
            dds_reference: DDS reference number string.

        Returns:
            ComplianceCheck with result = ValidationResult.PASS/FAIL.
        """
        check_id = f"CHK-{uuid.uuid4().hex[:12].upper()}"

        # DDS reference must be non-empty and match expected format
        # Expected: GL-DDS-YYYYMMDD-XXXXXX
        import re
        dds_pattern = re.compile(r"^GL-DDS-\d{8}-[A-Z0-9]{6,}$")

        if not dds_reference:
            result = ValidationResult.FAIL
            message = "No DDS reference number provided"
        elif not dds_pattern.match(dds_reference):
            result = ValidationResult.FAIL
            message = f"Invalid DDS reference format: '{dds_reference}'"
        else:
            result = ValidationResult.PASS
            message = f"DDS reference '{dds_reference}' is valid"

        self._checks_count += 1
        return ComplianceCheck(
            check_id=check_id,
            declaration_id=declaration_id,
            check_type="dds_reference_validation",
            result=result,
            message=message,
            article_reference="Article 4(2)",
            dds_reference=dds_reference,
        )

    async def check_cn_code_coverage(
        self,
        declaration_id: str,
        cn_codes: List[str],
    ) -> ComplianceCheck:
        """Check CN codes for EUDR coverage.

        Args:
            declaration_id: Declaration identifier.
            cn_codes: List of 8-digit CN codes.

        Returns:
            ComplianceCheck with EUDR coverage result.
        """
        check_id = f"CHK-{uuid.uuid4().hex[:12].upper()}"

        if not cn_codes:
            result = ValidationResult.FAIL
            message = "No CN codes provided for EUDR coverage check"
        else:
            # Check each CN code against EUDR commodity mappings
            eudr_codes = []
            non_eudr_codes = []
            for cn in cn_codes:
                found_eudr = False
                for commodity, codes in EUDR_COMMODITY_CN_CODES.items():
                    if cn in codes:
                        found_eudr = True
                        break
                if found_eudr:
                    eudr_codes.append(cn)
                else:
                    non_eudr_codes.append(cn)

            if eudr_codes and not non_eudr_codes:
                result = ValidationResult.PASS
                message = f"All {len(eudr_codes)} CN code(s) are EUDR-regulated"
            elif eudr_codes and non_eudr_codes:
                result = ValidationResult.WARNING
                message = (
                    f"{len(eudr_codes)} EUDR-regulated and "
                    f"{len(non_eudr_codes)} non-EUDR CN codes found"
                )
            else:
                result = ValidationResult.NOT_APPLICABLE
                message = "No EUDR-regulated CN codes found in declaration"

        self._checks_count += 1
        return ComplianceCheck(
            check_id=check_id,
            declaration_id=declaration_id,
            check_type="cn_code_eudr_coverage",
            result=result,
            message=message,
            article_reference="Annex I",
        )

    async def check_origin(
        self,
        declaration_id: str,
        declared_origin: str,
        supply_chain_origins: List[str],
    ) -> ComplianceCheck:
        """Check origin consistency.

        Args:
            declaration_id: Declaration identifier.
            declared_origin: Declared country code.
            supply_chain_origins: List of origin country codes from supply chain.

        Returns:
            ComplianceCheck with origin verification result.
        """
        check_id = f"CHK-{uuid.uuid4().hex[:12].upper()}"

        declared = declared_origin.upper()

        if not supply_chain_origins:
            result = ValidationResult.WARNING
            message = "No supply chain origin data available for cross-reference"
        elif declared in [s.upper() for s in supply_chain_origins]:
            result = ValidationResult.PASS
            message = f"Declared origin '{declared}' matches supply chain data"
        else:
            result = ValidationResult.FAIL
            message = (
                f"Declared origin '{declared}' not found in "
                f"supply chain origins: {supply_chain_origins}"
            )

        self._checks_count += 1
        return ComplianceCheck(
            check_id=check_id,
            declaration_id=declaration_id,
            check_type="origin_verification",
            result=result,
            message=message,
            article_reference="Article 9(1)(d)",
        )

    async def check_deforestation_free(
        self,
        declaration_id: str,
        deforestation_free: bool,
        dds_reference: str,
    ) -> ComplianceCheck:
        """Check deforestation-free declaration.

        Args:
            declaration_id: Declaration identifier.
            deforestation_free: Whether commodity is deforestation-free.
            dds_reference: DDS reference backing the claim.

        Returns:
            ComplianceCheck with deforestation-free result.
        """
        check_id = f"CHK-{uuid.uuid4().hex[:12].upper()}"

        if not dds_reference:
            result = ValidationResult.WARNING
            message = "No DDS reference to back deforestation-free claim"
        elif deforestation_free:
            result = ValidationResult.PASS
            message = "Commodity confirmed deforestation-free with DDS reference"
        else:
            result = ValidationResult.FAIL
            message = "Commodity NOT confirmed deforestation-free"

        self._checks_count += 1
        return ComplianceCheck(
            check_id=check_id,
            declaration_id=declaration_id,
            check_type="deforestation_free_declaration",
            result=result,
            message=message,
            article_reference="Article 3(a)",
            dds_reference=dds_reference,
        )

    async def check_risk_level(
        self,
        declaration_id: str,
        risk_level: str,
        country_code: str,
    ) -> ComplianceCheck:
        """Check risk level assessment.

        Args:
            declaration_id: Declaration identifier.
            risk_level: Risk level string (low/standard/high/critical).
            country_code: Country code.

        Returns:
            ComplianceCheck with risk level result.
        """
        check_id = f"CHK-{uuid.uuid4().hex[:12].upper()}"

        level = risk_level.lower()
        if level == "low":
            result = ValidationResult.PASS
            message = f"Low risk level for country '{country_code}'"
        elif level == "standard":
            result = ValidationResult.PASS
            message = f"Standard risk level for country '{country_code}'"
        elif level == "high":
            result = ValidationResult.WARNING
            message = f"High risk level for country '{country_code}' - enhanced due diligence required"
        elif level == "critical":
            result = ValidationResult.WARNING
            message = f"Critical risk level for country '{country_code}' - import may be blocked"
        else:
            result = ValidationResult.WARNING
            message = f"Unknown risk level '{risk_level}' for country '{country_code}'"

        self._checks_count += 1
        return ComplianceCheck(
            check_id=check_id,
            declaration_id=declaration_id,
            check_type="risk_level_check",
            result=result,
            message=message,
            article_reference="Article 29",
        )

    async def run_full_compliance_check(
        self,
        declaration_id: str,
        dds_reference: str = "",
        cn_codes: Optional[List[str]] = None,
        declared_origin: str = "",
        supply_chain_origins: Optional[List[str]] = None,
        deforestation_free: bool = False,
        risk_level: str = "standard",
        country_code: str = "",
    ) -> Dict[str, Any]:
        """Run all compliance checks and return a report.

        Args:
            declaration_id: Declaration identifier.
            dds_reference: DDS reference number.
            cn_codes: List of CN codes.
            declared_origin: Declared country of origin.
            supply_chain_origins: Supply chain origin countries.
            deforestation_free: Whether commodity is deforestation-free.
            risk_level: Risk level.
            country_code: Country code.

        Returns:
            Dict with checks, overall_status, provenance_hash.
        """
        checks: List[ComplianceCheck] = []

        # 1. DDS Reference Check
        dds_check = await self.check_dds_reference(declaration_id, dds_reference)
        checks.append(dds_check)

        # 2. CN Code EUDR Coverage
        cn_check = await self.check_cn_code_coverage(
            declaration_id, cn_codes or [],
        )
        checks.append(cn_check)

        # 3. Origin Check
        origin_check = await self.check_origin(
            declaration_id, declared_origin, supply_chain_origins or [],
        )
        checks.append(origin_check)

        # 4. Deforestation-Free Check
        deforestation_check = await self.check_deforestation_free(
            declaration_id, deforestation_free, dds_reference,
        )
        checks.append(deforestation_check)

        # 5. Risk Level Check
        risk_check = await self.check_risk_level(
            declaration_id, risk_level, country_code,
        )
        checks.append(risk_check)

        # Determine overall status
        all_pass = all(c.result == ValidationResult.PASS for c in checks)
        any_fail = any(c.result == ValidationResult.FAIL for c in checks)

        if all_pass:
            overall = ComplianceStatus.COMPLIANT.value
        elif any_fail:
            overall = ComplianceStatus.NON_COMPLIANT.value
        else:
            overall = ComplianceStatus.PARTIALLY_COMPLIANT.value

        # Compute provenance hash
        prov_data = {
            "declaration_id": declaration_id,
            "dds_reference": dds_reference,
            "cn_codes": sorted(cn_codes or []),
            "declared_origin": declared_origin,
            "overall_status": overall,
        }
        provenance_hash = self._provenance.compute_hash(prov_data)

        # Store for later retrieval
        self._checks[declaration_id] = checks

        return {
            "declaration_id": declaration_id,
            "checks": checks,
            "overall_status": overall,
            "provenance_hash": provenance_hash,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the Compliance Checker engine."""
        return {
            "engine": "ComplianceChecker",
            "status": "healthy",
            "checks_performed": self._checks_count,
        }

    # ------------------------------------------------------------------
    # Individual Check Methods (Legacy private interface)
    # ------------------------------------------------------------------

    async def _check_dds_reference(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check DDS reference number presence (Article 4(2))."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.DDS_REFERENCE]
        dds_refs = data.get("dds_reference_numbers", [])
        dds_ref = data.get("dds_reference_number", "")

        has_ref = bool(dds_refs) or bool(dds_ref)
        status = VerificationStatus.PASSED if has_ref else VerificationStatus.FAILED
        details = (
            f"DDS reference(s) found: {dds_refs or [dds_ref]}"
            if has_ref
            else "No DDS reference number provided. Required per Article 4(2)."
        )

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.DDS_REFERENCE,
            status=status,
            dds_reference_number=dds_ref or (dds_refs[0] if dds_refs else ""),
            article_reference=check_def["article"],
            details=details,
            evidence=dds_refs if dds_refs else ([dds_ref] if dds_ref else []),
        )

    async def _check_deforestation_free(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check deforestation-free status (Article 3(a))."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.DEFORESTATION_FREE]
        is_deforestation_free = data.get("deforestation_free", None)

        if is_deforestation_free is True:
            status = VerificationStatus.PASSED
            details = "Commodity confirmed deforestation-free (post 31 Dec 2020)."
        elif is_deforestation_free is False:
            status = VerificationStatus.FAILED
            details = (
                "Commodity NOT confirmed deforestation-free. "
                "Cannot be placed on EU market per Article 3(a)."
            )
        else:
            status = VerificationStatus.WARNING
            details = (
                "Deforestation-free status not provided. "
                "Verification required before submission."
            )

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.DEFORESTATION_FREE,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )

    async def _check_legality(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check legality of production (Article 3(b))."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.LEGALITY]
        is_legal = data.get("legality_verified", None)

        if is_legal is True:
            status = VerificationStatus.PASSED
            details = "Production legality verified per country of origin laws."
        elif is_legal is False:
            status = VerificationStatus.FAILED
            details = (
                "Production legality NOT verified. "
                "Legal compliance required per Article 3(b)."
            )
        else:
            status = VerificationStatus.WARNING
            details = "Legality verification status not provided."

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.LEGALITY,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )

    async def _check_geolocation(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check geolocation data availability (Article 9(1)(d))."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.GEOLOCATION]
        has_geolocation = data.get("geolocation_available", None)
        geo_coords = data.get("geolocation_coordinates", [])

        if has_geolocation is True or geo_coords:
            status = VerificationStatus.PASSED
            details = (
                f"Geolocation data available: "
                f"{len(geo_coords)} coordinate set(s) provided."
            )
        elif has_geolocation is False:
            status = VerificationStatus.FAILED
            details = (
                "Geolocation data NOT available. "
                "Required per Article 9(1)(d)."
            )
        else:
            status = VerificationStatus.WARNING
            details = "Geolocation data availability not confirmed."

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.GEOLOCATION,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )

    async def _check_supply_chain(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check supply chain traceability (Article 9(1)(a-c))."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.SUPPLY_CHAIN]
        sc_complete = data.get("supply_chain_complete", None)
        supplier_count = data.get("supplier_count", 0)

        if sc_complete is True:
            status = VerificationStatus.PASSED
            details = (
                f"Supply chain traceability complete: "
                f"{supplier_count} supplier(s) mapped."
            )
        elif sc_complete is False:
            status = VerificationStatus.FAILED
            details = (
                "Supply chain traceability incomplete. "
                "Full chain required per Article 9(1)(a-c)."
            )
        else:
            status = VerificationStatus.WARNING
            details = "Supply chain completeness not confirmed."

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.SUPPLY_CHAIN,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )

    async def _check_risk_assessment(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check risk assessment completion (Article 10)."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.RISK_ASSESSMENT]
        risk_assessed = data.get("risk_assessment_complete", None)
        risk_level = data.get("risk_level", "unknown")
        risk_mitigated = data.get("risk_mitigated", False)

        if risk_assessed is True:
            if risk_level in ("low", "negligible") or risk_mitigated:
                status = VerificationStatus.PASSED
                details = (
                    f"Risk assessment complete. Risk level: {risk_level}. "
                    f"Mitigated: {risk_mitigated}."
                )
            else:
                status = VerificationStatus.WARNING
                details = (
                    f"Risk assessment complete but risk level '{risk_level}' "
                    f"may require additional mitigation."
                )
        elif risk_assessed is False:
            status = VerificationStatus.FAILED
            details = "Risk assessment NOT completed. Required per Article 10."
        else:
            status = VerificationStatus.WARNING
            details = "Risk assessment status not provided."

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.RISK_ASSESSMENT,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )

    async def _check_country_benchmarking(
        self,
        declaration_id: str,
        data: Dict[str, Any],
    ) -> ComplianceCheck:
        """Check country benchmarking status (Article 29)."""
        check_def = _COMPLIANCE_CHECK_DEFINITIONS[ComplianceCheckType.COUNTRY_BENCHMARKING]
        country_risk = data.get("country_risk_level", "unknown")
        country = data.get("country_of_origin", "")

        if country_risk in ("low", "standard", "high"):
            status = VerificationStatus.PASSED
            details = (
                f"Country benchmarking assessed for '{country}': "
                f"risk level = {country_risk}."
            )
        else:
            status = VerificationStatus.WARNING
            details = (
                f"Country benchmarking not assessed for '{country}'. "
                f"Recommended per Article 29."
            )

        return ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:12].upper()}",
            declaration_id=declaration_id,
            check_type=ComplianceCheckType.COUNTRY_BENCHMARKING,
            status=status,
            article_reference=check_def["article"],
            details=details,
        )
