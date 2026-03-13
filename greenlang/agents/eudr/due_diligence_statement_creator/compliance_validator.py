# -*- coding: utf-8 -*-
"""
Compliance Validator Engine - AGENT-EUDR-037

Engine 5 of 7: Verifies all Article 4 mandatory fields are present and
valid in a DDS before submission. Checks geolocation completeness per
Article 9, risk assessment inclusion per Article 10, supply chain data
availability, compliance declaration text, and deforestation-free /
legally-produced declarations.

Algorithm:
    1. Check all 14 Article 4 mandatory fields for presence
    2. Validate geolocation data completeness per Article 9
    3. Verify risk assessment references are included
    4. Check supply chain traceability data completeness
    5. Validate compliance declaration text presence
    6. Check deforestation-free and legally-produced declarations
    7. Aggregate results into DDSValidationReport
    8. Compute provenance hash for audit trail

Zero-Hallucination Guarantees:
    - All validation via deterministic field presence checks
    - No LLM involvement in compliance determination
    - Pass/fail criteria codified from EUDR Article 4 text
    - Complete provenance trail for every validation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import DDSCreatorConfig, get_config
from .models import (
    ARTICLE_4_MANDATORY_FIELDS,
    ComplianceCheck,
    DDSStatement,
    DDSValidationReport,
    ValidationResult,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class ComplianceValidator:
    """Article 4 compliance validation engine.

    Verifies that a DDS contains all mandatory fields and data
    required by EUDR Article 4, Article 9 (geolocation), and
    Article 10 (risk assessment).

    Attributes:
        config: Agent configuration.
        _provenance: SHA-256 provenance tracker.

    Example:
        >>> validator = ComplianceValidator()
        >>> report = await validator.validate_statement(statement)
        >>> assert report.overall_result in (
        ...     ValidationResult.PASS, ValidationResult.FAIL
        ... )
    """

    def __init__(
        self,
        config: Optional[DDSCreatorConfig] = None,
    ) -> None:
        """Initialize the compliance validator engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._validation_count = 0
        logger.info("ComplianceValidator engine initialized")

    async def validate_statement(
        self,
        statement: DDSStatement,
    ) -> DDSValidationReport:
        """Validate a DDS against Article 4 requirements.

        Performs comprehensive validation of all mandatory fields,
        geolocation data, risk assessment inclusion, and supply chain
        completeness.

        Args:
            statement: DDSStatement to validate.

        Returns:
            DDSValidationReport with all check results.
        """
        start = time.monotonic()
        checks: List[ComplianceCheck] = []

        # 1. Check all Article 4 mandatory fields
        mandatory_complete = True
        for field_name in ARTICLE_4_MANDATORY_FIELDS:
            check_id = f"CHK-{uuid.uuid4().hex[:8].upper()}"
            val = self._get_field_value(statement, field_name)

            if val:
                checks.append(ComplianceCheck(
                    check_id=check_id,
                    field_name=field_name,
                    article_reference="Art. 4(2)",
                    result=ValidationResult.PASS,
                    message=f"Field '{field_name}' is present",
                    severity="info",
                ))
            else:
                mandatory_complete = False
                checks.append(ComplianceCheck(
                    check_id=check_id,
                    field_name=field_name,
                    article_reference="Art. 4(2)",
                    result=ValidationResult.FAIL,
                    message=f"Mandatory field '{field_name}' is missing",
                    severity="error",
                    suggested_fix=f"Provide value for '{field_name}'",
                ))

        # 2. Geolocation validation per Article 9
        geo_valid = len(statement.geolocations) > 0
        geo_check = ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:8].upper()}",
            field_name="geolocation_data",
            article_reference="Art. 9(1)(d)",
            result=ValidationResult.PASS if geo_valid else ValidationResult.FAIL,
            message=(
                f"Geolocation data present ({len(statement.geolocations)} plots)"
                if geo_valid
                else "No geolocation data attached"
            ),
            severity="info" if geo_valid else "error",
            suggested_fix="" if geo_valid else "Attach geolocation data for all production plots",
        )
        checks.append(geo_check)

        # 3. Risk assessment inclusion per Article 10
        risk_included = len(statement.risk_references) > 0
        risk_check = ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:8].upper()}",
            field_name="risk_assessment",
            article_reference="Art. 10",
            result=ValidationResult.PASS if risk_included else ValidationResult.WARNING,
            message=(
                f"Risk assessment included ({len(statement.risk_references)} references)"
                if risk_included
                else "No risk assessment data included"
            ),
            severity="info" if risk_included else "warning",
            suggested_fix="" if risk_included else "Include risk assessment from EUDR-016 to 025",
        )
        checks.append(risk_check)

        # 4. Supply chain completeness
        sc_complete = statement.supply_chain_data is not None
        sc_check = ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:8].upper()}",
            field_name="supply_chain",
            article_reference="Art. 9(1)(e)",
            result=ValidationResult.PASS if sc_complete else ValidationResult.WARNING,
            message=(
                "Supply chain data complete"
                if sc_complete
                else "No supply chain data attached"
            ),
            severity="info" if sc_complete else "warning",
            suggested_fix="" if sc_complete else "Compile supply chain data from EUDR-001 to 015",
        )
        checks.append(sc_check)

        # 5. Deforestation-free declaration
        deforestation_check = ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:8].upper()}",
            field_name="deforestation_free",
            article_reference="Art. 3(a)",
            result=ValidationResult.PASS if statement.deforestation_free else ValidationResult.WARNING,
            message=(
                "Deforestation-free declaration provided"
                if statement.deforestation_free
                else "Deforestation-free declaration not yet confirmed"
            ),
            severity="info" if statement.deforestation_free else "warning",
        )
        checks.append(deforestation_check)

        # 6. Legally produced declaration
        legal_check = ComplianceCheck(
            check_id=f"CHK-{uuid.uuid4().hex[:8].upper()}",
            field_name="legally_produced",
            article_reference="Art. 3(b)",
            result=ValidationResult.PASS if statement.legally_produced else ValidationResult.WARNING,
            message=(
                "Legally produced declaration provided"
                if statement.legally_produced
                else "Legally produced declaration not yet confirmed"
            ),
            severity="info" if statement.legally_produced else "warning",
        )
        checks.append(legal_check)

        # Aggregate results
        passed = sum(1 for c in checks if c.result == ValidationResult.PASS)
        failed = sum(1 for c in checks if c.result == ValidationResult.FAIL)
        warnings = sum(1 for c in checks if c.result == ValidationResult.WARNING)
        overall = ValidationResult.PASS if failed == 0 else ValidationResult.FAIL

        report = DDSValidationReport(
            report_id=f"VR-{uuid.uuid4().hex[:8].upper()}",
            statement_id=statement.statement_id,
            overall_result=overall,
            total_checks=len(checks),
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings,
            checks=checks,
            mandatory_fields_complete=mandatory_complete,
            geolocation_valid=geo_valid,
            risk_assessment_included=risk_included,
            supply_chain_complete=sc_complete,
            provenance_hash=self._provenance.compute_hash({
                "statement_id": statement.statement_id,
                "overall_result": overall.value,
                "total_checks": len(checks),
                "passed": passed,
                "failed": failed,
            }),
        )

        self._validation_count += 1
        elapsed = time.monotonic() - start
        logger.info(
            "DDS %s validation: %s (%d/%d passed, %d failed, "
            "%d warnings) in %.1fms",
            statement.statement_id, overall.value,
            passed, len(checks), failed, warnings, elapsed * 1000,
        )

        return report

    def _get_field_value(
        self,
        stmt: DDSStatement,
        field_name: str,
    ) -> Any:
        """Get the value of a mandatory field from a DDS.

        Args:
            stmt: DDSStatement to inspect.
            field_name: Field name to retrieve.

        Returns:
            Field value or None if empty/missing.
        """
        mapping: Dict[str, Any] = {
            "operator_name": stmt.operator_name,
            "operator_address": stmt.operator_address,
            "operator_eori_number": stmt.operator_eori_number,
            "commodity_type": stmt.commodities,
            "product_description": stmt.product_descriptions,
            "hs_code": stmt.hs_codes,
            "country_of_production": stmt.countries_of_production,
            "geolocation_of_plots": stmt.geolocations,
            "quantity": stmt.total_quantity if stmt.total_quantity > 0 else None,
            "supplier_information": stmt.supply_chain_data,
            "compliance_declaration": stmt.compliance_declaration,
            "risk_assessment_outcome": stmt.risk_references,
            "risk_mitigation_measures": stmt.risk_mitigation_measures,
            "date_of_statement": stmt.date_of_statement,
        }
        val = mapping.get(field_name)
        if isinstance(val, list):
            return val if len(val) > 0 else None
        return val

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Health check dictionary.
        """
        return {
            "engine": "ComplianceValidator",
            "status": "healthy",
            "validations_completed": self._validation_count,
        }
