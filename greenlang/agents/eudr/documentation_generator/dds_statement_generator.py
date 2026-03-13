# -*- coding: utf-8 -*-
"""
DDS Statement Generator Engine - AGENT-EUDR-030

Core engine that generates Due Diligence Statements per EUDR Article 12.
Produces DDS documents containing all mandatory information for submission
to the EU Information System, assembling operator data, product listings,
Article 9 packages, risk assessment summaries, mitigation summaries, and
compliance conclusions into a unified regulatory document.

Algorithm:
    1. Validate all mandatory inputs (operator, commodity, products)
    2. Build operator information section
    3. Build product listing section with HS/CN codes
    4. Incorporate Article 9 data (geolocation, suppliers)
    5. Incorporate risk assessment summary
    6. Incorporate mitigation measures summary (if applicable)
    7. Generate compliance conclusion
    8. Assign unique reference number
    9. Compute SHA-256 provenance hash
    10. Return DDSDocument

Zero-Hallucination Guarantees:
    - All numeric calculations use Decimal arithmetic
    - No LLM calls in the calculation path
    - Compliance conclusion via deterministic decision tree
    - Complete provenance trail for every DDS generation
    - Validation against mandatory field checklist

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Articles 4, 5, 9, 10, 11, 12, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import DocumentationGeneratorConfig, get_config
from .models import (
    AGENT_ID,
    AGENT_VERSION,
    Article9Element,
    Article9Package,
    DDSContent,
    DDSDocument,
    DDSStatus,
    DocumentType,
    EUDRCommodity,
    MitigationDoc,
    ProductEntry,
    RiskAssessmentDoc,
    RiskLevel,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compliance conclusion decision tree
# ---------------------------------------------------------------------------

_COMPLIANCE_CONCLUSIONS: Dict[str, str] = {
    "compliant": (
        "Based on the due diligence assessment, the products covered by "
        "this statement are compliant with Regulation (EU) 2023/1115. "
        "The risk of non-compliance is negligible."
    ),
    "compliant_standard_monitoring": (
        "Based on the due diligence assessment, the products are compliant "
        "with Regulation (EU) 2023/1115 with standard monitoring in place. "
        "Risk level is within acceptable bounds."
    ),
    "compliant_after_mitigation": (
        "Based on the due diligence assessment and after implementing "
        "risk mitigation measures per Article 11, the products are "
        "compliant with Regulation (EU) 2023/1115. The risk has been "
        "reduced to an acceptable level."
    ),
    "non_compliant": (
        "Based on the due diligence assessment, the operator cannot "
        "conclude that the risk of non-compliance is negligible. "
        "The products covered by this statement are NOT compliant "
        "with Regulation (EU) 2023/1115 and must NOT be placed on "
        "the market or exported."
    ),
    "pending": (
        "Due diligence assessment is in progress. Compliance conclusion "
        "will be determined upon completion of risk assessment and any "
        "required mitigation measures."
    ),
}

# ---------------------------------------------------------------------------
# DDS mandatory field requirements per Article 4(2) and Article 12
# ---------------------------------------------------------------------------

_DDS_MANDATORY_FIELDS: List[str] = [
    "operator_id",
    "commodity",
    "products",
    "article9_data",
    "risk_assessment",
    "compliance_conclusion",
    "reference_number",
]

# ---------------------------------------------------------------------------
# HS code prefixes by commodity (used for validation)
# ---------------------------------------------------------------------------

_COMMODITY_HS_PREFIXES: Dict[EUDRCommodity, List[str]] = {
    EUDRCommodity.CATTLE: ["0102", "0201", "0202", "4101", "4104"],
    EUDRCommodity.COCOA: ["1801", "1802", "1803", "1804", "1805", "1806"],
    EUDRCommodity.COFFEE: ["0901", "2101"],
    EUDRCommodity.PALM_OIL: ["1511", "1513", "3823"],
    EUDRCommodity.RUBBER: ["4001", "4002", "4005", "4006", "4007"],
    EUDRCommodity.SOYA: ["1201", "1507", "2304"],
    EUDRCommodity.WOOD: ["4401", "4403", "4407", "4408", "4409", "4412"],
}


class DDSStatementGenerator:
    """Generates Due Diligence Statements per EUDR Article 12.

    Produces DDS documents containing all mandatory information
    for submission to the EU Information System. Assembles operator,
    product, Article 9, risk assessment, and mitigation data into
    a unified regulatory document with provenance tracking.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.
        _sequence: Internal DDS reference number sequence counter.

    Example:
        >>> generator = DDSStatementGenerator()
        >>> dds = await generator.generate_dds(
        ...     operator_id="OP-001",
        ...     commodity=EUDRCommodity.COFFEE,
        ...     products=[product],
        ...     article9_package=article9,
        ... )
        >>> assert dds.dds_id.startswith("dds-")
        >>> assert dds.reference_number.startswith("DDS-")
    """

    def __init__(
        self,
        config: Optional[DocumentationGeneratorConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize DDSStatementGenerator.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._sequence: int = 0
        logger.info(
            "DDSStatementGenerator initialized: "
            "prefix=%s, schema=%s, max_products=%d",
            self._config.dds_reference_prefix,
            self._config.dds_schema_version,
            self._config.max_products_per_dds,
        )

    async def generate_dds(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        products: List[ProductEntry],
        article9_package: Article9Package,
        risk_doc: Optional[RiskAssessmentDoc] = None,
        mitigation_doc: Optional[MitigationDoc] = None,
    ) -> DDSDocument:
        """Generate a complete DDS document.

        Algorithm:
        1. Validate all mandatory inputs
        2. Build operator information section
        3. Build product listing section
        4. Incorporate Article 9 data (geolocation, suppliers)
        5. Incorporate risk assessment summary
        6. Incorporate mitigation measures summary (if applicable)
        7. Generate compliance conclusion
        8. Assign reference number
        9. Compute provenance hash
        10. Return DDSDocument

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity category.
            products: List of product entries.
            article9_package: Assembled Article 9 information.
            risk_doc: Risk assessment documentation (optional).
            mitigation_doc: Mitigation documentation (optional).

        Returns:
            DDSDocument with all sections populated.

        Raises:
            ValueError: If mandatory inputs are missing or invalid.
        """
        start_time = time.monotonic()
        dds_id = f"dds-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Generating DDS: id=%s, operator=%s, commodity=%s, "
            "products=%d",
            dds_id, operator_id, commodity.value, len(products),
        )

        # Step 1: Validate mandatory inputs
        self._validate_mandatory_inputs(
            operator_id, commodity, products, article9_package,
        )

        # Step 2: Build operator section
        operator_section = self._build_operator_section(operator_id)

        # Step 3: Build products section
        products_section = self._build_products_section(products)

        # Step 4: Build Article 9 section
        article9_section = self._build_article9_section(article9_package)

        # Step 5: Build risk summary section
        risk_summary = self._build_risk_summary_section(risk_doc)

        # Step 6: Build mitigation summary section
        mitigation_summary = self._build_mitigation_summary_section(
            mitigation_doc,
        )

        # Step 7: Determine compliance conclusion
        conclusion_key = self._determine_compliance_conclusion(
            risk_doc, mitigation_doc,
        )
        conclusion_text = _COMPLIANCE_CONCLUSIONS.get(
            conclusion_key, _COMPLIANCE_CONCLUSIONS["pending"],
        )

        # Step 8: Assign reference number
        reference_number = self._generate_reference_number(
            operator_id, commodity.value,
        )

        # Build DDS content
        content = DDSContent(
            operator_info=operator_section,
            products=[
                ProductEntry(**p) if isinstance(p, dict) else p
                for p in products
            ],
            article9_data=article9_section,
            risk_summary=risk_summary,
            mitigation_summary=mitigation_summary,
            conclusion=conclusion_text,
        )

        # Step 9: Compute provenance hash
        provenance_data: Dict[str, Any] = {
            "dds_id": dds_id,
            "reference_number": reference_number,
            "operator_id": operator_id,
            "commodity": commodity.value,
            "product_count": len(products),
            "article9_package_id": article9_package.package_id,
            "risk_doc_id": risk_doc.doc_id if risk_doc else None,
            "mitigation_doc_id": (
                mitigation_doc.doc_id if mitigation_doc else None
            ),
            "conclusion": conclusion_key,
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        # Build DDS document
        dds = DDSDocument(
            dds_id=dds_id,
            reference_number=reference_number,
            operator_id=operator_id,
            commodity=commodity,
            products=list(products),
            article9_ref=article9_package.package_id,
            risk_assessment_ref=(
                risk_doc.doc_id if risk_doc else ""
            ),
            mitigation_ref=(
                mitigation_doc.doc_id if mitigation_doc else ""
            ),
            status=DDSStatus.DRAFT,
            compliance_conclusion=conclusion_key,
            provenance_hash=provenance_hash,
        )

        # Record provenance entry
        self._provenance.create_entry(
            step="generate_dds",
            source="dds_statement_generator",
            input_hash=self._provenance.compute_hash(
                {"operator_id": operator_id, "commodity": commodity.value}
            ),
            output_hash=provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "DDS generated: id=%s, ref=%s, conclusion=%s, "
            "products=%d, elapsed=%.1fms",
            dds_id, reference_number, conclusion_key,
            len(products), elapsed_ms,
        )

        return dds

    def _generate_reference_number(
        self, operator_id: str, commodity: str,
    ) -> str:
        """Generate unique DDS reference number.

        Format: DDS-{operator_short}-{commodity}-{YYYYMMDD}-{seq}

        Args:
            operator_id: Operator identifier.
            commodity: Commodity value string.

        Returns:
            Formatted reference number string.
        """
        self._sequence += 1
        prefix = self._config.dds_reference_prefix
        op_short = operator_id[:8].upper().replace("-", "")
        commodity_short = commodity[:4].upper()
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        seq_str = str(self._sequence).zfill(5)
        return f"{prefix}-{op_short}-{commodity_short}-{date_str}-{seq_str}"

    def _build_operator_section(
        self, operator_id: str,
    ) -> Dict[str, Any]:
        """Build operator identification section.

        Args:
            operator_id: Operator identifier.

        Returns:
            Dictionary with operator information.
        """
        return {
            "operator_id": operator_id,
            "role": "operator",
            "registration_type": "eudr_registered",
            "generated_by": AGENT_ID,
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def _build_products_section(
        self, products: List[ProductEntry],
    ) -> List[Dict[str, Any]]:
        """Build product listing with HS/CN codes.

        Args:
            products: List of product entries.

        Returns:
            List of product dictionaries for DDS.
        """
        result: List[Dict[str, Any]] = []
        for idx, product in enumerate(products):
            entry: Dict[str, Any] = {
                "index": idx + 1,
                "product_id": product.product_id,
                "description": product.description,
                "hs_code": product.hs_code,
                "cn_code": getattr(product, "cn_code", ""),
                "quantity": str(product.quantity),
                "unit": product.unit,
            }
            result.append(entry)
        return result

    def _build_article9_section(
        self, article9: Article9Package,
    ) -> Dict[str, Any]:
        """Build Article 9 information section.

        Args:
            article9: Assembled Article 9 package.

        Returns:
            Dictionary with Article 9 summary for DDS.
        """
        return {
            "package_id": article9.package_id,
            "completeness_score": str(article9.completeness_score),
            "elements_present": len(article9.elements),
            "elements": article9.elements,
            "missing_elements": article9.missing_elements,
            "commodity": article9.commodity.value,
            "assembled_at": article9.assembled_at.isoformat(),
            "article_reference": "EUDR Article 9",
        }

    def _build_risk_summary_section(
        self, risk_doc: Optional[RiskAssessmentDoc],
    ) -> Dict[str, Any]:
        """Build risk assessment summary for DDS.

        Args:
            risk_doc: Risk assessment documentation (optional).

        Returns:
            Dictionary with risk summary data for DDS.
        """
        if risk_doc is None:
            return {
                "status": "not_assessed",
                "message": (
                    "Risk assessment has not been performed. "
                    "Required per EUDR Article 10."
                ),
                "article_reference": "EUDR Article 10",
            }

        return {
            "doc_id": risk_doc.doc_id,
            "assessment_id": risk_doc.assessment_id,
            "composite_score": str(risk_doc.composite_score),
            "risk_level": risk_doc.risk_level.value,
            "simplified_dd_eligible": risk_doc.simplified_dd_eligible,
            "country_benchmark": risk_doc.country_benchmark,
            "criterion_count": len(risk_doc.criterion_evaluations),
            "generated_at": risk_doc.generated_at.isoformat(),
            "article_reference": "EUDR Article 10",
        }

    def _build_mitigation_summary_section(
        self, mitigation_doc: Optional[MitigationDoc],
    ) -> Dict[str, Any]:
        """Build mitigation measures summary for DDS.

        Args:
            mitigation_doc: Mitigation documentation (optional).

        Returns:
            Dictionary with mitigation summary data for DDS.
        """
        if mitigation_doc is None:
            return {
                "status": "not_applicable",
                "message": (
                    "No mitigation measures were required based on "
                    "the risk assessment outcome."
                ),
                "article_reference": "EUDR Article 11",
            }

        return {
            "doc_id": mitigation_doc.doc_id,
            "strategy_id": mitigation_doc.strategy_id,
            "pre_score": str(mitigation_doc.pre_score),
            "post_score": str(mitigation_doc.post_score),
            "measure_count": len(mitigation_doc.measures_summary),
            "verification_result": mitigation_doc.verification_result,
            "generated_at": mitigation_doc.generated_at.isoformat(),
            "article_reference": "EUDR Article 11",
        }

    def _determine_compliance_conclusion(
        self,
        risk_doc: Optional[RiskAssessmentDoc],
        mitigation_doc: Optional[MitigationDoc],
    ) -> str:
        """Determine compliance conclusion based on risk/mitigation.

        Logic:
        - No risk doc → 'pending'
        - NEGLIGIBLE or LOW risk → 'compliant'
        - STANDARD risk → 'compliant_standard_monitoring'
        - HIGH/CRITICAL with successful mitigation → 'compliant_after_mitigation'
        - HIGH/CRITICAL without sufficient mitigation → 'non_compliant'

        Args:
            risk_doc: Risk assessment documentation.
            mitigation_doc: Mitigation documentation.

        Returns:
            Compliance conclusion key string.
        """
        if risk_doc is None:
            return "pending"

        risk_level = risk_doc.risk_level

        # Low/negligible risk: compliant
        if risk_level in (RiskLevel.NEGLIGIBLE, RiskLevel.LOW):
            return "compliant"

        # Standard risk: compliant with monitoring
        if risk_level == RiskLevel.STANDARD:
            return "compliant_standard_monitoring"

        # High/critical risk: need mitigation
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if mitigation_doc is not None:
                verification = mitigation_doc.verification_result
                if verification == "sufficient":
                    return "compliant_after_mitigation"
                if verification == "partial":
                    # Partial mitigation: check if post score is acceptable
                    if mitigation_doc.post_score <= Decimal("30"):
                        return "compliant_after_mitigation"
                    return "non_compliant"
            return "non_compliant"

        return "pending"

    def _compute_dds_hash(self, dds_content: Dict[str, Any]) -> str:
        """Compute SHA-256 of DDS content.

        Args:
            dds_content: DDS content dictionary.

        Returns:
            64-character hex SHA-256 hash string.
        """
        return self._provenance.compute_hash(dds_content)

    def _validate_mandatory_inputs(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        products: List[ProductEntry],
        article9_package: Article9Package,
    ) -> None:
        """Validate all mandatory DDS inputs.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity.
            products: Product entries.
            article9_package: Article 9 package.

        Raises:
            ValueError: If any mandatory input is missing or invalid.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError(
                "operator_id is required for DDS generation "
                "(EUDR Article 4(2))"
            )

        if not products:
            raise ValueError(
                "At least one product entry is required for DDS "
                "(EUDR Article 4(2))"
            )

        max_products = self._config.max_products_per_dds
        if len(products) > max_products:
            raise ValueError(
                f"Product count {len(products)} exceeds maximum "
                f"{max_products} per DDS"
            )

        if article9_package is None:
            raise ValueError(
                "Article 9 package is required for DDS "
                "(EUDR Article 9)"
            )

    def validate_dds_completeness(
        self, dds: DDSDocument,
    ) -> ValidationResult:
        """Validate DDS has all mandatory fields per Article 12.

        Checks that the DDS document contains all required information
        for regulatory submission, including operator identification,
        commodity classification, product listing, Article 9 data,
        risk assessment reference, and compliance conclusion.

        Args:
            dds: DDS document to validate.

        Returns:
            ValidationResult with completeness assessment.
        """
        start_time = time.monotonic()
        validation_id = f"val-{uuid.uuid4().hex[:12]}"
        issues: List[ValidationIssue] = []
        total_checks = 0
        passed_checks = 0

        # Check operator_id
        total_checks += 1
        if not dds.operator_id:
            issues.append(ValidationIssue(
                field="operator_id",
                severity=ValidationSeverity.ERROR,
                message="Operator ID is required (Article 4(2))",
                article_reference="EUDR Article 4(2)",
            ))
        else:
            passed_checks += 1

        # Check commodity
        total_checks += 1
        if dds.commodity is None:
            issues.append(ValidationIssue(
                field="commodity",
                severity=ValidationSeverity.ERROR,
                message="Commodity classification is required",
                article_reference="EUDR Article 1(1)",
            ))
        else:
            passed_checks += 1

        # Check products
        total_checks += 1
        if not dds.products:
            issues.append(ValidationIssue(
                field="products",
                severity=ValidationSeverity.ERROR,
                message="At least one product is required",
                article_reference="EUDR Article 9(1)",
            ))
        else:
            passed_checks += 1

        # Check Article 9 reference
        total_checks += 1
        if not dds.article9_ref:
            issues.append(ValidationIssue(
                field="article9_ref",
                severity=ValidationSeverity.ERROR,
                message="Article 9 package reference is required",
                article_reference="EUDR Article 9",
            ))
        else:
            passed_checks += 1

        # Check risk assessment reference
        total_checks += 1
        if not dds.risk_assessment_ref:
            issues.append(ValidationIssue(
                field="risk_assessment_ref",
                severity=ValidationSeverity.WARNING,
                message="Risk assessment reference is missing",
                article_reference="EUDR Article 10",
            ))
        else:
            passed_checks += 1

        # Check compliance conclusion
        total_checks += 1
        if not dds.compliance_conclusion:
            issues.append(ValidationIssue(
                field="compliance_conclusion",
                severity=ValidationSeverity.ERROR,
                message="Compliance conclusion is required",
                article_reference="EUDR Article 4(2)",
            ))
        else:
            passed_checks += 1

        # Check reference number
        total_checks += 1
        if not dds.reference_number:
            issues.append(ValidationIssue(
                field="reference_number",
                severity=ValidationSeverity.ERROR,
                message="DDS reference number is required",
                article_reference="EUDR Article 12",
            ))
        else:
            passed_checks += 1

        # Check provenance hash
        total_checks += 1
        if not dds.provenance_hash:
            issues.append(ValidationIssue(
                field="provenance_hash",
                severity=ValidationSeverity.WARNING,
                message="Provenance hash is missing for audit trail",
                article_reference="EUDR Article 31",
            ))
        else:
            passed_checks += 1

        # Check HS codes on products
        for idx, product in enumerate(dds.products):
            total_checks += 1
            if not product.hs_code:
                issues.append(ValidationIssue(
                    field=f"products[{idx}].hs_code",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Product '{product.product_id}' missing HS code"
                    ),
                    article_reference="EUDR Article 9(1)(g)",
                ))
            else:
                passed_checks += 1

        # Determine overall validity (no ERROR-level issues)
        errors = [
            i for i in issues
            if i.severity == ValidationSeverity.ERROR
        ]
        warnings = [
            i for i in issues
            if i.severity == ValidationSeverity.WARNING
        ]
        is_valid = len(errors) == 0

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "DDS validation: dds_id=%s, valid=%s, checks=%d/%d, "
            "errors=%d, warnings=%d, elapsed=%.1fms",
            dds.dds_id, is_valid, passed_checks, total_checks,
            len(errors), len(warnings), elapsed_ms,
        )

        return ValidationResult(
            validation_id=validation_id,
            document_id=dds.dds_id,
            document_type=DocumentType.DDS,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
        )

    def validate_hs_code_commodity_match(
        self,
        products: List[ProductEntry],
        commodity: EUDRCommodity,
    ) -> List[ValidationIssue]:
        """Validate HS codes match the declared commodity.

        Args:
            products: List of product entries.
            commodity: Declared EUDR commodity.

        Returns:
            List of validation issues for mismatched HS codes.
        """
        issues: List[ValidationIssue] = []
        valid_prefixes = _COMMODITY_HS_PREFIXES.get(commodity, [])

        if not valid_prefixes:
            return issues

        for idx, product in enumerate(products):
            if not product.hs_code:
                continue
            hs_prefix = product.hs_code[:4]
            if hs_prefix not in valid_prefixes:
                issues.append(ValidationIssue(
                    field=f"products[{idx}].hs_code",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"HS code '{product.hs_code}' may not match "
                        f"declared commodity '{commodity.value}'. "
                        f"Expected prefixes: {valid_prefixes}"
                    ),
                    article_reference="EUDR Article 9(1)(g)",
                ))

        return issues

    async def generate_batch_dds(
        self,
        operator_id: str,
        commodity: EUDRCommodity,
        product_batches: List[List[ProductEntry]],
        article9_package: Article9Package,
        risk_doc: Optional[RiskAssessmentDoc] = None,
        mitigation_doc: Optional[MitigationDoc] = None,
    ) -> List[DDSDocument]:
        """Generate multiple DDS documents for large product sets.

        Splits products into batches based on max_products_per_dds
        configuration and generates one DDS per batch.

        Args:
            operator_id: Operator identifier.
            commodity: EUDR commodity category.
            product_batches: Pre-split product batches.
            article9_package: Assembled Article 9 information.
            risk_doc: Risk assessment documentation.
            mitigation_doc: Mitigation documentation.

        Returns:
            List of DDSDocument instances.
        """
        start_time = time.monotonic()
        results: List[DDSDocument] = []

        logger.info(
            "Generating batch DDS: operator=%s, commodity=%s, "
            "batches=%d",
            operator_id, commodity.value, len(product_batches),
        )

        for batch_idx, products in enumerate(product_batches):
            dds = await self.generate_dds(
                operator_id=operator_id,
                commodity=commodity,
                products=products,
                article9_package=article9_package,
                risk_doc=risk_doc,
                mitigation_doc=mitigation_doc,
            )
            results.append(dds)
            logger.debug(
                "Batch DDS %d/%d generated: %s",
                batch_idx + 1, len(product_batches), dds.dds_id,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch DDS generation complete: count=%d, elapsed=%.1fms",
            len(results), elapsed_ms,
        )

        return results

    def get_conclusion_text(self, conclusion_key: str) -> str:
        """Get the full compliance conclusion text for a key.

        Args:
            conclusion_key: Conclusion key string.

        Returns:
            Full conclusion text or pending text if key unknown.
        """
        return _COMPLIANCE_CONCLUSIONS.get(
            conclusion_key, _COMPLIANCE_CONCLUSIONS["pending"],
        )

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration details.
        """
        return {
            "engine": "DDSStatementGenerator",
            "status": "available",
            "config": {
                "dds_reference_prefix": self._config.dds_reference_prefix,
                "dds_schema_version": self._config.dds_schema_version,
                "max_products_per_dds": self._config.max_products_per_dds,
                "include_provenance": self._config.include_provenance,
            },
            "current_sequence": self._sequence,
        }
