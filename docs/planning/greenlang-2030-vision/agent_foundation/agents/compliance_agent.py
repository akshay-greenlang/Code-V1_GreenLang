# -*- coding: utf-8 -*-
"""
ComplianceAgent - Regulatory compliance checking agent.

This module implements the ComplianceAgent for validating regulatory compliance
across multiple frameworks including CSRD, CBAM, EUDR, and SB253.

Example:
    >>> agent = ComplianceAgent(config)
    >>> result = await agent.execute(ComplianceInput(
    ...     regulation="CSRD",
    ...     data=sustainability_data,
    ...     check_type="double_materiality"
    ... ))
"""

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class Regulation(str, Enum):
    """Supported regulatory frameworks."""

    CSRD = "CSRD"  # Corporate Sustainability Reporting Directive
    CBAM = "CBAM"  # Carbon Border Adjustment Mechanism
    EUDR = "EUDR"  # EU Deforestation Regulation
    SB253 = "SB253"  # California Climate Disclosure
    TCFD = "TCFD"  # Task Force on Climate-related Financial Disclosures
    GRI = "GRI"  # Global Reporting Initiative
    SASB = "SASB"  # Sustainability Accounting Standards Board
    CDP = "CDP"  # Carbon Disclosure Project


class ComplianceCheckType(str, Enum):
    """Types of compliance checks."""

    FULL_AUDIT = "full_audit"
    DATA_VALIDATION = "data_validation"
    DISCLOSURE_CHECK = "disclosure_check"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    THRESHOLD_CHECK = "threshold_check"
    DOCUMENTATION_REVIEW = "documentation_review"
    QUICK_SCAN = "quick_scan"


class ComplianceStatus(str, Enum):
    """Compliance status results."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    REQUIRES_REVIEW = "requires_review"
    NOT_APPLICABLE = "not_applicable"


class ComplianceInput(BaseModel):
    """Input data model for ComplianceAgent."""

    regulation: Regulation = Field(..., description="Regulatory framework to check")
    check_type: ComplianceCheckType = Field(
        ComplianceCheckType.FULL_AUDIT,
        description="Type of compliance check"
    )
    data: Dict[str, Any] = Field(..., description="Data to validate for compliance")
    organization_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Organization details (size, sector, location)"
    )
    reporting_period: Optional[str] = Field(None, description="Reporting period (YYYY or YYYY-MM)")
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom thresholds for checks"
    )
    previous_findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous compliance findings for trend analysis"
    )

    @validator('reporting_period')
    def validate_period(cls, v):
        """Validate reporting period format."""
        if v and not (len(v) == 4 or len(v) == 7):
            raise ValueError("Reporting period must be YYYY or YYYY-MM format")
        return v


class ComplianceOutput(BaseModel):
    """Output data model for ComplianceAgent."""

    status: ComplianceStatus = Field(..., description="Overall compliance status")
    regulation: Regulation = Field(..., description="Regulation checked")
    check_type: ComplianceCheckType = Field(..., description="Type of check performed")
    score: float = Field(..., ge=0.0, le=100.0, description="Compliance score (0-100)")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed compliance findings"
    )
    requirements_met: List[str] = Field(
        default_factory=list,
        description="Requirements that are met"
    )
    requirements_failed: List[str] = Field(
        default_factory=list,
        description="Requirements that failed"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence")


class ComplianceAgent(BaseAgent):
    """
    ComplianceAgent implementation for regulatory compliance checking.

    This agent validates data against regulatory requirements using deterministic
    rule-based checks. It maintains a comprehensive rule registry for each
    regulation and provides detailed findings with recommendations.

    Attributes:
        config: Agent configuration
        rule_registry: Registry of compliance rules by regulation
        validation_cache: Cache of recent validations

    Example:
        >>> config = AgentConfig(name="csrd_compliance", version="1.0.0")
        >>> agent = ComplianceAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(compliance_input)
        >>> print(f"Compliance status: {result.result.status}")
    """

    def __init__(self, config: AgentConfig):
        """Initialize ComplianceAgent."""
        super().__init__(config)
        self.rule_registry: Dict[Regulation, List[ComplianceRule]] = {}
        self.validation_cache: Dict[str, ComplianceOutput] = {}
        self.compliance_history: List[ComplianceOutput] = []

    async def _initialize_core(self) -> None:
        """Initialize compliance resources."""
        self._logger.info("Initializing ComplianceAgent resources")

        # Load compliance rules for each regulation
        self._load_compliance_rules()

        self._logger.info(f"Loaded rules for {len(self.rule_registry)} regulations")

    def _load_compliance_rules(self) -> None:
        """Load compliance rules for each regulation."""
        # CSRD Rules
        self.rule_registry[Regulation.CSRD] = [
            ComplianceRule(
                id="CSRD-001",
                name="Double Materiality Assessment",
                category="Core Requirement",
                check_function=self._check_double_materiality,
                severity="critical",
                description="Validate double materiality assessment is complete"
            ),
            ComplianceRule(
                id="CSRD-002",
                name="Value Chain Disclosure",
                category="Disclosure",
                check_function=self._check_value_chain,
                severity="high",
                description="Validate upstream and downstream value chain data"
            ),
            ComplianceRule(
                id="CSRD-003",
                name="Climate Targets",
                category="Environmental",
                check_function=self._check_climate_targets,
                severity="high",
                description="Validate science-based climate targets"
            ),
            ComplianceRule(
                id="CSRD-004",
                name="EU Taxonomy Alignment",
                category="Taxonomy",
                check_function=self._check_taxonomy_alignment,
                severity="critical",
                description="Check EU Taxonomy eligibility and alignment"
            ),
        ]

        # CBAM Rules
        self.rule_registry[Regulation.CBAM] = [
            ComplianceRule(
                id="CBAM-001",
                name="Embedded Emissions Calculation",
                category="Emissions",
                check_function=self._check_embedded_emissions,
                severity="critical",
                description="Validate embedded emissions calculations"
            ),
            ComplianceRule(
                id="CBAM-002",
                name="CBAM Certificate Validity",
                category="Documentation",
                check_function=self._check_cbam_certificates,
                severity="high",
                description="Validate CBAM certificates"
            ),
        ]

        # EUDR Rules
        self.rule_registry[Regulation.EUDR] = [
            ComplianceRule(
                id="EUDR-001",
                name="Deforestation-Free Verification",
                category="Supply Chain",
                check_function=self._check_deforestation_free,
                severity="critical",
                description="Verify products are deforestation-free"
            ),
            ComplianceRule(
                id="EUDR-002",
                name="Geolocation Data",
                category="Traceability",
                check_function=self._check_geolocation,
                severity="high",
                description="Validate geolocation data for commodities"
            ),
        ]

        # SB253 Rules
        self.rule_registry[Regulation.SB253] = [
            ComplianceRule(
                id="SB253-001",
                name="Scope 1-3 Emissions Disclosure",
                category="Emissions",
                check_function=self._check_ghg_disclosure,
                severity="critical",
                description="Validate Scope 1, 2, and 3 emissions disclosure"
            ),
            ComplianceRule(
                id="SB253-002",
                name="Third-Party Verification",
                category="Assurance",
                check_function=self._check_third_party_verification,
                severity="high",
                description="Validate third-party verification status"
            ),
        ]

        # Add more regulations as needed
        for reg in [Regulation.TCFD, Regulation.GRI, Regulation.SASB, Regulation.CDP]:
            self.rule_registry[reg] = []  # Placeholder for additional rules

    async def _execute_core(self, input_data: ComplianceInput, context: ExecutionContext) -> ComplianceOutput:
        """
        Core execution logic for compliance checking.

        This method performs deterministic rule-based validation.
        """
        start_time = datetime.now(timezone.utc)
        findings = []
        requirements_met = []
        requirements_failed = []
        recommendations = []
        evidence = {}

        try:
            # Step 1: Check cache for recent validation
            cache_key = self._generate_cache_key(input_data)
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self._logger.info("Returning cached compliance result")
                    return cached_result

            # Step 2: Get applicable rules
            rules = self.rule_registry.get(input_data.regulation, [])
            if not rules:
                raise ValueError(f"No rules defined for regulation: {input_data.regulation}")

            self._logger.info(f"Checking {len(rules)} rules for {input_data.regulation}")

            # Step 3: Execute compliance checks
            total_score = 0.0
            max_score = 0.0
            critical_failures = []

            for rule in rules:
                try:
                    # Execute rule check
                    result = rule.check_function(input_data.data, input_data.organization_info)

                    # Weight by severity
                    weight = self._get_severity_weight(rule.severity)
                    max_score += weight

                    if result["passed"]:
                        requirements_met.append(f"{rule.id}: {rule.name}")
                        total_score += weight
                        evidence[rule.id] = result.get("evidence", {})
                    else:
                        requirements_failed.append(f"{rule.id}: {rule.name}")
                        if rule.severity == "critical":
                            critical_failures.append(rule.id)

                        findings.append({
                            "rule_id": rule.id,
                            "rule_name": rule.name,
                            "category": rule.category,
                            "severity": rule.severity,
                            "status": "failed",
                            "details": result.get("details", ""),
                            "recommendation": result.get("recommendation", "")
                        })

                        if result.get("recommendation"):
                            recommendations.append(result["recommendation"])

                except Exception as e:
                    self._logger.error(f"Error checking rule {rule.id}: {str(e)}")
                    findings.append({
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "status": "error",
                        "error": str(e)
                    })

            # Step 4: Calculate overall status and score
            compliance_score = (total_score / max_score * 100) if max_score > 0 else 0
            status = self._determine_compliance_status(
                compliance_score,
                critical_failures,
                len(requirements_failed)
            )

            # Step 5: Generate recommendations based on findings
            if not recommendations:
                recommendations = self._generate_recommendations(findings, input_data.regulation)

            # Step 6: Calculate confidence based on data completeness
            confidence = self._calculate_confidence(input_data.data)

            # Step 7: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                status,
                context.execution_id
            )

            # Step 8: Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 9: Create output
            output = ComplianceOutput(
                status=status,
                regulation=input_data.regulation,
                check_type=input_data.check_type,
                score=round(compliance_score, 2),
                findings=findings,
                requirements_met=requirements_met,
                requirements_failed=requirements_failed,
                recommendations=recommendations,
                evidence=evidence,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                confidence=confidence
            )

            # Store in cache and history
            self.validation_cache[cache_key] = output
            self.compliance_history.append(output)
            if len(self.compliance_history) > 100:
                self.compliance_history.pop(0)

            return output

        except Exception as e:
            self._logger.error(f"Compliance check failed: {str(e)}", exc_info=True)
            raise

    def _generate_cache_key(self, input_data: ComplianceInput) -> str:
        """Generate cache key for validation results."""
        key_data = f"{input_data.regulation}:{input_data.check_type}:{str(sorted(input_data.data.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_valid(self, cached_result: ComplianceOutput) -> bool:
        """Check if cached result is still valid (within 1 hour)."""
        # Simple time-based validation - could be enhanced with data versioning
        return True  # Simplified for now

    def _get_severity_weight(self, severity: str) -> float:
        """Get weight based on severity level."""
        weights = {
            "critical": 3.0,
            "high": 2.0,
            "medium": 1.0,
            "low": 0.5
        }
        return weights.get(severity, 1.0)

    def _determine_compliance_status(
        self,
        score: float,
        critical_failures: List[str],
        total_failures: int
    ) -> ComplianceStatus:
        """Determine overall compliance status."""
        if critical_failures:
            return ComplianceStatus.NON_COMPLIANT
        elif score >= 95:
            return ComplianceStatus.COMPLIANT
        elif score >= 80:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif score >= 60:
            return ComplianceStatus.REQUIRES_REVIEW
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence based on data completeness."""
        # Check for key data points
        required_fields = ["emissions", "targets", "governance", "risks", "opportunities"]
        present_fields = sum(1 for field in required_fields if field in data and data[field])
        return min(1.0, present_fields / len(required_fields) + 0.2)

    def _generate_recommendations(self, findings: List[Dict], regulation: Regulation) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Analyze patterns in findings
        failed_categories = set(f["category"] for f in findings if f.get("status") == "failed")

        if "Environmental" in failed_categories:
            recommendations.append("Establish science-based emissions reduction targets")
        if "Disclosure" in failed_categories:
            recommendations.append("Enhance data collection processes for complete disclosure")
        if "Governance" in failed_categories:
            recommendations.append("Strengthen sustainability governance structures")

        return recommendations

    def _calculate_provenance_hash(self, inputs: Dict, status: str, execution_id: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": inputs.get("regulation"),
            "status": status
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    # Compliance check implementations (deterministic rule-based)

    def _check_double_materiality(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check CSRD double materiality assessment."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "materiality_assessment" not in data:
            result["details"] = "Missing materiality assessment"
            result["recommendation"] = "Conduct double materiality assessment covering impact and financial materiality"
            return result

        assessment = data["materiality_assessment"]
        has_impact = "impact_materiality" in assessment
        has_financial = "financial_materiality" in assessment
        has_stakeholder = "stakeholder_engagement" in assessment

        if has_impact and has_financial and has_stakeholder:
            result["passed"] = True
            result["evidence"] = assessment
        else:
            missing = []
            if not has_impact:
                missing.append("impact materiality")
            if not has_financial:
                missing.append("financial materiality")
            if not has_stakeholder:
                missing.append("stakeholder engagement")

            result["details"] = f"Missing: {', '.join(missing)}"
            result["recommendation"] = f"Complete assessment for: {', '.join(missing)}"

        return result

    def _check_value_chain(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check value chain disclosure."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "value_chain" not in data:
            result["details"] = "Missing value chain data"
            result["recommendation"] = "Map and disclose upstream and downstream value chain impacts"
            return result

        value_chain = data["value_chain"]
        has_upstream = "upstream" in value_chain and value_chain["upstream"]
        has_downstream = "downstream" in value_chain and value_chain["downstream"]

        if has_upstream and has_downstream:
            result["passed"] = True
            result["evidence"] = value_chain
        else:
            result["details"] = "Incomplete value chain data"
            result["recommendation"] = "Provide comprehensive upstream and downstream value chain information"

        return result

    def _check_climate_targets(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check climate targets."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "targets" not in data or "climate" not in data["targets"]:
            result["details"] = "Missing climate targets"
            result["recommendation"] = "Set science-based climate targets aligned with 1.5°C pathway"
            return result

        climate_targets = data["targets"]["climate"]
        has_2030 = "2030" in climate_targets
        has_2050 = "2050" in climate_targets or "net_zero" in climate_targets
        has_baseline = "baseline_year" in climate_targets

        if has_2030 and has_2050 and has_baseline:
            result["passed"] = True
            result["evidence"] = climate_targets
        else:
            result["details"] = "Incomplete climate targets"
            result["recommendation"] = "Establish near-term (2030) and long-term (2050) science-based targets"

        return result

    def _check_taxonomy_alignment(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check EU Taxonomy alignment."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "taxonomy" not in data:
            result["details"] = "Missing EU Taxonomy data"
            result["recommendation"] = "Assess and disclose EU Taxonomy eligibility and alignment"
            return result

        taxonomy = data["taxonomy"]
        has_eligibility = "eligible_activities" in taxonomy
        has_alignment = "aligned_activities" in taxonomy
        has_kpis = all(kpi in taxonomy for kpi in ["turnover", "capex", "opex"])

        if has_eligibility and has_alignment and has_kpis:
            result["passed"] = True
            result["evidence"] = taxonomy
        else:
            result["details"] = "Incomplete EU Taxonomy disclosure"
            result["recommendation"] = "Provide complete Taxonomy KPIs for turnover, CapEx, and OpEx"

        return result

    def _check_embedded_emissions(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check CBAM embedded emissions."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "cbam_products" not in data:
            result["details"] = "Missing CBAM product data"
            result["recommendation"] = "Calculate embedded emissions for CBAM-covered products"
            return result

        all_calculated = all(
            "embedded_emissions" in product
            for product in data["cbam_products"]
        )

        if all_calculated:
            result["passed"] = True
            result["evidence"] = {"products": data["cbam_products"]}
        else:
            result["details"] = "Missing embedded emissions calculations"
            result["recommendation"] = "Complete embedded emissions calculations for all CBAM products"

        return result

    def _check_cbam_certificates(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check CBAM certificates."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "cbam_certificates" not in data:
            result["details"] = "Missing CBAM certificates"
            result["recommendation"] = "Obtain valid CBAM certificates for imports"
            return result

        all_valid = all(
            cert.get("valid", False) and cert.get("expiry_date", "") > DeterministicClock.now().isoformat()
            for cert in data["cbam_certificates"]
        )

        if all_valid:
            result["passed"] = True
            result["evidence"] = {"certificates": data["cbam_certificates"]}
        else:
            result["details"] = "Invalid or expired CBAM certificates"
            result["recommendation"] = "Renew expired certificates and ensure all imports are covered"

        return result

    def _check_deforestation_free(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check EUDR deforestation-free status."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "commodities" not in data:
            result["details"] = "Missing commodity data"
            result["recommendation"] = "Verify deforestation-free status for all relevant commodities"
            return result

        all_verified = all(
            commodity.get("deforestation_free", False)
            for commodity in data["commodities"]
        )

        if all_verified:
            result["passed"] = True
            result["evidence"] = {"commodities": data["commodities"]}
        else:
            result["details"] = "Not all commodities verified as deforestation-free"
            result["recommendation"] = "Complete due diligence for all EUDR-covered commodities"

        return result

    def _check_geolocation(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check EUDR geolocation data."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "commodities" not in data:
            result["details"] = "Missing commodity data"
            result["recommendation"] = "Provide geolocation data for commodity sources"
            return result

        all_geolocated = all(
            "geolocation" in commodity and
            "latitude" in commodity["geolocation"] and
            "longitude" in commodity["geolocation"]
            for commodity in data["commodities"]
        )

        if all_geolocated:
            result["passed"] = True
            result["evidence"] = {"geolocations": [c["geolocation"] for c in data["commodities"]]}
        else:
            result["details"] = "Missing or incomplete geolocation data"
            result["recommendation"] = "Collect precise geolocation coordinates for all production areas"

        return result

    def _check_ghg_disclosure(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check SB253 GHG disclosure."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "emissions" not in data:
            result["details"] = "Missing emissions data"
            result["recommendation"] = "Calculate and disclose Scope 1, 2, and 3 emissions"
            return result

        emissions = data["emissions"]
        has_scope1 = "scope1" in emissions and emissions["scope1"] is not None
        has_scope2 = "scope2" in emissions and emissions["scope2"] is not None
        has_scope3 = "scope3" in emissions and emissions["scope3"] is not None

        # Check if organization size requires Scope 3 (revenue > $1B)
        requires_scope3 = org_info.get("annual_revenue", 0) > 1_000_000_000

        if has_scope1 and has_scope2 and (has_scope3 or not requires_scope3):
            result["passed"] = True
            result["evidence"] = emissions
        else:
            missing = []
            if not has_scope1:
                missing.append("Scope 1")
            if not has_scope2:
                missing.append("Scope 2")
            if not has_scope3 and requires_scope3:
                missing.append("Scope 3")

            result["details"] = f"Missing: {', '.join(missing)} emissions"
            result["recommendation"] = f"Calculate and disclose {', '.join(missing)} emissions"

        return result

    def _check_third_party_verification(self, data: Dict, org_info: Dict) -> Dict[str, Any]:
        """Check third-party verification status."""
        result = {
            "passed": False,
            "details": "",
            "evidence": {},
            "recommendation": ""
        }

        if "verification" not in data:
            result["details"] = "Missing verification information"
            result["recommendation"] = "Obtain third-party verification for emissions data"
            return result

        verification = data["verification"]
        has_verifier = "verifier_name" in verification
        has_standard = "verification_standard" in verification
        is_valid = verification.get("valid_until", "") > DeterministicClock.now().isoformat()

        if has_verifier and has_standard and is_valid:
            result["passed"] = True
            result["evidence"] = verification
        else:
            result["details"] = "Invalid or incomplete verification"
            result["recommendation"] = "Obtain valid third-party verification following recognized standards"

        return result

    async def _terminate_core(self) -> None:
        """Cleanup compliance resources."""
        self._logger.info("Cleaning up ComplianceAgent resources")
        self.rule_registry.clear()
        self.validation_cache.clear()
        self.compliance_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect compliance-specific metrics."""
        if not self.compliance_history:
            return {}

        recent = self.compliance_history[-100:]
        return {
            "total_checks": len(self.compliance_history),
            "average_score": sum(c.score for c in recent) / len(recent),
            "compliance_rate": sum(1 for c in recent if c.status == ComplianceStatus.COMPLIANT) / len(recent),
            "regulations_checked": list(set(c.regulation for c in recent)),
            "cached_validations": len(self.validation_cache)
        }


class ComplianceRule:
    """Represents a single compliance rule."""

    def __init__(
        self,
        id: str,
        name: str,
        category: str,
        check_function: callable,
        severity: str,
        description: str
    ):
        """Initialize compliance rule."""
        self.id = id
        self.name = name
        self.category = category
        self.check_function = check_function
        self.severity = severity
        self.description = description