# -*- coding: utf-8 -*-
"""
Certificate Management Workflow
=================================

Four-phase CBAM certificate lifecycle management workflow covering obligation
assessment, purchase planning, quarterly holding compliance, and annual
surrender execution.

Regulatory Context:
    Per EU CBAM Regulation 2023/956:
    - Article 20: CBAM certificates are purchased from the national competent
      authority at the EU ETS auction price (weekly weighted average)
    - Article 22(1): Authorized declarants must surrender certificates equal
      to embedded emissions by May 31 of each year
    - Article 22(2): By end of each quarter, declarants must hold at least
      50% of the estimated annual certificate obligation (calculated from
      previous year's imports or pro-rated actual data)
    - Article 23: Certificates not surrendered within two years of purchase
      may be repurchased by the national authority at the original price
    - Article 26: Penalties for non-compliance (EUR 100/tCO2e for missed
      certificates, adjusted annually to EU ETS price)

    Certificate pricing:
        - Published weekly by European Commission based on EU ETS auction
        - 1 certificate = 1 tonne CO2e
        - No secondary market (certificates are non-transferable)

Phases:
    1. Obligation assessment - Calculate annual certificate requirement
    2. Purchase planning - Plan purchases against EU ETS auction schedule
    3. Holding compliance - Verify quarterly 50% holding requirement
    4. Surrender execution - Execute annual certificate surrender by May 31

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class HoldingComplianceStatus(str, Enum):
    """Quarterly holding compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    APPROACHING_THRESHOLD = "approaching_threshold"


class SurrenderStatus(str, Enum):
    """Certificate surrender execution status."""
    READY = "ready"
    SHORTFALL = "shortfall"
    SURRENDERED = "surrendered"
    OVERDUE = "overdue"


# =============================================================================
# CONSTANTS
# =============================================================================

# Penalty per tCO2e for non-compliance (Article 26)
PENALTY_EUR_PER_TCO2E = 100.0

# Minimum quarterly holding percentage (Article 22(2))
QUARTERLY_HOLDING_PCT = 50.0

# Certificate validity period in years (Article 23)
CERTIFICATE_VALIDITY_YEARS = 2


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class EmissionData(BaseModel):
    """Input emission data for certificate obligation calculation."""
    year: int = Field(..., ge=2026, description="Reporting year")
    total_embedded_emissions_tco2e: float = Field(..., ge=0)
    direct_emissions_tco2e: float = Field(default=0.0, ge=0)
    indirect_emissions_tco2e: float = Field(default=0.0, ge=0)
    free_allocation_deduction_tco2e: float = Field(default=0.0, ge=0)
    carbon_price_deduction_tco2e: float = Field(default=0.0, ge=0)
    sector_breakdown: Dict[str, float] = Field(default_factory=dict)
    quarterly_estimates: Dict[str, float] = Field(
        default_factory=dict,
        description="Quarterly emission estimates: {'Q1': 100.0, ...}",
    )


class PurchasePlan(BaseModel):
    """Certificate purchase plan entry."""
    purchase_date: str = Field(..., description="Planned purchase date YYYY-MM-DD")
    certificates_to_purchase: float = Field(..., ge=0)
    estimated_price_eur: float = Field(..., ge=0)
    estimated_cost_eur: float = Field(..., ge=0)
    purpose: str = Field(default="", description="quarterly_holding or annual_surrender")


class QuarterlyHoldingCheck(BaseModel):
    """Result of a quarterly 50% holding requirement check."""
    quarter: str = Field(..., description="Quarter label e.g. 'Q1'")
    estimated_annual_obligation: float = Field(default=0.0, ge=0)
    required_holding: float = Field(default=0.0, ge=0, description="50% of estimated annual")
    actual_holding: float = Field(default=0.0, ge=0, description="Certificates currently held")
    status: HoldingComplianceStatus = Field(...)
    shortfall: float = Field(default=0.0, ge=0)


class CertificateManagementResult(BaseModel):
    """Complete result from the certificate management workflow."""
    workflow_name: str = Field(default="certificate_management")
    status: PhaseStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    year: int = Field(...)
    annual_obligation_certificates: float = Field(default=0.0, ge=0)
    certificates_held: float = Field(default=0.0, ge=0)
    certificates_to_purchase: float = Field(default=0.0, ge=0)
    estimated_annual_cost_eur: float = Field(default=0.0, ge=0)
    purchase_plan: List[PurchasePlan] = Field(default_factory=list)
    holding_checks: List[QuarterlyHoldingCheck] = Field(default_factory=list)
    holding_compliant: bool = Field(default=False)
    surrender_status: SurrenderStatus = Field(default=SurrenderStatus.SHORTFALL)
    surrender_deadline: str = Field(default="")
    potential_penalty_eur: float = Field(default=0.0, ge=0)
    provenance_hash: str = Field(default="")
    execution_id: str = Field(default="")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)


# =============================================================================
# CERTIFICATE MANAGEMENT WORKFLOW
# =============================================================================


class CertificateManagementWorkflow:
    """
    Four-phase CBAM certificate lifecycle management workflow.

    Manages the complete certificate lifecycle from obligation calculation
    through purchase planning, quarterly holding compliance verification,
    and annual surrender execution.

    All certificate calculations use Decimal arithmetic with ROUND_HALF_UP
    for zero-hallucination compliance.

    Attributes:
        config: Optional configuration dict.
        logger: Logger instance.
        _execution_id: Unique execution identifier.
        _phase_results: Accumulated phase results.

    Example:
        >>> wf = CertificateManagementWorkflow()
        >>> result = await wf.execute(
        ...     config={"ets_price_eur": 80.0},
        ...     emission_data=EmissionData(year=2026, total_embedded_emissions_tco2e=1000),
        ...     year=2026,
        ... )
        >>> assert result.annual_obligation_certificates > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the CertificateManagementWorkflow.

        Args:
            config: Optional configuration dict.
        """
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.CertificateManagementWorkflow")
        self._execution_id: str = str(uuid.uuid4())
        self._phase_results: List[PhaseResult] = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        config: Optional[Dict[str, Any]],
        emission_data: EmissionData,
        year: int,
    ) -> CertificateManagementResult:
        """
        Execute the full 4-phase certificate management workflow.

        Args:
            config: Execution-level config overrides.
            emission_data: Emission data for certificate calculation.
            year: Reporting year.

        Returns:
            CertificateManagementResult with obligation, purchase plan,
            holding checks, and surrender status.
        """
        started_at = datetime.utcnow()
        merged_config = {**self.config, **(config or {})}
        surrender_deadline = f"{year + 1}-05-31"

        self.logger.info(
            "Starting certificate management execution_id=%s year=%d",
            self._execution_id, year,
        )

        context: Dict[str, Any] = {
            "config": merged_config,
            "emission_data": emission_data,
            "year": year,
            "execution_id": self._execution_id,
            "surrender_deadline": surrender_deadline,
        }

        phase_handlers = [
            ("obligation_assessment", self._phase_1_obligation_assessment),
            ("purchase_planning", self._phase_2_purchase_planning),
            ("holding_compliance", self._phase_3_holding_compliance),
            ("surrender_execution", self._phase_4_surrender_execution),
        ]

        overall_status = PhaseStatus.COMPLETED

        for phase_name, handler in phase_handlers:
            phase_start = datetime.utcnow()
            self.logger.info("Starting phase: %s", phase_name)

            try:
                phase_result = await handler(context)
                phase_result.duration_seconds = (
                    datetime.utcnow() - phase_start
                ).total_seconds()
            except Exception as exc:
                self.logger.error("Phase '%s' failed: %s", phase_name, exc, exc_info=True)
                phase_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    duration_seconds=(datetime.utcnow() - phase_start).total_seconds(),
                    outputs={"error": str(exc)},
                    provenance_hash=self._hash({"error": str(exc)}),
                )

            self._phase_results.append(phase_result)

            if phase_result.status == PhaseStatus.FAILED:
                overall_status = PhaseStatus.FAILED
                if phase_name == "obligation_assessment":
                    break

        completed_at = datetime.utcnow()

        # Extract final values
        annual_obligation = context.get("annual_obligation", 0.0)
        certificates_held = context.get("certificates_held", 0.0)
        certificates_to_purchase = context.get("certificates_to_purchase", 0.0)
        estimated_cost = context.get("estimated_annual_cost_eur", 0.0)
        purchase_plan = context.get("purchase_plan", [])
        holding_checks = context.get("holding_checks", [])
        holding_compliant = context.get("holding_compliant", False)
        surrender_status = context.get("surrender_status", SurrenderStatus.SHORTFALL)
        potential_penalty = context.get("potential_penalty_eur", 0.0)

        provenance = self._hash({
            "execution_id": self._execution_id,
            "phases": [p.provenance_hash for p in self._phase_results],
            "obligation": annual_obligation,
        })

        self.logger.info(
            "Certificate management finished execution_id=%s obligation=%.4f cost=%.2f EUR",
            self._execution_id, annual_obligation, estimated_cost,
        )

        return CertificateManagementResult(
            status=overall_status,
            phases=self._phase_results,
            year=year,
            annual_obligation_certificates=annual_obligation,
            certificates_held=certificates_held,
            certificates_to_purchase=certificates_to_purchase,
            estimated_annual_cost_eur=estimated_cost,
            purchase_plan=purchase_plan,
            holding_checks=holding_checks,
            holding_compliant=holding_compliant,
            surrender_status=surrender_status,
            surrender_deadline=surrender_deadline,
            potential_penalty_eur=potential_penalty,
            provenance_hash=provenance,
            execution_id=self._execution_id,
            started_at=started_at,
            completed_at=completed_at,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Obligation Assessment
    # -------------------------------------------------------------------------

    async def _phase_1_obligation_assessment(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Calculate annual certificate requirement from emission data.

        Per Article 22(1):
            Net obligation = total_emissions - free_allocation_deduction - carbon_price_deduction
            Certificates required = net obligation (1 cert = 1 tCO2e)
        """
        phase_name = "obligation_assessment"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        emission_data: EmissionData = context["emission_data"]
        year = context["year"]

        total = Decimal(str(emission_data.total_embedded_emissions_tco2e))
        free_alloc = Decimal(str(emission_data.free_allocation_deduction_tco2e))
        carbon_ded = Decimal(str(emission_data.carbon_price_deduction_tco2e))

        # Net certificate obligation
        net_obligation = max(
            Decimal("0"),
            (total - free_alloc - carbon_ded).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ),
        )

        # Check current certificate balance
        balance = await self._fetch_certificate_balance(context)
        certificates_held = Decimal(str(balance.get("certificates_held", 0)))

        # Calculate purchase requirement
        purchase_needed = max(
            Decimal("0"),
            (net_obligation - certificates_held).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ),
        )

        # Estimated cost at current ETS price
        ets_price = Decimal(str(context["config"].get("ets_price_eur", 80.0)))
        estimated_cost = (purchase_needed * ets_price).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Potential penalty for non-compliance
        potential_penalty = (
            net_obligation * Decimal(str(PENALTY_EUR_PER_TCO2E))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        context["annual_obligation"] = float(net_obligation)
        context["certificates_held"] = float(certificates_held)
        context["certificates_to_purchase"] = float(purchase_needed)
        context["estimated_annual_cost_eur"] = float(estimated_cost)
        context["potential_penalty_eur"] = float(potential_penalty)

        outputs["total_emissions_tco2e"] = float(total)
        outputs["free_allocation_deduction"] = float(free_alloc)
        outputs["carbon_price_deduction"] = float(carbon_ded)
        outputs["net_obligation_certificates"] = float(net_obligation)
        outputs["certificates_held"] = float(certificates_held)
        outputs["purchase_needed"] = float(purchase_needed)
        outputs["estimated_cost_eur"] = float(estimated_cost)
        outputs["ets_price_eur"] = float(ets_price)
        outputs["potential_penalty_eur"] = float(potential_penalty)

        # Sector breakdown
        sector_obligations: Dict[str, float] = {}
        for sector, emissions in emission_data.sector_breakdown.items():
            sector_share = Decimal(str(emissions)) / total if total > 0 else Decimal("0")
            sector_obligations[sector] = float(
                (net_obligation * sector_share).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            )
        outputs["sector_obligations"] = sector_obligations

        if float(purchase_needed) > 0:
            warnings.append(
                f"Need to purchase {float(purchase_needed):.4f} certificates "
                f"(estimated cost: EUR {float(estimated_cost):,.2f})"
            )

        self.logger.info(
            "Phase 1 complete: obligation=%.4f held=%.4f purchase=%.4f cost=%.2f EUR",
            float(net_obligation), float(certificates_held),
            float(purchase_needed), float(estimated_cost),
        )

        provenance = self._hash({
            "phase": phase_name,
            "obligation": float(net_obligation),
            "held": float(certificates_held),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Purchase Planning
    # -------------------------------------------------------------------------

    async def _phase_2_purchase_planning(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Plan certificate purchases against EU ETS auction schedule.

        Creates a purchase plan that:
            - Spreads purchases across the year to manage cash flow
            - Ensures quarterly 50% holding requirement is met
            - Considers EU ETS price forecasts for optimal timing
        """
        phase_name = "purchase_planning"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        year = context["year"]
        total_to_purchase = Decimal(str(context.get("certificates_to_purchase", 0)))
        ets_price = Decimal(str(context["config"].get("ets_price_eur", 80.0)))
        annual_obligation = Decimal(str(context.get("annual_obligation", 0)))
        certificates_held = Decimal(str(context.get("certificates_held", 0)))

        # Quarterly holding requirement: 50% of estimated annual by end of each quarter
        quarterly_requirement = (annual_obligation * Decimal("0.5")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Plan purchases across quarters
        purchase_plan: List[PurchasePlan] = []
        remaining_to_purchase = total_to_purchase

        if remaining_to_purchase <= 0:
            # Already have enough certificates
            context["purchase_plan"] = []
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                outputs={"purchase_plan": [], "total_cost_eur": 0.0, "message": "Sufficient certificates held"},
                provenance_hash=self._hash({"phase": phase_name, "purchases": 0}),
            )

        # Spread purchases across 4 quarters with front-loading for holding compliance
        quarter_dates = [
            (f"{year}-01-15", "Q1 early purchase (holding compliance)"),
            (f"{year}-04-15", "Q2 purchase (holding compliance)"),
            (f"{year}-07-15", "Q3 purchase"),
            (f"{year}-10-15", "Q4 final purchase"),
        ]

        # Front-load to meet quarterly holding: buy 50% in Q1, rest spread
        q1_purchase = min(remaining_to_purchase, quarterly_requirement - certificates_held)
        q1_purchase = max(Decimal("0"), q1_purchase)
        remaining_after_q1 = remaining_to_purchase - q1_purchase

        # Distribute remaining across Q2-Q4
        per_remaining_quarter = (
            remaining_after_q1 / Decimal("3")
        ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if remaining_after_q1 > 0 else Decimal("0")

        quarter_amounts = [
            q1_purchase,
            per_remaining_quarter,
            per_remaining_quarter,
            remaining_after_q1 - (per_remaining_quarter * 2),  # Remainder in Q4
        ]

        total_planned_cost = Decimal("0")
        for (date, purpose), amount in zip(quarter_dates, quarter_amounts):
            if amount > 0:
                cost = (amount * ets_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                total_planned_cost += cost
                purchase_plan.append(PurchasePlan(
                    purchase_date=date,
                    certificates_to_purchase=float(amount),
                    estimated_price_eur=float(ets_price),
                    estimated_cost_eur=float(cost),
                    purpose=purpose,
                ))

        context["purchase_plan"] = purchase_plan

        outputs["purchase_plan"] = [p.model_dump() for p in purchase_plan]
        outputs["total_planned_purchases"] = float(total_to_purchase)
        outputs["total_planned_cost_eur"] = float(total_planned_cost)
        outputs["quarterly_holding_requirement"] = float(quarterly_requirement)

        if float(total_planned_cost) > 50_000:
            warnings.append(
                f"Annual certificate cost estimated at EUR {float(total_planned_cost):,.2f}. "
                "Consider cost optimization through actual supplier data collection."
            )

        self.logger.info(
            "Phase 2 complete: %d purchase entries, total cost=%.2f EUR",
            len(purchase_plan), float(total_planned_cost),
        )

        provenance = self._hash({
            "phase": phase_name,
            "entries": len(purchase_plan),
            "cost": float(total_planned_cost),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Holding Compliance
    # -------------------------------------------------------------------------

    async def _phase_3_holding_compliance(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Verify quarterly 50% holding requirement is met.

        Per Article 22(2), by the end of each quarter, authorized CBAM
        declarants must hold at least 50% of their estimated annual
        certificate obligation.
        """
        phase_name = "holding_compliance"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_obligation = Decimal(str(context.get("annual_obligation", 0)))
        required_holding = (annual_obligation * Decimal("0.5")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )

        # Check holding for each quarter
        holding_checks: List[QuarterlyHoldingCheck] = []
        all_compliant = True

        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            # Fetch actual holding at end of quarter
            actual_holding = await self._fetch_quarterly_holding(
                context, quarter,
            )

            actual = Decimal(str(actual_holding))
            shortfall = max(Decimal("0"), required_holding - actual)

            if actual >= required_holding:
                status = HoldingComplianceStatus.COMPLIANT
            elif actual >= required_holding * Decimal("0.9"):
                status = HoldingComplianceStatus.APPROACHING_THRESHOLD
                warnings.append(
                    f"{quarter}: Holding at {float(actual):.4f} is approaching "
                    f"the {float(required_holding):.4f} requirement"
                )
            else:
                status = HoldingComplianceStatus.NON_COMPLIANT
                all_compliant = False
                warnings.append(
                    f"{quarter}: NON-COMPLIANT. Holding {float(actual):.4f} "
                    f"below required {float(required_holding):.4f} "
                    f"(shortfall: {float(shortfall):.4f})"
                )

            holding_checks.append(QuarterlyHoldingCheck(
                quarter=quarter,
                estimated_annual_obligation=float(annual_obligation),
                required_holding=float(required_holding),
                actual_holding=float(actual),
                status=status,
                shortfall=float(shortfall),
            ))

        context["holding_checks"] = holding_checks
        context["holding_compliant"] = all_compliant

        outputs["holding_checks"] = [hc.model_dump() for hc in holding_checks]
        outputs["overall_compliant"] = all_compliant
        outputs["required_holding_per_quarter"] = float(required_holding)

        self.logger.info(
            "Phase 3 complete: overall_compliant=%s, required=%.4f per quarter",
            all_compliant, float(required_holding),
        )

        provenance = self._hash({
            "phase": phase_name,
            "compliant": all_compliant,
            "required": float(required_holding),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Surrender Execution
    # -------------------------------------------------------------------------

    async def _phase_4_surrender_execution(
        self, context: Dict[str, Any]
    ) -> PhaseResult:
        """
        Execute annual certificate surrender by May 31 deadline.

        Validates that sufficient certificates are held, generates the
        surrender instruction, and tracks the surrender status.

        Per Article 22(1), all certificates must be surrendered by May 31
        of the year following the reporting year.
        """
        phase_name = "surrender_execution"
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_obligation = Decimal(str(context.get("annual_obligation", 0)))
        certificates_held = Decimal(str(context.get("certificates_held", 0)))
        surrender_deadline = context["surrender_deadline"]

        # Check deadline
        deadline_date = datetime.strptime(surrender_deadline, "%Y-%m-%d")
        days_until = (deadline_date - datetime.utcnow()).days

        # Determine surrender status
        shortfall = max(Decimal("0"), annual_obligation - certificates_held)

        if days_until < 0:
            surrender_status = SurrenderStatus.OVERDUE
            warnings.append(
                f"SURRENDER DEADLINE HAS PASSED ({surrender_deadline}). "
                f"Penalties may apply at EUR {PENALTY_EUR_PER_TCO2E}/tCO2e per Article 26."
            )
        elif float(shortfall) == 0:
            surrender_status = SurrenderStatus.READY
        else:
            surrender_status = SurrenderStatus.SHORTFALL
            warnings.append(
                f"Certificate shortfall: {float(shortfall):.4f} certificates needed. "
                f"Purchase before {surrender_deadline}."
            )

        # Generate surrender instruction
        surrender_instruction = {
            "declaration_year": context["year"],
            "certificates_to_surrender": float(annual_obligation),
            "certificates_available": float(certificates_held),
            "shortfall": float(shortfall),
            "surrender_status": surrender_status.value,
            "deadline": surrender_deadline,
            "days_until_deadline": days_until,
            "prepared_at": datetime.utcnow().isoformat(),
        }

        # Calculate potential penalties
        penalty = Decimal("0")
        if surrender_status in (SurrenderStatus.SHORTFALL, SurrenderStatus.OVERDUE):
            penalty = (shortfall * Decimal(str(PENALTY_EUR_PER_TCO2E))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if days_until < 0:
                warnings.append(
                    f"Potential penalty: EUR {float(penalty):,.2f} for {float(shortfall):.4f} "
                    f"unsurrendered certificates"
                )

        context["surrender_status"] = surrender_status
        context["potential_penalty_eur"] = float(penalty)

        outputs["surrender_instruction"] = surrender_instruction
        outputs["surrender_status"] = surrender_status.value
        outputs["shortfall"] = float(shortfall)
        outputs["days_until_deadline"] = days_until
        outputs["potential_penalty_eur"] = float(penalty)

        self.logger.info(
            "Phase 4 complete: status=%s shortfall=%.4f days_until=%d",
            surrender_status.value, float(shortfall), days_until,
        )

        provenance = self._hash({
            "phase": phase_name,
            "status": surrender_status.value,
            "shortfall": float(shortfall),
        })

        return PhaseResult(
            phase_name=phase_name,
            status=PhaseStatus.COMPLETED,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=provenance,
        )

    # =========================================================================
    # ASYNC STUBS
    # =========================================================================

    async def _fetch_certificate_balance(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch current certificate balance from CBAM registry."""
        await asyncio.sleep(0)
        return {"certificates_held": 0, "last_updated": datetime.utcnow().isoformat()}

    async def _fetch_quarterly_holding(
        self, context: Dict[str, Any], quarter: str
    ) -> float:
        """Fetch certificate holding at end of a specific quarter."""
        await asyncio.sleep(0)
        return 0.0

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _hash(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        return hashlib.sha256(str(data).encode("utf-8")).hexdigest()
