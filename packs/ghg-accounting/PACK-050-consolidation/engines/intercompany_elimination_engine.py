"""
PACK-050 GHG Consolidation Pack - Intercompany Elimination Engine
====================================================================

Eliminates double-counted emissions arising from intra-group
transactions during multi-entity GHG consolidation.  When one group
entity generates energy (Scope 1) and transfers it to another group
entity whose Scope 2 includes that same energy, the consolidated
inventory would otherwise double-count those emissions.  This engine
identifies, records, reconciles and eliminates such overlaps.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Consolidation
      requires removal of intra-group double-counting.
    - GHG Protocol Corporate Standard (Chapter 4): Setting
      operational boundaries - distinguishing direct and indirect
      emissions for inter-entity transfers.
    - ISO 14064-1:2018 (Clause 5.2.4): Quantification shall avoid
      double-counting of GHG emissions between entities within the
      consolidation boundary.
    - ESRS E1-6: Gross scope disclosures must exclude intra-group
      double-counting at the consolidated level.

Calculation Methodology:
    Elimination Amount:
        elimination = min(seller_scope1_from_transfer,
                          buyer_scope2_from_transfer)

    Net Consolidated Emissions:
        net_consolidated = sum(entity_emissions) - sum(eliminations)

    Transfer Reconciliation Variance:
        variance = seller_recorded - buyer_recorded
        (should be zero for a fully reconciled transfer)

    Partial Elimination:
        When only a portion of a transfer is intra-group:
        elimination = total_transfer_emissions * intra_group_pct / 100

Capabilities:
    - Register intra-group energy transfers (electricity, steam,
      heat, cooling) between group entities
    - Eliminate double-counted emissions at the consolidated level
    - Handle waste transfers between group entities
    - Handle product/service transfers (embedded emissions)
    - Maintain a transfer register of all intra-group flows
    - Reconcile elimination amounts (seller vs buyer records)
    - Support partial eliminations for mixed intra/extra-group flows

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TransferType(str, Enum):
    """Types of intra-group transfers that may cause double-counting."""
    ELECTRICITY = "ELECTRICITY"
    STEAM = "STEAM"
    HEAT = "HEAT"
    COOLING = "COOLING"
    WASTE = "WASTE"
    PRODUCT = "PRODUCT"
    SERVICE = "SERVICE"
    TRANSPORT = "TRANSPORT"
    OTHER = "OTHER"

class EliminationScope(str, Enum):
    """Emission scope from which the elimination is deducted."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"

class ReconciliationStatus(str, Enum):
    """Reconciliation outcome between seller and buyer records."""
    RECONCILED = "RECONCILED"
    VARIANCE = "VARIANCE"
    UNRECONCILED = "UNRECONCILED"
    PENDING = "PENDING"

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_RECONCILIATION_TOLERANCE_PCT = Decimal("2")
DEFAULT_TRANSFER_TYPE_SCOPE_MAP: Dict[str, str] = {
    TransferType.ELECTRICITY.value: EliminationScope.SCOPE_2_LOCATION.value,
    TransferType.STEAM.value: EliminationScope.SCOPE_2_LOCATION.value,
    TransferType.HEAT.value: EliminationScope.SCOPE_2_LOCATION.value,
    TransferType.COOLING.value: EliminationScope.SCOPE_2_LOCATION.value,
    TransferType.WASTE.value: EliminationScope.SCOPE_3.value,
    TransferType.PRODUCT.value: EliminationScope.SCOPE_3.value,
    TransferType.SERVICE.value: EliminationScope.SCOPE_3.value,
    TransferType.TRANSPORT.value: EliminationScope.SCOPE_3.value,
    TransferType.OTHER.value: EliminationScope.SCOPE_3.value,
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TransferRecord(BaseModel):
    """A single intra-group transfer between two entities.

    Records the seller (source) and buyer (destination) entity,
    the type and quantity of the transfer, and the emissions
    amounts reported by each side.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    transfer_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this transfer.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for this transfer.",
    )
    seller_entity_id: str = Field(
        ...,
        description="Entity that generates / sells the energy or product.",
    )
    buyer_entity_id: str = Field(
        ...,
        description="Entity that receives / purchases the energy or product.",
    )
    transfer_type: str = Field(
        ...,
        description="Type of intra-group transfer.",
    )
    quantity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Physical quantity transferred (e.g. MWh, tonnes).",
    )
    quantity_unit: str = Field(
        default="MWh",
        description="Unit of the quantity field.",
    )
    seller_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions reported by seller for this transfer (tCO2e).",
    )
    buyer_emissions_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions reported by buyer for this transfer (tCO2e).",
    )
    seller_scope: str = Field(
        default=EliminationScope.SCOPE_1.value,
        description="Scope under which seller reports these emissions.",
    )
    buyer_scope: str = Field(
        default=EliminationScope.SCOPE_2_LOCATION.value,
        description="Scope under which buyer reports these emissions.",
    )
    intra_group_pct: Decimal = Field(
        default=Decimal("100"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of transfer that is intra-group (for partial).",
    )
    description: Optional[str] = Field(
        None,
        description="Description of the transfer.",
    )
    evidence_reference: Optional[str] = Field(
        None,
        description="Reference to supporting documentation.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When the transfer was registered.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "quantity", "seller_emissions_tco2e", "buyer_emissions_tco2e",
        "intra_group_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("transfer_type")
    @classmethod
    def _validate_transfer_type(cls, v: str) -> str:
        valid = {tt.value for tt in TransferType}
        if v.upper() not in valid:
            logger.warning("Transfer type '%s' not standard; accepted.", v)
        return v.upper()

class EliminationEntry(BaseModel):
    """A single double-counting elimination entry.

    Records the amount eliminated from the buyer entity's scope
    to prevent double-counting at the consolidated level.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    elimination_id: str = Field(
        default_factory=_new_uuid,
        description="Unique elimination identifier.",
    )
    transfer_id: str = Field(
        ...,
        description="Transfer that triggered this elimination.",
    )
    seller_entity_id: str = Field(
        ...,
        description="Selling entity.",
    )
    buyer_entity_id: str = Field(
        ...,
        description="Buying entity.",
    )
    transfer_type: str = Field(
        ...,
        description="Type of transfer.",
    )
    elimination_amount_tco2e: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Emissions eliminated (tCO2e).",
    )
    eliminated_from_scope: str = Field(
        ...,
        description="Scope from which emissions are eliminated.",
    )
    is_partial: bool = Field(
        default=False,
        description="Whether this is a partial elimination.",
    )
    intra_group_pct: Decimal = Field(
        default=Decimal("100"),
        description="Intra-group percentage applied.",
    )
    rationale: str = Field(
        default="",
        description="Explanation for the elimination.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When the elimination was calculated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "elimination_amount_tco2e", "intra_group_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

class TransferReconciliation(BaseModel):
    """Reconciliation result for a single transfer.

    Compares the emissions amount reported by the seller against
    the amount reported by the buyer.  A reconciled transfer has
    zero or near-zero variance.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    transfer_id: str = Field(
        ...,
        description="The reconciled transfer.",
    )
    seller_entity_id: str = Field(
        ...,
        description="Selling entity.",
    )
    buyer_entity_id: str = Field(
        ...,
        description="Buying entity.",
    )
    seller_amount: Decimal = Field(
        ...,
        description="Seller-reported emissions (tCO2e).",
    )
    buyer_amount: Decimal = Field(
        ...,
        description="Buyer-reported emissions (tCO2e).",
    )
    variance: Decimal = Field(
        default=Decimal("0"),
        description="Absolute variance (seller - buyer).",
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"),
        description="Variance as percentage of buyer amount.",
    )
    status: str = Field(
        default=ReconciliationStatus.PENDING.value,
        description="Reconciliation status.",
    )
    tolerance_pct: Decimal = Field(
        default=DEFAULT_RECONCILIATION_TOLERANCE_PCT,
        description="Tolerance threshold used.",
    )
    notes: Optional[str] = Field(
        None,
        description="Reconciliation notes.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "seller_amount", "buyer_amount", "variance",
        "variance_pct", "tolerance_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

class EliminationResult(BaseModel):
    """Aggregated result of all eliminations for a reporting period.

    Summarises the total eliminations applied, the net consolidated
    emissions, and per-scope breakdowns.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year.",
    )
    total_entity_emissions: Decimal = Field(
        default=Decimal("0"),
        description="Sum of all entity emissions before eliminations.",
    )
    total_eliminations: Decimal = Field(
        default=Decimal("0"),
        description="Total emissions eliminated (tCO2e).",
    )
    net_consolidated: Decimal = Field(
        default=Decimal("0"),
        description="Net consolidated emissions after eliminations.",
    )
    eliminations_by_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Eliminations broken down by transfer type.",
    )
    eliminations_by_scope: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Eliminations broken down by scope eliminated.",
    )
    elimination_entries: List[EliminationEntry] = Field(
        default_factory=list,
        description="All individual elimination entries.",
    )
    transfer_count: int = Field(
        default=0,
        description="Number of transfers processed.",
    )
    elimination_count: int = Field(
        default=0,
        description="Number of eliminations applied.",
    )
    reconciliation_results: List[TransferReconciliation] = Field(
        default_factory=list,
        description="Reconciliation results for each transfer.",
    )
    unreconciled_count: int = Field(
        default=0,
        description="Number of transfers with variance outside tolerance.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this result was generated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "total_entity_emissions", "total_eliminations",
        "net_consolidated", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IntercompanyEliminationEngine:
    """Eliminates double-counted emissions from intra-group transfers.

    Implements the full intercompany elimination pipeline: transfer
    registration, reconciliation, elimination calculation, and
    net consolidated total computation.

    Attributes:
        _transfers: Dict mapping transfer_id to TransferRecord.
        _eliminations: Dict mapping elimination_id to EliminationEntry.
        _results: Dict mapping result_id to EliminationResult.

    Example:
        >>> engine = IntercompanyEliminationEngine()
        >>> transfer = engine.register_transfer({
        ...     "reporting_year": 2025,
        ...     "seller_entity_id": "ENT-A",
        ...     "buyer_entity_id": "ENT-B",
        ...     "transfer_type": "ELECTRICITY",
        ...     "quantity": "50000",
        ...     "seller_emissions_tco2e": "12500",
        ...     "buyer_emissions_tco2e": "12500",
        ... })
        >>> result = engine.calculate_eliminations(
        ...     reporting_year=2025,
        ...     entity_emissions={"ENT-A": Decimal("30000"),
        ...                       "ENT-B": Decimal("20000")},
        ... )
        >>> assert result.net_consolidated < result.total_entity_emissions
    """

    def __init__(self) -> None:
        """Initialise the IntercompanyEliminationEngine."""
        self._transfers: Dict[str, TransferRecord] = {}
        self._eliminations: Dict[str, EliminationEntry] = {}
        self._results: Dict[str, EliminationResult] = {}
        logger.info(
            "IntercompanyEliminationEngine v%s initialised.", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Transfer Registration
    # ------------------------------------------------------------------

    def register_transfer(
        self,
        transfer_data: Dict[str, Any],
    ) -> TransferRecord:
        """Register an intra-group transfer in the transfer register.

        Validates the transfer data, assigns a unique transfer_id if
        not provided, and stores the record.

        Args:
            transfer_data: Dictionary of transfer attributes. Must
                include: reporting_year, seller_entity_id,
                buyer_entity_id, transfer_type, quantity,
                seller_emissions_tco2e, buyer_emissions_tco2e.

        Returns:
            The created TransferRecord with provenance hash.

        Raises:
            ValueError: If required fields are missing or invalid.
            ValueError: If seller and buyer are the same entity.
        """
        logger.info(
            "Registering transfer: %s -> %s (%s).",
            transfer_data.get("seller_entity_id", "?"),
            transfer_data.get("buyer_entity_id", "?"),
            transfer_data.get("transfer_type", "?"),
        )

        seller = transfer_data.get("seller_entity_id", "")
        buyer = transfer_data.get("buyer_entity_id", "")
        if seller == buyer:
            raise ValueError(
                f"Seller and buyer cannot be the same entity: '{seller}'."
            )

        if "transfer_id" not in transfer_data or not transfer_data["transfer_id"]:
            transfer_data["transfer_id"] = _new_uuid()

        # Apply default scope mapping based on transfer type
        ttype = transfer_data.get("transfer_type", "OTHER").upper()
        if "buyer_scope" not in transfer_data:
            transfer_data["buyer_scope"] = DEFAULT_TRANSFER_TYPE_SCOPE_MAP.get(
                ttype, EliminationScope.SCOPE_2_LOCATION.value
            )

        transfer = TransferRecord(**transfer_data)
        transfer.provenance_hash = _compute_hash(transfer)
        self._transfers[transfer.transfer_id] = transfer

        logger.info(
            "Transfer '%s' registered: %s -> %s, %s %s, "
            "seller=%s tCO2e, buyer=%s tCO2e.",
            transfer.transfer_id,
            transfer.seller_entity_id,
            transfer.buyer_entity_id,
            transfer.quantity,
            transfer.quantity_unit,
            transfer.seller_emissions_tco2e,
            transfer.buyer_emissions_tco2e,
        )
        return transfer

    def register_transfers_batch(
        self,
        transfers: List[Dict[str, Any]],
    ) -> List[TransferRecord]:
        """Register multiple transfers in batch.

        Args:
            transfers: List of transfer data dictionaries.

        Returns:
            List of successfully registered TransferRecords.
        """
        logger.info("Registering %d transfer(s) in batch.", len(transfers))
        results: List[TransferRecord] = []
        for i, td in enumerate(transfers):
            try:
                record = self.register_transfer(td)
                results.append(record)
            except (ValueError, TypeError) as exc:
                logger.error(
                    "Failed to register transfer %d: %s.", i, exc
                )
        logger.info(
            "Batch complete: %d of %d transfer(s) registered.",
            len(results), len(transfers),
        )
        return results

    # ------------------------------------------------------------------
    # Elimination Calculation
    # ------------------------------------------------------------------

    def calculate_eliminations(
        self,
        reporting_year: int,
        entity_emissions: Dict[str, Union[Decimal, str, int, float]],
        tolerance_pct: Optional[Union[Decimal, str, int, float]] = None,
    ) -> EliminationResult:
        """Calculate all eliminations for a reporting year.

        For each registered transfer in the year:
        1. Reconcile seller vs buyer emissions.
        2. Calculate the elimination amount as
           min(seller_emissions, buyer_emissions) * intra_group_pct / 100.
        3. Aggregate all eliminations.
        4. Compute net consolidated = sum(entity_emissions) - sum(eliminations).

        Args:
            reporting_year: The year to process.
            entity_emissions: Dict mapping entity_id to total emissions
                (tCO2e) before eliminations.
            tolerance_pct: Reconciliation tolerance percentage.

        Returns:
            EliminationResult with all eliminations and net total.
        """
        logger.info(
            "Calculating eliminations for year %d with %d entity(ies).",
            reporting_year, len(entity_emissions),
        )

        tol = _decimal(
            tolerance_pct if tolerance_pct is not None
            else DEFAULT_RECONCILIATION_TOLERANCE_PCT
        )

        # Filter transfers for this year
        year_transfers = [
            t for t in self._transfers.values()
            if t.reporting_year == reporting_year
        ]
        logger.info(
            "Found %d transfer(s) for year %d.", len(year_transfers), reporting_year
        )

        # Process each transfer
        elimination_entries: List[EliminationEntry] = []
        reconciliation_results: List[TransferReconciliation] = []
        unreconciled_count = 0

        for transfer in year_transfers:
            # Step 1: Reconcile
            recon = self._reconcile_single_transfer(transfer, tol)
            reconciliation_results.append(recon)
            if recon.status == ReconciliationStatus.VARIANCE.value:
                unreconciled_count += 1

            # Step 2: Calculate elimination
            elimination = self._calculate_single_elimination(transfer)
            elimination.provenance_hash = _compute_hash(elimination)
            elimination_entries.append(elimination)
            self._eliminations[elimination.elimination_id] = elimination

        # Step 3: Aggregate
        total_entity = sum(
            (_decimal(v) for v in entity_emissions.values()), Decimal("0")
        )
        total_eliminated = sum(
            (e.elimination_amount_tco2e for e in elimination_entries),
            Decimal("0"),
        )
        net_consolidated = _round2(total_entity - total_eliminated)

        # Breakdowns
        by_type: Dict[str, Decimal] = {}
        by_scope: Dict[str, Decimal] = {}
        for entry in elimination_entries:
            by_type[entry.transfer_type] = (
                by_type.get(entry.transfer_type, Decimal("0"))
                + entry.elimination_amount_tco2e
            )
            by_scope[entry.eliminated_from_scope] = (
                by_scope.get(entry.eliminated_from_scope, Decimal("0"))
                + entry.elimination_amount_tco2e
            )

        for k in by_type:
            by_type[k] = _round2(by_type[k])
        for k in by_scope:
            by_scope[k] = _round2(by_scope[k])

        result = EliminationResult(
            reporting_year=reporting_year,
            total_entity_emissions=_round2(total_entity),
            total_eliminations=_round2(total_eliminated),
            net_consolidated=net_consolidated,
            eliminations_by_type=by_type,
            eliminations_by_scope=by_scope,
            elimination_entries=elimination_entries,
            transfer_count=len(year_transfers),
            elimination_count=len(elimination_entries),
            reconciliation_results=reconciliation_results,
            unreconciled_count=unreconciled_count,
        )
        result.provenance_hash = _compute_hash(result)
        self._results[result.result_id] = result

        logger.info(
            "Eliminations complete: total_entity=%s, eliminated=%s, "
            "net=%s tCO2e, %d elimination(s), %d unreconciled.",
            result.total_entity_emissions,
            result.total_eliminations,
            result.net_consolidated,
            len(elimination_entries),
            unreconciled_count,
        )
        return result

    def _calculate_single_elimination(
        self,
        transfer: TransferRecord,
    ) -> EliminationEntry:
        """Calculate elimination for a single transfer.

        The elimination amount is the minimum of the seller's and
        buyer's reported emissions, scaled by the intra-group
        percentage.

        Args:
            transfer: The transfer to eliminate.

        Returns:
            EliminationEntry for this transfer.
        """
        seller_amt = transfer.seller_emissions_tco2e
        buyer_amt = transfer.buyer_emissions_tco2e
        base_elimination = min(seller_amt, buyer_amt)

        # Apply intra-group percentage for partial eliminations
        ig_pct = transfer.intra_group_pct
        is_partial = ig_pct < Decimal("100")

        if is_partial:
            elimination_amt = _round2(
                base_elimination * ig_pct / Decimal("100")
            )
        else:
            elimination_amt = _round2(base_elimination)

        rationale = (
            f"Eliminate min(seller={seller_amt}, buyer={buyer_amt})"
        )
        if is_partial:
            rationale += f" * {ig_pct}% intra-group"

        return EliminationEntry(
            transfer_id=transfer.transfer_id,
            seller_entity_id=transfer.seller_entity_id,
            buyer_entity_id=transfer.buyer_entity_id,
            transfer_type=transfer.transfer_type,
            elimination_amount_tco2e=elimination_amt,
            eliminated_from_scope=transfer.buyer_scope,
            is_partial=is_partial,
            intra_group_pct=ig_pct,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile_transfers(
        self,
        reporting_year: int,
        tolerance_pct: Optional[Union[Decimal, str, int, float]] = None,
    ) -> List[TransferReconciliation]:
        """Reconcile all transfers for a reporting year.

        Compares seller-reported emissions against buyer-reported
        emissions for each transfer.  Flags variances outside the
        tolerance threshold.

        Args:
            reporting_year: The year to reconcile.
            tolerance_pct: Acceptable variance percentage.

        Returns:
            List of TransferReconciliation results.
        """
        logger.info(
            "Reconciling transfers for year %d.", reporting_year
        )
        tol = _decimal(
            tolerance_pct if tolerance_pct is not None
            else DEFAULT_RECONCILIATION_TOLERANCE_PCT
        )

        year_transfers = [
            t for t in self._transfers.values()
            if t.reporting_year == reporting_year
        ]

        results: List[TransferReconciliation] = []
        for transfer in year_transfers:
            recon = self._reconcile_single_transfer(transfer, tol)
            results.append(recon)

        reconciled = sum(
            1 for r in results
            if r.status == ReconciliationStatus.RECONCILED.value
        )
        logger.info(
            "Reconciliation complete: %d of %d transfer(s) reconciled.",
            reconciled, len(results),
        )
        return results

    def _reconcile_single_transfer(
        self,
        transfer: TransferRecord,
        tolerance_pct: Decimal,
    ) -> TransferReconciliation:
        """Reconcile a single transfer's seller vs buyer amounts.

        Args:
            transfer: The transfer to reconcile.
            tolerance_pct: Acceptable variance percentage.

        Returns:
            TransferReconciliation with variance analysis.
        """
        seller_amt = transfer.seller_emissions_tco2e
        buyer_amt = transfer.buyer_emissions_tco2e

        variance = _round2(seller_amt - buyer_amt)
        variance_pct = _round2(
            _safe_divide(
                abs(variance), buyer_amt
            ) * Decimal("100")
        ) if buyer_amt != Decimal("0") else Decimal("0")

        if abs(variance_pct) <= tolerance_pct:
            status = ReconciliationStatus.RECONCILED.value
        else:
            status = ReconciliationStatus.VARIANCE.value

        recon = TransferReconciliation(
            transfer_id=transfer.transfer_id,
            seller_entity_id=transfer.seller_entity_id,
            buyer_entity_id=transfer.buyer_entity_id,
            seller_amount=seller_amt,
            buyer_amount=buyer_amt,
            variance=variance,
            variance_pct=variance_pct,
            status=status,
            tolerance_pct=tolerance_pct,
        )
        recon.provenance_hash = _compute_hash(recon)
        return recon

    # ------------------------------------------------------------------
    # Net Consolidated Calculation
    # ------------------------------------------------------------------

    def get_net_consolidated(
        self,
        reporting_year: int,
        entity_emissions: Dict[str, Union[Decimal, str, int, float]],
    ) -> Dict[str, Any]:
        """Compute net consolidated emissions after eliminations.

        If eliminations have already been calculated for the year,
        uses the cached result.  Otherwise, calculates on the fly.

        Args:
            reporting_year: The year to compute.
            entity_emissions: Dict mapping entity_id to total emissions.

        Returns:
            Dictionary with entity totals, eliminations, and net.
        """
        # Check for existing result
        existing = [
            r for r in self._results.values()
            if r.reporting_year == reporting_year
        ]

        if existing:
            result = existing[-1]
        else:
            result = self.calculate_eliminations(
                reporting_year=reporting_year,
                entity_emissions=entity_emissions,
            )

        total_entity = sum(
            (_decimal(v) for v in entity_emissions.values()), Decimal("0")
        )

        summary = {
            "reporting_year": reporting_year,
            "entity_count": len(entity_emissions),
            "total_entity_emissions": str(_round2(total_entity)),
            "total_eliminations": str(result.total_eliminations),
            "net_consolidated": str(result.net_consolidated),
            "elimination_count": result.elimination_count,
            "unreconciled_count": result.unreconciled_count,
        }
        summary["provenance_hash"] = _compute_hash(summary)

        logger.info(
            "Net consolidated for year %d: %s tCO2e.",
            reporting_year, result.net_consolidated,
        )
        return summary

    # ------------------------------------------------------------------
    # Elimination Log and Accessors
    # ------------------------------------------------------------------

    def get_elimination_log(
        self,
        reporting_year: Optional[int] = None,
        entity_id: Optional[str] = None,
    ) -> List[EliminationEntry]:
        """Retrieve the elimination log, optionally filtered.

        Args:
            reporting_year: Filter by year (via transfer lookup).
            entity_id: Filter by seller or buyer entity.

        Returns:
            List of EliminationEntry records.
        """
        entries = list(self._eliminations.values())

        if reporting_year is not None:
            year_transfer_ids = {
                t.transfer_id for t in self._transfers.values()
                if t.reporting_year == reporting_year
            }
            entries = [
                e for e in entries
                if e.transfer_id in year_transfer_ids
            ]

        if entity_id is not None:
            entries = [
                e for e in entries
                if e.seller_entity_id == entity_id
                or e.buyer_entity_id == entity_id
            ]

        logger.info("Elimination log: %d entries returned.", len(entries))
        return entries

    def get_transfer(self, transfer_id: str) -> TransferRecord:
        """Retrieve a transfer by ID.

        Args:
            transfer_id: The transfer ID.

        Returns:
            The TransferRecord.

        Raises:
            KeyError: If not found.
        """
        if transfer_id not in self._transfers:
            raise KeyError(f"Transfer '{transfer_id}' not found.")
        return self._transfers[transfer_id]

    def get_transfers_for_year(
        self,
        reporting_year: int,
    ) -> List[TransferRecord]:
        """Get all transfers for a reporting year.

        Args:
            reporting_year: The year to query.

        Returns:
            List of TransferRecords.
        """
        return [
            t for t in self._transfers.values()
            if t.reporting_year == reporting_year
        ]

    def get_transfers_for_entity(
        self,
        entity_id: str,
    ) -> List[TransferRecord]:
        """Get all transfers involving an entity.

        Args:
            entity_id: The entity to query.

        Returns:
            List of TransferRecords where entity is seller or buyer.
        """
        return [
            t for t in self._transfers.values()
            if t.seller_entity_id == entity_id
            or t.buyer_entity_id == entity_id
        ]

    def get_result(self, result_id: str) -> EliminationResult:
        """Retrieve an elimination result by ID.

        Args:
            result_id: The result ID.

        Returns:
            The EliminationResult.

        Raises:
            KeyError: If not found.
        """
        if result_id not in self._results:
            raise KeyError(f"Elimination result '{result_id}' not found.")
        return self._results[result_id]

    def get_all_results(self) -> List[EliminationResult]:
        """Return all elimination results.

        Returns:
            List of all EliminationResults.
        """
        return list(self._results.values())

    def get_transfer_summary(
        self,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Get a summary of transfers for a year.

        Args:
            reporting_year: The year to summarise.

        Returns:
            Dictionary with transfer count, total quantities, and
            emissions by transfer type.
        """
        transfers = self.get_transfers_for_year(reporting_year)
        by_type: Dict[str, Dict[str, Decimal]] = {}

        for t in transfers:
            if t.transfer_type not in by_type:
                by_type[t.transfer_type] = {
                    "count": Decimal("0"),
                    "total_quantity": Decimal("0"),
                    "total_seller_emissions": Decimal("0"),
                    "total_buyer_emissions": Decimal("0"),
                }
            by_type[t.transfer_type]["count"] += Decimal("1")
            by_type[t.transfer_type]["total_quantity"] += t.quantity
            by_type[t.transfer_type]["total_seller_emissions"] += (
                t.seller_emissions_tco2e
            )
            by_type[t.transfer_type]["total_buyer_emissions"] += (
                t.buyer_emissions_tco2e
            )

        # Convert to serialisable
        serialised: Dict[str, Any] = {}
        for ttype, vals in by_type.items():
            serialised[ttype] = {
                k: str(_round2(v)) for k, v in vals.items()
            }

        summary = {
            "reporting_year": reporting_year,
            "total_transfers": len(transfers),
            "by_type": serialised,
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
