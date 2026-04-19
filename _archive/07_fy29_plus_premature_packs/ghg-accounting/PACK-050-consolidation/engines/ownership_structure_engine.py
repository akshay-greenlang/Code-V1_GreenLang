"""
PACK-050 GHG Consolidation Pack - Ownership Structure Engine
====================================================================

Resolves equity chains, computes effective ownership percentages
through multi-tier holding structures, assesses control type
(operational vs financial), identifies minority interests, and
tracks ownership change history for GHG consolidation.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Equity share
      approach requires calculation of effective ownership through
      complex holding structures.
    - IAS 28 / IFRS 11: Defines associate (20-50%) and joint
      venture relationships for equity method.
    - IFRS 10: Defines control (operational and financial) for
      consolidation purposes.
    - GHG Protocol Corporate Standard (Chapter 5): Ownership
      changes trigger base year recalculation.

Calculation Methodology:
    Effective Ownership:
        If A owns X% of B, and B owns Y% of C, then:
        A's effective ownership of C = X/100 * Y/100 * 100
        (product of ownership percentages through the chain)

    Control Assessment:
        operational_control: True if the reporting entity has
            authority to introduce and implement operating policies.
        financial_control: True if the reporting entity has
            ability to direct financial and operating policies
            with a view to gaining economic benefits.

    Minority Interest:
        minority = ownership_pct < 50 AND may still have control
        (e.g. via shareholder agreements, board seats, veto rights)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  2 of 5
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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

class ControlType(str, Enum):
    """Types of control under GHG Protocol."""
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    NO_CONTROL = "NO_CONTROL"
    JOINT_CONTROL = "JOINT_CONTROL"

class OwnershipCategory(str, Enum):
    """Classification of ownership stake."""
    WHOLLY_OWNED = "WHOLLY_OWNED"
    MAJORITY = "MAJORITY"
    JOINT_VENTURE = "JOINT_VENTURE"
    ASSOCIATE = "ASSOCIATE"
    MINORITY = "MINORITY"
    NO_STAKE = "NO_STAKE"

class ChangeReason(str, Enum):
    """Reasons for ownership changes."""
    ACQUISITION = "ACQUISITION"
    PARTIAL_ACQUISITION = "PARTIAL_ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    PARTIAL_DIVESTITURE = "PARTIAL_DIVESTITURE"
    MERGER = "MERGER"
    SHARE_ISSUANCE = "SHARE_ISSUANCE"
    SHARE_BUYBACK = "SHARE_BUYBACK"
    RESTRUCTURING = "RESTRUCTURING"
    OTHER = "OTHER"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class OwnershipRecord(BaseModel):
    """Records a direct ownership link between two entities.

    Represents that owner_entity_id holds ownership_pct percent
    of target_entity_id, along with control indicators.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    record_id: str = Field(
        default_factory=_new_uuid,
        description="Unique record identifier.",
    )
    owner_entity_id: str = Field(
        ...,
        description="ID of the owning entity.",
    )
    target_entity_id: str = Field(
        ...,
        description="ID of the entity being owned.",
    )
    ownership_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Direct ownership percentage (0-100).",
    )
    has_operational_control: bool = Field(
        default=False,
        description="Whether owner has operational control of target.",
    )
    has_financial_control: bool = Field(
        default=False,
        description="Whether owner has financial control of target.",
    )
    manages_operations: bool = Field(
        default=False,
        description="Whether owner manages day-to-day operations.",
    )
    directs_policies: bool = Field(
        default=False,
        description="Whether owner directs financial/operating policies.",
    )
    has_board_majority: bool = Field(
        default=False,
        description="Whether owner holds majority of board seats.",
    )
    has_veto_rights: bool = Field(
        default=False,
        description="Whether owner has veto rights on key decisions.",
    )
    effective_from: Optional[date] = Field(
        None,
        description="Date from which this ownership is effective.",
    )
    effective_to: Optional[date] = Field(
        None,
        description="Date until which this ownership is effective.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes on the ownership relationship.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this record was created.",
    )

    @field_validator("ownership_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

class EquityChain(BaseModel):
    """Represents a resolved equity chain from reporting entity to target.

    Contains each link in the chain and the computed effective
    ownership percentage.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    chain_id: str = Field(
        default_factory=_new_uuid,
        description="Unique chain identifier.",
    )
    reporting_entity_id: str = Field(
        ...,
        description="The top-level reporting entity.",
    )
    target_entity_id: str = Field(
        ...,
        description="The entity at the end of the chain.",
    )
    chain_links: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of chain links with entity_id and pct.",
    )
    effective_ownership_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Computed effective ownership percentage.",
    )
    chain_depth: int = Field(
        default=0,
        description="Number of links in the chain.",
    )
    ownership_category: str = Field(
        default="NO_STAKE",
        description="Classification of effective ownership.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this chain was resolved.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator("effective_ownership_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

class ControlAssessment(BaseModel):
    """Result of assessing control type for an entity.

    Determines whether the reporting entity has operational
    control, financial control, or no control over the target.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    assessment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique assessment identifier.",
    )
    reporting_entity_id: str = Field(
        ...,
        description="The reporting entity.",
    )
    target_entity_id: str = Field(
        ...,
        description="The entity being assessed.",
    )
    control_type: str = Field(
        ...,
        description="Determined control type.",
    )
    ownership_pct: Decimal = Field(
        ...,
        description="Direct or effective ownership percentage.",
    )
    has_operational_control: bool = Field(
        default=False,
        description="Whether operational control exists.",
    )
    has_financial_control: bool = Field(
        default=False,
        description="Whether financial control exists.",
    )
    assessment_basis: List[str] = Field(
        default_factory=list,
        description="Basis for the control determination.",
    )
    inclusion_pct_equity: Decimal = Field(
        default=Decimal("0"),
        description="Inclusion percentage under equity share.",
    )
    inclusion_pct_operational: Decimal = Field(
        default=Decimal("0"),
        description="Inclusion percentage under operational control.",
    )
    inclusion_pct_financial: Decimal = Field(
        default=Decimal("0"),
        description="Inclusion percentage under financial control.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this assessment was performed.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator("ownership_pct", "inclusion_pct_equity",
                     "inclusion_pct_operational", "inclusion_pct_financial",
                     mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

class OwnershipChange(BaseModel):
    """Records a change in ownership percentage over time."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    change_id: str = Field(
        default_factory=_new_uuid,
        description="Unique change identifier.",
    )
    owner_entity_id: str = Field(
        ...,
        description="The owning entity.",
    )
    target_entity_id: str = Field(
        ...,
        description="The target entity.",
    )
    previous_pct: Decimal = Field(
        ...,
        description="Ownership percentage before change.",
    )
    new_pct: Decimal = Field(
        ...,
        description="Ownership percentage after change.",
    )
    change_reason: str = Field(
        ...,
        description="Reason for the ownership change.",
    )
    effective_date: date = Field(
        ...,
        description="Date the change takes effect.",
    )
    description: Optional[str] = Field(
        None,
        description="Additional description.",
    )
    triggers_base_year_restatement: bool = Field(
        default=False,
        description="Whether this change triggers base year restatement.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this change was recorded.",
    )

    @field_validator("previous_pct", "new_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("change_reason")
    @classmethod
    def _validate_reason(cls, v: str) -> str:
        valid = {cr.value for cr in ChangeReason}
        if v.upper() not in valid:
            logger.warning("Change reason '%s' not standard; accepted.", v)
        return v.upper()

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

# Ownership classification thresholds
OWNERSHIP_THRESHOLDS: Dict[str, Tuple[Decimal, Decimal]] = {
    OwnershipCategory.WHOLLY_OWNED.value: (Decimal("100"), Decimal("100")),
    OwnershipCategory.MAJORITY.value: (Decimal("50"), Decimal("99.99")),
    OwnershipCategory.JOINT_VENTURE.value: (Decimal("50"), Decimal("50")),
    OwnershipCategory.ASSOCIATE.value: (Decimal("20"), Decimal("49.99")),
    OwnershipCategory.MINORITY.value: (Decimal("0.01"), Decimal("19.99")),
}

def _classify_ownership(pct: Decimal) -> str:
    """Classify ownership percentage into a standard category.

    Args:
        pct: Ownership percentage.

    Returns:
        String classification from OwnershipCategory.
    """
    if pct >= Decimal("100"):
        return OwnershipCategory.WHOLLY_OWNED.value
    elif pct > Decimal("50"):
        return OwnershipCategory.MAJORITY.value
    elif pct == Decimal("50"):
        return OwnershipCategory.JOINT_VENTURE.value
    elif pct >= Decimal("20"):
        return OwnershipCategory.ASSOCIATE.value
    elif pct > Decimal("0"):
        return OwnershipCategory.MINORITY.value
    return OwnershipCategory.NO_STAKE.value

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OwnershipStructureEngine:
    """Resolves equity chains and assesses control for GHG consolidation.

    Tracks ownership links between entities, resolves multi-tier
    effective ownership, assesses operational and financial control,
    identifies minority interests, and maintains a change history.

    Attributes:
        _records: Dict mapping record_id to OwnershipRecord.
        _changes: List of OwnershipChange records.
        _change_log: Append-only audit log.

    Example:
        >>> engine = OwnershipStructureEngine()
        >>> engine.set_ownership({
        ...     "owner_entity_id": "parent-01",
        ...     "target_entity_id": "sub-01",
        ...     "ownership_pct": "60",
        ...     "has_operational_control": True,
        ... })
        >>> chain = engine.resolve_equity_chain("parent-01", "sub-01")
        >>> assert chain.effective_ownership_pct == Decimal("60")
    """

    def __init__(self) -> None:
        """Initialise the OwnershipStructureEngine with empty state."""
        self._records: Dict[str, OwnershipRecord] = {}
        self._ownership_index: Dict[str, Dict[str, OwnershipRecord]] = {}
        self._changes: List[OwnershipChange] = []
        self._change_log: List[Dict[str, Any]] = []
        logger.info("OwnershipStructureEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Ownership CRUD
    # ------------------------------------------------------------------

    def set_ownership(
        self, ownership_data: Dict[str, Any],
    ) -> OwnershipRecord:
        """Set or update an ownership link between two entities.

        If an existing link between the same owner and target exists,
        it is replaced and the change is logged.

        Args:
            ownership_data: Dictionary with ownership attributes.
                Required: owner_entity_id, target_entity_id, ownership_pct.

        Returns:
            The created or updated OwnershipRecord.

        Raises:
            ValueError: If self-ownership is attempted.
        """
        owner_id = ownership_data.get("owner_entity_id", "")
        target_id = ownership_data.get("target_entity_id", "")

        if owner_id == target_id:
            raise ValueError("An entity cannot own itself.")

        logger.info(
            "Setting ownership: '%s' -> '%s' at %s%%.",
            owner_id, target_id, ownership_data.get("ownership_pct", "N/A"),
        )

        # Check for existing link
        existing = self._find_record(owner_id, target_id)
        if existing is not None:
            previous_pct = existing.ownership_pct
            new_pct = _decimal(ownership_data.get("ownership_pct", "0"))

            # Log the change
            change = OwnershipChange(
                owner_entity_id=owner_id,
                target_entity_id=target_id,
                previous_pct=previous_pct,
                new_pct=new_pct,
                change_reason=ownership_data.get(
                    "change_reason", ChangeReason.OTHER.value
                ),
                effective_date=ownership_data.get(
                    "effective_date", date.today()
                ),
                description=ownership_data.get("description"),
            )
            self._changes.append(change)

            # Remove old record
            del self._records[existing.record_id]

        record = OwnershipRecord(**ownership_data)
        self._records[record.record_id] = record

        # Update index
        if owner_id not in self._ownership_index:
            self._ownership_index[owner_id] = {}
        self._ownership_index[owner_id][target_id] = record

        self._change_log.append({
            "event": "OWNERSHIP_SET",
            "owner": owner_id,
            "target": target_id,
            "ownership_pct": str(record.ownership_pct),
            "timestamp": utcnow().isoformat(),
        })

        logger.info(
            "Ownership set: '%s' owns %s%% of '%s'.",
            owner_id, record.ownership_pct, target_id,
        )
        return record

    def _find_record(
        self, owner_id: str, target_id: str,
    ) -> Optional[OwnershipRecord]:
        """Find an existing ownership record between two entities.

        Args:
            owner_id: Owner entity ID.
            target_id: Target entity ID.

        Returns:
            The OwnershipRecord if found, else None.
        """
        owner_links = self._ownership_index.get(owner_id, {})
        return owner_links.get(target_id)

    def remove_ownership(
        self, owner_entity_id: str, target_entity_id: str,
    ) -> None:
        """Remove an ownership link between two entities.

        Args:
            owner_entity_id: Owner entity ID.
            target_entity_id: Target entity ID.

        Raises:
            KeyError: If the link does not exist.
        """
        record = self._find_record(owner_entity_id, target_entity_id)
        if record is None:
            raise KeyError(
                f"No ownership link from '{owner_entity_id}' "
                f"to '{target_entity_id}'."
            )

        del self._records[record.record_id]
        del self._ownership_index[owner_entity_id][target_entity_id]

        self._change_log.append({
            "event": "OWNERSHIP_REMOVED",
            "owner": owner_entity_id,
            "target": target_entity_id,
            "timestamp": utcnow().isoformat(),
        })

        logger.info(
            "Ownership removed: '%s' -> '%s'.",
            owner_entity_id, target_entity_id,
        )

    # ------------------------------------------------------------------
    # Equity Chain Resolution
    # ------------------------------------------------------------------

    def resolve_equity_chain(
        self,
        reporting_entity_id: str,
        target_entity_id: str,
    ) -> EquityChain:
        """Resolve the effective ownership through a multi-tier chain.

        Traces all paths from reporting_entity to target_entity through
        intermediary holding companies. Computes effective ownership as
        the product of ownership percentages along the chain.

        Formula:
            effective_pct = link1_pct/100 * link2_pct/100 * ... * 100

        Args:
            reporting_entity_id: The top-level reporting entity.
            target_entity_id: The entity at the end of the chain.

        Returns:
            EquityChain with computed effective ownership.
        """
        logger.info(
            "Resolving equity chain: '%s' -> '%s'.",
            reporting_entity_id, target_entity_id,
        )

        # Direct ownership check
        direct = self._find_record(reporting_entity_id, target_entity_id)
        if direct is not None:
            category = _classify_ownership(direct.ownership_pct)
            result = EquityChain(
                reporting_entity_id=reporting_entity_id,
                target_entity_id=target_entity_id,
                chain_links=[{
                    "from": reporting_entity_id,
                    "to": target_entity_id,
                    "ownership_pct": str(direct.ownership_pct),
                }],
                effective_ownership_pct=direct.ownership_pct,
                chain_depth=1,
                ownership_category=category,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Multi-tier resolution via DFS
        all_paths = self._find_all_paths(
            reporting_entity_id, target_entity_id
        )

        if not all_paths:
            result = EquityChain(
                reporting_entity_id=reporting_entity_id,
                target_entity_id=target_entity_id,
                chain_links=[],
                effective_ownership_pct=Decimal("0"),
                chain_depth=0,
                ownership_category=OwnershipCategory.NO_STAKE.value,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Compute effective ownership for each path and sum
        total_effective = Decimal("0")
        all_chain_links: List[Dict[str, Any]] = []

        for path in all_paths:
            path_pct = Decimal("1")
            path_links: List[Dict[str, Any]] = []

            for i in range(len(path) - 1):
                from_id = path[i]
                to_id = path[i + 1]
                record = self._find_record(from_id, to_id)
                if record is None:
                    path_pct = Decimal("0")
                    break
                link_pct = record.ownership_pct
                path_pct = path_pct * link_pct / Decimal("100")
                path_links.append({
                    "from": from_id,
                    "to": to_id,
                    "ownership_pct": str(link_pct),
                })

            effective_for_path = _round4(path_pct * Decimal("100"))
            total_effective += effective_for_path

            for link in path_links:
                link["path_effective_pct"] = str(effective_for_path)
            all_chain_links.extend(path_links)

        total_effective = min(_round4(total_effective), Decimal("100"))
        category = _classify_ownership(total_effective)

        result = EquityChain(
            reporting_entity_id=reporting_entity_id,
            target_entity_id=target_entity_id,
            chain_links=all_chain_links,
            effective_ownership_pct=total_effective,
            chain_depth=max(len(p) - 1 for p in all_paths) if all_paths else 0,
            ownership_category=category,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Equity chain resolved: '%s' -> '%s' = %s%% (%s).",
            reporting_entity_id, target_entity_id,
            total_effective, category,
        )
        return result

    def _find_all_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> List[List[str]]:
        """Find all ownership paths from source to target via DFS.

        Args:
            source: Starting entity ID.
            target: Target entity ID.
            max_depth: Maximum chain depth to prevent infinite loops.

        Returns:
            List of paths, each path is a list of entity IDs.
        """
        paths: List[List[str]] = []
        stack: List[Tuple[str, List[str]]] = [(source, [source])]

        while stack:
            current, path = stack.pop()
            if len(path) > max_depth + 1:
                continue

            targets = self._ownership_index.get(current, {})
            for next_id in targets:
                if next_id in path:
                    continue  # Avoid cycles
                new_path = path + [next_id]
                if next_id == target:
                    paths.append(new_path)
                else:
                    stack.append((next_id, new_path))

        return paths

    def get_effective_ownership(
        self,
        reporting_entity_id: str,
        target_entity_id: str,
    ) -> Decimal:
        """Get the effective ownership percentage (convenience method).

        Args:
            reporting_entity_id: The reporting entity.
            target_entity_id: The target entity.

        Returns:
            Effective ownership percentage (0-100).
        """
        chain = self.resolve_equity_chain(
            reporting_entity_id, target_entity_id
        )
        return chain.effective_ownership_pct

    # ------------------------------------------------------------------
    # Control Assessment
    # ------------------------------------------------------------------

    def assess_control(
        self,
        reporting_entity_id: str,
        target_entity_id: str,
    ) -> ControlAssessment:
        """Assess the control type over a target entity.

        Determines operational control, financial control, or no
        control based on ownership records and control indicators.

        Logic:
            - operational_control if manages_operations is True
            - financial_control if directs_policies is True
            - Majority ownership (>50%) implies financial control
            - Board majority or veto rights strengthen control claim

        Args:
            reporting_entity_id: The reporting entity.
            target_entity_id: The target entity to assess.

        Returns:
            ControlAssessment with determined control type.
        """
        logger.info(
            "Assessing control: '%s' over '%s'.",
            reporting_entity_id, target_entity_id,
        )

        chain = self.resolve_equity_chain(
            reporting_entity_id, target_entity_id
        )
        effective_pct = chain.effective_ownership_pct

        # Gather control indicators from direct link
        direct = self._find_record(reporting_entity_id, target_entity_id)
        manages_ops = False
        directs_pol = False
        has_board_majority = False
        has_veto = False
        assessment_basis: List[str] = []

        if direct is not None:
            manages_ops = direct.manages_operations
            directs_pol = direct.directs_policies
            has_board_majority = direct.has_board_majority
            has_veto = direct.has_veto_rights

        # Also check indirect records for control flags
        for record in self._records.values():
            if (record.owner_entity_id == reporting_entity_id
                    and record.target_entity_id == target_entity_id):
                if record.has_operational_control:
                    manages_ops = True
                if record.has_financial_control:
                    directs_pol = True
                if record.has_board_majority:
                    has_board_majority = True
                if record.has_veto_rights:
                    has_veto = True

        # Determine operational control
        has_oper_ctrl = manages_ops
        if manages_ops:
            assessment_basis.append(
                "Manages day-to-day operations of the target entity."
            )

        # Determine financial control
        has_fin_ctrl = directs_pol
        if directs_pol:
            assessment_basis.append(
                "Directs financial and operating policies of the target."
            )
        if effective_pct > Decimal("50") and not has_fin_ctrl:
            has_fin_ctrl = True
            assessment_basis.append(
                f"Majority ownership ({effective_pct}% > 50%) "
                f"implies financial control."
            )
        if has_board_majority and not has_fin_ctrl:
            has_fin_ctrl = True
            assessment_basis.append(
                "Board majority implies financial control."
            )

        # Additional indicators
        if has_veto:
            assessment_basis.append(
                "Veto rights on key decisions strengthen control claim."
            )

        # Determine control type
        if has_oper_ctrl and has_fin_ctrl:
            control_type = ControlType.OPERATIONAL_CONTROL.value
            assessment_basis.append(
                "Both operational and financial control exist; "
                "classified as operational control."
            )
        elif has_oper_ctrl:
            control_type = ControlType.OPERATIONAL_CONTROL.value
        elif has_fin_ctrl:
            control_type = ControlType.FINANCIAL_CONTROL.value
        elif effective_pct == Decimal("50"):
            control_type = ControlType.JOINT_CONTROL.value
            assessment_basis.append(
                "50% ownership classified as joint control."
            )
        else:
            control_type = ControlType.NO_CONTROL.value
            assessment_basis.append(
                f"No control indicators found for {effective_pct}% ownership."
            )

        # Compute inclusion percentages for each approach
        incl_equity = effective_pct
        incl_operational = Decimal("100") if has_oper_ctrl else Decimal("0")
        incl_financial = Decimal("100") if has_fin_ctrl else Decimal("0")

        result = ControlAssessment(
            reporting_entity_id=reporting_entity_id,
            target_entity_id=target_entity_id,
            control_type=control_type,
            ownership_pct=effective_pct,
            has_operational_control=has_oper_ctrl,
            has_financial_control=has_fin_ctrl,
            assessment_basis=assessment_basis,
            inclusion_pct_equity=incl_equity,
            inclusion_pct_operational=incl_operational,
            inclusion_pct_financial=incl_financial,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Control assessment: '%s' over '%s' = %s (equity=%s%%).",
            reporting_entity_id, target_entity_id,
            control_type, effective_pct,
        )
        return result

    # ------------------------------------------------------------------
    # Minority Interests
    # ------------------------------------------------------------------

    def get_minority_interests(
        self,
        reporting_entity_id: str,
    ) -> List[ControlAssessment]:
        """Identify all entities where reporting entity is a minority owner.

        A minority interest is defined as ownership < 50% but the
        entity may still have some form of control or significant
        influence.

        Args:
            reporting_entity_id: The reporting entity.

        Returns:
            List of ControlAssessments for minority-owned entities.
        """
        logger.info(
            "Identifying minority interests for '%s'.",
            reporting_entity_id,
        )

        targets = self._ownership_index.get(reporting_entity_id, {})
        minorities: List[ControlAssessment] = []

        for target_id, record in targets.items():
            if record.ownership_pct < Decimal("50"):
                assessment = self.assess_control(
                    reporting_entity_id, target_id,
                )
                minorities.append(assessment)

        logger.info(
            "Found %d minority interest(s) for '%s'.",
            len(minorities), reporting_entity_id,
        )
        return minorities

    def get_jv_partners(
        self,
        target_entity_id: str,
    ) -> List[Dict[str, Any]]:
        """Identify all JV partners for a given entity.

        Finds all owners of the target entity, useful for verifying
        that equity shares sum to 100%.

        Args:
            target_entity_id: The entity to find partners for.

        Returns:
            List of dicts with owner_entity_id and ownership_pct.
        """
        logger.info(
            "Identifying JV partners for '%s'.", target_entity_id,
        )

        partners: List[Dict[str, Any]] = []
        total_pct = Decimal("0")

        for record in self._records.values():
            if record.target_entity_id == target_entity_id:
                partners.append({
                    "owner_entity_id": record.owner_entity_id,
                    "ownership_pct": str(record.ownership_pct),
                    "has_operational_control": record.has_operational_control,
                    "has_financial_control": record.has_financial_control,
                })
                total_pct += record.ownership_pct

        logger.info(
            "Found %d partner(s) for '%s', total ownership=%s%%.",
            len(partners), target_entity_id, _round2(total_pct),
        )
        return partners

    # ------------------------------------------------------------------
    # Ownership History
    # ------------------------------------------------------------------

    def get_ownership_history(
        self,
        owner_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None,
    ) -> List[OwnershipChange]:
        """Get ownership change history, optionally filtered.

        Args:
            owner_entity_id: Filter by owner entity.
            target_entity_id: Filter by target entity.

        Returns:
            List of OwnershipChange records.
        """
        result = list(self._changes)

        if owner_entity_id is not None:
            result = [
                c for c in result
                if c.owner_entity_id == owner_entity_id
            ]
        if target_entity_id is not None:
            result = [
                c for c in result
                if c.target_entity_id == target_entity_id
            ]

        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_records(self) -> List[OwnershipRecord]:
        """Return all ownership records.

        Returns:
            List of all OwnershipRecords.
        """
        return list(self._records.values())

    def get_direct_ownership(
        self,
        owner_entity_id: str,
    ) -> List[OwnershipRecord]:
        """Get all entities directly owned by the given entity.

        Args:
            owner_entity_id: The owning entity.

        Returns:
            List of OwnershipRecords for direct holdings.
        """
        return [
            r for r in self._records.values()
            if r.owner_entity_id == owner_entity_id
        ]

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries.
        """
        return list(self._change_log)
