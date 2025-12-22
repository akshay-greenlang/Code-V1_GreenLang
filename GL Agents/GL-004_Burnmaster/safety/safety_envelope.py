"""
SafetyEnvelope - Pre-approved operating envelope validation for combustion optimization.

This module defines and validates safety envelopes that constrain all optimizer actions.
CRITICAL: All setpoints must be validated within envelope before ANY write operation.

Example:
    >>> envelope = SafetyEnvelope(unit_id="BLR-001")
    >>> envelope.define_envelope("BLR-001", limits)
    >>> validation = envelope.validate_within_envelope(setpoint)
    >>> if not validation.is_valid:
    ...     # BLOCK the setpoint write
    ...     pass
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)


class EnvelopeStatus(str, Enum):
    """Status of safety envelope."""
    ACTIVE = "active"
    SHRUNK = "shrunk"
    EXPANDED = "expanded"
    SUSPENDED = "suspended"


class LimitType(str, Enum):
    """Type of limit constraint."""
    HARD = "hard"  # Absolute limit, never exceed
    SOFT = "soft"  # Warning limit, can approach
    OPERATIONAL = "operational"  # Normal operating range


class ParameterLimit(BaseModel):
    """Individual parameter limit definition."""
    parameter_name: str = Field(..., description="Name of the parameter")
    min_value: float = Field(..., description="Minimum allowed value")
    max_value: float = Field(..., description="Maximum allowed value")
    unit: str = Field(..., description="Engineering unit")
    limit_type: LimitType = Field(default=LimitType.HARD, description="Type of limit")
    description: str = Field(default="", description="Description of this limit")

    @validator('max_value')
    def max_greater_than_min(cls, v, values):
        """Validate max is greater than min."""
        if 'min_value' in values and v <= values['min_value']:
            raise ValueError('max_value must be greater than min_value')
        return v


class EnvelopeLimits(BaseModel):
    """Complete set of envelope limits for a unit."""
    unit_id: str = Field(..., description="Unit identifier")
    o2_min: float = Field(..., ge=0.5, le=10.0, description="Min O2 percentage")
    o2_max: float = Field(..., ge=1.0, le=15.0, description="Max O2 percentage")
    co_max: float = Field(..., ge=0, le=1000, description="Max CO in ppm")
    nox_max: float = Field(..., ge=0, le=500, description="Max NOx in ppm")
    draft_min: float = Field(..., description="Min furnace draft in inwc")
    draft_max: float = Field(..., description="Max furnace draft in inwc")
    flame_signal_min: float = Field(..., ge=0, description="Min flame signal")
    steam_temp_max: float = Field(..., description="Max steam temperature F")
    steam_pressure_max: float = Field(..., description="Max steam pressure psig")
    firing_rate_min: float = Field(..., ge=0, description="Min firing rate percentage")
    firing_rate_max: float = Field(..., le=100, description="Max firing rate percentage")
    additional_limits: Dict[str, ParameterLimit] = Field(default_factory=dict)

    @validator('o2_max')
    def o2_max_greater_than_min(cls, v, values):
        """Validate O2 max greater than min."""
        if 'o2_min' in values and v <= values['o2_min']:
            raise ValueError('o2_max must be greater than o2_min')
        return v


class Setpoint(BaseModel):
    """Setpoint to be validated against envelope."""
    parameter_name: str = Field(..., description="Parameter being set")
    value: float = Field(..., description="Setpoint value")
    unit: str = Field(..., description="Engineering unit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = Field(default="optimizer", description="Source of setpoint")


class EnvelopeValidation(BaseModel):
    """Result of envelope validation."""
    is_valid: bool = Field(..., description="Whether setpoint is within envelope")
    setpoint: Setpoint = Field(..., description="The validated setpoint")
    distance_to_limit: float = Field(..., description="Distance to nearest limit")
    limit_name: str = Field(..., description="Name of nearest limit")
    margin_percentage: float = Field(..., description="Margin as percentage of range")
    warnings: List[str] = Field(default_factory=list)
    blocking_reason: Optional[str] = Field(None, description="Reason for blocking if invalid")
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class SafetyEnvelope:
    """
    SafetyEnvelope enforces pre-approved operating boundaries for combustion optimization.

    CRITICAL SAFETY INVARIANT:
    - All optimizer setpoints MUST pass envelope validation before write
    - No action may exceed envelope limits
    - Envelope can only shrink automatically (more conservative)
    - Envelope expansion requires explicit approval

    Attributes:
        unit_id: Identifier for the combustion unit
        limits: Current envelope limits
        status: Current envelope status
        shrink_history: Record of envelope shrinkage events

    Example:
        >>> envelope = SafetyEnvelope(unit_id="BLR-001")
        >>> limits = EnvelopeLimits(unit_id="BLR-001", o2_min=2.0, ...)
        >>> envelope.define_envelope("BLR-001", limits)
        >>> validation = envelope.validate_within_envelope(setpoint)
        >>> assert validation.is_valid, "Setpoint blocked by safety envelope"
    """

    def __init__(self, unit_id: str):
        """Initialize SafetyEnvelope for a specific unit."""
        self.unit_id = unit_id
        self.limits: Optional[EnvelopeLimits] = None
        self.status = EnvelopeStatus.SUSPENDED
        self.shrink_history: List[Dict[str, Any]] = []
        self.expand_history: List[Dict[str, Any]] = []
        self._creation_time = datetime.utcnow()
        logger.info(f"SafetyEnvelope initialized for unit {unit_id}")

    def define_envelope(
        self,
        unit_id: str,
        limits: Dict[str, Any]
    ) -> 'SafetyEnvelope':
        """
        Define the safety envelope with validated limits.

        Args:
            unit_id: Unit identifier (must match envelope unit_id)
            limits: Dictionary of limit definitions

        Returns:
            Self for method chaining

        Raises:
            ValueError: If unit_id mismatch or invalid limits
        """
        if unit_id != self.unit_id:
            raise ValueError(f"Unit ID mismatch: expected {self.unit_id}, got {unit_id}")

        # Create validated limits from dictionary
        if isinstance(limits, dict):
            self.limits = EnvelopeLimits(unit_id=unit_id, **limits)
        elif isinstance(limits, EnvelopeLimits):
            self.limits = limits
        else:
            raise ValueError("limits must be dict or EnvelopeLimits")

        self.status = EnvelopeStatus.ACTIVE
        logger.info(f"Safety envelope defined for {unit_id}: {self.limits}")
        return self

    def validate_within_envelope(self, setpoint: Setpoint) -> EnvelopeValidation:
        """
        Validate that a setpoint is within the safety envelope.

        CRITICAL: This method MUST be called before ANY setpoint write.

        Args:
            setpoint: The setpoint to validate

        Returns:
            EnvelopeValidation with is_valid=True if within envelope

        Raises:
            ValueError: If envelope not defined
        """
        if self.limits is None:
            raise ValueError("Envelope not defined - cannot validate")

        if self.status == EnvelopeStatus.SUSPENDED:
            logger.warning(f"Envelope suspended for {self.unit_id} - blocking all setpoints")
            return self._create_blocked_validation(
                setpoint, "Envelope suspended - observe only mode"
            )

        # Get limits for this parameter
        limit_min, limit_max = self._get_parameter_limits(setpoint.parameter_name)

        if limit_min is None or limit_max is None:
            logger.warning(f"No limits defined for {setpoint.parameter_name}")
            return self._create_blocked_validation(
                setpoint, f"No limits defined for {setpoint.parameter_name}"
            )

        # Check if within limits
        is_valid = limit_min <= setpoint.value <= limit_max

        # Calculate distance to nearest limit
        distance_to_min = abs(setpoint.value - limit_min)
        distance_to_max = abs(setpoint.value - limit_max)

        if distance_to_min < distance_to_max:
            distance_to_limit = distance_to_min
            limit_name = f"{setpoint.parameter_name}_min"
        else:
            distance_to_limit = distance_to_max
            limit_name = f"{setpoint.parameter_name}_max"

        # Calculate margin percentage
        range_size = limit_max - limit_min
        margin_percentage = (distance_to_limit / range_size * 100) if range_size > 0 else 0

        # Generate warnings for approaching limits
        warnings = []
        if margin_percentage < 10:
            warnings.append(f"CAUTION: Within 10% of {limit_name}")
        elif margin_percentage < 20:
            warnings.append(f"Advisory: Within 20% of {limit_name}")

        # Create provenance hash
        provenance_str = f"{setpoint.json()}{is_valid}{distance_to_limit}{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        validation = EnvelopeValidation(
            is_valid=is_valid,
            setpoint=setpoint,
            distance_to_limit=distance_to_limit,
            limit_name=limit_name,
            margin_percentage=margin_percentage,
            warnings=warnings,
            blocking_reason=None if is_valid else f"Value {setpoint.value} outside limits [{limit_min}, {limit_max}]",
            provenance_hash=provenance_hash
        )

        if not is_valid:
            logger.warning(
                f"ENVELOPE VIOLATION: {setpoint.parameter_name}={setpoint.value} "
                f"outside [{limit_min}, {limit_max}] - BLOCKING SETPOINT"
            )

        return validation

    def compute_distance_to_limit(self, value: float, limit: float) -> float:
        """
        Compute absolute distance from value to limit.

        Args:
            value: Current value
            limit: Limit value

        Returns:
            Absolute distance to limit
        """
        return abs(value - limit)

    def shrink_envelope(self, factor: float, reason: str) -> 'SafetyEnvelope':
        """
        Shrink envelope to be more conservative (automatic, no approval needed).

        This operation makes limits more restrictive for safety.
        - Minimum limits increase
        - Maximum limits decrease

        Args:
            factor: Shrink factor (0 < factor < 1, e.g., 0.9 = 10% shrink)
            reason: Reason for shrinking envelope

        Returns:
            Self with updated limits

        Raises:
            ValueError: If factor not in valid range
        """
        if not 0 < factor < 1:
            raise ValueError("Shrink factor must be between 0 and 1")

        if self.limits is None:
            raise ValueError("Cannot shrink undefined envelope")

        # Store original for audit
        original_limits = deepcopy(self.limits)

        # Calculate shrunk limits (move toward center of range)
        new_limits = self.limits.dict()

        for param in ['o2', 'draft', 'firing_rate']:
            min_key = f"{param}_min"
            max_key = f"{param}_max"
            if min_key in new_limits and max_key in new_limits:
                range_size = new_limits[max_key] - new_limits[min_key]
                shrink_amount = range_size * (1 - factor) / 2
                new_limits[min_key] += shrink_amount
                new_limits[max_key] -= shrink_amount

        # Reduce max limits
        for max_param in ['co_max', 'nox_max', 'steam_temp_max', 'steam_pressure_max']:
            if max_param in new_limits:
                new_limits[max_param] *= factor

        # Update limits
        self.limits = EnvelopeLimits(**new_limits)
        self.status = EnvelopeStatus.SHRUNK

        # Record shrink event
        shrink_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'factor': factor,
            'reason': reason,
            'original_limits': original_limits.dict(),
            'new_limits': self.limits.dict(),
            'provenance_hash': hashlib.sha256(
                f"{factor}{reason}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
        }
        self.shrink_history.append(shrink_record)

        logger.info(f"Envelope shrunk by factor {factor}: {reason}")
        return self

    def expand_envelope(self, factor: float, approval: str) -> 'SafetyEnvelope':
        """
        Expand envelope (REQUIRES explicit approval string).

        CRITICAL: Expansion is a safety-sensitive operation.

        Args:
            factor: Expansion factor (> 1, e.g., 1.1 = 10% expansion)
            approval: Approval string with authorizer info

        Returns:
            Self with updated limits

        Raises:
            ValueError: If factor invalid or approval missing/invalid
        """
        if factor <= 1:
            raise ValueError("Expansion factor must be greater than 1")

        if not approval or len(approval) < 10:
            raise ValueError("Valid approval string required for envelope expansion")

        if self.limits is None:
            raise ValueError("Cannot expand undefined envelope")

        # Validate approval format (example: "APPROVED-USERID-TIMESTAMP")
        if not approval.startswith("APPROVED-"):
            raise ValueError("Approval must start with 'APPROVED-'")

        # Store original for audit
        original_limits = deepcopy(self.limits)

        # Calculate expanded limits
        new_limits = self.limits.dict()

        for param in ['o2', 'draft', 'firing_rate']:
            min_key = f"{param}_min"
            max_key = f"{param}_max"
            if min_key in new_limits and max_key in new_limits:
                range_size = new_limits[max_key] - new_limits[min_key]
                expand_amount = range_size * (factor - 1) / 2
                new_limits[min_key] -= expand_amount
                new_limits[max_key] += expand_amount

        # Expand max limits
        for max_param in ['co_max', 'nox_max', 'steam_temp_max', 'steam_pressure_max']:
            if max_param in new_limits:
                new_limits[max_param] *= factor

        # Update limits
        self.limits = EnvelopeLimits(**new_limits)
        self.status = EnvelopeStatus.EXPANDED

        # Record expand event
        expand_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'factor': factor,
            'approval': approval,
            'original_limits': original_limits.dict(),
            'new_limits': self.limits.dict(),
            'provenance_hash': hashlib.sha256(
                f"{factor}{approval}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()
        }
        self.expand_history.append(expand_record)

        logger.warning(f"Envelope EXPANDED by factor {factor} with approval: {approval}")
        return self

    def _get_parameter_limits(self, parameter_name: str) -> tuple:
        """Get min/max limits for a parameter."""
        if self.limits is None:
            return None, None

        limits_dict = self.limits.dict()

        # Map parameter names to limit keys
        param_mapping = {
            'o2': ('o2_min', 'o2_max'),
            'co': (0, 'co_max'),
            'nox': (0, 'nox_max'),
            'draft': ('draft_min', 'draft_max'),
            'flame_signal': ('flame_signal_min', float('inf')),
            'steam_temp': (0, 'steam_temp_max'),
            'steam_pressure': (0, 'steam_pressure_max'),
            'firing_rate': ('firing_rate_min', 'firing_rate_max'),
        }

        if parameter_name.lower() in param_mapping:
            min_key, max_key = param_mapping[parameter_name.lower()]
            min_val = limits_dict.get(min_key, 0) if isinstance(min_key, str) else min_key
            max_val = limits_dict.get(max_key, float('inf')) if isinstance(max_key, str) else max_key
            return min_val, max_val

        # Check additional limits
        if parameter_name in limits_dict.get('additional_limits', {}):
            limit = limits_dict['additional_limits'][parameter_name]
            return limit['min_value'], limit['max_value']

        return None, None

    def _create_blocked_validation(
        self,
        setpoint: Setpoint,
        reason: str
    ) -> EnvelopeValidation:
        """Create a blocked validation result."""
        provenance_hash = hashlib.sha256(
            f"{setpoint.json()}{reason}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return EnvelopeValidation(
            is_valid=False,
            setpoint=setpoint,
            distance_to_limit=0,
            limit_name="N/A",
            margin_percentage=0,
            warnings=[],
            blocking_reason=reason,
            provenance_hash=provenance_hash
        )
