"""
GL-011 FuelCraft - Inventory Calculator

Deterministic inventory balance calculations:
- Tank level calculations with temperature correction
- Loss factor modeling (boil-off, handling)
- Heel/minimum level constraints
- Overfill protection
- Inventory balance: s_{k,t} = s_{k,t-1} + inflow - outflow - losses

Standards:
- API MPMS Chapter 12 (Tank Measurement)
- API MPMS Chapter 11 (Temperature Volume Correction)
- NFPA 30 (Flammable Liquids)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json


class TransactionType(Enum):
    """Type of inventory transaction."""
    RECEIPT = "receipt"           # Inflow from procurement
    WITHDRAWAL = "withdrawal"     # Outflow for consumption
    TRANSFER_IN = "transfer_in"   # Transfer from another tank
    TRANSFER_OUT = "transfer_out" # Transfer to another tank
    ADJUSTMENT = "adjustment"     # Manual adjustment
    LOSS = "loss"                 # Boil-off, evaporation, handling loss


class TankType(Enum):
    """Type of storage tank."""
    FIXED_ROOF = "fixed_roof"         # Atmospheric tank
    FLOATING_ROOF = "floating_roof"   # Internal/external floating roof
    PRESSURIZED = "pressurized"       # Pressure vessel (LPG, LNG)
    UNDERGROUND = "underground"       # Underground storage


class LossType(Enum):
    """Type of inventory loss."""
    BOIL_OFF = "boil_off"           # Cryogenic liquids (LNG)
    EVAPORATION = "evaporation"     # Working/breathing losses
    HANDLING = "handling"           # Transfer losses
    MEASUREMENT = "measurement"     # Measurement uncertainty


@dataclass
class TankConfiguration:
    """
    Tank physical and operational parameters.
    """
    tank_id: str
    tank_type: TankType
    capacity_m3: Decimal              # Total capacity
    min_level_m3: Decimal             # Heel / minimum operating level
    max_level_m3: Decimal             # Maximum fill (overfill protection)
    safe_fill_pct: Decimal            # Safe fill percentage (typically 95%)
    # Loss parameters
    daily_loss_rate_pct: Decimal      # Daily loss rate as % of inventory
    transfer_loss_pct: Decimal        # Loss per transfer operation
    # Temperature parameters
    reference_temp_c: Decimal         # Reference temperature (typically 15C)
    current_temp_c: Decimal           # Current tank temperature
    # Fuel type
    fuel_type: str
    density_kg_m3: Decimal            # Fuel density at reference temp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tank_id": self.tank_id,
            "tank_type": self.tank_type.value,
            "capacity_m3": str(self.capacity_m3),
            "min_level_m3": str(self.min_level_m3),
            "max_level_m3": str(self.max_level_m3),
            "safe_fill_pct": str(self.safe_fill_pct),
            "daily_loss_rate_pct": str(self.daily_loss_rate_pct),
            "transfer_loss_pct": str(self.transfer_loss_pct),
            "reference_temp_c": str(self.reference_temp_c),
            "current_temp_c": str(self.current_temp_c),
            "fuel_type": self.fuel_type,
            "density_kg_m3": str(self.density_kg_m3)
        }


@dataclass
class InventoryTransaction:
    """
    Single inventory transaction.
    """
    transaction_id: str
    tank_id: str
    transaction_type: TransactionType
    quantity_m3: Decimal              # Volume at reference temperature
    quantity_kg: Decimal              # Mass
    transaction_time: datetime
    source_document: str              # Bill of lading, meter ticket, etc.
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "tank_id": self.tank_id,
            "transaction_type": self.transaction_type.value,
            "quantity_m3": str(self.quantity_m3),
            "quantity_kg": str(self.quantity_kg),
            "transaction_time": self.transaction_time.isoformat(),
            "source_document": self.source_document,
            "notes": self.notes
        }


@dataclass
class TankState:
    """
    Current state of a tank.
    """
    tank_id: str
    timestamp: datetime
    level_m3: Decimal                 # Current level (volume at reference temp)
    level_kg: Decimal                 # Current level (mass)
    level_pct: Decimal                # Percentage of capacity
    available_capacity_m3: Decimal    # Remaining capacity
    is_at_minimum: bool               # At or below heel
    is_near_maximum: bool             # Within 5% of max
    temperature_c: Decimal            # Current temperature
    # Derived
    energy_mj: Decimal                # Energy content
    days_of_supply: Optional[Decimal] # Days at current consumption rate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tank_id": self.tank_id,
            "timestamp": self.timestamp.isoformat(),
            "level_m3": str(self.level_m3),
            "level_kg": str(self.level_kg),
            "level_pct": str(self.level_pct),
            "available_capacity_m3": str(self.available_capacity_m3),
            "is_at_minimum": self.is_at_minimum,
            "is_near_maximum": self.is_near_maximum,
            "temperature_c": str(self.temperature_c),
            "energy_mj": str(self.energy_mj),
            "days_of_supply": str(self.days_of_supply) if self.days_of_supply else None
        }


@dataclass
class InventoryInput:
    """Input for inventory calculation."""
    tank_config: TankConfiguration
    initial_level_m3: Decimal
    transactions: List[InventoryTransaction]
    period_start: datetime
    period_end: datetime
    daily_consumption_m3: Optional[Decimal] = None  # For days of supply calc


@dataclass
class InventoryResult:
    """
    Result of inventory calculation with provenance.
    """
    tank_id: str
    period_start: datetime
    period_end: datetime
    # Starting state
    opening_level_m3: Decimal
    opening_level_kg: Decimal
    # Transactions summary
    total_receipts_m3: Decimal
    total_withdrawals_m3: Decimal
    total_losses_m3: Decimal
    # Ending state
    closing_level_m3: Decimal
    closing_level_kg: Decimal
    closing_level_pct: Decimal
    # Balance check
    balance_check_passed: bool
    balance_discrepancy_m3: Decimal
    # Constraints
    min_level_violated: bool
    max_level_violated: bool
    # Detailed states
    daily_states: List[TankState]
    # Provenance
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "tank_id": self.tank_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "opening_level_m3": str(self.opening_level_m3),
            "closing_level_m3": str(self.closing_level_m3),
            "total_receipts_m3": str(self.total_receipts_m3),
            "total_withdrawals_m3": str(self.total_withdrawals_m3),
            "total_losses_m3": str(self.total_losses_m3),
            "balance_check_passed": self.balance_check_passed,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tank_id": self.tank_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "opening_level_m3": str(self.opening_level_m3),
            "opening_level_kg": str(self.opening_level_kg),
            "total_receipts_m3": str(self.total_receipts_m3),
            "total_withdrawals_m3": str(self.total_withdrawals_m3),
            "total_losses_m3": str(self.total_losses_m3),
            "closing_level_m3": str(self.closing_level_m3),
            "closing_level_kg": str(self.closing_level_kg),
            "closing_level_pct": str(self.closing_level_pct),
            "balance_check_passed": self.balance_check_passed,
            "balance_discrepancy_m3": str(self.balance_discrepancy_m3),
            "min_level_violated": self.min_level_violated,
            "max_level_violated": self.max_level_violated,
            "daily_states": [s.to_dict() for s in self.daily_states],
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }


class InventoryCalculator:
    """
    Deterministic inventory balance calculator.

    Implements inventory balance equation:
    s_{k,t} = s_{k,t-1} + inflow - outflow - losses

    Provides ZERO-HALLUCINATION calculations for:
    - Tank level tracking with temperature correction
    - Loss factor modeling (boil-off, evaporation, handling)
    - Constraint validation (min/max levels)
    - Mass balance verification

    All calculations use Decimal arithmetic.
    """

    NAME: str = "InventoryCalculator"
    VERSION: str = "1.0.0"

    PRECISION: int = 6

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate(
        self,
        inventory_input: InventoryInput,
        precision: int = 6
    ) -> InventoryResult:
        """
        Calculate inventory balance over period - DETERMINISTIC.

        Args:
            inventory_input: Input parameters
            precision: Output decimal places

        Returns:
            InventoryResult with full provenance
        """
        config = inventory_input.tank_config
        transactions = sorted(inventory_input.transactions,
                             key=lambda t: t.transaction_time)

        steps: List[Dict[str, Any]] = []
        daily_states: List[TankState] = []

        # Step 1: Initialize with opening balance
        current_level = inventory_input.initial_level_m3
        opening_level = current_level
        opening_mass = current_level * config.density_kg_m3

        steps.append({
            "step": 1,
            "operation": "initialize",
            "opening_level_m3": str(opening_level),
            "opening_mass_kg": str(opening_mass)
        })

        # Step 2: Process transactions and calculate daily losses
        total_receipts = Decimal("0")
        total_withdrawals = Decimal("0")
        total_losses = Decimal("0")
        min_violated = False
        max_violated = False

        # Generate daily periods
        current_date = inventory_input.period_start.date()
        end_date = inventory_input.period_end.date()

        while current_date <= end_date:
            day_start = datetime.combine(current_date, datetime.min.time(),
                                        tzinfo=timezone.utc)
            day_end = datetime.combine(current_date, datetime.max.time(),
                                      tzinfo=timezone.utc)

            # Get transactions for this day
            day_transactions = [
                t for t in transactions
                if day_start <= t.transaction_time <= day_end
            ]

            # Process transactions
            day_receipts = Decimal("0")
            day_withdrawals = Decimal("0")

            for txn in day_transactions:
                if txn.transaction_type in [TransactionType.RECEIPT, TransactionType.TRANSFER_IN]:
                    # Apply transfer loss on receipt
                    net_receipt = txn.quantity_m3 * (Decimal("1") - config.transfer_loss_pct / Decimal("100"))
                    current_level += net_receipt
                    day_receipts += txn.quantity_m3
                    total_receipts += txn.quantity_m3

                elif txn.transaction_type in [TransactionType.WITHDRAWAL, TransactionType.TRANSFER_OUT]:
                    current_level -= txn.quantity_m3
                    day_withdrawals += txn.quantity_m3
                    total_withdrawals += txn.quantity_m3

                elif txn.transaction_type == TransactionType.LOSS:
                    current_level -= txn.quantity_m3
                    total_losses += txn.quantity_m3

            # Apply daily losses (boil-off, evaporation)
            daily_loss = current_level * config.daily_loss_rate_pct / Decimal("100")
            current_level -= daily_loss
            total_losses += daily_loss

            # Check constraints
            if current_level < config.min_level_m3:
                min_violated = True
            if current_level > config.max_level_m3:
                max_violated = True

            # Create daily state
            state = self._create_tank_state(
                config, current_level, datetime.combine(current_date, datetime.max.time(),
                                                       tzinfo=timezone.utc),
                inventory_input.daily_consumption_m3
            )
            daily_states.append(state)

            current_date += timedelta(days=1)

        # Step 3: Calculate closing balance
        closing_level = current_level
        closing_mass = closing_level * config.density_kg_m3
        closing_pct = (closing_level / config.capacity_m3) * Decimal("100")

        steps.append({
            "step": 3,
            "operation": "calculate_closing",
            "closing_level_m3": str(closing_level),
            "closing_mass_kg": str(closing_mass),
            "closing_pct": str(closing_pct)
        })

        # Step 4: Verify mass balance
        expected_closing = opening_level + total_receipts - total_withdrawals - total_losses
        discrepancy = abs(closing_level - expected_closing)
        balance_ok = discrepancy < Decimal("0.001")  # 1 liter tolerance

        steps.append({
            "step": 4,
            "operation": "verify_balance",
            "expected_closing_m3": str(expected_closing),
            "actual_closing_m3": str(closing_level),
            "discrepancy_m3": str(discrepancy),
            "balance_passed": balance_ok
        })

        # Apply precision
        quantize_str = "0." + "0" * precision

        return InventoryResult(
            tank_id=config.tank_id,
            period_start=inventory_input.period_start,
            period_end=inventory_input.period_end,
            opening_level_m3=opening_level.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            opening_level_kg=opening_mass.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            total_receipts_m3=total_receipts.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            total_withdrawals_m3=total_withdrawals.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            total_losses_m3=total_losses.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            closing_level_m3=closing_level.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            closing_level_kg=closing_mass.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            closing_level_pct=closing_pct.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            balance_check_passed=balance_ok,
            balance_discrepancy_m3=discrepancy.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP),
            min_level_violated=min_violated,
            max_level_violated=max_violated,
            daily_states=daily_states,
            calculation_steps=steps
        )

    def calculate_temperature_correction(
        self,
        volume_observed: Union[float, Decimal],
        observed_temp_c: Union[float, Decimal],
        reference_temp_c: Union[float, Decimal],
        api_gravity: Union[float, Decimal]
    ) -> Decimal:
        """
        Calculate temperature-corrected volume per API MPMS Chapter 11.

        Args:
            volume_observed: Observed volume
            observed_temp_c: Observed temperature (C)
            reference_temp_c: Reference temperature (C)
            api_gravity: API gravity of product

        Returns:
            Volume at reference temperature
        """
        vol = Decimal(str(volume_observed))
        t_obs = Decimal(str(observed_temp_c))
        t_ref = Decimal(str(reference_temp_c))
        api = Decimal(str(api_gravity))

        # Calculate VCF (simplified - production uses full ASTM tables)
        sg = Decimal("141.5") / (api + Decimal("131.5"))
        alpha = Decimal("613.9723") / (sg * sg) / Decimal("1000000")
        delta_t = t_obs - t_ref

        vcf = Decimal("1.0") - alpha * delta_t
        vcf = max(Decimal("0.9"), min(Decimal("1.1"), vcf))

        return (vol * vcf).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def calculate_loss_factor(
        self,
        tank_config: TankConfiguration,
        period_days: int
    ) -> Decimal:
        """
        Calculate expected loss over period.

        Args:
            tank_config: Tank configuration
            period_days: Number of days

        Returns:
            Expected loss factor (fraction)
        """
        daily_rate = tank_config.daily_loss_rate_pct / Decimal("100")

        # Compound loss over period
        # Remaining = (1 - daily_rate)^days
        # Loss = 1 - Remaining
        remaining = (Decimal("1") - daily_rate) ** period_days
        loss_factor = Decimal("1") - remaining

        return loss_factor.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def check_overfill(
        self,
        tank_config: TankConfiguration,
        current_level_m3: Decimal,
        planned_receipt_m3: Decimal
    ) -> Tuple[bool, Decimal]:
        """
        Check if planned receipt would cause overfill.

        Args:
            tank_config: Tank configuration
            current_level_m3: Current tank level
            planned_receipt_m3: Planned receipt volume

        Returns:
            Tuple of (is_safe, max_safe_receipt)
        """
        safe_capacity = tank_config.capacity_m3 * tank_config.safe_fill_pct / Decimal("100")
        available = safe_capacity - current_level_m3

        is_safe = planned_receipt_m3 <= available
        max_safe = max(Decimal("0"), available)

        return is_safe, max_safe.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _create_tank_state(
        self,
        config: TankConfiguration,
        level_m3: Decimal,
        timestamp: datetime,
        daily_consumption: Optional[Decimal]
    ) -> TankState:
        """Create tank state snapshot."""
        level_kg = level_m3 * config.density_kg_m3
        level_pct = (level_m3 / config.capacity_m3) * Decimal("100")
        available = config.max_level_m3 - level_m3

        # Days of supply
        days_supply = None
        if daily_consumption and daily_consumption > Decimal("0"):
            usable = max(Decimal("0"), level_m3 - config.min_level_m3)
            days_supply = usable / daily_consumption

        # Energy (assuming diesel LHV of 43 MJ/kg)
        lhv_mj_kg = Decimal("43.0")  # Default - should come from fuel properties
        energy_mj = level_kg * lhv_mj_kg

        return TankState(
            tank_id=config.tank_id,
            timestamp=timestamp,
            level_m3=level_m3.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            level_kg=level_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            level_pct=level_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            available_capacity_m3=available.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            is_at_minimum=level_m3 <= config.min_level_m3,
            is_near_maximum=level_pct >= Decimal("95"),
            temperature_c=config.current_temp_c,
            energy_mj=energy_mj.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            days_of_supply=days_supply.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) if days_supply else None
        )
