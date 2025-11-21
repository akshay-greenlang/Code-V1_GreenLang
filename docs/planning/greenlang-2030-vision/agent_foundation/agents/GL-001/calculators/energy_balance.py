# -*- coding: utf-8 -*-
"""
Energy Balance Validator - Zero Hallucination Guarantee

Validates energy conservation using first law of thermodynamics with
deterministic calculations and complete provenance tracking.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ISO 50001, ASME EA-4-2010
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from provenance import ProvenanceTracker, ProvenanceRecord
import hashlib
from greenlang.determinism import FinancialDecimal


class EnergyFlowType(Enum):
    """Types of energy flows in the system."""
    INPUT = "input"
    OUTPUT = "output"
    STORED = "stored"
    LOST = "lost"


@dataclass
class EnergyFlow:
    """Represents an energy flow in the system."""
    flow_id: str
    flow_type: EnergyFlowType
    description: str
    value_kw: float
    measurement_uncertainty_percent: float = 2.0
    timestamp: str = ""


@dataclass
class EnergyBalanceData:
    """Input data for energy balance validation."""
    # Energy inputs
    fuel_energy_kw: float
    electrical_energy_kw: float
    steam_import_kw: float
    recovered_heat_kw: float

    # Energy outputs
    process_heat_output_kw: float
    steam_export_kw: float
    electricity_generation_kw: float
    useful_work_kw: float

    # Energy losses
    flue_gas_loss_kw: float
    radiation_loss_kw: float
    blowdown_loss_kw: float
    condensate_loss_kw: float
    unaccounted_loss_kw: float

    # Storage changes
    thermal_storage_change_kw: float = 0.0

    # System parameters
    measurement_period_hours: float = 1.0
    ambient_temperature_c: float = 20.0


class EnergyBalanceValidator:
    """
    Validates energy balance using thermodynamic principles.

    Zero Hallucination Guarantee:
    - Pure mathematical validation (First Law of Thermodynamics)
    - No LLM inference
    - Bit-perfect reproducibility
    - Complete provenance tracking
    """

    # Tolerance for energy balance (% of total input)
    BALANCE_TOLERANCE_PERCENT = Decimal('2.0')

    # Physical constants
    JOULE_TO_KWH = Decimal('2.778e-7')
    KW_TO_BTU_PER_HR = Decimal('3412.14')

    def __init__(self, version: str = "1.0.0"):
        """Initialize validator with version tracking."""
        self.version = version

    def validate(self, energy_data: EnergyBalanceData) -> Dict:
        """
        Validate energy balance and identify discrepancies.

        First Law: ΣE_in = ΣE_out + ΣE_stored + ΣE_lost

        Args:
            energy_data: Energy flow measurements

        Returns:
            Validation report with discrepancies and recommendations
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"energy_balance_{id(energy_data)}",
            calculation_type="energy_balance_validation",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs(energy_data.__dict__)

        # Step 1: Calculate total energy input
        total_input = self._calculate_total_input(energy_data, tracker)

        # Step 2: Calculate total energy output
        total_output = self._calculate_total_output(energy_data, tracker)

        # Step 3: Calculate total losses
        total_losses = self._calculate_total_losses(energy_data, tracker)

        # Step 4: Calculate storage changes
        storage_change = self._calculate_storage_change(energy_data, tracker)

        # Step 5: Calculate energy balance
        balance_result = self._calculate_balance(
            total_input, total_output, total_losses, storage_change, tracker
        )

        # Step 6: Identify violations
        violations = self._identify_violations(balance_result, energy_data, tracker)

        # Step 7: Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            total_input, total_output, total_losses, tracker
        )

        # Step 8: Generate corrective actions
        corrective_actions = self._generate_corrective_actions(
            violations, balance_result, tracker
        )

        # Step 9: Perform sankey analysis
        sankey_data = self._generate_sankey_data(
            energy_data, total_input, total_output, total_losses, tracker
        )

        # Final result
        result = {
            'balance_status': balance_result['status'],
            'total_input_kw': FinancialDecimal.from_string(total_input),
            'total_output_kw': FinancialDecimal.from_string(total_output),
            'total_losses_kw': FinancialDecimal.from_string(total_losses),
            'storage_change_kw': float(storage_change),
            'imbalance_kw': float(balance_result['imbalance']),
            'imbalance_percent': float(balance_result['imbalance_percent']),
            'conservation_verified': balance_result['conservation_verified'],
            'violations': violations,
            'efficiency_metrics': efficiency_metrics,
            'corrective_actions': corrective_actions,
            'sankey_diagram': sankey_data,
            'provenance': tracker.get_provenance_record(
                balance_result['imbalance']
            ).to_dict()
        }

        return result

    def _calculate_total_input(
        self,
        data: EnergyBalanceData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate total energy input to the system."""
        inputs = {
            'fuel_energy': Decimal(str(data.fuel_energy_kw)),
            'electrical_energy': Decimal(str(data.electrical_energy_kw)),
            'steam_import': Decimal(str(data.steam_import_kw)),
            'recovered_heat': Decimal(str(data.recovered_heat_kw))
        }

        total_input = sum(inputs.values())

        tracker.record_step(
            operation="summation",
            description="Calculate total energy input",
            inputs=inputs,
            output_value=total_input,
            output_name="total_input_kw",
            formula="E_in = E_fuel + E_elec + E_steam + E_recovered",
            units="kW"
        )

        return total_input

    def _calculate_total_output(
        self,
        data: EnergyBalanceData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate total useful energy output."""
        outputs = {
            'process_heat': Decimal(str(data.process_heat_output_kw)),
            'steam_export': Decimal(str(data.steam_export_kw)),
            'electricity_generation': Decimal(str(data.electricity_generation_kw)),
            'useful_work': Decimal(str(data.useful_work_kw))
        }

        total_output = sum(outputs.values())

        tracker.record_step(
            operation="summation",
            description="Calculate total energy output",
            inputs=outputs,
            output_value=total_output,
            output_name="total_output_kw",
            formula="E_out = E_process + E_steam_exp + E_elec_gen + E_work",
            units="kW"
        )

        return total_output

    def _calculate_total_losses(
        self,
        data: EnergyBalanceData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate total energy losses."""
        losses = {
            'flue_gas': Decimal(str(data.flue_gas_loss_kw)),
            'radiation': Decimal(str(data.radiation_loss_kw)),
            'blowdown': Decimal(str(data.blowdown_loss_kw)),
            'condensate': Decimal(str(data.condensate_loss_kw)),
            'unaccounted': Decimal(str(data.unaccounted_loss_kw))
        }

        total_losses = sum(losses.values())

        tracker.record_step(
            operation="summation",
            description="Calculate total energy losses",
            inputs=losses,
            output_value=total_losses,
            output_name="total_losses_kw",
            formula="E_loss = E_flue + E_rad + E_bd + E_cond + E_unac",
            units="kW"
        )

        return total_losses

    def _calculate_storage_change(
        self,
        data: EnergyBalanceData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate change in stored energy."""
        storage_change = Decimal(str(data.thermal_storage_change_kw))

        tracker.record_step(
            operation="direct_measurement",
            description="Record thermal storage change",
            inputs={'thermal_storage_change': storage_change},
            output_value=storage_change,
            output_name="storage_change_kw",
            formula="ΔE_stored = E_stored_final - E_stored_initial",
            units="kW"
        )

        return storage_change

    def _calculate_balance(
        self,
        total_input: Decimal,
        total_output: Decimal,
        total_losses: Decimal,
        storage_change: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate energy balance and check conservation."""
        # First Law: E_in = E_out + E_losses + ΔE_stored
        right_side = total_output + total_losses + storage_change
        imbalance = total_input - right_side

        if total_input > 0:
            imbalance_percent = (abs(imbalance) / total_input) * Decimal('100')
        else:
            imbalance_percent = Decimal('0')

        # Check if within tolerance
        conservation_verified = imbalance_percent <= self.BALANCE_TOLERANCE_PERCENT

        # Determine status
        if conservation_verified:
            status = "BALANCED"
        elif imbalance > 0:
            status = "ENERGY_SURPLUS"
        else:
            status = "ENERGY_DEFICIT"

        result = {
            'status': status,
            'imbalance': imbalance,
            'imbalance_percent': imbalance_percent,
            'conservation_verified': conservation_verified,
            'left_side': total_input,
            'right_side': right_side
        }

        tracker.record_step(
            operation="balance_check",
            description="Verify energy conservation (First Law)",
            inputs={
                'total_input': total_input,
                'total_output': total_output,
                'total_losses': total_losses,
                'storage_change': storage_change
            },
            output_value=imbalance,
            output_name="energy_imbalance_kw",
            formula="Imbalance = E_in - (E_out + E_loss + ΔE_stored)",
            units="kW"
        )

        return result

    def _identify_violations(
        self,
        balance_result: Dict,
        data: EnergyBalanceData,
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """Identify specific conservation violations."""
        violations = []

        # Check overall balance
        if not balance_result['conservation_verified']:
            violations.append({
                'type': 'ENERGY_CONSERVATION',
                'severity': 'HIGH',
                'description': f"Energy imbalance of {float(balance_result['imbalance']):.2f} kW "
                             f"({float(balance_result['imbalance_percent']):.2f}% of input)",
                'threshold_percent': float(self.BALANCE_TOLERANCE_PERCENT),
                'actual_percent': float(balance_result['imbalance_percent'])
            })

        # Check for negative values (physically impossible)
        if data.fuel_energy_kw < 0:
            violations.append({
                'type': 'NEGATIVE_INPUT',
                'severity': 'CRITICAL',
                'description': 'Negative fuel energy input (impossible)',
                'value': data.fuel_energy_kw
            })

        # Check for excessive unaccounted losses
        total_input = Decimal(str(balance_result['left_side']))
        unaccounted = Decimal(str(data.unaccounted_loss_kw))
        if total_input > 0:
            unaccounted_percent = (unaccounted / total_input) * Decimal('100')
            if unaccounted_percent > Decimal('5'):
                violations.append({
                    'type': 'EXCESSIVE_UNACCOUNTED_LOSS',
                    'severity': 'MEDIUM',
                    'description': f"Unaccounted losses {float(unaccounted_percent):.2f}% exceed 5% threshold",
                    'threshold_percent': 5.0,
                    'actual_percent': float(unaccounted_percent)
                })

        # Check efficiency bounds
        if total_input > 0:
            efficiency = (Decimal(str(data.process_heat_output_kw)) / total_input) * Decimal('100')
            if efficiency > Decimal('100'):
                violations.append({
                    'type': 'EFFICIENCY_VIOLATION',
                    'severity': 'CRITICAL',
                    'description': f"Efficiency {float(efficiency):.2f}% exceeds 100% (impossible)",
                    'value': float(efficiency)
                })

        tracker.record_step(
            operation="violation_check",
            description="Identify conservation law violations",
            inputs={'num_checks': 4},
            output_value=len(violations),
            output_name="violation_count",
            formula="Rule-based violation detection",
            units="count"
        )

        return violations

    def _calculate_efficiency_metrics(
        self,
        total_input: Decimal,
        total_output: Decimal,
        total_losses: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate various efficiency metrics."""
        metrics = {}

        if total_input > 0:
            # First Law Efficiency
            first_law_eff = (total_output / total_input) * Decimal('100')
            metrics['first_law_efficiency_percent'] = float(
                first_law_eff.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            )

            # Loss ratio
            loss_ratio = (total_losses / total_input) * Decimal('100')
            metrics['loss_ratio_percent'] = float(
                loss_ratio.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            )

            # Energy Performance Indicator (EnPI)
            enpi = total_output / total_input if total_output > 0 else Decimal('0')
            metrics['energy_performance_indicator'] = float(
                enpi.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            )
        else:
            metrics['first_law_efficiency_percent'] = 0.0
            metrics['loss_ratio_percent'] = 0.0
            metrics['energy_performance_indicator'] = 0.0

        tracker.record_step(
            operation="efficiency_calculation",
            description="Calculate efficiency metrics",
            inputs={
                'total_input': total_input,
                'total_output': total_output,
                'total_losses': total_losses
            },
            output_value=metrics['first_law_efficiency_percent'],
            output_name="first_law_efficiency",
            formula="η = (E_out / E_in) × 100",
            units="%"
        )

        return metrics

    def _generate_corrective_actions(
        self,
        violations: List[Dict],
        balance_result: Dict,
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """Generate corrective actions for violations."""
        actions = []

        for violation in violations:
            if violation['type'] == 'ENERGY_CONSERVATION':
                if balance_result['status'] == 'ENERGY_SURPLUS':
                    actions.append({
                        'action': 'INVESTIGATE_UNMEASURED_OUTPUTS',
                        'priority': 'HIGH',
                        'description': 'Check for unmeasured steam vents, leaks, or bypasses',
                        'expected_impact': 'Identify missing energy flows'
                    })
                else:
                    actions.append({
                        'action': 'CALIBRATE_METERS',
                        'priority': 'HIGH',
                        'description': 'Calibrate fuel and steam flow meters',
                        'expected_impact': 'Improve measurement accuracy'
                    })

            elif violation['type'] == 'EXCESSIVE_UNACCOUNTED_LOSS':
                actions.append({
                    'action': 'CONDUCT_ENERGY_AUDIT',
                    'priority': 'MEDIUM',
                    'description': 'Perform detailed energy audit to identify hidden losses',
                    'expected_impact': f"Reduce unaccounted losses by {violation['actual_percent']/2:.1f}%"
                })

            elif violation['type'] == 'NEGATIVE_INPUT':
                actions.append({
                    'action': 'CHECK_SENSOR_WIRING',
                    'priority': 'CRITICAL',
                    'description': 'Verify sensor wiring and signal conditioning',
                    'expected_impact': 'Correct measurement errors'
                })

        tracker.record_step(
            operation="action_generation",
            description="Generate corrective actions",
            inputs={'violation_count': len(violations)},
            output_value=len(actions),
            output_name="action_count",
            formula="Rule-based action generation",
            units="count"
        )

        return actions

    def _generate_sankey_data(
        self,
        data: EnergyBalanceData,
        total_input: Decimal,
        total_output: Decimal,
        total_losses: Decimal,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Generate data for Sankey diagram visualization."""
        # Create nodes and links for energy flow visualization
        nodes = [
            {'id': 'input', 'label': 'Total Input'},
            {'id': 'fuel', 'label': 'Fuel'},
            {'id': 'electricity_in', 'label': 'Electricity In'},
            {'id': 'steam_import', 'label': 'Steam Import'},
            {'id': 'recovered', 'label': 'Recovered Heat'},
            {'id': 'system', 'label': 'System'},
            {'id': 'process_heat', 'label': 'Process Heat'},
            {'id': 'steam_export', 'label': 'Steam Export'},
            {'id': 'electricity_out', 'label': 'Electricity Out'},
            {'id': 'useful_work', 'label': 'Useful Work'},
            {'id': 'losses', 'label': 'Total Losses'}
        ]

        links = []

        # Input flows
        if data.fuel_energy_kw > 0:
            links.append({
                'source': 'fuel',
                'target': 'input',
                'value': float(data.fuel_energy_kw)
            })

        if data.electrical_energy_kw > 0:
            links.append({
                'source': 'electricity_in',
                'target': 'input',
                'value': float(data.electrical_energy_kw)
            })

        # System flows
        links.append({
            'source': 'input',
            'target': 'system',
            'value': FinancialDecimal.from_string(total_input)
        })

        # Output flows
        if data.process_heat_output_kw > 0:
            links.append({
                'source': 'system',
                'target': 'process_heat',
                'value': float(data.process_heat_output_kw)
            })

        # Loss flows
        if total_losses > 0:
            links.append({
                'source': 'system',
                'target': 'losses',
                'value': FinancialDecimal.from_string(total_losses)
            })

        sankey_data = {
            'nodes': nodes,
            'links': links,
            'total_flow': FinancialDecimal.from_string(total_input)
        }

        tracker.record_step(
            operation="visualization_prep",
            description="Prepare Sankey diagram data",
            inputs={'num_nodes': len(nodes), 'num_links': len(links)},
            output_value=len(links),
            output_name="flow_count",
            formula="Graph representation of energy flows",
            units="flows"
        )

        return sankey_data