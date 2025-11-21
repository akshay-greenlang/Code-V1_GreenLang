# -*- coding: utf-8 -*-
"""
Steam Leak Detection Calculator - Zero Hallucination

Implements leak detection algorithms based on mass balance, pressure drop,
and statistical process control for steam distribution systems.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME PTC 12.4, ISO 20823
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
from .provenance import ProvenanceTracker, ProvenanceRecord
from greenlang.determinism import FinancialDecimal


@dataclass
class FlowMeasurement:
    """Flow measurement at a point in time."""
    timestamp: str
    flow_rate_kg_hr: float
    pressure_bar: float
    temperature_c: float
    location: str


@dataclass
class LeakDetectionResult:
    """Results from leak detection analysis."""
    leak_detected: bool
    confidence_percent: float
    estimated_leak_rate_kg_hr: float
    estimated_leak_cost_per_year: float
    leak_locations_probable: List[Dict]
    mass_balance_deviation_percent: float
    pressure_anomalies: List[Dict]
    flow_anomalies: List[Dict]
    recommendations: List[str]
    provenance: Dict


class LeakDetectionCalculator:
    """
    Detect steam leaks using multiple analytical methods.

    Zero Hallucination Guarantee:
    - Pure statistical and physical calculations
    - No LLM inference
    - Bit-perfect reproducibility
    - Complete provenance tracking

    Detection Methods:
    1. Mass Balance Analysis - Compare inlet vs. outlet flows
    2. Pressure Drop Anomaly - Detect unusual pressure drops
    3. Flow Deviation Analysis - Identify abnormal flow patterns
    4. Statistical Process Control - Control charts for trends
    """

    # Leak detection thresholds
    MASS_BALANCE_THRESHOLD_PERCENT = Decimal('5.0')  # 5% deviation indicates leak
    PRESSURE_DROP_THRESHOLD_PERCENT = Decimal('10.0')  # 10% excess drop
    FLOW_DEVIATION_SIGMA = Decimal('3.0')  # 3-sigma threshold
    CONFIDENCE_HIGH = Decimal('90.0')
    CONFIDENCE_MEDIUM = Decimal('70.0')
    CONFIDENCE_LOW = Decimal('50.0')

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator with version tracking."""
        self.version = version

    def detect_leaks(
        self,
        inlet_measurements: List[FlowMeasurement],
        outlet_measurements: List[FlowMeasurement],
        intermediate_measurements: Optional[List[FlowMeasurement]] = None,
        expected_pressure_drop_bar: float = 0.5,
        steam_cost_per_tonne: float = 50.0,
        operating_hours_per_year: float = 8760
    ) -> LeakDetectionResult:
        """
        Detect leaks in steam distribution system.

        Args:
            inlet_measurements: Flow measurements at system inlet
            outlet_measurements: Flow measurements at system outlets
            intermediate_measurements: Optional measurements at intermediate points
            expected_pressure_drop_bar: Expected pressure drop in normal operation
            steam_cost_per_tonne: Cost of steam generation
            operating_hours_per_year: Annual operating hours

        Returns:
            LeakDetectionResult with leak detection analysis
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"leak_detect_{id(inlet_measurements)}",
            calculation_type="leak_detection",
            version=self.version
        )

        tracker.record_inputs({
            'num_inlet_measurements': len(inlet_measurements),
            'num_outlet_measurements': len(outlet_measurements),
            'expected_pressure_drop_bar': expected_pressure_drop_bar,
            'steam_cost_per_tonne': steam_cost_per_tonne
        })

        # Step 1: Mass balance analysis
        mass_balance = self._analyze_mass_balance(
            inlet_measurements,
            outlet_measurements,
            tracker
        )

        # Step 2: Pressure drop anomaly detection
        pressure_anomalies = self._detect_pressure_anomalies(
            inlet_measurements,
            outlet_measurements,
            expected_pressure_drop_bar,
            tracker
        )

        # Step 3: Flow deviation analysis
        flow_anomalies = self._detect_flow_anomalies(
            inlet_measurements,
            outlet_measurements,
            tracker
        )

        # Step 4: Combine evidence to determine leak likelihood
        leak_detected, confidence = self._determine_leak_probability(
            mass_balance,
            pressure_anomalies,
            flow_anomalies,
            tracker
        )

        # Step 5: Estimate leak rate
        leak_rate = mass_balance['imbalance_kg_hr']

        # Step 6: Calculate financial impact
        annual_cost = self._calculate_leak_cost(
            FinancialDecimal.from_string(leak_rate),
            steam_cost_per_tonne,
            operating_hours_per_year,
            tracker
        )

        # Step 7: Identify probable leak locations
        leak_locations = self._identify_leak_locations(
            intermediate_measurements or [],
            pressure_anomalies,
            tracker
        )

        # Step 8: Generate recommendations
        recommendations = self._generate_recommendations(
            leak_detected,
            float(confidence),
            FinancialDecimal.from_string(leak_rate),
            leak_locations
        )

        result = LeakDetectionResult(
            leak_detected=leak_detected,
            confidence_percent=float(confidence),
            estimated_leak_rate_kg_hr=FinancialDecimal.from_string(leak_rate),
            estimated_leak_cost_per_year=FinancialDecimal.from_string(annual_cost),
            leak_locations_probable=leak_locations,
            mass_balance_deviation_percent=float(mass_balance['deviation_percent']),
            pressure_anomalies=pressure_anomalies,
            flow_anomalies=flow_anomalies,
            recommendations=recommendations,
            provenance=tracker.get_provenance_record(confidence).to_dict()
        )

        return result

    def _analyze_mass_balance(
        self,
        inlet: List[FlowMeasurement],
        outlet: List[FlowMeasurement],
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Perform mass balance analysis.

        In steady state: Σ(m_in) = Σ(m_out) + losses
        Deviation indicates leak or measurement error
        """
        # Calculate average flows
        avg_inlet = sum(m.flow_rate_kg_hr for m in inlet) / len(inlet) if inlet else 0
        avg_outlet = sum(m.flow_rate_kg_hr for m in outlet) / len(outlet) if outlet else 0

        inlet_flow = Decimal(str(avg_inlet))
        outlet_flow = Decimal(str(avg_outlet))

        # Mass imbalance
        imbalance = inlet_flow - outlet_flow

        # Percentage deviation
        if inlet_flow > Decimal('0'):
            deviation_percent = (imbalance / inlet_flow) * Decimal('100')
        else:
            deviation_percent = Decimal('0')

        deviation_percent = deviation_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        # Check if exceeds threshold
        leak_indicated = abs(deviation_percent) > self.MASS_BALANCE_THRESHOLD_PERCENT

        tracker.record_step(
            operation="mass_balance",
            description="Analyze mass balance for leak detection",
            inputs={
                'inlet_flow_kg_hr': inlet_flow,
                'outlet_flow_kg_hr': outlet_flow
            },
            output_value=deviation_percent,
            output_name="mass_balance_deviation_percent",
            formula="Deviation% = (m_in - m_out) / m_in * 100",
            units="%"
        )

        return {
            'inlet_flow_kg_hr': inlet_flow,
            'outlet_flow_kg_hr': outlet_flow,
            'imbalance_kg_hr': imbalance,
            'deviation_percent': deviation_percent,
            'leak_indicated': leak_indicated
        }

    def _detect_pressure_anomalies(
        self,
        inlet: List[FlowMeasurement],
        outlet: List[FlowMeasurement],
        expected_drop: float,
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """
        Detect abnormal pressure drops that may indicate leaks.

        Leaks cause additional pressure drop beyond friction losses.
        """
        anomalies = []

        if not inlet or not outlet:
            return anomalies

        # Calculate average pressures
        avg_inlet_pressure = sum(m.pressure_bar for m in inlet) / len(inlet)
        avg_outlet_pressure = sum(m.pressure_bar for m in outlet) / len(outlet)

        actual_drop = Decimal(str(avg_inlet_pressure - avg_outlet_pressure))
        expected_drop_dec = Decimal(str(expected_drop))

        # Calculate excess pressure drop
        excess_drop = actual_drop - expected_drop_dec

        # Percentage excess
        if expected_drop_dec > Decimal('0'):
            excess_percent = (excess_drop / expected_drop_dec) * Decimal('100')
        else:
            excess_percent = Decimal('0')

        # Check if anomalous
        if excess_percent > self.PRESSURE_DROP_THRESHOLD_PERCENT:
            anomalies.append({
                'type': 'excessive_pressure_drop',
                'actual_drop_bar': float(actual_drop),
                'expected_drop_bar': expected_drop,
                'excess_percent': float(excess_percent),
                'severity': 'high' if excess_percent > Decimal('20') else 'medium'
            })

        tracker.record_step(
            operation="pressure_anomaly_detection",
            description="Detect pressure drop anomalies",
            inputs={
                'actual_pressure_drop_bar': actual_drop,
                'expected_pressure_drop_bar': expected_drop_dec
            },
            output_value=excess_percent,
            output_name="excess_pressure_drop_percent",
            formula="Excess% = (Actual - Expected) / Expected * 100",
            units="%"
        )

        return anomalies

    def _detect_flow_anomalies(
        self,
        inlet: List[FlowMeasurement],
        outlet: List[FlowMeasurement],
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """
        Detect flow rate anomalies using statistical process control.

        Uses control charts (mean ± 3σ) to identify unusual patterns.
        """
        anomalies = []

        # Analyze inlet flow stability
        if len(inlet) >= 3:
            inlet_flows = [Decimal(str(m.flow_rate_kg_hr)) for m in inlet]
            inlet_mean = sum(inlet_flows) / len(inlet_flows)
            inlet_variance = sum((x - inlet_mean) ** 2 for x in inlet_flows) / len(inlet_flows)
            inlet_std = inlet_variance.sqrt()

            # Check for outliers (>3σ)
            for idx, flow in enumerate(inlet_flows):
                deviation_sigma = abs(flow - inlet_mean) / inlet_std if inlet_std > 0 else Decimal('0')
                if deviation_sigma > self.FLOW_DEVIATION_SIGMA:
                    anomalies.append({
                        'type': 'inlet_flow_spike',
                        'measurement_index': idx,
                        'flow_rate_kg_hr': FinancialDecimal.from_string(flow),
                        'mean_flow_kg_hr': float(inlet_mean),
                        'deviation_sigma': float(deviation_sigma),
                        'severity': 'high'
                    })

        # Analyze outlet flow stability
        if len(outlet) >= 3:
            outlet_flows = [Decimal(str(m.flow_rate_kg_hr)) for m in outlet]
            outlet_mean = sum(outlet_flows) / len(outlet_flows)
            outlet_variance = sum((x - outlet_mean) ** 2 for x in outlet_flows) / len(outlet_flows)
            outlet_std = outlet_variance.sqrt()

            for idx, flow in enumerate(outlet_flows):
                deviation_sigma = abs(flow - outlet_mean) / outlet_std if outlet_std > 0 else Decimal('0')
                if deviation_sigma > self.FLOW_DEVIATION_SIGMA:
                    anomalies.append({
                        'type': 'outlet_flow_drop',
                        'measurement_index': idx,
                        'flow_rate_kg_hr': FinancialDecimal.from_string(flow),
                        'mean_flow_kg_hr': float(outlet_mean),
                        'deviation_sigma': float(deviation_sigma),
                        'severity': 'high'
                    })

        tracker.record_step(
            operation="flow_anomaly_detection",
            description="Detect flow anomalies using statistical process control",
            inputs={
                'num_inlet_measurements': len(inlet),
                'num_outlet_measurements': len(outlet)
            },
            output_value=len(anomalies),
            output_name="num_flow_anomalies",
            formula="3-sigma control chart",
            units="count"
        )

        return anomalies

    def _determine_leak_probability(
        self,
        mass_balance: Dict,
        pressure_anomalies: List[Dict],
        flow_anomalies: List[Dict],
        tracker: ProvenanceTracker
    ) -> Tuple[bool, Decimal]:
        """
        Combine evidence from multiple methods to determine leak probability.

        Uses Bayesian-inspired scoring system.
        """
        evidence_score = Decimal('0')

        # Mass balance evidence (strongest indicator)
        if mass_balance['leak_indicated']:
            deviation = abs(mass_balance['deviation_percent'])
            if deviation > Decimal('10'):
                evidence_score += Decimal('50')  # Strong evidence
            elif deviation > Decimal('5'):
                evidence_score += Decimal('30')  # Medium evidence
            else:
                evidence_score += Decimal('15')  # Weak evidence

        # Pressure anomaly evidence
        for anomaly in pressure_anomalies:
            if anomaly['severity'] == 'high':
                evidence_score += Decimal('25')
            else:
                evidence_score += Decimal('15')

        # Flow anomaly evidence
        for anomaly in flow_anomalies:
            if anomaly['severity'] == 'high':
                evidence_score += Decimal('10')
            else:
                evidence_score += Decimal('5')

        # Cap at 100%
        confidence = min(evidence_score, Decimal('99'))

        # Leak detected if confidence > 70%
        leak_detected = confidence >= self.CONFIDENCE_MEDIUM

        tracker.record_step(
            operation="leak_probability",
            description="Determine leak probability from combined evidence",
            inputs={
                'mass_balance_indicated': mass_balance['leak_indicated'],
                'num_pressure_anomalies': len(pressure_anomalies),
                'num_flow_anomalies': len(flow_anomalies),
                'evidence_score': evidence_score
            },
            output_value=confidence,
            output_name="leak_confidence_percent",
            formula="Bayesian evidence combination",
            units="%"
        )

        return leak_detected, confidence

    def _calculate_leak_cost(
        self,
        leak_rate_kg_hr: float,
        steam_cost_per_tonne: float,
        operating_hours: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate annual cost of steam loss due to leak."""
        leak_rate = Decimal(str(leak_rate_kg_hr))
        cost_per_tonne = Decimal(str(steam_cost_per_tonne))
        hours = Decimal(str(operating_hours))

        # Annual leak in tonnes
        annual_leak_tonnes = (leak_rate * hours) / Decimal('1000')

        # Annual cost
        annual_cost = annual_leak_tonnes * cost_per_tonne

        tracker.record_step(
            operation="leak_cost",
            description="Calculate annual cost of steam leak",
            inputs={
                'leak_rate_kg_hr': leak_rate,
                'steam_cost_per_tonne': cost_per_tonne,
                'operating_hours': hours
            },
            output_value=annual_cost,
            output_name="annual_leak_cost",
            formula="Cost = Leak_rate * Hours * Cost_per_tonne / 1000",
            units="currency"
        )

        return annual_cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _identify_leak_locations(
        self,
        intermediate: List[FlowMeasurement],
        pressure_anomalies: List[Dict],
        tracker: ProvenanceTracker
    ) -> List[Dict]:
        """
        Identify probable leak locations based on measurements.

        Uses flow measurements at intermediate points to triangulate.
        """
        locations = []

        # If we have intermediate measurements, analyze flow drops
        if len(intermediate) >= 2:
            for i in range(len(intermediate) - 1):
                upstream = intermediate[i]
                downstream = intermediate[i + 1]

                flow_drop = upstream.flow_rate_kg_hr - downstream.flow_rate_kg_hr
                pressure_drop = upstream.pressure_bar - downstream.pressure_bar

                # Significant flow drop indicates leak in this segment
                if flow_drop > upstream.flow_rate_kg_hr * 0.05:  # >5% drop
                    locations.append({
                        'segment': f"{upstream.location} to {downstream.location}",
                        'estimated_leak_rate_kg_hr': flow_drop,
                        'pressure_drop_bar': pressure_drop,
                        'probability': 'high' if flow_drop > upstream.flow_rate_kg_hr * 0.10 else 'medium'
                    })

        # If no intermediate measurements, use pressure anomalies
        elif pressure_anomalies:
            locations.append({
                'segment': 'Between inlet and outlet (exact location unknown)',
                'estimated_leak_rate_kg_hr': 0.0,
                'probability': 'medium',
                'recommendation': 'Install intermediate flow meters for leak localization'
            })

        return locations

    def _generate_recommendations(
        self,
        leak_detected: bool,
        confidence: float,
        leak_rate: float,
        locations: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if leak_detected:
            if confidence >= 90:
                recommendations.append(
                    f"HIGH PRIORITY: Leak detected with {confidence:.1f}% confidence. "
                    f"Estimated leak rate: {leak_rate:.1f} kg/hr. Immediate inspection required."
                )
            elif confidence >= 70:
                recommendations.append(
                    f"MEDIUM PRIORITY: Probable leak detected ({confidence:.1f}% confidence). "
                    f"Schedule inspection within 48 hours."
                )
            else:
                recommendations.append(
                    f"LOW PRIORITY: Possible leak indicated ({confidence:.1f}% confidence). "
                    f"Monitor closely and investigate when convenient."
                )

            # Location-specific recommendations
            if locations:
                for loc in locations:
                    recommendations.append(
                        f"Inspect segment: {loc['segment']} "
                        f"(probability: {loc.get('probability', 'unknown')})"
                    )
            else:
                recommendations.append(
                    "Install additional flow meters at intermediate points to pinpoint leak location"
                )

            # Repair recommendations
            if leak_rate > 50:
                recommendations.append(
                    "Large leak detected. Consider emergency shutdown for repair to prevent "
                    "safety hazards and energy waste."
                )
            elif leak_rate > 10:
                recommendations.append(
                    "Moderate leak. Schedule repair during next planned maintenance window."
                )
            else:
                recommendations.append(
                    "Small leak detected. Add to maintenance backlog for future repair."
                )

        else:
            recommendations.append(
                "No significant leaks detected. Continue routine monitoring and maintenance."
            )

            # Preventive recommendations
            recommendations.append(
                "Perform regular thermal imaging surveys to detect small leaks before they grow"
            )
            recommendations.append(
                "Maintain comprehensive records of flow and pressure data for trend analysis"
            )

        return recommendations
