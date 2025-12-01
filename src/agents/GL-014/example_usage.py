#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO - Example Usage Guide

This module demonstrates comprehensive usage of the GL-014 Heat Exchanger
Optimizer agent, including:

1. Basic Heat Exchanger Analysis
2. Fouling Prediction and Monitoring
3. Cleaning Schedule Optimization
4. Fleet Management for Multiple Exchangers
5. Integration with Process Historians

Prerequisites:
    pip install requests aiohttp python-dotenv

Environment Variables:
    GL014_API_URL - Base URL for the GL-014 API (default: http://localhost:8000)
    GL014_API_KEY - API key for authentication (optional for local)

Usage:
    # Run all examples
    python example_usage.py

    # Run specific example
    python example_usage.py --example basic

Author: GreenLang AI Agent Factory
License: Apache-2.0
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

# Third-party imports (install with: pip install requests aiohttp python-dotenv)
try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("Please install required packages: pip install requests aiohttp python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("GL014_API_URL", "http://localhost:8000")
API_KEY = os.getenv("GL014_API_KEY", "")


# =============================================================================
# Helper Classes
# =============================================================================


@dataclass
class TemperatureData:
    """Temperature measurements for heat exchanger."""

    hot_inlet_temp_c: float
    hot_outlet_temp_c: float
    cold_inlet_temp_c: float
    cold_outlet_temp_c: float
    ambient_temp_c: float = 25.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for API calls."""
        return {
            "hot_inlet_temp_c": self.hot_inlet_temp_c,
            "hot_outlet_temp_c": self.hot_outlet_temp_c,
            "cold_inlet_temp_c": self.cold_inlet_temp_c,
            "cold_outlet_temp_c": self.cold_outlet_temp_c,
            "ambient_temp_c": self.ambient_temp_c,
        }


@dataclass
class FlowData:
    """Flow rate measurements for heat exchanger."""

    hot_mass_flow_kg_s: float
    cold_mass_flow_kg_s: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for API calls."""
        return {
            "hot_mass_flow_kg_s": self.hot_mass_flow_kg_s,
            "cold_mass_flow_kg_s": self.cold_mass_flow_kg_s,
        }


@dataclass
class PressureData:
    """Pressure measurements for heat exchanger."""

    shell_inlet_pressure_kpa: float
    shell_outlet_pressure_kpa: float
    tube_inlet_pressure_kpa: float
    tube_outlet_pressure_kpa: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for API calls."""
        return {
            "shell_inlet_pressure_kpa": self.shell_inlet_pressure_kpa,
            "shell_outlet_pressure_kpa": self.shell_outlet_pressure_kpa,
            "tube_inlet_pressure_kpa": self.tube_inlet_pressure_kpa,
            "tube_outlet_pressure_kpa": self.tube_outlet_pressure_kpa,
        }


@dataclass
class ExchangerParameters:
    """Design parameters for heat exchanger."""

    design_heat_duty_kw: float
    design_u_w_m2k: float
    clean_u_w_m2k: float
    heat_transfer_area_m2: float
    exchanger_type: str = "shell_and_tube"
    flow_arrangement: str = "counterflow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "design_heat_duty_kw": self.design_heat_duty_kw,
            "design_u_w_m2k": self.design_u_w_m2k,
            "clean_u_w_m2k": self.clean_u_w_m2k,
            "heat_transfer_area_m2": self.heat_transfer_area_m2,
            "exchanger_type": self.exchanger_type,
            "flow_arrangement": self.flow_arrangement,
        }


# =============================================================================
# API Client
# =============================================================================


class GL014Client:
    """
    Client for interacting with the GL-014 EXCHANGER-PRO API.

    Example:
        >>> client = GL014Client(base_url="http://localhost:8000")
        >>> result = client.analyze("HX-001", temperature_data, flow_data, parameters)
        >>> print(f"U-value: {result['performance_metrics']['current_u_w_m2k']}")
    """

    def __init__(
        self,
        base_url: str = API_URL,
        api_key: Optional[str] = API_KEY,
        timeout: int = 30,
    ):
        """
        Initialize the GL-014 API client.

        Args:
            base_url: Base URL for the API (e.g., http://localhost:8000)
            api_key: API key for authentication (optional for local development)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

        # Set default headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Add API key if provided
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status dictionary

        Example:
            >>> client.health_check()
            {'status': 'healthy', 'version': '1.0.0', 'agent': 'GL-014'}
        """
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def analyze(
        self,
        exchanger_id: str,
        temperature_data: TemperatureData,
        flow_data: FlowData,
        exchanger_parameters: ExchangerParameters,
        pressure_data: Optional[PressureData] = None,
    ) -> Dict[str, Any]:
        """
        Analyze heat exchanger performance.

        Args:
            exchanger_id: Unique identifier for the heat exchanger
            temperature_data: Temperature measurements
            flow_data: Flow rate measurements
            exchanger_parameters: Design parameters
            pressure_data: Optional pressure measurements

        Returns:
            Analysis results including performance metrics, fouling analysis,
            cleaning schedule, and economic impact

        Example:
            >>> result = client.analyze(
            ...     "HX-001",
            ...     TemperatureData(150, 80, 25, 65),
            ...     FlowData(10.0, 15.0),
            ...     ExchangerParameters(3000, 500, 550, 100)
            ... )
            >>> print(f"Fouling state: {result['fouling_analysis']['fouling_state']}")
        """
        payload = {
            "exchanger_id": exchanger_id,
            "temperature_data": temperature_data.to_dict(),
            "flow_data": flow_data.to_dict(),
            "exchanger_parameters": exchanger_parameters.to_dict(),
        }

        if pressure_data:
            payload["pressure_data"] = pressure_data.to_dict()

        response = self.session.post(
            f"{self.base_url}/api/v1/analyze",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def batch_analyze(
        self,
        analyses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple heat exchangers in batch.

        Args:
            analyses: List of analysis requests

        Returns:
            List of analysis results

        Example:
            >>> results = client.batch_analyze([
            ...     {"exchanger_id": "HX-001", ...},
            ...     {"exchanger_id": "HX-002", ...},
            ... ])
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/analyze/batch",
            json={"exchangers": analyses},
            timeout=self.timeout * 2,  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()

    def get_fouling(self, exchanger_id: str) -> Dict[str, Any]:
        """
        Get fouling analysis for a specific exchanger.

        Args:
            exchanger_id: Heat exchanger identifier

        Returns:
            Fouling analysis details
        """
        response = self.session.get(
            f"{self.base_url}/api/v1/exchangers/{exchanger_id}/fouling",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_cleaning_schedule(self, exchanger_id: str) -> Dict[str, Any]:
        """
        Get cleaning schedule for a specific exchanger.

        Args:
            exchanger_id: Heat exchanger identifier

        Returns:
            Cleaning schedule details
        """
        response = self.session.get(
            f"{self.base_url}/api/v1/exchangers/{exchanger_id}/cleaning-schedule",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_economic_impact(self, exchanger_id: str) -> Dict[str, Any]:
        """
        Get economic impact analysis for a specific exchanger.

        Args:
            exchanger_id: Heat exchanger identifier

        Returns:
            Economic impact analysis
        """
        response = self.session.get(
            f"{self.base_url}/api/v1/exchangers/{exchanger_id}/economic-impact",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def register_exchanger(self, exchanger_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new heat exchanger.

        Args:
            exchanger_config: Heat exchanger configuration

        Returns:
            Registered exchanger details
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/exchangers",
            json=exchanger_config,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def list_exchangers(self) -> List[Dict[str, Any]]:
        """
        List all registered heat exchangers.

        Returns:
            List of heat exchanger configurations
        """
        response = self.session.get(
            f"{self.base_url}/api/v1/exchangers",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Example 1: Basic Heat Exchanger Analysis
# =============================================================================


def example_basic_analysis():
    """
    Example: Basic heat exchanger performance analysis.

    This example demonstrates:
    - Creating a GL-014 client
    - Submitting temperature and flow data
    - Interpreting the analysis results
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic Heat Exchanger Analysis")
    print("=" * 70)

    # Initialize client
    client = GL014Client()

    # Check API health
    try:
        health = client.health_check()
        print(f"\nAPI Status: {health['status']}")
        print(f"Agent Version: {health['version']}")
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to GL-014 API")
        print("Make sure the service is running: docker-compose up -d")
        return

    # Define heat exchanger data
    temperature_data = TemperatureData(
        hot_inlet_temp_c=150.0,
        hot_outlet_temp_c=80.0,
        cold_inlet_temp_c=25.0,
        cold_outlet_temp_c=65.0,
    )

    flow_data = FlowData(
        hot_mass_flow_kg_s=10.0,
        cold_mass_flow_kg_s=15.0,
    )

    parameters = ExchangerParameters(
        design_heat_duty_kw=3000.0,
        design_u_w_m2k=500.0,
        clean_u_w_m2k=550.0,
        heat_transfer_area_m2=100.0,
        exchanger_type="shell_and_tube",
        flow_arrangement="counterflow",
    )

    # Perform analysis
    print("\nSubmitting analysis request...")
    result = client.analyze(
        exchanger_id="HX-001",
        temperature_data=temperature_data,
        flow_data=flow_data,
        exchanger_parameters=parameters,
    )

    # Display results
    print("\n--- Performance Metrics ---")
    perf = result["performance_metrics"]
    print(f"  Current Heat Duty:    {perf['current_heat_duty_kw']:.1f} kW")
    print(f"  Design Heat Duty:     {perf['design_heat_duty_kw']:.1f} kW")
    print(f"  Heat Duty Ratio:      {perf['heat_duty_ratio']:.2%}")
    print(f"  Current U-value:      {perf['current_u_w_m2k']:.1f} W/m2K")
    print(f"  Design U-value:       {perf['design_u_w_m2k']:.1f} W/m2K")
    print(f"  U-value Ratio:        {perf['u_ratio']:.2%}")
    print(f"  LMTD:                 {perf['lmtd_k']:.1f} K")
    print(f"  Effectiveness:        {perf['effectiveness']:.2%}")
    print(f"  Performance Status:   {perf['performance_status']}")

    print("\n--- Fouling Analysis ---")
    fouling = result["fouling_analysis"]
    print(f"  Total Fouling Factor: {fouling['total_fouling_factor']:.6f} m2K/W")
    print(f"  Shell-Side Fouling:   {fouling['shell_side_fouling_factor']:.6f} m2K/W")
    print(f"  Tube-Side Fouling:    {fouling['tube_side_fouling_factor']:.6f} m2K/W")
    print(f"  Fouling State:        {fouling['fouling_state']}")
    print(f"  Days to Threshold:    {fouling['predicted_days_to_threshold']}")

    print("\n--- Cleaning Schedule ---")
    cleaning = result["cleaning_schedule"]
    print(f"  Recommended Date:     {cleaning['recommended_cleaning_date']}")
    print(f"  Cleaning Method:      {cleaning['recommended_cleaning_method']}")
    print(f"  Urgency Level:        {cleaning['urgency_level']}")
    print(f"  Estimated Cost:       ${cleaning['estimated_cleaning_cost']:,.0f}")
    print(f"  Payback Period:       {cleaning['payback_period_days']} days")

    print("\n--- Economic Impact ---")
    econ = result["economic_impact"]
    print(f"  Daily Energy Loss:    ${econ['daily_energy_cost_loss']:,.2f}")
    print(f"  Monthly Energy Loss:  ${econ['monthly_energy_cost_loss']:,.2f}")
    print(f"  Annual Energy Loss:   ${econ['annual_energy_cost_loss']:,.2f}")
    print(f"  Cleaning ROI:         {econ['cleaning_roi_percent']:.0f}%")

    print("\n--- Provenance ---")
    print(f"  Hash: {result['provenance_hash'][:60]}...")

    return result


# =============================================================================
# Example 2: Fouling Prediction
# =============================================================================


def example_fouling_prediction():
    """
    Example: Fouling prediction and monitoring.

    This example demonstrates:
    - Tracking fouling progression over time
    - Predicting time to cleaning threshold
    - Comparing Kern-Seaton and Ebert-Panchal models
    """
    print("\n" + "=" * 70)
    print("Example 2: Fouling Prediction and Monitoring")
    print("=" * 70)

    # Simulate fouling progression with degrading U-value
    days = [0, 30, 60, 90, 120, 150, 180]
    u_values = [500.0, 480.0, 455.0, 425.0, 390.0, 350.0, 310.0]

    print("\n--- Simulated Fouling Progression ---")
    print("\n  Day    U-value   Fouling Factor   Cleanliness")
    print("  ---    -------   --------------   -----------")

    u_clean = 550.0  # Clean U-value

    for day, u_current in zip(days, u_values):
        # Calculate fouling resistance: R_f = (1/U_fouled) - (1/U_clean)
        r_f = (1 / u_current) - (1 / u_clean)
        cleanliness = (u_current / u_clean) * 100

        print(f"  {day:3d}    {u_current:7.1f}     {r_f:.6f}        {cleanliness:.1f}%")

    # Predict using Kern-Seaton model
    print("\n--- Kern-Seaton Model Prediction ---")
    r_f_max = 0.002  # Asymptotic fouling resistance (m2K/W)
    tau = 180  # Time constant (days)

    print("\n  Asymptotic R_f (R_f_max): ", r_f_max, "m2K/W")
    print("  Time Constant (tau):      ", tau, "days")
    print("\n  Predicted fouling resistance at key times:")
    print(f"    At tau (63.2%):         {r_f_max * 0.632:.6f} m2K/W")
    print(f"    At 3*tau (95%):         {r_f_max * 0.95:.6f} m2K/W")
    print(f"    At 5*tau (99%):         {r_f_max * 0.99:.6f} m2K/W")

    # Determine cleaning threshold
    cleaning_threshold = 0.001  # m2K/W
    print(f"\n  Cleaning threshold:       {cleaning_threshold} m2K/W")

    # Calculate time to threshold
    import math

    time_to_threshold = -tau * math.log(1 - cleaning_threshold / r_f_max)
    print(f"  Predicted time to threshold: {time_to_threshold:.0f} days")

    return {
        "days": days,
        "u_values": u_values,
        "time_to_threshold": time_to_threshold,
    }


# =============================================================================
# Example 3: Cleaning Optimization
# =============================================================================


def example_cleaning_optimization():
    """
    Example: Cleaning schedule optimization.

    This example demonstrates:
    - Comparing cleaning methods
    - Cost-benefit analysis
    - Optimal cleaning interval calculation
    """
    print("\n" + "=" * 70)
    print("Example 3: Cleaning Schedule Optimization")
    print("=" * 70)

    # Define cleaning methods with their characteristics
    cleaning_methods = {
        "chemical_online": {
            "name": "Online Chemical Cleaning",
            "cost": 8000,
            "downtime_hours": 0,
            "effectiveness": 0.75,
            "duration_hours": 8,
        },
        "chemical_offline": {
            "name": "Offline Chemical Cleaning",
            "cost": 18000,
            "downtime_hours": 24,
            "effectiveness": 0.85,
            "duration_hours": 24,
        },
        "mechanical": {
            "name": "Mechanical Cleaning",
            "cost": 30000,
            "downtime_hours": 48,
            "effectiveness": 0.95,
            "duration_hours": 48,
        },
        "hydroblast": {
            "name": "Hydroblast Cleaning",
            "cost": 40000,
            "downtime_hours": 36,
            "effectiveness": 0.92,
            "duration_hours": 36,
        },
    }

    # Economic parameters
    energy_loss_per_day = 500  # $/day at current fouling
    downtime_cost_per_hour = 5000  # $/hour
    operating_days_per_year = 350

    print("\n--- Cleaning Method Comparison ---")
    print("\n  Method                    Cost      Downtime  Effectiveness  Total Cost*")
    print("  ------                    ----      --------  -------------  ----------")

    results = []
    for method_id, method in cleaning_methods.items():
        total_cost = method["cost"] + (method["downtime_hours"] * downtime_cost_per_hour)

        # Calculate annual savings
        energy_savings = energy_loss_per_day * method["effectiveness"] * operating_days_per_year
        net_annual_benefit = energy_savings - total_cost
        roi = (net_annual_benefit / total_cost) * 100

        results.append({
            "method": method["name"],
            "total_cost": total_cost,
            "annual_savings": energy_savings,
            "roi": roi,
        })

        print(
            f"  {method['name']:<25} ${method['cost']:>7,}  {method['downtime_hours']:>4} hrs     "
            f"{method['effectiveness']:>6.0%}        ${total_cost:>9,}"
        )

    print("\n  * Total cost includes downtime at $5,000/hour")

    # Calculate optimal cleaning frequency
    print("\n--- Optimal Cleaning Frequency Analysis ---")

    # Simplified model: cleaning when marginal benefit = marginal cost
    fouling_rate = 0.0001  # m2K/W per month
    energy_cost_sensitivity = 50000  # $/year per 0.0001 m2K/W fouling

    for result in results:
        # Simple payback calculation
        monthly_savings = result["annual_savings"] / 12
        payback_months = result["total_cost"] / monthly_savings if monthly_savings > 0 else float("inf")

        print(f"\n  {result['method']}:")
        print(f"    Monthly savings:        ${monthly_savings:,.0f}")
        print(f"    Payback period:         {payback_months:.1f} months")
        print(f"    Annual ROI:             {result['roi']:.0f}%")

    # Recommend best method
    best_method = max(results, key=lambda x: x["roi"])
    print(f"\n  Recommended: {best_method['method']} (ROI: {best_method['roi']:.0f}%)")

    return results


# =============================================================================
# Example 4: Fleet Management
# =============================================================================


def example_fleet_management():
    """
    Example: Fleet management for multiple heat exchangers.

    This example demonstrates:
    - Batch analysis of multiple exchangers
    - Fleet-wide cleaning schedule optimization
    - Resource planning
    """
    print("\n" + "=" * 70)
    print("Example 4: Fleet Management")
    print("=" * 70)

    # Define a fleet of heat exchangers
    fleet = [
        {
            "exchanger_id": "HX-101",
            "name": "Crude Preheat #1",
            "area_m2": 250,
            "u_current": 380,
            "u_clean": 500,
            "fouling_state": "moderate",
            "days_since_cleaning": 180,
            "criticality": "high",
        },
        {
            "exchanger_id": "HX-102",
            "name": "Crude Preheat #2",
            "area_m2": 250,
            "u_current": 420,
            "u_clean": 500,
            "fouling_state": "light",
            "days_since_cleaning": 90,
            "criticality": "high",
        },
        {
            "exchanger_id": "HX-201",
            "name": "Overhead Condenser",
            "area_m2": 400,
            "u_current": 550,
            "u_clean": 600,
            "fouling_state": "light",
            "days_since_cleaning": 120,
            "criticality": "critical",
        },
        {
            "exchanger_id": "HX-301",
            "name": "Product Cooler #1",
            "area_m2": 150,
            "u_current": 280,
            "u_clean": 400,
            "fouling_state": "heavy",
            "days_since_cleaning": 365,
            "criticality": "medium",
        },
        {
            "exchanger_id": "HX-302",
            "name": "Product Cooler #2",
            "area_m2": 150,
            "u_current": 350,
            "u_clean": 400,
            "fouling_state": "moderate",
            "days_since_cleaning": 200,
            "criticality": "medium",
        },
    ]

    print("\n--- Fleet Overview ---")
    print("\n  ID       Name                    Area    U-ratio  Fouling   Priority")
    print("  --       ----                    ----    -------  -------   --------")

    # Calculate metrics for each exchanger
    fleet_metrics = []
    for hx in fleet:
        u_ratio = hx["u_current"] / hx["u_clean"]
        r_f = (1 / hx["u_current"]) - (1 / hx["u_clean"])

        # Calculate priority score (higher = more urgent)
        priority_score = (1 - u_ratio) * 100
        if hx["criticality"] == "critical":
            priority_score *= 1.5
        elif hx["criticality"] == "high":
            priority_score *= 1.2

        fleet_metrics.append({
            **hx,
            "u_ratio": u_ratio,
            "r_f": r_f,
            "priority_score": priority_score,
        })

        print(
            f"  {hx['exchanger_id']:<8} {hx['name']:<22} {hx['area_m2']:>5} m2  "
            f"{u_ratio:>6.1%}   {hx['fouling_state']:<9} {priority_score:.1f}"
        )

    # Sort by priority
    fleet_metrics.sort(key=lambda x: x["priority_score"], reverse=True)

    print("\n--- Recommended Cleaning Schedule ---")
    print("\n  Priority  Exchanger  Name                    Recommended   Est. Cost")
    print("  --------  ---------  ----                    -----------   ---------")

    total_cost = 0
    cleaning_schedule = []
    base_date = datetime.now()

    for i, hx in enumerate(fleet_metrics):
        # Determine cleaning method based on fouling state
        if hx["fouling_state"] in ["heavy", "severe"]:
            method = "mechanical"
            cost = 30000
        elif hx["fouling_state"] == "moderate":
            method = "chemical_offline"
            cost = 18000
        else:
            method = "chemical_online"
            cost = 8000

        # Schedule based on priority (spread over time)
        scheduled_date = base_date + timedelta(days=i * 14)  # 2 weeks apart

        cleaning_schedule.append({
            "exchanger_id": hx["exchanger_id"],
            "name": hx["name"],
            "date": scheduled_date,
            "method": method,
            "cost": cost,
        })

        total_cost += cost

        print(
            f"  {i + 1:<8} {hx['exchanger_id']:<10} {hx['name']:<22} "
            f"{scheduled_date.strftime('%Y-%m-%d')}   ${cost:>7,}"
        )

    print(f"\n  Total Estimated Cost: ${total_cost:,}")

    # Resource planning summary
    print("\n--- Resource Planning Summary ---")
    method_counts = {}
    for item in cleaning_schedule:
        method_counts[item["method"]] = method_counts.get(item["method"], 0) + 1

    print("\n  Cleaning methods required:")
    for method, count in method_counts.items():
        print(f"    {method}: {count} exchangers")

    print(f"\n  Timeline: {cleaning_schedule[0]['date'].strftime('%Y-%m-%d')} to "
          f"{cleaning_schedule[-1]['date'].strftime('%Y-%m-%d')}")

    return fleet_metrics, cleaning_schedule


# =============================================================================
# Example 5: Process Historian Integration
# =============================================================================


def example_historian_integration():
    """
    Example: Integration with process historians.

    This example demonstrates:
    - Configuring tag mappings for OSIsoft PI
    - Retrieving historical data
    - Setting up continuous monitoring
    """
    print("\n" + "=" * 70)
    print("Example 5: Process Historian Integration")
    print("=" * 70)

    # Note: This example uses mock data since actual historian connectivity
    # requires specific infrastructure

    # Tag mapping configuration for OSIsoft PI
    tag_mapping = {
        "exchanger_id": "HX-001",
        "historian_type": "osisoft_pi",
        "tags": {
            "hot_inlet_temp": "UNIT1.HX001.TI101.PV",
            "hot_outlet_temp": "UNIT1.HX001.TI102.PV",
            "cold_inlet_temp": "UNIT1.HX001.TI103.PV",
            "cold_outlet_temp": "UNIT1.HX001.TI104.PV",
            "hot_flow": "UNIT1.HX001.FI101.PV",
            "cold_flow": "UNIT1.HX001.FI102.PV",
            "shell_inlet_pressure": "UNIT1.HX001.PI101.PV",
            "shell_outlet_pressure": "UNIT1.HX001.PI102.PV",
        },
        "polling_interval_seconds": 60,
        "data_quality_threshold": 0.95,
    }

    print("\n--- Tag Mapping Configuration ---")
    print(f"\n  Exchanger: {tag_mapping['exchanger_id']}")
    print(f"  Historian: {tag_mapping['historian_type']}")
    print(f"  Polling:   {tag_mapping['polling_interval_seconds']}s")
    print("\n  Tag Mappings:")
    for measurement, tag in tag_mapping["tags"].items():
        print(f"    {measurement:<25} -> {tag}")

    # Simulated historian data retrieval
    print("\n--- Simulated Historical Data Retrieval ---")

    # Mock data representing a 24-hour period with hourly readings
    mock_data = []
    base_time = datetime.now() - timedelta(hours=24)

    for hour in range(24):
        timestamp = base_time + timedelta(hours=hour)

        # Simulate slight variations and gradual fouling effect
        fouling_factor = 1.0 + (hour * 0.001)  # Slight increase over time

        mock_data.append({
            "timestamp": timestamp.isoformat(),
            "hot_inlet_temp": 150.0 + (hour * 0.1),  # Slight variation
            "hot_outlet_temp": 80.0 + (hour * 0.2 * fouling_factor),  # Increases with fouling
            "cold_inlet_temp": 25.0 + (hour * 0.05),
            "cold_outlet_temp": 65.0 - (hour * 0.1 * fouling_factor),  # Decreases with fouling
            "hot_flow": 10.0 + (0.1 if hour % 3 == 0 else 0),  # Slight variations
            "cold_flow": 15.0 - (0.1 if hour % 4 == 0 else 0),
        })

    # Display sample of retrieved data
    print("\n  Time                    T_hot_in  T_hot_out  T_cold_in  T_cold_out  Flow_hot  Flow_cold")
    print("  ----                    --------  ---------  ---------  ----------  --------  ---------")

    for i in [0, 6, 12, 18, 23]:
        d = mock_data[i]
        print(
            f"  {d['timestamp'][11:19]}               "
            f"{d['hot_inlet_temp']:>6.1f}     {d['hot_outlet_temp']:>6.1f}     "
            f"{d['cold_inlet_temp']:>6.1f}      {d['cold_outlet_temp']:>6.1f}      "
            f"{d['hot_flow']:>5.1f}     {d['cold_flow']:>6.1f}"
        )

    # Calculate trends
    print("\n--- Trend Analysis ---")

    u_values = []
    for d in mock_data:
        # Simplified U-value calculation for demonstration
        q_hot = d["hot_flow"] * 2100 * (d["hot_inlet_temp"] - d["hot_outlet_temp"])
        lmtd_approx = ((d["hot_inlet_temp"] - d["cold_outlet_temp"]) +
                       (d["hot_outlet_temp"] - d["cold_inlet_temp"])) / 2
        u_approx = q_hot / (100 * lmtd_approx) if lmtd_approx > 0 else 0
        u_values.append(u_approx)

    avg_u = sum(u_values) / len(u_values)
    trend = (u_values[-1] - u_values[0]) / u_values[0] * 100 if u_values[0] > 0 else 0

    print(f"\n  Average U-value:      {avg_u:.1f} W/m2K")
    print(f"  24-hour U-value trend: {trend:+.2f}%")

    if trend < -1:
        print("  Status: Declining - fouling may be accelerating")
    elif trend > 1:
        print("  Status: Improving - possible process change")
    else:
        print("  Status: Stable - normal operation")

    # Continuous monitoring configuration
    print("\n--- Continuous Monitoring Setup ---")

    monitoring_config = {
        "analysis_interval_minutes": 15,
        "alert_thresholds": {
            "u_ratio_warning": 0.85,
            "u_ratio_critical": 0.70,
            "fouling_rate_warning": 0.00001,  # m2K/W per day
            "fouling_rate_critical": 0.00005,
        },
        "alert_destinations": [
            {"type": "email", "address": "operator@example.com"},
            {"type": "sms", "number": "+1234567890"},
            {"type": "webhook", "url": "https://alerts.example.com/gl014"},
        ],
    }

    print(f"\n  Analysis interval: Every {monitoring_config['analysis_interval_minutes']} minutes")
    print("\n  Alert thresholds:")
    for threshold, value in monitoring_config["alert_thresholds"].items():
        print(f"    {threshold}: {value}")
    print("\n  Alert destinations configured: ", len(monitoring_config["alert_destinations"]))

    return tag_mapping, mock_data


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """
    Run all GL-014 examples.

    Usage:
        python example_usage.py              # Run all examples
        python example_usage.py --example basic   # Run specific example
    """
    parser = argparse.ArgumentParser(
        description="GL-014 EXCHANGER-PRO Example Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python example_usage.py              Run all examples
    python example_usage.py --example basic   Basic analysis
    python example_usage.py --example fouling   Fouling prediction
    python example_usage.py --example cleaning   Cleaning optimization
    python example_usage.py --example fleet   Fleet management
    python example_usage.py --example historian   Historian integration
        """,
    )

    parser.add_argument(
        "--example",
        choices=["basic", "fouling", "cleaning", "fleet", "historian", "all"],
        default="all",
        help="Which example to run (default: all)",
    )

    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"GL-014 API URL (default: {API_URL})",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GL-014 EXCHANGER-PRO - Example Usage Guide")
    print("=" * 70)
    print(f"\nAPI URL: {args.api_url}")
    print(f"Time: {datetime.now().isoformat()}")

    examples = {
        "basic": example_basic_analysis,
        "fouling": example_fouling_prediction,
        "cleaning": example_cleaning_optimization,
        "fleet": example_fleet_management,
        "historian": example_historian_integration,
    }

    if args.example == "all":
        for name, func in examples.items():
            try:
                func()
            except requests.exceptions.ConnectionError:
                print(f"\nSkipping {name} example - API not available")
                print("Run 'docker-compose up -d' to start the GL-014 service")
            except Exception as e:
                print(f"\nError in {name} example: {e}")
    else:
        try:
            examples[args.example]()
        except requests.exceptions.ConnectionError:
            print(f"\nError: Cannot connect to GL-014 API at {args.api_url}")
            print("Run 'docker-compose up -d' to start the GL-014 service")
        except Exception as e:
            print(f"\nError: {e}")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run 'docker-compose up -d' to start GL-014")
    print("  2. Visit http://localhost:8000/docs for API documentation")
    print("  3. See README.md for full documentation")
    print("=" * 70)


if __name__ == "__main__":
    main()
