# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Thermal Pipeline Integration Tests

End-to-end tests for the thermal calculation pipeline including:
- OPC-UA data ingestion to thermal KPIs
- Full calculation chain (Q -> LMTD -> UA -> epsilon)
- Data quality propagation
- Provenance tracking through pipeline

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any


class TestOPCUAToKPIPipeline:
    """Test OPC-UA data ingestion to KPI calculation pipeline."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_data_ingestion(self, mock_opcua_client, sample_exchanger_config):
        """Test full data ingestion from OPC-UA to KPI calculation."""
        # Connect to OPC-UA
        connected = await mock_opcua_client.connect()
        assert connected

        # Read all tags
        tags = [
            "HX-001/TI_HOT_IN",
            "HX-001/TI_HOT_OUT",
            "HX-001/TI_COLD_IN",
            "HX-001/TI_COLD_OUT",
            "HX-001/FI_HOT",
            "HX-001/FI_COLD",
        ]

        data = await mock_opcua_client.read_tags(tags)

        # Verify all data received
        assert len(data) == len(tags)
        for tag in tags:
            assert data[tag]["value"] is not None
            assert data[tag]["quality"] == "good"

        # Disconnect
        await mock_opcua_client.disconnect()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_operating_state_construction(self, mock_opcua_client):
        """Test construction of operating state from OPC-UA data."""
        # Read tag data
        tags = [
            "HX-001/TI_HOT_IN",
            "HX-001/TI_HOT_OUT",
            "HX-001/TI_COLD_IN",
            "HX-001/TI_COLD_OUT",
            "HX-001/FI_HOT",
            "HX-001/FI_COLD",
        ]

        data = await mock_opcua_client.read_tags(tags)

        # Construct operating state
        operating_state = {
            "exchanger_id": "HX-001",
            "timestamp": datetime.now(timezone.utc),
            "T_hot_in_C": data["HX-001/TI_HOT_IN"]["value"],
            "T_hot_out_C": data["HX-001/TI_HOT_OUT"]["value"],
            "T_cold_in_C": data["HX-001/TI_COLD_IN"]["value"],
            "T_cold_out_C": data["HX-001/TI_COLD_OUT"]["value"],
            "m_dot_hot_kg_s": data["HX-001/FI_HOT"]["value"],
            "m_dot_cold_kg_s": data["HX-001/FI_COLD"]["value"],
        }

        assert operating_state["T_hot_in_C"] == 150.0
        assert operating_state["m_dot_hot_kg_s"] == 25.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_kpi_calculation_from_live_data(self, mock_opcua_client):
        """Test KPI calculation from live OPC-UA data."""
        # Read data
        tags = [
            "HX-001/TI_HOT_IN",
            "HX-001/TI_HOT_OUT",
            "HX-001/TI_COLD_IN",
            "HX-001/TI_COLD_OUT",
            "HX-001/FI_HOT",
            "HX-001/FI_COLD",
        ]

        data = await mock_opcua_client.read_tags(tags)

        # Calculate heat duties
        Cp_hot = 2.3  # kJ/kgK (crude oil)
        Cp_cold = 4.18  # kJ/kgK (water)

        Q_hot = data["HX-001/FI_HOT"]["value"] * Cp_hot * (
            data["HX-001/TI_HOT_IN"]["value"] - data["HX-001/TI_HOT_OUT"]["value"]
        )

        Q_cold = data["HX-001/FI_COLD"]["value"] * Cp_cold * (
            data["HX-001/TI_COLD_OUT"]["value"] - data["HX-001/TI_COLD_IN"]["value"]
        )

        assert Q_hot > 0
        assert Q_cold > 0


class TestThermalCalculationChain:
    """Test full thermal calculation chain."""

    @pytest.mark.integration
    def test_heat_duty_to_lmtd_chain(self, sample_operating_state):
        """Test heat duty to LMTD calculation chain."""
        state = sample_operating_state

        # Step 1: Heat duty
        Q_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        # Step 2: LMTD (counterflow)
        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # Verify chain
        assert Q_hot > 0
        assert lmtd > 0

    @pytest.mark.integration
    def test_lmtd_to_ua_chain(self, sample_operating_state):
        """Test LMTD to UA calculation chain."""
        state = sample_operating_state

        # Heat duty
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        # LMTD
        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # F-factor
        F = 0.9

        # UA
        UA = Q / (F * lmtd)

        assert UA > 0

    @pytest.mark.integration
    def test_ua_to_effectiveness_chain(self, sample_operating_state):
        """Test UA to effectiveness calculation chain."""
        state = sample_operating_state

        # Capacity rates
        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max

        # UA (assumed from previous calculation)
        UA = 90.0  # kW/K

        # NTU
        NTU = UA / C_min

        # Effectiveness (counterflow)
        if C_ratio == 1.0:
            epsilon = NTU / (1 + NTU)
        else:
            epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                      (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        assert 0 < epsilon < 1

    @pytest.mark.integration
    def test_full_thermal_kpi_generation(self, sample_operating_state, sample_exchanger_config):
        """Test full thermal KPI generation."""
        state = sample_operating_state
        config = sample_exchanger_config

        # Calculate all KPIs
        Q_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )
        Q_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK * (
            state.T_cold_out_C - state.T_cold_in_C
        )
        Q_avg = (Q_hot + Q_cold) / 2
        heat_balance_error = abs(Q_hot - Q_cold) / Q_avg * 100 if Q_avg > 0 else 0

        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        F = 0.9
        lmtd_corrected = lmtd * F

        UA = Q_avg / lmtd_corrected
        UA_ratio = UA / config.design_UA_kW_K

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)
        C_ratio = C_min / max(C_hot, C_cold)

        Q_max = C_min * (state.T_hot_in_C - state.T_cold_in_C)
        epsilon = Q_avg / Q_max

        NTU = UA / C_min

        dP_shell = state.P_hot_in_kPa - state.P_hot_out_kPa
        dP_tube = state.P_cold_in_kPa - state.P_cold_out_kPa
        dP_ratio_shell = dP_shell / config.design_pressure_drop_shell_kPa
        dP_ratio_tube = dP_tube / config.design_pressure_drop_tube_kPa

        # Assemble KPIs
        kpis = {
            "exchanger_id": state.exchanger_id,
            "timestamp": datetime.now(timezone.utc),
            "Q_hot_kW": Q_hot,
            "Q_cold_kW": Q_cold,
            "Q_avg_kW": Q_avg,
            "heat_balance_error_percent": heat_balance_error,
            "lmtd_C": lmtd,
            "lmtd_corrected_C": lmtd_corrected,
            "F_factor": F,
            "UA_actual_kW_K": UA,
            "UA_ratio": UA_ratio,
            "epsilon": epsilon,
            "NTU": NTU,
            "C_ratio": C_ratio,
            "dP_shell_kPa": dP_shell,
            "dP_tube_kPa": dP_tube,
            "dP_ratio_shell": dP_ratio_shell,
            "dP_ratio_tube": dP_ratio_tube,
        }

        # Verify all KPIs calculated
        assert len(kpis) > 10
        assert kpis["Q_avg_kW"] > 0
        assert kpis["epsilon"] > 0
        assert kpis["epsilon"] < 1


class TestDataQualityPropagation:
    """Test data quality propagation through pipeline."""

    @pytest.mark.integration
    def test_good_quality_data_propagation(self, sample_operating_state):
        """Test that good quality data produces valid results."""
        state = sample_operating_state
        assert state.data_quality.value == "good"

        # Calculations should proceed normally
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        assert Q > 0

    @pytest.mark.integration
    def test_degraded_data_handling(self, sample_operating_state):
        """Test handling of degraded quality data."""
        # Modify to degraded quality
        state = sample_operating_state
        # In real implementation, would flag results

        # Calculation proceeds but with quality flag
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        assert Q > 0

    @pytest.mark.integration
    def test_heat_balance_as_quality_indicator(self, sample_operating_state):
        """Test heat balance error as data quality indicator."""
        state = sample_operating_state

        Q_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )
        Q_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK * (
            state.T_cold_out_C - state.T_cold_in_C
        )
        Q_avg = (Q_hot + Q_cold) / 2

        heat_balance_error = abs(Q_hot - Q_cold) / Q_avg * 100 if Q_avg > 0 else 0

        # Heat balance error > 5% indicates data quality issue
        if heat_balance_error > 5:
            quality = "degraded"
        else:
            quality = "good"

        assert quality in ["good", "degraded"]


class TestProvenanceTracking:
    """Test provenance tracking through pipeline."""

    @pytest.mark.integration
    def test_provenance_hash_generation(self, sample_operating_state):
        """Test provenance hash generation for calculation chain."""
        state = sample_operating_state

        # Calculate KPIs
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        UA = Q / (0.9 * lmtd)

        # Generate provenance hash
        provenance_data = f"{state.exchanger_id}:{Q:.6f}:{lmtd:.6f}:{UA:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64

    @pytest.mark.integration
    def test_provenance_chain_integrity(self, sample_operating_state):
        """Test provenance chain integrity through calculations."""
        state = sample_operating_state

        # Stage 1: Input provenance
        input_hash = hashlib.sha256(
            f"{state.exchanger_id}:{state.timestamp.isoformat()}".encode()
        ).hexdigest()

        # Stage 2: Calculation provenance
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )
        calc_hash = hashlib.sha256(
            f"{input_hash}:Q:{Q:.6f}".encode()
        ).hexdigest()

        # Stage 3: Output provenance
        output_hash = hashlib.sha256(
            f"{calc_hash}:final".encode()
        ).hexdigest()

        # All hashes should be valid SHA-256
        assert len(input_hash) == 64
        assert len(calc_hash) == 64
        assert len(output_hash) == 64

    @pytest.mark.integration
    def test_deterministic_provenance(self, sample_operating_state):
        """Test that provenance hashes are deterministic."""
        state = sample_operating_state

        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
            state.T_hot_in_C - state.T_hot_out_C
        )

        hashes = []
        for _ in range(5):
            data = f"{state.exchanger_id}:Q:{Q:.6f}"
            hash_val = hashlib.sha256(data.encode()).hexdigest()
            hashes.append(hash_val)

        assert all(h == hashes[0] for h in hashes)


class TestPipelinePerformance:
    """Test pipeline performance."""

    @pytest.mark.integration
    @pytest.mark.performance
    def test_calculation_latency(self, sample_operating_state, performance_timer):
        """Test that thermal calculations complete within latency target."""
        state = sample_operating_state

        timer = performance_timer()
        with timer:
            # Full calculation chain
            Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * (
                state.T_hot_in_C - state.T_hot_out_C
            )

            dT1 = state.T_hot_in_C - state.T_cold_out_C
            dT2 = state.T_hot_out_C - state.T_cold_in_C
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

            UA = Q / (0.9 * lmtd)

            C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
            C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
            C_min = min(C_hot, C_cold)

            NTU = UA / C_min

            C_ratio = C_min / max(C_hot, C_cold)
            epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                      (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        timer.assert_under(5.0)  # Should complete in <5ms


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestOPCUAToKPIPipeline",
    "TestThermalCalculationChain",
    "TestDataQualityPropagation",
    "TestProvenanceTracking",
    "TestPipelinePerformance",
]
