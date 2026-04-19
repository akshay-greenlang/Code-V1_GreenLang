# -*- coding: utf-8 -*-
"""
Golden Test for Byte-Exact Replay (INTL-103 DoD Gap 4)

Verifies that tool runtime execution produces deterministic, reproducible output.
This test ensures that running the same scenario multiple times yields byte-identical results.

DoD Requirement: Create golden test with byte-exact replay to validate determinism.
"""

import json
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from greenlang.intelligence.runtime.tools import Tool, ToolRegistry, ToolRuntime
from greenlang.intelligence.runtime.schemas import Quantity


class TestGoldenReplay:
    """Golden tests for byte-exact reproducibility"""

    @pytest.fixture
    def golden_dir(self):
        """Path to golden files directory"""
        return Path(__file__).parent.parent / "goldens"

    @pytest.fixture
    def calc_emissions_tool(self):
        """Deterministic emissions calculation tool"""
        def calculate_emissions(energy_kwh: float, intensity_kg_per_kwh: float):
            """Calculate emissions from energy and intensity"""
            return {
                "energy": {"value": energy_kwh, "unit": "kWh"},
                "intensity": {"value": intensity_kg_per_kwh, "unit": "kgCO2e/kWh"},
                "emissions": {
                    "value": energy_kwh * intensity_kg_per_kwh,
                    "unit": "kgCO2e"
                },
            }

        return Tool(
            name="calc_emissions",
            description="Calculate emissions from energy and carbon intensity",
            args_schema={
                "type": "object",
                "required": ["energy_kwh", "intensity_kg_per_kwh"],
                "properties": {
                    "energy_kwh": {"type": "number"},
                    "intensity_kg_per_kwh": {"type": "number"},
                },
            },
            result_schema={
                "type": "object",
                "required": ["energy", "intensity", "emissions"],
                "properties": {
                    "energy": {"$ref": "greenlang://schemas/quantity.json"},
                    "intensity": {"$ref": "greenlang://schemas/quantity.json"},
                    "emissions": {"$ref": "greenlang://schemas/quantity.json"},
                },
            },
            fn=lambda energy_kwh, intensity_kg_per_kwh: calculate_emissions(
                energy_kwh, intensity_kg_per_kwh
            ),
        )

    def test_golden_no_naked_numbers_scenario(
        self, calc_emissions_tool, golden_dir
    ):
        """
        Golden test: Verify byte-exact replay of 'no naked numbers' scenario

        Scenario:
        1. User asks: "Calculate emissions for 1000 kWh at 0.5 kgCO2e/kWh"
        2. Assistant calls calc_emissions(1000, 0.5)
        3. Tool returns emissions data with Quantities
        4. Assistant produces final message with {{claim:i}} macros

        Expected: Running this scenario multiple times produces identical JSON output
        """
        # Setup
        registry = ToolRegistry()
        registry.register(calc_emissions_tool)

        mock_provider = Mock()

        # Mock provider behavior: two-step interaction
        call_count = [0]

        def mock_chat_step(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: assistant invokes tool
                return {
                    "kind": "tool_call",
                    "tool_name": "calc_emissions",
                    "arguments": {
                        "energy_kwh": 1000,
                        "intensity_kg_per_kwh": 0.5
                    }
                }
            else:
                # Second call: assistant produces final message
                return {
                    "kind": "final",
                    "final": {
                        "message": (
                            "Total emissions are {{claim:0}}."
                        ),
                        "claims": [
                            {
                                "source_call_id": "tc_1",
                                "path": "$.emissions",
                                "quantity": {"value": 500.0, "unit": "kgCO2e"}
                            }
                        ]
                    }
                }

        mock_provider.chat_step.side_effect = mock_chat_step

        runtime = ToolRuntime(mock_provider, registry)

        # Execute scenario
        result = runtime.run(
            system_prompt="You are a climate assistant. Help calculate emissions.",
            user_msg="Calculate emissions for 1000 kWh at 0.5 kgCO2e/kWh"
        )

        # Normalize result to stable JSON format
        result_json = json.dumps(result, sort_keys=True, indent=2)
        result_hash = hashlib.sha256(result_json.encode()).hexdigest()

        # Golden file path
        golden_file = golden_dir / "runtime_no_naked_numbers.json"
        golden_hash_file = golden_dir / "runtime_no_naked_numbers.sha256"

        # If golden doesn't exist, create it (first run)
        if not golden_file.exists():
            golden_dir.mkdir(parents=True, exist_ok=True)
            golden_file.write_text(result_json, encoding="utf-8")
            golden_hash_file.write_text(result_hash, encoding="utf-8")
            pytest.skip(
                f"Golden file created: {golden_file}\n"
                f"Run test again to verify reproducibility."
            )

        # Load golden reference
        golden_json = golden_file.read_text(encoding="utf-8")
        golden_hash = golden_hash_file.read_text(encoding="utf-8").strip()

        # Verify byte-exact match
        assert result_json == golden_json, (
            f"Output does not match golden reference!\n"
            f"Expected hash: {golden_hash}\n"
            f"Actual hash:   {result_hash}\n"
            f"Golden file: {golden_file}\n"
            f"To update golden, delete {golden_file} and re-run."
        )

        assert result_hash == golden_hash, (
            f"SHA256 hash mismatch!\n"
            f"Expected: {golden_hash}\n"
            f"Actual:   {result_hash}"
        )

    def test_golden_replay_multiple_runs(self, calc_emissions_tool):
        """
        Verify that running the same scenario 10 times produces identical output

        This validates determinism without relying on a stored golden file.
        """
        registry = ToolRegistry()
        registry.register(calc_emissions_tool)

        results = []

        for i in range(10):
            # Reset mock provider for each run
            mock_provider = Mock()
            call_count = [0]

            def mock_chat_step(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {
                        "kind": "tool_call",
                        "tool_name": "calc_emissions",
                        "arguments": {
                            "energy_kwh": 1234.5,
                            "intensity_kg_per_kwh": 0.789
                        }
                    }
                else:
                    return {
                        "kind": "final",
                        "final": {
                            "message": "Emissions: {{claim:0}}",
                            "claims": [
                                {
                                    "source_call_id": "tc_1",
                                    "path": "$.emissions",
                                    "quantity": {
                                        "value": 1234.5 * 0.789,
                                        "unit": "kgCO2e"
                                    }
                                }
                            ]
                        }
                    }

            mock_provider.chat_step.side_effect = mock_chat_step
            runtime = ToolRuntime(mock_provider, registry)

            result = runtime.run(
                system_prompt="Climate assistant",
                user_msg="Calculate emissions"
            )

            # Serialize to stable JSON
            result_json = json.dumps(result, sort_keys=True, indent=2)
            results.append(result_json)

        # Verify all 10 runs produced identical output
        first_result = results[0]
        for i, result in enumerate(results[1:], start=1):
            assert result == first_result, (
                f"Run {i+1} produced different output than run 1!\n"
                f"This indicates non-deterministic behavior."
            )

        # Verify hash consistency
        hashes = [hashlib.sha256(r.encode()).hexdigest() for r in results]
        assert len(set(hashes)) == 1, (
            f"Multiple different hashes detected: {set(hashes)}\n"
            f"Runs should be byte-identical but got {len(set(hashes))} unique outputs."
        )
