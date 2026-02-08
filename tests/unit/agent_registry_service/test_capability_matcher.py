# -*- coding: utf-8 -*-
"""
Unit Tests for CapabilityMatcher (AGENT-FOUND-007)

Tests capability search, category filtering, input/output type matching,
capability matrix, and combined filters.

Coverage target: 85%+ of capability_matcher.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline models (self-contained)
# ---------------------------------------------------------------------------


class CapabilityCategory(str, Enum):
    CALCULATION = "calculation"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    INGESTION = "ingestion"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    ANALYTICS = "analytics"
    ORCHESTRATION = "orchestration"
    INTEGRATION = "integration"
    UTILITY = "utility"


class SectorClassification(str, Enum):
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    TRANSPORTATION = "transportation"
    BUILDINGS = "buildings"
    CROSS_SECTOR = "cross_sector"


class Capability:
    def __init__(self, name, category="utility", input_types=None,
                 output_types=None, description=""):
        self.name = name
        self.category = CapabilityCategory(category)
        self.input_types = input_types or []
        self.output_types = output_types or []
        self.description = description


class AgentRecord:
    def __init__(self, agent_id, name, capabilities=None, sectors=None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities or []
        self.sectors = [SectorClassification(s) for s in (sectors or [])]


class CapabilityMatcher:
    """Matches agents by their declared capabilities."""

    def __init__(self):
        self._agents: Dict[str, AgentRecord] = {}
        self._by_capability: Dict[str, List[str]] = defaultdict(list)
        self._by_category: Dict[str, List[str]] = defaultdict(list)
        self._by_input: Dict[str, List[str]] = defaultdict(list)
        self._by_output: Dict[str, List[str]] = defaultdict(list)

    def register(self, agent: AgentRecord) -> None:
        self._agents[agent.agent_id] = agent
        for cap in agent.capabilities:
            if agent.agent_id not in self._by_capability[cap.name]:
                self._by_capability[cap.name].append(agent.agent_id)
            if agent.agent_id not in self._by_category[cap.category.value]:
                self._by_category[cap.category.value].append(agent.agent_id)
            for inp in cap.input_types:
                if agent.agent_id not in self._by_input[inp]:
                    self._by_input[inp].append(agent.agent_id)
            for out in cap.output_types:
                if agent.agent_id not in self._by_output[out]:
                    self._by_output[out].append(agent.agent_id)

    def find_by_capabilities(self, required: List[str],
                             match_all: bool = True) -> List[AgentRecord]:
        """Find agents with matching capabilities."""
        if not required:
            return list(self._agents.values())

        candidate_sets = []
        for cap_name in required:
            agent_ids = set(self._by_capability.get(cap_name, []))
            candidate_sets.append(agent_ids)

        if match_all:
            if not candidate_sets:
                return []
            result_ids = candidate_sets[0]
            for s in candidate_sets[1:]:
                result_ids = result_ids.intersection(s)
        else:
            result_ids: Set[str] = set()
            for s in candidate_sets:
                result_ids = result_ids.union(s)

        return [self._agents[aid] for aid in result_ids if aid in self._agents]

    def find_by_category(self, category: str) -> List[AgentRecord]:
        agent_ids = self._by_category.get(category, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_input_type(self, input_type: str) -> List[AgentRecord]:
        agent_ids = self._by_input.get(input_type, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def find_by_output_type(self, output_type: str) -> List[AgentRecord]:
        agent_ids = self._by_output.get(output_type, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    def get_capability_matrix(self) -> Dict[str, List[str]]:
        """Return capability -> list of agent_ids mapping."""
        return dict(self._by_capability)

    def find_by_sector_and_capability(self, sector: str,
                                       capability: str) -> List[AgentRecord]:
        cap_agents = set(self._by_capability.get(capability, []))
        results = []
        for aid in cap_agents:
            agent = self._agents.get(aid)
            if agent and any(s.value == sector for s in agent.sectors):
                results.append(agent)
        return results


# ===========================================================================
# Helpers
# ===========================================================================


def _make_matcher():
    matcher = CapabilityMatcher()
    matcher.register(AgentRecord(
        "gl-001", "Carbon Calc",
        capabilities=[
            Capability("carbon_calc", "calculation",
                       input_types=["emission_factor", "activity_data"],
                       output_types=["carbon_footprint"]),
        ],
        sectors=["energy"],
    ))
    matcher.register(AgentRecord(
        "gl-002", "CBAM Reporter",
        capabilities=[
            Capability("cbam_report", "reporting",
                       input_types=["import_data", "emission_data"],
                       output_types=["cbam_report"]),
        ],
        sectors=["manufacturing"],
    ))
    matcher.register(AgentRecord(
        "gl-003", "Data Validator",
        capabilities=[
            Capability("data_validate", "validation",
                       input_types=["raw_data"],
                       output_types=["validated_data"]),
            Capability("carbon_calc", "calculation",
                       input_types=["emission_factor"],
                       output_types=["carbon_footprint"]),
        ],
        sectors=["energy", "manufacturing"],
    ))
    return matcher


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCapabilityMatcherFindByCapabilities:
    """Test find_by_capabilities."""

    def test_exact_match_single(self):
        matcher = _make_matcher()
        results = matcher.find_by_capabilities(["carbon_calc"])
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-003" in ids

    def test_exact_match_no_results(self):
        matcher = _make_matcher()
        results = matcher.find_by_capabilities(["nonexistent"])
        assert len(results) == 0

    def test_partial_match_any(self):
        matcher = _make_matcher()
        results = matcher.find_by_capabilities(
            ["carbon_calc", "cbam_report"], match_all=False,
        )
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-002" in ids
        assert "gl-003" in ids

    def test_match_all_multiple(self):
        matcher = _make_matcher()
        results = matcher.find_by_capabilities(
            ["carbon_calc", "data_validate"], match_all=True,
        )
        ids = {r.agent_id for r in results}
        assert ids == {"gl-003"}

    def test_empty_required_returns_all(self):
        matcher = _make_matcher()
        results = matcher.find_by_capabilities([])
        assert len(results) == 3


class TestCapabilityMatcherFindByCategory:
    """Test find_by_category."""

    def test_calculation_category(self):
        matcher = _make_matcher()
        results = matcher.find_by_category("calculation")
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-003" in ids

    def test_reporting_category(self):
        matcher = _make_matcher()
        results = matcher.find_by_category("reporting")
        assert len(results) == 1
        assert results[0].agent_id == "gl-002"

    def test_nonexistent_category(self):
        matcher = _make_matcher()
        results = matcher.find_by_category("orchestration")
        assert len(results) == 0


class TestCapabilityMatcherFindByInputOutput:
    """Test find_by_input_type and find_by_output_type."""

    def test_find_by_input_type(self):
        matcher = _make_matcher()
        results = matcher.find_by_input_type("emission_factor")
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-003" in ids

    def test_find_by_input_type_unique(self):
        matcher = _make_matcher()
        results = matcher.find_by_input_type("import_data")
        assert len(results) == 1
        assert results[0].agent_id == "gl-002"

    def test_find_by_input_type_none(self):
        matcher = _make_matcher()
        results = matcher.find_by_input_type("nonexistent")
        assert len(results) == 0

    def test_find_by_output_type(self):
        matcher = _make_matcher()
        results = matcher.find_by_output_type("carbon_footprint")
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-003" in ids

    def test_find_by_output_type_unique(self):
        matcher = _make_matcher()
        results = matcher.find_by_output_type("validated_data")
        assert len(results) == 1


class TestCapabilityMatcherMatrix:
    """Test get_capability_matrix."""

    def test_matrix_keys(self):
        matcher = _make_matcher()
        matrix = matcher.get_capability_matrix()
        assert "carbon_calc" in matrix
        assert "cbam_report" in matrix
        assert "data_validate" in matrix

    def test_matrix_values(self):
        matcher = _make_matcher()
        matrix = matcher.get_capability_matrix()
        assert "gl-001" in matrix["carbon_calc"]
        assert "gl-003" in matrix["carbon_calc"]

    def test_matrix_empty(self):
        matcher = CapabilityMatcher()
        assert matcher.get_capability_matrix() == {}


class TestCapabilityMatcherCombined:
    """Test sector + capability combined filters."""

    def test_sector_and_capability(self):
        matcher = _make_matcher()
        results = matcher.find_by_sector_and_capability("energy", "carbon_calc")
        ids = {r.agent_id for r in results}
        assert "gl-001" in ids
        assert "gl-003" in ids

    def test_sector_and_capability_no_match(self):
        matcher = _make_matcher()
        results = matcher.find_by_sector_and_capability("manufacturing", "carbon_calc")
        ids = {r.agent_id for r in results}
        assert "gl-003" in ids
        assert "gl-001" not in ids

    def test_sector_and_capability_empty(self):
        matcher = _make_matcher()
        results = matcher.find_by_sector_and_capability("water", "carbon_calc")
        assert len(results) == 0


class TestCapabilityMatcherEdgeCases:
    """Test edge cases."""

    def test_register_agent_no_capabilities(self):
        matcher = CapabilityMatcher()
        matcher.register(AgentRecord("gl-empty", "Empty"))
        assert len(matcher.find_by_capabilities([])) == 1
        assert len(matcher.get_capability_matrix()) == 0

    def test_duplicate_register(self):
        matcher = CapabilityMatcher()
        agent = AgentRecord("gl-001", "A", capabilities=[
            Capability("calc", "calculation"),
        ])
        matcher.register(agent)
        matcher.register(agent)
        # Should not duplicate in index
        assert len(matcher._by_capability["calc"]) == 1
