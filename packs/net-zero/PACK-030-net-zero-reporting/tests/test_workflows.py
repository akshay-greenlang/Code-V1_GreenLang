# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - Workflows.

Tests all 8 workflows: SBTi Progress, CDP Questionnaire, TCFD Disclosure,
GRI 305, ISSB IFRS S2, SEC Climate, CSRD ESRS E1, and Multi-Framework
Full Report. Tests workflow/config/input/result class instantiation, the
workflow registry, and registry utility functions.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from workflows import (
    # 1. SBTi
    SBTiProgressWorkflow, SBTiProgressConfig, SBTiProgressInput, SBTiProgressResult,
    # 2. CDP
    CDPQuestionnaireWorkflow, CDPQuestionnaireConfig, CDPQuestionnaireInput, CDPQuestionnaireResult,
    # 3. TCFD
    TCFDDisclosureWorkflow, TCFDDisclosureConfig, TCFDDisclosureInput, TCFDDisclosureResult,
    # 4. GRI
    GRI305Workflow, GRI305Config, GRI305Input, GRI305Result,
    # 5. ISSB (actual name is IFRSS2Workflow, NOT ISSBifrsS2Workflow)
    IFRSS2Workflow, IFRSS2Config, IFRSS2Input, IFRSS2Result,
    # 6. SEC
    SECClimateWorkflow, SECClimateConfig, SECClimateInput, SECClimateResult,
    # 7. CSRD (actual name is CSRDESRSE1Workflow, NOT CSRDE1Workflow)
    CSRDESRSE1Workflow, CSRDE1Config, CSRDE1Input, CSRDE1Result,
    # 8. Multi-Framework
    MultiFrameworkWorkflow, MultiFrameworkConfig, MultiFrameworkInput, MultiFrameworkResult,
    # Registry
    WORKFLOW_REGISTRY, get_workflow, get_workflow_config, get_workflow_input, list_workflows,
)

from .conftest import timed_block, FRAMEWORKS, LANGUAGES


# ========================================================================
# Module-Level Metadata
# ========================================================================


class TestWorkflowModuleMetadata:
    def test_registry_is_dict(self):
        assert isinstance(WORKFLOW_REGISTRY, dict)

    def test_registry_has_8_workflows(self):
        assert len(WORKFLOW_REGISTRY) == 8

    def test_each_entry_has_required_keys(self):
        for name, entry in WORKFLOW_REGISTRY.items():
            assert "class" in entry, f"{name} missing 'class'"
            assert "config_class" in entry, f"{name} missing 'config_class'"
            assert "input_class" in entry, f"{name} missing 'input_class'"
            assert "result_class" in entry, f"{name} missing 'result_class'"
            assert "phases" in entry, f"{name} missing 'phases'"
            assert "description" in entry, f"{name} missing 'description'"
            assert "dag" in entry, f"{name} missing 'dag'"


# ========================================================================
# Registry Utility Functions
# ========================================================================


class TestRegistryFunctions:
    @pytest.mark.parametrize("name", list(WORKFLOW_REGISTRY.keys()))
    def test_get_workflow(self, name):
        wf = get_workflow(name)
        assert wf is not None

    @pytest.mark.parametrize("name", list(WORKFLOW_REGISTRY.keys()))
    def test_get_workflow_config(self, name):
        config_class = get_workflow_config(name)
        assert config_class is not None

    @pytest.mark.parametrize("name", list(WORKFLOW_REGISTRY.keys()))
    def test_get_workflow_input(self, name):
        input_class = get_workflow_input(name)
        assert input_class is not None

    def test_list_workflows(self):
        wf_list = list_workflows()
        assert isinstance(wf_list, list)
        assert len(wf_list) == 8
        for item in wf_list:
            assert "name" in item
            assert "phases" in item
            assert "description" in item

    def test_get_workflow_unknown_raises(self):
        with pytest.raises(KeyError):
            get_workflow("nonexistent_workflow_xyz")


# ========================================================================
# Workflow Class Instantiation
# ========================================================================


_WORKFLOW_CLASSES = [
    ("SBTiProgressWorkflow", SBTiProgressWorkflow),
    ("CDPQuestionnaireWorkflow", CDPQuestionnaireWorkflow),
    ("TCFDDisclosureWorkflow", TCFDDisclosureWorkflow),
    ("GRI305Workflow", GRI305Workflow),
    ("IFRSS2Workflow", IFRSS2Workflow),
    ("SECClimateWorkflow", SECClimateWorkflow),
    ("CSRDESRSE1Workflow", CSRDESRSE1Workflow),
    ("MultiFrameworkWorkflow", MultiFrameworkWorkflow),
]


class TestWorkflowInstantiation:
    @pytest.mark.parametrize("name,cls", _WORKFLOW_CLASSES, ids=[n for n, _ in _WORKFLOW_CLASSES])
    def test_workflow_instantiates(self, name, cls):
        wf = cls()
        assert wf is not None

    @pytest.mark.parametrize("name,cls", _WORKFLOW_CLASSES, ids=[n for n, _ in _WORKFLOW_CLASSES])
    def test_workflow_has_execute(self, name, cls):
        wf = cls()
        assert hasattr(wf, "execute") or hasattr(wf, "run")


# ========================================================================
# Config Class Instantiation
# ========================================================================


_CONFIG_CLASSES = [
    ("SBTiProgressConfig", SBTiProgressConfig),
    ("CDPQuestionnaireConfig", CDPQuestionnaireConfig),
    ("TCFDDisclosureConfig", TCFDDisclosureConfig),
    ("GRI305Config", GRI305Config),
    ("IFRSS2Config", IFRSS2Config),
    ("SECClimateConfig", SECClimateConfig),
    ("CSRDE1Config", CSRDE1Config),
    ("MultiFrameworkConfig", MultiFrameworkConfig),
]


class TestConfigInstantiation:
    @pytest.mark.parametrize("name,cls", _CONFIG_CLASSES, ids=[n for n, _ in _CONFIG_CLASSES])
    def test_config_instantiates(self, name, cls):
        config = cls()
        assert config is not None


# ========================================================================
# Input Class Instantiation
# ========================================================================


_INPUT_CLASSES = [
    ("SBTiProgressInput", SBTiProgressInput),
    ("CDPQuestionnaireInput", CDPQuestionnaireInput),
    ("TCFDDisclosureInput", TCFDDisclosureInput),
    ("GRI305Input", GRI305Input),
    ("IFRSS2Input", IFRSS2Input),
    ("SECClimateInput", SECClimateInput),
    ("CSRDE1Input", CSRDE1Input),
    ("MultiFrameworkInput", MultiFrameworkInput),
]


class TestInputInstantiation:
    @pytest.mark.parametrize("name,cls", _INPUT_CLASSES, ids=[n for n, _ in _INPUT_CLASSES])
    def test_input_instantiates(self, name, cls):
        inp = cls()
        assert inp is not None


# ========================================================================
# Result Class Existence
# ========================================================================


_RESULT_CLASSES = [
    ("SBTiProgressResult", SBTiProgressResult),
    ("CDPQuestionnaireResult", CDPQuestionnaireResult),
    ("TCFDDisclosureResult", TCFDDisclosureResult),
    ("GRI305Result", GRI305Result),
    ("IFRSS2Result", IFRSS2Result),
    ("SECClimateResult", SECClimateResult),
    ("CSRDE1Result", CSRDE1Result),
    ("MultiFrameworkResult", MultiFrameworkResult),
]


class TestResultClassExistence:
    @pytest.mark.parametrize("name,cls", _RESULT_CLASSES, ids=[n for n, _ in _RESULT_CLASSES])
    def test_result_class_exists(self, name, cls):
        assert cls is not None


# ========================================================================
# SBTi Progress Workflow Detail
# ========================================================================


class TestSBTiProgressWorkflowDetail:
    def test_sbti_phases(self):
        entry = WORKFLOW_REGISTRY["sbti_progress_report"]
        assert entry["phases"] == 8

    def test_sbti_dag(self):
        entry = WORKFLOW_REGISTRY["sbti_progress_report"]
        assert "AggregateTargetData" in entry["dag"]


# ========================================================================
# CDP Questionnaire Workflow Detail
# ========================================================================


class TestCDPQuestionnaireWorkflowDetail:
    def test_cdp_phases(self):
        entry = WORKFLOW_REGISTRY["cdp_questionnaire"]
        assert entry["phases"] == 8


# ========================================================================
# TCFD Disclosure Workflow Detail
# ========================================================================


class TestTCFDDisclosureWorkflowDetail:
    def test_tcfd_phases(self):
        entry = WORKFLOW_REGISTRY["tcfd_disclosure"]
        assert entry["phases"] == 8


# ========================================================================
# GRI 305 Workflow Detail
# ========================================================================


class TestGRI305WorkflowDetail:
    def test_gri_phases(self):
        entry = WORKFLOW_REGISTRY["gri_305_disclosure"]
        assert entry["phases"] == 8


# ========================================================================
# ISSB IFRS S2 Workflow Detail
# ========================================================================


class TestIFRSS2WorkflowDetail:
    def test_issb_phases(self):
        entry = WORKFLOW_REGISTRY["issb_ifrs_s2"]
        assert entry["phases"] == 7


# ========================================================================
# SEC Climate Workflow Detail
# ========================================================================


class TestSECClimateWorkflowDetail:
    def test_sec_phases(self):
        entry = WORKFLOW_REGISTRY["sec_climate_disclosure"]
        assert entry["phases"] == 8


# ========================================================================
# CSRD ESRS E1 Workflow Detail
# ========================================================================


class TestCSRDESRSE1WorkflowDetail:
    def test_csrd_phases(self):
        entry = WORKFLOW_REGISTRY["csrd_esrs_e1"]
        assert entry["phases"] == 12


# ========================================================================
# Multi-Framework Workflow Detail
# ========================================================================


class TestMultiFrameworkWorkflowDetail:
    def test_multi_phases(self):
        entry = WORKFLOW_REGISTRY["multi_framework_report"]
        assert entry["phases"] == 7


# ========================================================================
# Performance
# ========================================================================


class TestWorkflowPerformance:
    def test_all_instantiation_fast(self):
        with timed_block("all_workflow_instantiation", max_seconds=2.0):
            for name, entry in WORKFLOW_REGISTRY.items():
                entry["class"]()
                entry["config_class"]()
                entry["input_class"]()
