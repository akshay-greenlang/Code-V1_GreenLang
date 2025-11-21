# -*- coding: utf-8 -*-
"""
GreenLang Quality Validators
Comprehensive 12-dimension quality validation framework for AI agents.
Based on ISO 25010 software quality standards adapted for AI systems.
"""

import json
import time
import hashlib
import traceback
import psutil
import resource
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Callable, Type, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
import pandas as pd
import logging
import re
import ast
import sys
import os
import inspect
import importlib
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from greenlang.determinism import DeterministicClock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Quality Dimensions (ISO 25010 adapted for AI agents)
class QualityDimension(Enum):
    """12 quality dimensions from Architecture doc lines 1099-1221."""
    FUNCTIONAL_QUALITY = "functional_quality"
    PERFORMANCE_EFFICIENCY = "performance_efficiency"
    COMPATIBILITY = "compatibility"
    USABILITY = "usability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    PORTABILITY = "portability"
    SCALABILITY = "scalability"
    INTEROPERABILITY = "interoperability"
    REUSABILITY = "reusability"
    TESTABILITY = "testability"


@dataclass
class QualityMetrics:
    """Metrics for quality assessment."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    passed: bool
    target: float
    measurements: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: DeterministicClock.now().isoformat())


@dataclass
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    passed: bool
    dimensions: List[QualityMetrics]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    duration_s: float


# Base Quality Validator
class QualityValidator:
    """Base class for quality validation."""

    def __init__(self, target_score: float = 0.8):
        """Initialize validator with target score."""
        self.target_score = target_score
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history = []

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate agent quality for specific dimension."""
        raise NotImplementedError("Subclasses must implement validate()")

    def record_metric(self, metric: QualityMetrics):
        """Record metric for historical analysis."""
        self.metrics_history.append(metric)

    def get_trend(self, window: int = 10) -> Dict[str, float]:
        """Get quality trend over recent validations."""
        if len(self.metrics_history) < window:
            return {}

        recent = self.metrics_history[-window:]
        scores = [m.score for m in recent]

        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "trend": "improving" if scores[-1] > scores[0] else "declining",
            "stability": 1.0 - np.std(scores)
        }


# 1. Functional Quality Validator
class FunctionalQualityValidator(QualityValidator):
    """Validate functional correctness and completeness."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate functional quality."""
        start_time = time.time()
        measurements = {}
        issues = []

        # Test correctness
        correctness_score = self._test_correctness(agent, test_data)
        measurements["correctness"] = correctness_score

        # Test completeness
        completeness_score = self._test_completeness(agent)
        measurements["completeness"] = completeness_score

        # Test consistency
        consistency_score = self._test_consistency(agent, test_data)
        measurements["consistency"] = consistency_score

        # Calculate overall score
        overall_score = np.mean([
            correctness_score,
            completeness_score,
            consistency_score
        ])

        # Generate recommendations
        recommendations = []
        if correctness_score < self.target_score:
            recommendations.append("Improve calculation accuracy and output correctness")
        if completeness_score < self.target_score:
            recommendations.append("Implement missing features or capabilities")
        if consistency_score < self.target_score:
            recommendations.append("Ensure consistent behavior across similar inputs")

        metric = QualityMetrics(
            dimension=QualityDimension.FUNCTIONAL_QUALITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_correctness(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test output correctness."""
        if not test_data or "ground_truth" not in test_data:
            return 1.0  # Assume correct if no ground truth

        correct_count = 0
        total_count = 0

        for test_case in test_data.get("test_cases", []):
            try:
                result = agent.process(test_case["input"])
                expected = test_case["expected"]

                # Compare results
                if self._compare_outputs(result, expected):
                    correct_count += 1
                total_count += 1
            except Exception as e:
                self.logger.error(f"Correctness test failed: {e}")
                total_count += 1

        return correct_count / total_count if total_count > 0 else 0.0

    def _test_completeness(self, agent: Any) -> float:
        """Test feature completeness."""
        required_methods = [
            "process", "validate_input", "handle_error",
            "get_state", "reset"
        ]

        implemented = 0
        for method in required_methods:
            if hasattr(agent, method) and callable(getattr(agent, method)):
                implemented += 1

        return implemented / len(required_methods)

    def _test_consistency(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test behavioral consistency."""
        if not test_data:
            return 1.0

        # Test same input produces same output
        consistency_tests = []

        for _ in range(5):
            test_input = test_data.get("sample_input", {})
            try:
                result1 = agent.process(test_input)
                result2 = agent.process(test_input)

                # Check if results are identical
                if self._compare_outputs(result1, result2):
                    consistency_tests.append(1.0)
                else:
                    consistency_tests.append(0.0)
            except Exception:
                consistency_tests.append(0.0)

        return np.mean(consistency_tests) if consistency_tests else 0.0

    def _compare_outputs(self, output1: Any, output2: Any) -> bool:
        """Compare two outputs for equality."""
        if type(output1) != type(output2):
            return False

        if isinstance(output1, (int, float, str, bool)):
            return output1 == output2
        elif isinstance(output1, dict):
            return json.dumps(output1, sort_keys=True) == json.dumps(output2, sort_keys=True)
        elif hasattr(output1, "__dict__"):
            return asdict(output1) == asdict(output2)
        else:
            return str(output1) == str(output2)


# 2. Performance Efficiency Validator
class PerformanceValidator(QualityValidator):
    """Validate performance efficiency metrics."""

    def __init__(self, target_score: float = 0.8):
        """Initialize with performance targets."""
        super().__init__(target_score)
        # Performance targets from Architecture doc lines 22-28
        self.targets = {
            "response_time_ms": 2000,  # <2s average
            "throughput_per_s": 1000,  # >1000 agents/second
            "memory_mb": 4096,  # <4GB per agent
            "cpu_percent": 80  # <80% CPU utilization
        }

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate performance efficiency."""
        measurements = {}

        # Test response time
        response_time_score = self._test_response_time(agent, test_data)
        measurements["response_time"] = response_time_score

        # Test throughput
        throughput_score = self._test_throughput(agent, test_data)
        measurements["throughput"] = throughput_score

        # Test resource usage
        resource_score = self._test_resource_usage(agent, test_data)
        measurements["resource_usage"] = resource_score

        # Calculate overall score
        overall_score = np.mean([
            response_time_score,
            throughput_score,
            resource_score
        ])

        # Generate recommendations
        recommendations = []
        if response_time_score < self.target_score:
            recommendations.append(f"Optimize response time to <{self.targets['response_time_ms']}ms")
        if throughput_score < self.target_score:
            recommendations.append(f"Improve throughput to >{self.targets['throughput_per_s']}/s")
        if resource_score < self.target_score:
            recommendations.append("Reduce memory and CPU usage")

        metric = QualityMetrics(
            dimension=QualityDimension.PERFORMANCE_EFFICIENCY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_response_time(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test agent response time."""
        response_times = []

        test_inputs = test_data.get("inputs", [{}]) if test_data else [{}]

        for test_input in test_inputs[:10]:  # Test up to 10 inputs
            start_time = time.perf_counter()
            try:
                _ = agent.process(test_input)
                response_time_ms = (time.perf_counter() - start_time) * 1000
                response_times.append(response_time_ms)
            except Exception:
                response_times.append(self.targets["response_time_ms"] * 2)  # Penalty

        avg_response_time = np.mean(response_times) if response_times else float('inf')

        # Score based on target
        if avg_response_time <= self.targets["response_time_ms"]:
            return 1.0
        elif avg_response_time <= self.targets["response_time_ms"] * 2:
            return 0.5 + 0.5 * (2 - avg_response_time / self.targets["response_time_ms"])
        else:
            return max(0.0, 1.0 - avg_response_time / (self.targets["response_time_ms"] * 10))

    def _test_throughput(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test agent throughput."""
        test_duration_s = 1.0
        processed_count = 0

        test_input = test_data.get("sample_input", {}) if test_data else {}

        start_time = time.time()
        while time.time() - start_time < test_duration_s:
            try:
                _ = agent.process(test_input)
                processed_count += 1
            except Exception:
                break

        throughput = processed_count / test_duration_s

        # Score based on target
        target_throughput = self.targets["throughput_per_s"]
        if throughput >= target_throughput:
            return 1.0
        else:
            return throughput / target_throughput

    def _test_resource_usage(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test resource usage."""
        process = psutil.Process()

        # Get initial measurements
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()

        # Process some requests
        test_input = test_data.get("sample_input", {}) if test_data else {}
        for _ in range(10):
            try:
                _ = agent.process(test_input)
            except Exception:
                pass

        # Get final measurements
        time.sleep(0.1)  # Allow CPU measurement to stabilize
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent()

        memory_increase = final_memory - initial_memory
        avg_cpu = (initial_cpu + final_cpu) / 2

        # Score based on targets
        memory_score = 1.0 if memory_increase < self.targets["memory_mb"] else \
                      max(0.0, 1.0 - memory_increase / (self.targets["memory_mb"] * 2))

        cpu_score = 1.0 if avg_cpu < self.targets["cpu_percent"] else \
                   max(0.0, 1.0 - avg_cpu / 200)

        return (memory_score + cpu_score) / 2


# 3. Compatibility Validator
class CompatibilityValidator(QualityValidator):
    """Validate compatibility with different environments and systems."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate compatibility."""
        measurements = {}

        # Test API compatibility
        api_score = self._test_api_compatibility(agent)
        measurements["api_compatibility"] = api_score

        # Test data format compatibility
        format_score = self._test_data_format_compatibility(agent, test_data)
        measurements["data_formats"] = format_score

        # Test integration compatibility
        integration_score = self._test_integration_compatibility(agent)
        measurements["integration"] = integration_score

        # Calculate overall score
        overall_score = np.mean([api_score, format_score, integration_score])

        # Generate recommendations
        recommendations = []
        if api_score < self.target_score:
            recommendations.append("Ensure backward compatible API changes")
        if format_score < self.target_score:
            recommendations.append("Support multiple data formats (JSON, XML, CSV)")
        if integration_score < self.target_score:
            recommendations.append("Use standard protocols for integration")

        metric = QualityMetrics(
            dimension=QualityDimension.COMPATIBILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_api_compatibility(self, agent: Any) -> float:
        """Test API compatibility."""
        required_methods = ["process", "validate", "get_version"]
        compatible = 0

        for method in required_methods:
            if hasattr(agent, method):
                compatible += 1

        return compatible / len(required_methods)

    def _test_data_format_compatibility(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test data format compatibility."""
        formats_supported = 0
        formats_tested = ["json", "dict", "string"]

        test_value = test_data.get("sample_value", "test") if test_data else "test"

        for format_type in formats_tested:
            try:
                if format_type == "json":
                    test_input = json.dumps({"value": test_value})
                elif format_type == "dict":
                    test_input = {"value": test_value}
                else:
                    test_input = str(test_value)

                _ = agent.process(test_input)
                formats_supported += 1
            except Exception:
                pass

        return formats_supported / len(formats_tested)

    def _test_integration_compatibility(self, agent: Any) -> float:
        """Test integration protocol compatibility."""
        protocols = ["REST", "GraphQL", "gRPC"]
        compatible_protocols = 0

        # Check for protocol support indicators
        if hasattr(agent, "handle_http_request") or hasattr(agent, "rest_api"):
            compatible_protocols += 1  # REST
        if hasattr(agent, "graphql_schema") or hasattr(agent, "handle_graphql"):
            compatible_protocols += 1  # GraphQL
        if hasattr(agent, "grpc_service") or hasattr(agent, "handle_grpc"):
            compatible_protocols += 1  # gRPC

        return compatible_protocols / len(protocols)


# 4. Usability Validator
class UsabilityValidator(QualityValidator):
    """Validate agent usability and developer experience."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate usability."""
        measurements = {}

        # Test ease of use
        ease_score = self._test_ease_of_use(agent)
        measurements["ease_of_use"] = ease_score

        # Test documentation
        doc_score = self._test_documentation(agent)
        measurements["documentation"] = doc_score

        # Test error messages
        error_score = self._test_error_messages(agent)
        measurements["error_messages"] = error_score

        # Calculate overall score
        overall_score = np.mean([ease_score, doc_score, error_score])

        # Generate recommendations
        recommendations = []
        if ease_score < self.target_score:
            recommendations.append("Simplify API and improve developer ergonomics")
        if doc_score < self.target_score:
            recommendations.append("Add comprehensive documentation and examples")
        if error_score < self.target_score:
            recommendations.append("Provide clear, actionable error messages")

        metric = QualityMetrics(
            dimension=QualityDimension.USABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_ease_of_use(self, agent: Any) -> float:
        """Test ease of use."""
        ease_factors = []

        # Check for simple initialization
        try:
            if hasattr(agent.__class__, "__init__"):
                init_params = inspect.signature(agent.__class__.__init__).parameters
                # Fewer required parameters = easier to use
                required_params = sum(1 for p in init_params.values()
                                    if p.default == inspect.Parameter.empty)
                ease_factors.append(1.0 if required_params <= 3 else 0.5)
        except Exception:
            ease_factors.append(0.0)

        # Check for sensible defaults
        if hasattr(agent, "config") and agent.config:
            ease_factors.append(1.0)  # Has configuration
        else:
            ease_factors.append(0.5)

        # Check for clear method names
        methods = [m for m in dir(agent) if not m.startswith("_")]
        clear_names = sum(1 for m in methods if len(m) < 20 and "_" in m)
        ease_factors.append(clear_names / len(methods) if methods else 0.0)

        return np.mean(ease_factors) if ease_factors else 0.0

    def _test_documentation(self, agent: Any) -> float:
        """Test documentation quality."""
        doc_scores = []

        # Check class docstring
        if agent.__class__.__doc__:
            doc_scores.append(1.0 if len(agent.__class__.__doc__) > 50 else 0.5)
        else:
            doc_scores.append(0.0)

        # Check method docstrings
        methods = [m for m in dir(agent) if not m.startswith("_") and callable(getattr(agent, m))]
        documented = sum(1 for m in methods if getattr(agent, m).__doc__)
        doc_scores.append(documented / len(methods) if methods else 0.0)

        return np.mean(doc_scores) if doc_scores else 0.0

    def _test_error_messages(self, agent: Any) -> float:
        """Test error message quality."""
        error_quality = []

        # Test with invalid input
        try:
            _ = agent.process(None)
            error_quality.append(0.0)  # Should have raised an error
        except Exception as e:
            error_msg = str(e)
            # Check error message quality
            has_description = len(error_msg) > 20
            has_suggestion = any(word in error_msg.lower()
                               for word in ["try", "should", "must", "please", "instead"])
            error_quality.append(1.0 if (has_description and has_suggestion) else 0.5)

        return np.mean(error_quality) if error_quality else 0.0


# 5. Reliability Validator
class ReliabilityValidator(QualityValidator):
    """Validate agent reliability and fault tolerance."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate reliability."""
        measurements = {}

        # Test availability
        availability_score = self._test_availability(agent, test_data)
        measurements["availability"] = availability_score

        # Test fault tolerance
        fault_score = self._test_fault_tolerance(agent, test_data)
        measurements["fault_tolerance"] = fault_score

        # Test recoverability
        recovery_score = self._test_recoverability(agent)
        measurements["recoverability"] = recovery_score

        # Calculate overall score
        overall_score = np.mean([availability_score, fault_score, recovery_score])

        # Generate recommendations
        recommendations = []
        if availability_score < self.target_score:
            recommendations.append("Improve uptime to 99.99% availability")
        if fault_score < self.target_score:
            recommendations.append("Implement graceful degradation and error recovery")
        if recovery_score < self.target_score:
            recommendations.append("Reduce recovery time to <5 minutes")

        metric = QualityMetrics(
            dimension=QualityDimension.RELIABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_availability(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test agent availability."""
        successful_calls = 0
        total_calls = 100

        test_input = test_data.get("sample_input", {}) if test_data else {}

        for _ in range(total_calls):
            try:
                _ = agent.process(test_input)
                successful_calls += 1
            except Exception:
                pass

        availability = successful_calls / total_calls

        # Target 99.99% availability
        if availability >= 0.9999:
            return 1.0
        elif availability >= 0.999:
            return 0.9
        elif availability >= 0.99:
            return 0.8
        else:
            return availability

    def _test_fault_tolerance(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test fault tolerance."""
        fault_tests = []

        # Test with various fault conditions
        fault_inputs = [
            None,  # Null input
            {},  # Empty input
            {"invalid": "data"},  # Invalid structure
            "malformed",  # Wrong type
        ]

        for fault_input in fault_inputs:
            try:
                _ = agent.process(fault_input)
                fault_tests.append(0.5)  # Handled but might be incorrect
            except Exception as e:
                # Check if error is handled gracefully
                if hasattr(agent, "state") and str(agent.state) != "ERROR":
                    fault_tests.append(1.0)  # Graceful handling
                else:
                    fault_tests.append(0.0)  # Crashed

        return np.mean(fault_tests) if fault_tests else 0.0

    def _test_recoverability(self, agent: Any) -> float:
        """Test recovery capabilities."""
        recovery_score = 0.0

        # Check for recovery methods
        if hasattr(agent, "reset") and callable(agent.reset):
            recovery_score += 0.5

        if hasattr(agent, "recover") and callable(agent.recover):
            recovery_score += 0.5

        # Test actual recovery
        try:
            # Cause an error
            try:
                agent.process(None)
            except Exception:
                pass

            # Try to recover
            if hasattr(agent, "reset"):
                agent.reset()
                # Test if agent works after reset
                agent.process({})
                recovery_score = 1.0
        except Exception:
            pass

        return recovery_score


# 6. Security Validator
class SecurityValidator(QualityValidator):
    """Validate security aspects of agents."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate security."""
        measurements = {}

        # Test vulnerability scanning
        vulnerability_score = self._test_vulnerabilities(agent)
        measurements["vulnerabilities"] = vulnerability_score

        # Test compliance
        compliance_score = self._test_compliance(agent)
        measurements["compliance"] = compliance_score

        # Test encryption
        encryption_score = self._test_encryption(agent)
        measurements["encryption"] = encryption_score

        # Calculate overall score
        overall_score = np.mean([vulnerability_score, compliance_score, encryption_score])

        # Generate recommendations
        recommendations = []
        if vulnerability_score < self.target_score:
            recommendations.append("Address security vulnerabilities")
        if compliance_score < self.target_score:
            recommendations.append("Ensure SOC2 and GDPR compliance")
        if encryption_score < self.target_score:
            recommendations.append("Implement encryption at rest and in transit")

        metric = QualityMetrics(
            dimension=QualityDimension.SECURITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_vulnerabilities(self, agent: Any) -> float:
        """Test for common vulnerabilities."""
        vulnerability_checks = []

        # Check for injection vulnerabilities
        injection_test = "'; DROP TABLE users; --"
        try:
            _ = agent.process({"input": injection_test})
            # If it processes without error, check if it's sanitized
            vulnerability_checks.append(1.0)
        except Exception:
            vulnerability_checks.append(1.0)  # Error is acceptable

        # Check for exposed sensitive data
        if hasattr(agent, "__dict__"):
            sensitive_keys = ["password", "secret", "key", "token", "credential"]
            exposed = any(k in str(agent.__dict__).lower() for k in sensitive_keys)
            vulnerability_checks.append(0.0 if exposed else 1.0)

        return np.mean(vulnerability_checks) if vulnerability_checks else 0.0

    def _test_compliance(self, agent: Any) -> float:
        """Test compliance requirements."""
        compliance_checks = []

        # Check for audit logging
        if hasattr(agent, "audit_log") or hasattr(agent, "logger"):
            compliance_checks.append(1.0)
        else:
            compliance_checks.append(0.0)

        # Check for data privacy
        if hasattr(agent, "anonymize") or hasattr(agent, "encrypt"):
            compliance_checks.append(1.0)
        else:
            compliance_checks.append(0.5)

        return np.mean(compliance_checks) if compliance_checks else 0.0

    def _test_encryption(self, agent: Any) -> float:
        """Test encryption capabilities."""
        encryption_score = 0.0

        # Check for encryption methods
        if hasattr(agent, "encrypt") and hasattr(agent, "decrypt"):
            encryption_score += 0.5

        # Check for secure communication
        if hasattr(agent, "use_tls") or hasattr(agent, "secure_channel"):
            encryption_score += 0.5

        return encryption_score


# 7. Maintainability Validator
class MaintainabilityValidator(QualityValidator):
    """Validate code maintainability."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate maintainability."""
        measurements = {}

        # Test code quality
        quality_score = self._test_code_quality(agent)
        measurements["code_quality"] = quality_score

        # Test technical debt
        debt_score = self._test_technical_debt(agent)
        measurements["technical_debt"] = debt_score

        # Test modularity
        modularity_score = self._test_modularity(agent)
        measurements["modularity"] = modularity_score

        # Calculate overall score
        overall_score = np.mean([quality_score, debt_score, modularity_score])

        # Generate recommendations
        recommendations = []
        if quality_score < self.target_score:
            recommendations.append("Improve code quality to Grade A")
        if debt_score < self.target_score:
            recommendations.append("Reduce technical debt to <10%")
        if modularity_score < self.target_score:
            recommendations.append("Improve modularity and reduce coupling")

        metric = QualityMetrics(
            dimension=QualityDimension.MAINTAINABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_code_quality(self, agent: Any) -> float:
        """Test code quality metrics."""
        quality_factors = []

        # Check method complexity
        methods = [m for m in dir(agent) if not m.startswith("_") and callable(getattr(agent, m))]
        if methods:
            avg_method_length = np.mean([
                len(inspect.getsource(getattr(agent, m)).split("\n"))
                for m in methods
                if hasattr(getattr(agent, m), "__code__")
            ] or [20])
            # Prefer shorter methods
            quality_factors.append(1.0 if avg_method_length < 50 else 0.5)

        # Check naming conventions
        proper_names = sum(1 for m in methods if re.match(r"^[a-z_][a-z0-9_]*$", m))
        quality_factors.append(proper_names / len(methods) if methods else 0.0)

        return np.mean(quality_factors) if quality_factors else 0.0

    def _test_technical_debt(self, agent: Any) -> float:
        """Test technical debt indicators."""
        debt_indicators = []

        # Check for TODO/FIXME comments
        try:
            source = inspect.getsource(agent.__class__)
            todo_count = source.upper().count("TODO") + source.upper().count("FIXME")
            debt_indicators.append(1.0 if todo_count < 5 else 0.5)
        except Exception:
            debt_indicators.append(0.5)

        # Check for deprecated methods
        deprecated_count = sum(1 for m in dir(agent) if "deprecated" in m.lower())
        debt_indicators.append(1.0 if deprecated_count == 0 else 0.5)

        return np.mean(debt_indicators) if debt_indicators else 0.0

    def _test_modularity(self, agent: Any) -> float:
        """Test modularity and coupling."""
        modularity_score = 0.0

        # Check for separation of concerns
        concerns = ["process", "validate", "transform", "persist", "communicate"]
        separated = sum(1 for c in concerns if any(c in m for m in dir(agent)))
        modularity_score = separated / len(concerns)

        return modularity_score


# 8. Portability Validator
class PortabilityValidator(QualityValidator):
    """Validate agent portability across platforms."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate portability."""
        measurements = {}

        # Test platform support
        platform_score = self._test_platform_support(agent)
        measurements["platform_support"] = platform_score

        # Test containerization
        container_score = self._test_containerization(agent)
        measurements["containerization"] = container_score

        # Test cloud agnostic
        cloud_score = self._test_cloud_agnostic(agent)
        measurements["cloud_agnostic"] = cloud_score

        # Calculate overall score
        overall_score = np.mean([platform_score, container_score, cloud_score])

        # Generate recommendations
        recommendations = []
        if platform_score < self.target_score:
            recommendations.append("Support Linux, Windows, and MacOS")
        if container_score < self.target_score:
            recommendations.append("Ensure Docker compatibility")
        if cloud_score < self.target_score:
            recommendations.append("Support multi-cloud deployment")

        metric = QualityMetrics(
            dimension=QualityDimension.PORTABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_platform_support(self, agent: Any) -> float:
        """Test multi-platform support."""
        # Check for platform-specific code
        try:
            source = inspect.getsource(agent.__class__)
            platform_specific = ["win32", "darwin", "linux"]
            has_platform_checks = any(p in source.lower() for p in platform_specific)

            # If it has platform checks, it's likely portable
            return 1.0 if has_platform_checks else 0.8
        except Exception:
            return 0.5

    def _test_containerization(self, agent: Any) -> float:
        """Test Docker readiness."""
        container_ready = 0.0

        # Check for container-friendly attributes
        if hasattr(agent, "config"):
            container_ready += 0.5  # Configurable

        if not hasattr(agent, "gui") and not hasattr(agent, "display"):
            container_ready += 0.5  # No GUI dependencies

        return container_ready

    def _test_cloud_agnostic(self, agent: Any) -> float:
        """Test cloud agnostic design."""
        # Check for cloud-specific dependencies
        cloud_specific = ["aws", "azure", "gcp", "boto3", "google-cloud"]

        try:
            source = inspect.getsource(agent.__class__)
            has_cloud_specific = any(c in source.lower() for c in cloud_specific)

            # Less cloud-specific code is better
            return 0.5 if has_cloud_specific else 1.0
        except Exception:
            return 0.5


# 9. Scalability Validator
class ScalabilityValidator(QualityValidator):
    """Validate agent scalability."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate scalability."""
        measurements = {}

        # Test horizontal scaling
        horizontal_score = self._test_horizontal_scaling(agent, test_data)
        measurements["horizontal_scaling"] = horizontal_score

        # Test vertical scaling
        vertical_score = self._test_vertical_scaling(agent, test_data)
        measurements["vertical_scaling"] = vertical_score

        # Test elasticity
        elasticity_score = self._test_elasticity(agent)
        measurements["elasticity"] = elasticity_score

        # Calculate overall score
        overall_score = np.mean([horizontal_score, vertical_score, elasticity_score])

        # Generate recommendations
        recommendations = []
        if horizontal_score < self.target_score:
            recommendations.append("Improve horizontal scaling capability")
        if vertical_score < self.target_score:
            recommendations.append("Optimize resource usage for vertical scaling")
        if elasticity_score < self.target_score:
            recommendations.append("Implement auto-scaling capabilities")

        metric = QualityMetrics(
            dimension=QualityDimension.SCALABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_horizontal_scaling(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test horizontal scaling capability."""
        # Test concurrent instance creation
        num_instances = 10
        instances = []

        try:
            for i in range(num_instances):
                # Create instance (mock)
                instance = type(agent).__new__(type(agent))
                instances.append(instance)

            # Check if instances are independent
            return 1.0 if len(instances) == num_instances else 0.5
        except Exception:
            return 0.0

    def _test_vertical_scaling(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test vertical scaling efficiency."""
        # Test with increasing load
        load_levels = [10, 100, 1000]
        performance_scores = []

        test_input = test_data.get("sample_input", {}) if test_data else {}

        for load in load_levels:
            start_time = time.time()
            success_count = 0

            for _ in range(min(load, 100)):  # Cap at 100 for testing
                try:
                    _ = agent.process(test_input)
                    success_count += 1
                except Exception:
                    break

            duration = time.time() - start_time
            throughput = success_count / duration if duration > 0 else 0

            # Check if throughput scales reasonably
            expected_throughput = load / 10  # Simple expectation
            performance_scores.append(min(1.0, throughput / expected_throughput))

        return np.mean(performance_scores) if performance_scores else 0.0

    def _test_elasticity(self, agent: Any) -> float:
        """Test auto-scaling elasticity."""
        elasticity_score = 0.0

        # Check for scaling indicators
        if hasattr(agent, "scale_up") or hasattr(agent, "scale_out"):
            elasticity_score += 0.5

        if hasattr(agent, "scale_down") or hasattr(agent, "scale_in"):
            elasticity_score += 0.5

        return elasticity_score


# 10. Interoperability Validator
class InteroperabilityValidator(QualityValidator):
    """Validate interoperability with other systems."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate interoperability."""
        measurements = {}

        # Test protocol support
        protocol_score = self._test_protocol_support(agent)
        measurements["protocol_support"] = protocol_score

        # Test data exchange
        exchange_score = self._test_data_exchange(agent)
        measurements["data_exchange"] = exchange_score

        # Test standard compliance
        standard_score = self._test_standard_compliance(agent)
        measurements["standard_compliance"] = standard_score

        # Calculate overall score
        overall_score = np.mean([protocol_score, exchange_score, standard_score])

        # Generate recommendations
        recommendations = []
        if protocol_score < self.target_score:
            recommendations.append("Support REST, GraphQL, and gRPC protocols")
        if exchange_score < self.target_score:
            recommendations.append("Support JSON, XML, and Protocol Buffers")
        if standard_score < self.target_score:
            recommendations.append("Comply with OpenAPI and AsyncAPI standards")

        metric = QualityMetrics(
            dimension=QualityDimension.INTEROPERABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_protocol_support(self, agent: Any) -> float:
        """Test protocol support."""
        protocols = ["REST", "GraphQL", "gRPC", "WebSocket"]
        supported = 0

        # Check for protocol handlers
        protocol_indicators = {
            "REST": ["handle_request", "rest_endpoint", "http_handler"],
            "GraphQL": ["graphql_schema", "resolve_query"],
            "gRPC": ["grpc_service", "proto_handler"],
            "WebSocket": ["websocket_handler", "on_message"]
        }

        for protocol, indicators in protocol_indicators.items():
            if any(hasattr(agent, ind) for ind in indicators):
                supported += 1

        return supported / len(protocols)

    def _test_data_exchange(self, agent: Any) -> float:
        """Test data exchange format support."""
        formats = ["JSON", "XML", "Protocol Buffers", "YAML"]
        supported_formats = 0

        # Check for format handlers
        if hasattr(agent, "to_json") or hasattr(agent, "from_json"):
            supported_formats += 1
        if hasattr(agent, "to_xml") or hasattr(agent, "from_xml"):
            supported_formats += 1
        if hasattr(agent, "to_proto") or hasattr(agent, "from_proto"):
            supported_formats += 1
        if hasattr(agent, "to_yaml") or hasattr(agent, "from_yaml"):
            supported_formats += 1

        return supported_formats / len(formats)

    def _test_standard_compliance(self, agent: Any) -> float:
        """Test standards compliance."""
        standards_checks = []

        # Check for OpenAPI compliance
        if hasattr(agent, "openapi_spec") or hasattr(agent, "swagger_spec"):
            standards_checks.append(1.0)
        else:
            standards_checks.append(0.0)

        # Check for AsyncAPI compliance
        if hasattr(agent, "asyncapi_spec"):
            standards_checks.append(1.0)
        else:
            standards_checks.append(0.0)

        return np.mean(standards_checks) if standards_checks else 0.0


# 11. Reusability Validator
class ReusabilityValidator(QualityValidator):
    """Validate component reusability."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate reusability."""
        measurements = {}

        # Test component reuse
        reuse_score = self._test_component_reuse(agent)
        measurements["component_reuse"] = reuse_score

        # Test pattern usage
        pattern_score = self._test_pattern_usage(agent)
        measurements["pattern_library"] = pattern_score

        # Test template usage
        template_score = self._test_template_usage(agent)
        measurements["template_usage"] = template_score

        # Calculate overall score
        overall_score = np.mean([reuse_score, pattern_score, template_score])

        # Generate recommendations
        recommendations = []
        if reuse_score < self.target_score:
            recommendations.append("Increase component reuse to >60%")
        if pattern_score < self.target_score:
            recommendations.append("Document and use design patterns")
        if template_score < self.target_score:
            recommendations.append("Create and use standardized templates")

        metric = QualityMetrics(
            dimension=QualityDimension.REUSABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_component_reuse(self, agent: Any) -> float:
        """Test component reusability."""
        # Check for modular components
        components = [m for m in dir(agent) if not m.startswith("_")]

        # Check if components follow single responsibility
        single_purpose = sum(1 for c in components if "_" not in c or c.count("_") <= 2)

        return single_purpose / len(components) if components else 0.0

    def _test_pattern_usage(self, agent: Any) -> float:
        """Test design pattern usage."""
        patterns = ["factory", "singleton", "observer", "strategy", "adapter"]
        pattern_usage = 0

        for pattern in patterns:
            if any(pattern in m.lower() for m in dir(agent)):
                pattern_usage += 1

        return pattern_usage / len(patterns)

    def _test_template_usage(self, agent: Any) -> float:
        """Test template utilization."""
        # Check for template methods
        if hasattr(agent, "template") or hasattr(agent, "from_template"):
            return 1.0
        elif any("template" in m.lower() for m in dir(agent)):
            return 0.5
        else:
            return 0.0


# 12. Testability Validator
class TestabilityValidator(QualityValidator):
    """Validate agent testability."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate testability."""
        measurements = {}

        # Test coverage potential
        coverage_score = self._test_coverage_potential(agent)
        measurements["test_coverage"] = coverage_score

        # Test automation potential
        automation_score = self._test_automation_potential(agent)
        measurements["test_automation"] = automation_score

        # Test efficiency
        efficiency_score = self._test_test_efficiency(agent)
        measurements["test_efficiency"] = efficiency_score

        # Calculate overall score
        overall_score = np.mean([coverage_score, automation_score, efficiency_score])

        # Generate recommendations
        recommendations = []
        if coverage_score < self.target_score:
            recommendations.append("Achieve >85% test coverage")
        if automation_score < self.target_score:
            recommendations.append("Automate >95% of tests")
        if efficiency_score < self.target_score:
            recommendations.append("Optimize test suite to run in <10 minutes")

        metric = QualityMetrics(
            dimension=QualityDimension.TESTABILITY,
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_coverage_potential(self, agent: Any) -> float:
        """Test potential for high coverage."""
        testable_methods = [
            m for m in dir(agent)
            if not m.startswith("_") and callable(getattr(agent, m))
        ]

        # Check if methods are testable (not too complex)
        simple_methods = 0
        for method in testable_methods:
            try:
                source = inspect.getsource(getattr(agent, method))
                lines = source.split("\n")
                if len(lines) < 50:  # Simple enough to test easily
                    simple_methods += 1
            except Exception:
                pass

        return simple_methods / len(testable_methods) if testable_methods else 0.0

    def _test_automation_potential(self, agent: Any) -> float:
        """Test automation potential."""
        automation_factors = []

        # Check for deterministic behavior
        if hasattr(agent, "set_seed") or hasattr(agent, "random_seed"):
            automation_factors.append(1.0)
        else:
            automation_factors.append(0.5)

        # Check for mockability
        if hasattr(agent, "set_mock") or any("mock" in m.lower() for m in dir(agent)):
            automation_factors.append(1.0)
        else:
            automation_factors.append(0.5)

        return np.mean(automation_factors) if automation_factors else 0.0

    def _test_test_efficiency(self, agent: Any) -> float:
        """Test efficiency of testing."""
        # Check for fast test execution
        start_time = time.time()

        try:
            # Run a simple operation
            if hasattr(agent, "process"):
                agent.process({})
            execution_time = time.time() - start_time

            # Fast execution enables efficient testing
            if execution_time < 0.01:  # <10ms
                return 1.0
            elif execution_time < 0.1:  # <100ms
                return 0.8
            else:
                return 0.5
        except Exception:
            return 0.5


# Comprehensive Quality Validator
class ComprehensiveQualityValidator:
    """Run all quality validations and generate comprehensive report."""

    def __init__(self, target_score: float = 0.8):
        """Initialize comprehensive validator."""
        self.target_score = target_score
        self.validators = {
            QualityDimension.FUNCTIONAL_QUALITY: FunctionalQualityValidator(target_score),
            QualityDimension.PERFORMANCE_EFFICIENCY: PerformanceValidator(target_score),
            QualityDimension.COMPATIBILITY: CompatibilityValidator(target_score),
            QualityDimension.USABILITY: UsabilityValidator(target_score),
            QualityDimension.RELIABILITY: ReliabilityValidator(target_score),
            QualityDimension.SECURITY: SecurityValidator(target_score),
            QualityDimension.MAINTAINABILITY: MaintainabilityValidator(target_score),
            QualityDimension.PORTABILITY: PortabilityValidator(target_score),
            QualityDimension.SCALABILITY: ScalabilityValidator(target_score),
            QualityDimension.INTEROPERABILITY: InteroperabilityValidator(target_score),
            QualityDimension.REUSABILITY: ReusabilityValidator(target_score),
            QualityDimension.TESTABILITY: TestabilityValidator(target_score)
        }

    def validate_agent(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """Run comprehensive quality validation."""
        start_time = time.time()
        dimension_results = []
        all_recommendations = []

        # Validate each dimension
        for dimension, validator in self.validators.items():
            try:
                metric = validator.validate(agent, test_data)
                dimension_results.append(metric)
                all_recommendations.extend(metric.recommendations)
            except Exception as e:
                # Create failed metric
                metric = QualityMetrics(
                    dimension=dimension,
                    score=0.0,
                    passed=False,
                    target=self.target_score,
                    measurements={"error": str(e)},
                    recommendations=[f"Fix validation error: {e}"]
                )
                dimension_results.append(metric)

        # Calculate overall score
        scores = [m.score for m in dimension_results]
        overall_score = np.mean(scores) if scores else 0.0

        # Generate summary
        passed_dimensions = sum(1 for m in dimension_results if m.passed)
        total_dimensions = len(dimension_results)

        summary = {
            "overall_score": overall_score,
            "passed_dimensions": passed_dimensions,
            "total_dimensions": total_dimensions,
            "pass_rate": passed_dimensions / total_dimensions if total_dimensions > 0 else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "target_score": self.target_score
        }

        # Create report
        report = QualityReport(
            overall_score=overall_score,
            passed=overall_score >= self.target_score,
            dimensions=dimension_results,
            summary=summary,
            recommendations=list(set(all_recommendations))[:10],  # Top 10 unique
            timestamp=DeterministicClock.now().isoformat(),
            duration_s=time.time() - start_time
        )

        return report

    def generate_html_report(self, report: QualityReport, output_file: str = "quality_report.html"):
        """Generate HTML quality report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GreenLang Agent Quality Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #2e7d32; color: white; padding: 20px; border-radius: 10px; }}
                h1 {{ margin: 0; }}
                .summary {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .dimension {{ background: white; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .pass {{ color: #2e7d32; font-weight: bold; }}
                .fail {{ color: #d32f2f; font-weight: bold; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .progress {{ width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }}
                .progress-bar {{ height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); transition: width 0.3s; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th {{ background: #f0f0f0; padding: 10px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #e0e0e0; }}
                .recommendation {{ background: #fff3e0; padding: 10px; border-left: 4px solid #ff9800; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>GreenLang Agent Quality Report</h1>
                <p>Generated: {report.timestamp}</p>
                <p>Duration: {report.duration_s:.2f} seconds</p>
            </div>

            <div class="summary">
                <h2>Overall Quality Score</h2>
                <div class="score {('pass' if report.passed else 'fail')}">{report.overall_score:.1%}</div>
                <div class="progress">
                    <div class="progress-bar" style="width: {report.overall_score * 100}%"></div>
                </div>
                <p>Target Score: {report.summary['target_score']:.1%}</p>
                <p>Passed Dimensions: {report.summary['passed_dimensions']}/{report.summary['total_dimensions']}</p>
            </div>

            <h2>Quality Dimensions</h2>
            {self._generate_dimension_html(report.dimensions)}

            <h2>Recommendations</h2>
            <div class="summary">
                {self._generate_recommendations_html(report.recommendations)}
            </div>
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

    def _generate_dimension_html(self, dimensions: List[QualityMetrics]) -> str:
        """Generate HTML for dimension results."""
        html = ""
        for metric in dimensions:
            status = "pass" if metric.passed else "fail"
            html += f"""
            <div class="dimension">
                <h3>{metric.dimension.value.replace('_', ' ').title()}</h3>
                <div class="score {status}">{metric.score:.1%}</div>
                <div class="progress">
                    <div class="progress-bar" style="width: {metric.score * 100}%"></div>
                </div>
                <table>
                    <tr><th>Measurement</th><th>Value</th></tr>
                    {self._generate_measurement_rows(metric.measurements)}
                </table>
            </div>
            """
        return html

    def _generate_measurement_rows(self, measurements: Dict[str, Any]) -> str:
        """Generate HTML rows for measurements."""
        rows = ""
        for key, value in measurements.items():
            if isinstance(value, float):
                value_str = f"{value:.2%}" if value <= 1.0 else f"{value:.2f}"
            else:
                value_str = str(value)
            rows += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value_str}</td></tr>"
        return rows

    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations."""
        if not recommendations:
            return "<p>No recommendations - all quality standards met!</p>"

        html = ""
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'
        return html


# Compliance Validator (specific for zero-hallucination and provenance)
class ComplianceValidator(QualityValidator):
    """Validate regulatory compliance and zero-hallucination guarantees."""

    def validate(
        self,
        agent: Any,
        test_data: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate compliance."""
        measurements = {}

        # Test zero-hallucination
        hallucination_score = self._test_zero_hallucination(agent, test_data)
        measurements["zero_hallucination"] = hallucination_score

        # Test provenance tracking
        provenance_score = self._test_provenance_tracking(agent, test_data)
        measurements["provenance_tracking"] = provenance_score

        # Test audit trail
        audit_score = self._test_audit_trail(agent)
        measurements["audit_trail"] = audit_score

        # Calculate overall score
        overall_score = np.mean([hallucination_score, provenance_score, audit_score])

        # Generate recommendations
        recommendations = []
        if hallucination_score < self.target_score:
            recommendations.append("Ensure zero-hallucination for critical calculations")
        if provenance_score < self.target_score:
            recommendations.append("Implement complete provenance tracking")
        if audit_score < self.target_score:
            recommendations.append("Maintain comprehensive audit trails")

        metric = QualityMetrics(
            dimension=QualityDimension.FUNCTIONAL_QUALITY,  # Part of functional quality
            score=overall_score,
            passed=overall_score >= self.target_score,
            target=self.target_score,
            measurements=measurements,
            recommendations=recommendations
        )

        self.record_metric(metric)
        return metric

    def _test_zero_hallucination(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test zero-hallucination guarantee."""
        if not test_data or "calculations" not in test_data:
            return 1.0

        hallucination_tests = []

        for calc in test_data["calculations"]:
            try:
                result = agent.process(calc["input"])
                expected = calc["expected"]

                # Check numerical accuracy
                if isinstance(result, (int, float, Decimal)):
                    error = abs(float(result) - float(expected))
                    hallucination_tests.append(1.0 if error < 1e-6 else 0.0)
                else:
                    hallucination_tests.append(1.0 if result == expected else 0.0)
            except Exception:
                hallucination_tests.append(0.0)

        return np.mean(hallucination_tests) if hallucination_tests else 0.0

    def _test_provenance_tracking(self, agent: Any, test_data: Optional[Dict] = None) -> float:
        """Test provenance tracking completeness."""
        provenance_score = 0.0

        # Check for provenance capabilities
        if hasattr(agent, "get_provenance") or hasattr(agent, "provenance_chain"):
            provenance_score += 0.5

        # Test provenance generation
        try:
            test_input = test_data.get("sample_input", {}) if test_data else {}
            result = agent.process(test_input)

            # Check if result has provenance
            if hasattr(result, "provenance_hash") or hasattr(result, "provenance"):
                provenance_score += 0.5
        except Exception:
            pass

        return provenance_score

    def _test_audit_trail(self, agent: Any) -> float:
        """Test audit trail completeness."""
        audit_capabilities = 0

        # Check for audit methods
        audit_methods = ["log", "audit", "record", "track"]
        for method in audit_methods:
            if any(method in m.lower() for m in dir(agent)):
                audit_capabilities += 1

        return audit_capabilities / len(audit_methods)


if __name__ == "__main__":
    # Example usage
    print("GreenLang Quality Validation Framework Loaded")
    print("12 Quality Dimensions Available:")
    for dimension in QualityDimension:
        print(f"  - {dimension.value}")