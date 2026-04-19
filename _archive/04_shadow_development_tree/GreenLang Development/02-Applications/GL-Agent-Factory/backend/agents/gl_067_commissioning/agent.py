"""GL-067: Commissioning Agent (COMMISSIONING).

Manages building and system commissioning processes.

Standards: ASHRAE Guideline 0, ISO 16813
"""
import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CommissioningPhase(str, Enum):
    DESIGN = "DESIGN"
    CONSTRUCTION = "CONSTRUCTION"
    FUNCTIONAL_TESTING = "FUNCTIONAL_TESTING"
    ACCEPTANCE = "ACCEPTANCE"
    WARRANTY = "WARRANTY"
    ONGOING = "ONGOING"


class SystemStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    ISSUES_FOUND = "ISSUES_FOUND"


class CommissioningTest(BaseModel):
    test_id: str
    test_name: str
    system_name: str
    status: SystemStatus = Field(default=SystemStatus.NOT_STARTED)
    expected_value: float
    measured_value: Optional[float] = None
    tolerance_pct: float = Field(default=10)
    passed: Optional[bool] = None


class CommissioningInput(BaseModel):
    project_id: str
    project_name: str = Field(default="Project")
    phase: CommissioningPhase = Field(default=CommissioningPhase.FUNCTIONAL_TESTING)
    tests: List[CommissioningTest] = Field(default_factory=list)
    total_systems: int = Field(default=10, ge=1)
    start_date: datetime = Field(default_factory=datetime.utcnow)
    target_completion_date: Optional[datetime] = None
    energy_baseline_kwh: float = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommissioningOutput(BaseModel):
    project_id: str
    phase: str
    tests_total: int
    tests_passed: int
    tests_failed: int
    tests_pending: int
    pass_rate_pct: float
    systems_verified: int
    issues_found: int
    energy_savings_verified_kwh: float
    status_summary: str
    recommendations: List[str]
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


class CommissioningAgent:
    AGENT_ID = "GL-067"
    AGENT_NAME = "COMMISSIONING"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"CommissioningAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = CommissioningInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _evaluate_test(self, test: CommissioningTest) -> bool:
        """Evaluate if a test passes within tolerance."""
        if test.measured_value is None:
            return False

        tolerance = test.expected_value * (test.tolerance_pct / 100)
        lower = test.expected_value - tolerance
        upper = test.expected_value + tolerance

        return lower <= test.measured_value <= upper

    def _process(self, inp: CommissioningInput) -> CommissioningOutput:
        recommendations = []

        # Evaluate all tests
        passed = 0
        failed = 0
        pending = 0
        issues = 0

        for test in inp.tests:
            if test.status == SystemStatus.NOT_STARTED:
                pending += 1
            elif test.status == SystemStatus.IN_PROGRESS:
                pending += 1
            elif test.status == SystemStatus.ISSUES_FOUND:
                failed += 1
                issues += 1
            else:
                if self._evaluate_test(test):
                    passed += 1
                else:
                    failed += 1
                    issues += 1

        total_tests = len(inp.tests)
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0

        # Systems verified (unique systems with all tests passed)
        system_tests = {}
        for test in inp.tests:
            if test.system_name not in system_tests:
                system_tests[test.system_name] = {"total": 0, "passed": 0}
            system_tests[test.system_name]["total"] += 1
            if test.status == SystemStatus.COMPLETE and self._evaluate_test(test):
                system_tests[test.system_name]["passed"] += 1

        verified = sum(1 for s in system_tests.values() if s["total"] > 0 and s["passed"] == s["total"])

        # Energy savings estimate (10% typical for new commissioning)
        savings_verified = inp.energy_baseline_kwh * 0.10 * (pass_rate / 100)

        # Status summary
        if pass_rate >= 95 and pending == 0:
            status = "COMMISSIONING COMPLETE - All systems verified"
        elif pass_rate >= 80:
            status = "ON TRACK - Minor issues to resolve"
        elif issues > total_tests * 0.2:
            status = "AT RISK - Significant issues found"
        else:
            status = "IN PROGRESS"

        # Recommendations
        if failed > 0:
            recommendations.append(f"{failed} tests failed - schedule retesting after corrections")
        if pending > total_tests * 0.3:
            recommendations.append(f"{pending} tests pending - accelerate testing schedule")
        if issues > 5:
            recommendations.append(f"{issues} issues logged - prioritize resolution meeting")

        # Phase-specific recommendations
        if inp.phase == CommissioningPhase.FUNCTIONAL_TESTING and pass_rate < 80:
            recommendations.append("Consider extending functional testing phase")
        if inp.phase == CommissioningPhase.ACCEPTANCE and issues > 0:
            recommendations.append("Resolve all issues before final acceptance")
        if inp.phase == CommissioningPhase.WARRANTY:
            recommendations.append("Document all seasonal testing requirements")

        calc_hash = hashlib.sha256(json.dumps({
            "project": inp.project_id,
            "phase": inp.phase.value,
            "pass_rate": round(pass_rate, 1)
        }).encode()).hexdigest()

        return CommissioningOutput(
            project_id=inp.project_id,
            phase=inp.phase.value,
            tests_total=total_tests,
            tests_passed=passed,
            tests_failed=failed,
            tests_pending=pending,
            pass_rate_pct=round(pass_rate, 1),
            systems_verified=verified,
            issues_found=issues,
            energy_savings_verified_kwh=round(savings_verified, 0),
            status_summary=status,
            recommendations=recommendations,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )


PACK_SPEC = {
    "schema_version": "2.0.0", "id": "GL-067", "name": "COMMISSIONING", "version": "1.0.0",
    "summary": "Building and system commissioning management",
    "standards": [{"ref": "ASHRAE Guideline 0"}, {"ref": "ISO 16813"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
