# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH End-to-End Tests

End-to-end tests for complete workflow validation of the
EmissionsComplianceAgent. Tests simulate real-world operational
scenarios from data acquisition through regulatory reporting.

Test Modules:
    - test_complete_workflow.py: Complete operational workflows (12+ tests)

Test Scenarios:
    - Daily Compliance Workflow: 24-hour monitoring cycle
    - Quarterly Reporting Workflow: EPA ECMPS submission
    - Violation Response Workflow: Detection to corrective action
    - Annual Emissions Inventory: Full year aggregation

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

__all__ = [
    "test_complete_workflow",
]
