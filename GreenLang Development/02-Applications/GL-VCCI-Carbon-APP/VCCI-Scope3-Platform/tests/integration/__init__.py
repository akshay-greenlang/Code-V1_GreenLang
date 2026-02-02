# -*- coding: utf-8 -*-
# Integration Tests
# End-to-end integration tests

"""
Integration Tests
=================

End-to-end integration tests covering complete workflows.

Test Scenarios:
--------------
- Complete Scope 3 pipeline (intake → calculate → analyze → report)
- Multi-tenant isolation
- ERP data extraction → calculation → reporting
- Supplier engagement workflows
- Report generation (GHG Protocol, CDP, SBTi)

Test Files:
----------
- test_e2e_pipeline.py: Complete pipeline tests (20+ scenarios)
- test_multi_tenant.py: Multi-tenancy tests (15+ scenarios)
- test_erp_to_report.py: ERP integration → report generation (15+ scenarios)
"""

__version__ = "1.0.0"
