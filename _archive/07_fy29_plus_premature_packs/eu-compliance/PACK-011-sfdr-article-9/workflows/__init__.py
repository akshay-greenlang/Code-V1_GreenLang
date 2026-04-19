"""
PACK-011 SFDR Article 9 Pack - Workflows module.

Provides eight workflow orchestrators for Article 9 financial product
compliance under EU SFDR Regulation 2019/2088.

Workflows:
    1. AnnexIIIDisclosureWorkflow - Annex III pre-contractual disclosure (5 phases)
    2. AnnexVReportingWorkflow - Annex V periodic reporting (4 phases)
    3. SustainableVerificationWorkflow - Sustainable investment verification (4 phases)
    4. ImpactReportingWorkflow - Impact measurement and reporting (4 phases)
    5. BenchmarkMonitoringWorkflow - EU Climate Benchmark monitoring (4 phases)
    6. PAIMandatoryWorkflow - Mandatory PAI indicator assessment (4 phases)
    7. DowngradeMonitoringWorkflow - Article 9 downgrade risk monitoring (4 phases)
    8. RegulatoryUpdateWorkflow - Regulatory change management (3 phases)
"""

from .annex_iii_disclosure import (
    AnnexIIIDisclosureInput,
    AnnexIIIDisclosureResult,
    AnnexIIIDisclosureWorkflow,
)
from .annex_v_reporting import (
    AnnexVReportingInput,
    AnnexVReportingResult,
    AnnexVReportingWorkflow,
)
from .benchmark_monitoring import (
    BenchmarkMonitoringInput,
    BenchmarkMonitoringResult,
    BenchmarkMonitoringWorkflow,
)
from .downgrade_monitoring import (
    DowngradeMonitoringInput,
    DowngradeMonitoringResult,
    DowngradeMonitoringWorkflow,
)
from .impact_reporting import (
    ImpactReportingInput,
    ImpactReportingResult,
    ImpactReportingWorkflow,
)
from .pai_mandatory_workflow import (
    PAIMandatoryInput,
    PAIMandatoryResult,
    PAIMandatoryWorkflow,
)
from .regulatory_update import (
    RegulatoryUpdateInput,
    RegulatoryUpdateResult,
    RegulatoryUpdateWorkflow,
)
from .sustainable_verification import (
    SustainableVerificationInput,
    SustainableVerificationResult,
    SustainableVerificationWorkflow,
)

__all__ = [
    # Annex III Disclosure (5-phase)
    "AnnexIIIDisclosureWorkflow",
    "AnnexIIIDisclosureInput",
    "AnnexIIIDisclosureResult",
    # Annex V Reporting (4-phase)
    "AnnexVReportingWorkflow",
    "AnnexVReportingInput",
    "AnnexVReportingResult",
    # Sustainable Verification (4-phase)
    "SustainableVerificationWorkflow",
    "SustainableVerificationInput",
    "SustainableVerificationResult",
    # Impact Reporting (4-phase)
    "ImpactReportingWorkflow",
    "ImpactReportingInput",
    "ImpactReportingResult",
    # Benchmark Monitoring (4-phase)
    "BenchmarkMonitoringWorkflow",
    "BenchmarkMonitoringInput",
    "BenchmarkMonitoringResult",
    # PAI Mandatory (4-phase)
    "PAIMandatoryWorkflow",
    "PAIMandatoryInput",
    "PAIMandatoryResult",
    # Downgrade Monitoring (4-phase)
    "DowngradeMonitoringWorkflow",
    "DowngradeMonitoringInput",
    "DowngradeMonitoringResult",
    # Regulatory Update (3-phase)
    "RegulatoryUpdateWorkflow",
    "RegulatoryUpdateInput",
    "RegulatoryUpdateResult",
]
