"""
EUDR Deforestation Compliance Agent - Validates supply chain compliance with the EU Deforestation Regulation (EU) 2023/1115. Covers 7 regulated commodities: cattle, cocoa, coffee, palm oil, rubber, soya, and wood. Ensures products are deforestation-free (cutoff date: December 31, 2020) and legally produced. Generates EU Due Diligence Statements (DDS) for regulatory submission.


Version: 1.0.0
License: Apache-2.0
"""

from eudr_compliance_v1.agent import EudrDeforestationComplianceAgentAgent
from eudr_compliance_v1.tools import *

__all__ = [
    "EudrDeforestationComplianceAgentAgent",
]

__version__ = "1.0.0"
