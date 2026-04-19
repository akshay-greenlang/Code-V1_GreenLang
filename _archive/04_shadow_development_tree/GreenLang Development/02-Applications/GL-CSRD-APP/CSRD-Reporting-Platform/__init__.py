# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform
======================================

A comprehensive GreenLang-powered platform for EU Corporate Sustainability
Reporting Directive (CSRD) compliance.

Features:
- 6-agent pipeline processing 1,082 ESRS data points
- Zero-hallucination calculations
- AI-powered double materiality assessment
- XBRL/ESEF report generation
- Complete audit trail

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "GreenLang CSRD Team"

from .csrd_pipeline import CSRDPipeline

__all__ = ["CSRDPipeline"]
