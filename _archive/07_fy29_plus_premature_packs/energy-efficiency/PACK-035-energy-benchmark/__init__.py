# -*- coding: utf-8 -*-
"""
PACK-035: Energy Benchmark Pack
=================================

Comprehensive energy benchmarking platform for comparing facility energy
performance against peers, standards, and best practices. Covers EUI
calculation, weather normalisation, peer comparison, sector-specific
benchmarking, portfolio management, performance gap analysis, regulatory
compliance, target setting, trend analysis, and continuous monitoring.

Key capabilities:
  - EUI calculation with multiple accounting boundaries (site, source, primary)
  - Weather normalisation using degree-day regression (2P-5P models)
  - Peer comparison against ENERGY STAR, CIBSE TM46, DIN V 18599, BPIE
  - Portfolio benchmarking for 1-1000+ facilities with mixed building types
  - End-use disaggregated gap analysis (lighting, HVAC, process, plug loads)
  - Performance ratings: EPC A-G, ENERGY STAR 1-100, NABERS stars, CRREM
  - Continuous monitoring with CUSUM, SPC, and automated alerting
  - Regulatory compliance: EPBD, EED, MEES, LL97, NABERS
  - Target setting with peer context and SBTi alignment

Regulatory references:
  - EED: Directive (EU) 2023/1791 (Article 8/11 benchmarking)
  - EPBD: Directive 2024/1275 (EPC, DEC, MEPS, ZEB)
  - ISO 50001:2018 / ISO 50006:2014 (EnPI benchmarking)
  - ASHRAE Standard 100-2018 (building energy benchmarking)
  - ENERGY STAR Portfolio Manager (US benchmarking methodology)
  - CIBSE TM46:2008 (UK building energy benchmarks)
  - GHG Protocol (carbon intensity benchmarking)

Category: Energy Efficiency Packs
Pack Tier: Professional
Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
Status: Production Ready
"""

__version__: str = "1.0.0"
__pack__: str = "PACK-035"
__pack_name__: str = "Energy Benchmark Pack"
__category__: str = "energy-efficiency"
