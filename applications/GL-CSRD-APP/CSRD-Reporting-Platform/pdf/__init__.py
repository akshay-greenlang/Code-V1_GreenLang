# -*- coding: utf-8 -*-
"""
GL-CSRD-APP PDF Report Generation Module
=========================================

WeasyPrint-based professional CSRD report generation with multi-language
support, inline SVG charts, and full ESRS section coverage.

This module replaces the placeholder ``PDFGenerator`` that was previously
defined inside ``agents/reporting_agent.py`` with a production-grade
implementation that produces A4-formatted PDF reports suitable for EU
regulatory submission alongside the iXBRL/ESEF package.

Key capabilities:
    - A4 layout with @page rules (headers, footers, page numbers)
    - Inline SVG charts: bar, pie, line, stacked_bar, waterfall
    - Multi-language support (EN, DE, FR, ES) with locale-aware formatting
    - Auto-generated table of contents
    - KPI dashboard with metric cards
    - Materiality matrix scatter plot
    - Compliance summary with progress bars
    - SHA-256 provenance hash for every generated PDF
    - Thread-safe singleton PDFGenerator

Usage::

    from pdf import PDFGenerator, ReportTemplate, ReportStyles

    gen = PDFGenerator()
    pdf_bytes = gen.generate_csrd_report(
        report_data=pipeline_output,
        company_profile=company_profile,
        materiality=materiality_assessment,
        metrics=calculated_metrics,
        locale="en",
    )
    Path("csrd_report.pdf").write_bytes(pdf_bytes)

Version: 1.1.0
Author: GreenLang CSRD Team
License: MIT
"""

from pdf.pdf_generator import (
    PDFGenerator,
    ReportTemplate,
    ReportStyles,
    ReportSection,
    DataTable,
    ChartSpec,
    BRAND_COLORS,
)

__all__ = [
    "PDFGenerator",
    "ReportTemplate",
    "ReportStyles",
    "ReportSection",
    "DataTable",
    "ChartSpec",
    "BRAND_COLORS",
]

__version__ = "1.1.0"
