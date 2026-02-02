# -*- coding: utf-8 -*-
"""
Audit Trail Generator
GL-VCCI Scope 3 Platform

Generates comprehensive audit documentation for reports.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class AuditTrailGenerator:
    """
    Generates audit trail documentation for sustainability reports.

    Features:
    - Complete calculation provenance
    - Data lineage tracking
    - Methodology documentation
    - Audit-ready evidence packages
    """

    def __init__(self):
        """Initialize audit trail generator."""
        pass

    def generate_audit_package(
        self,
        emissions_data: Dict[str, Any],
        calculations: List[Dict[str, Any]],
        report_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive audit package.

        Args:
            emissions_data: Emissions data used
            calculations: Calculation details
            report_metadata: Report metadata

        Returns:
            Audit package dictionary
        """
        logger.info("Generating audit trail package")

        package = {
            "audit_id": self._generate_audit_id(report_metadata),
            "generated_at": DeterministicClock.utcnow().isoformat(),
            "report_metadata": report_metadata,
            "data_lineage": self._build_data_lineage(emissions_data),
            "calculation_evidence": self._build_calculation_evidence(calculations),
            "methodology_documentation": self._build_methodology_docs(calculations),
            "data_quality_evidence": self._build_dqi_evidence(emissions_data),
            "provenance_chains": self._extract_provenance_chains(emissions_data),
            "integrity_hashes": self._compute_integrity_hashes(emissions_data, calculations),
        }

        logger.info(f"Audit package generated: {package['audit_id']}")
        return package

    def generate_methodology_appendix(
        self,
        emissions_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate methodology appendix for report.

        Args:
            emissions_data: Emissions data

        Returns:
            Methodology documentation
        """
        return {
            "title": "Calculation Methodology",
            "sections": [
                {
                    "section": "Scope 1",
                    "description": "Direct GHG emissions from owned or controlled sources",
                    "methodology": "Activity-based calculation using emission factors",
                    "standards": ["GHG Protocol", "ISO 14064-1"],
                },
                {
                    "section": "Scope 2",
                    "description": "Indirect emissions from purchased electricity, heat, steam",
                    "methodology": "Location-based and market-based methods",
                    "standards": ["GHG Protocol Scope 2 Guidance"],
                },
                {
                    "section": "Scope 3",
                    "description": "All other indirect emissions in value chain",
                    "methodology": self._get_scope3_methodologies(emissions_data),
                    "standards": ["GHG Protocol Corporate Value Chain (Scope 3) Standard"],
                },
            ],
            "emission_factors": self._document_emission_factors(emissions_data),
            "gwp_standard": "IPCC AR6",
            "uncertainty_approach": "Monte Carlo simulation (10,000 iterations)",
        }

    def generate_limitations_disclosure(
        self,
        emissions_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate limitations and exclusions disclosure.

        Args:
            emissions_data: Emissions data

        Returns:
            Limitations documentation
        """
        return {
            "title": "Limitations and Exclusions",
            "scope3_categories_excluded": self._get_excluded_categories(emissions_data),
            "data_gaps": self._identify_data_gaps(emissions_data),
            "assumptions": self._document_key_assumptions(emissions_data),
            "data_quality_limitations": self._document_dqi_limitations(emissions_data),
            "boundary_exclusions": [],
            "future_improvements": [
                "Expand Scope 3 category coverage",
                "Increase supplier-specific data collection",
                "Implement continuous monitoring systems",
            ],
        }

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    def _generate_audit_id(self, metadata: Dict[str, Any]) -> str:
        """Generate unique audit ID."""
        timestamp = DeterministicClock.utcnow().isoformat()
        data = f"{metadata.get('report_id', '')}_{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16].upper()

    def _build_data_lineage(self, emissions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build data lineage documentation."""
        return {
            "data_sources": emissions_data.get("data_sources", []),
            "collection_period": {
                "start": emissions_data.get("reporting_period_start"),
                "end": emissions_data.get("reporting_period_end"),
            },
            "processing_steps": [
                "Data ingestion and validation",
                "Entity resolution and matching",
                "Calculation and aggregation",
                "Quality assurance checks",
            ],
            "transformations": [
                "Unit conversions to tCO2e",
                "Geographic aggregation",
                "Category classification",
            ],
        }

    def _build_calculation_evidence(self, calculations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build calculation evidence."""
        evidence = []
        for calc in calculations[:100]:  # Limit to 100 for brevity
            evidence.append({
                "calculation_id": calc.get("calculation_id", "unknown"),
                "category": calc.get("category"),
                "input_values": calc.get("inputs", {}),
                "emission_factor": calc.get("emission_factor"),
                "result_kgco2e": calc.get("emissions_kgco2e"),
                "data_quality_score": calc.get("dqi_score"),
            })
        return evidence

    def _build_methodology_docs(self, calculations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build methodology documentation."""
        categories = {}
        for calc in calculations:
            cat = calc.get("category")
            if cat and cat not in categories:
                categories[cat] = {
                    "category": cat,
                    "method": calc.get("calculation_method", "unknown"),
                    "tier": calc.get("tier"),
                    "standard": "GHG Protocol",
                }

        return {
            "categories": list(categories.values()),
            "overall_approach": "Activity-based calculations with emission factors",
            "standards_compliance": ["GHG Protocol", "ISO 14064-1", "ISO 14083"],
        }

    def _build_dqi_evidence(self, emissions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build data quality evidence."""
        return {
            "overall_dqi_score": emissions_data.get("avg_dqi_score"),
            "dqi_by_scope": emissions_data.get("data_quality_by_scope", {}),
            "assessment_method": "Pedigree matrix approach",
            "quality_dimensions": [
                "Temporal correlation",
                "Geographic correlation",
                "Technological correlation",
                "Completeness",
                "Reliability",
            ],
        }

    def _extract_provenance_chains(self, emissions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract provenance chains."""
        chains = emissions_data.get("provenance_chains", [])
        return chains[:50]  # First 50 chains

    def _compute_integrity_hashes(
        self,
        emissions_data: Dict[str, Any],
        calculations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Compute integrity hashes."""
        return {
            "emissions_data_hash": hashlib.sha256(
                json.dumps(emissions_data, sort_keys=True).encode()
            ).hexdigest(),
            "calculations_hash": hashlib.sha256(
                json.dumps(calculations, sort_keys=True).encode()
            ).hexdigest(),
        }

    def _get_scope3_methodologies(self, emissions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get Scope 3 methodologies by category."""
        methodologies = []
        categories = emissions_data.get("scope3_categories", {})

        for cat_num, emissions in categories.items():
            methodologies.append({
                "category": cat_num,
                "category_name": self._get_category_name(cat_num),
                "method": self._get_category_method(cat_num),
            })

        return methodologies

    def _get_category_name(self, cat_num: int) -> str:
        """Get Scope 3 category name."""
        names = {
            1: "Purchased Goods & Services",
            4: "Upstream Transportation & Distribution",
            6: "Business Travel",
        }
        return names.get(cat_num, f"Category {cat_num}")

    def _get_category_method(self, cat_num: int) -> str:
        """Get calculation method for category."""
        methods = {
            1: "Supplier-specific PCF or spend-based with EEIO factors",
            4: "Distance-based calculation per ISO 14083",
            6: "Distance-based for flights, nights for hotels",
        }
        return methods.get(cat_num, "Activity-based calculation")

    def _document_emission_factors(self, emissions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document emission factors used."""
        # This would extract from provenance chains
        return [
            {
                "source": "DEFRA 2024",
                "coverage": "Road transport, UK",
                "vintage": 2024,
            },
            {
                "source": "EPA 2024",
                "coverage": "Electricity, US grid",
                "vintage": 2024,
            },
        ]

    def _get_excluded_categories(self, emissions_data: Dict[str, Any]) -> List[int]:
        """Get excluded Scope 3 categories."""
        all_categories = set(range(1, 16))
        covered = set(emissions_data.get("scope3_categories", {}).keys())
        return sorted(all_categories - covered)

    def _identify_data_gaps(self, emissions_data: Dict[str, Any]) -> List[str]:
        """Identify data gaps."""
        gaps = []

        if emissions_data.get("avg_dqi_score", 100) < 80:
            gaps.append("Some data quality below target (DQI < 80)")

        excluded_cats = self._get_excluded_categories(emissions_data)
        if excluded_cats:
            gaps.append(f"Scope 3 categories not calculated: {excluded_cats}")

        return gaps

    def _document_key_assumptions(self, emissions_data: Dict[str, Any]) -> List[str]:
        """Document key assumptions."""
        return [
            "All purchased goods allocated to reporting period based on delivery date",
            "Load factors for transport assumed at industry average where actual data unavailable",
            "Radiative forcing factor of 1.9 applied to aviation emissions",
            "Market-based Scope 2 uses supplier-specific emission factors where available",
        ]

    def _document_dqi_limitations(self, emissions_data: Dict[str, Any]) -> List[str]:
        """Document data quality limitations."""
        dqi = emissions_data.get("avg_dqi_score", 100)

        limitations = []
        if dqi < 90:
            limitations.append("Some activity data based on estimates rather than primary measurements")
        if dqi < 80:
            limitations.append("Limited supplier-specific emission factors available")
        if dqi < 70:
            limitations.append("Significant reliance on spend-based calculations")

        return limitations


__all__ = ["AuditTrailGenerator"]
