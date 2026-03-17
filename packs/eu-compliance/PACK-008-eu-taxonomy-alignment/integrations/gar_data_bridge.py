"""
GAR Data Bridge - PACK-008 EU Taxonomy Alignment

This module aggregates counterparty taxonomy data for financial institution
Green Asset Ratio (GAR) and Banking Book Taxonomy Alignment Ratio (BTAR)
calculation. It supports exposure classification, EPC rating integration
for real estate, and de minimis threshold handling.

GAR coverage:
- Counterparty taxonomy data aggregation
- Exposure classification (corporate loans, debt securities, equity, mortgages, project finance)
- EPC (Energy Performance Certificate) rating integration for real estate
- GAR stock and flow calculation inputs
- BTAR calculation for banking books
- De minimis threshold handling
- Template 6-10 data preparation for EBA Pillar 3

Example:
    >>> config = GARDataConfig(
    ...     exposure_types=["corporate_loans", "mortgages"],
    ...     epc_source="national_registry"
    ... )
    >>> bridge = GARDataBridge(config)
    >>> portfolio = await bridge.aggregate_exposures(portfolio_data)
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class GARDataConfig(BaseModel):
    """Configuration for GAR Data Bridge."""

    exposure_types: List[str] = Field(
        default=[
            "corporate_loans", "debt_securities", "equity_holdings",
            "residential_mortgages", "commercial_mortgages",
            "project_finance", "auto_loans"
        ],
        description="Exposure types to include in GAR calculation"
    )
    epc_source: Literal["national_registry", "manual_input", "estimated"] = Field(
        default="national_registry",
        description="Source for EPC ratings"
    )
    include_btar: bool = Field(
        default=True,
        description="Include Banking Book Taxonomy Alignment Ratio"
    )
    de_minimis_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="De minimis threshold for SME counterparties"
    )
    reporting_date: str = Field(
        default="2025-12-31",
        description="GAR reference date (YYYY-MM-DD)"
    )
    consolidation_scope: Literal["solo", "consolidated", "sub_consolidated"] = Field(
        default="consolidated",
        description="Prudential consolidation scope"
    )


class GARDataBridge:
    """
    Bridge for GAR/BTAR data aggregation for financial institutions.

    Aggregates counterparty taxonomy data, classifies exposures, integrates
    EPC ratings, and prepares data for GAR/BTAR computation and EBA
    Pillar 3 template generation.

    Example:
        >>> config = GARDataConfig(exposure_types=["corporate_loans", "mortgages"])
        >>> bridge = GARDataBridge(config)
        >>> bridge.inject_service(counterparty_data_service)
        >>> exposures = await bridge.aggregate_exposures(portfolio_data)
    """

    # Exposure classification hierarchy
    EXPOSURE_CLASSES: Dict[str, Dict[str, Any]] = {
        "corporate_loans": {
            "eba_category": "Loans and advances",
            "template": "Template 7",
            "requires_counterparty_data": True,
            "nfrd_applicable": True
        },
        "debt_securities": {
            "eba_category": "Debt securities",
            "template": "Template 7",
            "requires_counterparty_data": True,
            "nfrd_applicable": True
        },
        "equity_holdings": {
            "eba_category": "Equity instruments",
            "template": "Template 7",
            "requires_counterparty_data": True,
            "nfrd_applicable": True
        },
        "residential_mortgages": {
            "eba_category": "Loans collateralised by residential immovable property",
            "template": "Template 8",
            "requires_counterparty_data": False,
            "requires_epc": True,
            "nfrd_applicable": False
        },
        "commercial_mortgages": {
            "eba_category": "Loans collateralised by commercial immovable property",
            "template": "Template 8",
            "requires_counterparty_data": False,
            "requires_epc": True,
            "nfrd_applicable": False
        },
        "project_finance": {
            "eba_category": "Other on-balance-sheet exposures (project finance loans)",
            "template": "Template 7",
            "requires_counterparty_data": True,
            "nfrd_applicable": True
        },
        "auto_loans": {
            "eba_category": "Motor vehicle loans",
            "template": "Template 9",
            "requires_counterparty_data": False,
            "nfrd_applicable": False
        }
    }

    # EPC rating to taxonomy alignment mapping
    EPC_ALIGNMENT_MAP: Dict[str, Dict[str, Any]] = {
        "A": {"aligned": True, "top_15_percent": True, "nzeb_compliant": True},
        "B": {"aligned": True, "top_15_percent": True, "nzeb_compliant": False},
        "C": {"aligned": False, "top_15_percent": False, "nzeb_compliant": False},
        "D": {"aligned": False, "top_15_percent": False, "nzeb_compliant": False},
        "E": {"aligned": False, "top_15_percent": False, "nzeb_compliant": False},
        "F": {"aligned": False, "top_15_percent": False, "nzeb_compliant": False},
        "G": {"aligned": False, "top_15_percent": False, "nzeb_compliant": False}
    }

    def __init__(self, config: GARDataConfig):
        """Initialize GAR data bridge."""
        self.config = config
        self._service: Any = None
        logger.info(
            f"GARDataBridge initialized "
            f"(exposure_types={len(config.exposure_types)}, "
            f"scope={config.consolidation_scope})"
        )

    def inject_service(self, service: Any) -> None:
        """Inject real counterparty data service."""
        self._service = service
        logger.info("Injected GAR data service")

    async def import_counterparty_data(
        self,
        counterparties: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Import counterparty taxonomy data for GAR calculation.

        Args:
            counterparties: List of counterparty records with taxonomy data

        Returns:
            Processed counterparty data with taxonomy alignment indicators
        """
        try:
            if self._service and hasattr(self._service, "import_counterparty_data"):
                return await self._service.import_counterparty_data(counterparties)

            processed = []
            nfrd_covered = 0
            non_nfrd = 0

            for cp in counterparties:
                is_nfrd = cp.get("nfrd_subject", False)
                if is_nfrd:
                    nfrd_covered += 1
                else:
                    non_nfrd += 1

                processed.append({
                    "counterparty_id": cp.get("id", ""),
                    "name": cp.get("name", ""),
                    "nfrd_subject": is_nfrd,
                    "turnover_ratio": cp.get("turnover_ratio", 0.0),
                    "capex_ratio": cp.get("capex_ratio", 0.0),
                    "sector": cp.get("nace_sector", ""),
                    "country": cp.get("country", ""),
                    "exposure_amount": cp.get("exposure_amount", 0.0)
                })

            return {
                "total_counterparties": len(counterparties),
                "nfrd_covered": nfrd_covered,
                "non_nfrd": non_nfrd,
                "counterparties": processed,
                "provenance_hash": self._calculate_hash(processed),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Counterparty data import failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def aggregate_exposures(
        self,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate exposures by classification for GAR calculation.

        Args:
            portfolio: Portfolio data with exposure details

        Returns:
            Aggregated exposure data by class and taxonomy alignment
        """
        try:
            if self._service and hasattr(self._service, "aggregate_exposures"):
                return await self._service.aggregate_exposures(portfolio)

            exposures_by_class = {}
            total_on_balance = 0.0
            total_taxonomy_eligible = 0.0
            total_taxonomy_aligned = 0.0

            for exp_type in self.config.exposure_types:
                class_info = self.EXPOSURE_CLASSES.get(exp_type, {})
                class_data = portfolio.get(exp_type, {})

                amount = class_data.get("total_amount", 0.0)
                eligible = class_data.get("taxonomy_eligible", 0.0)
                aligned = class_data.get("taxonomy_aligned", 0.0)

                total_on_balance += amount
                total_taxonomy_eligible += eligible
                total_taxonomy_aligned += aligned

                exposures_by_class[exp_type] = {
                    "eba_category": class_info.get("eba_category", ""),
                    "template": class_info.get("template", ""),
                    "total_amount": amount,
                    "taxonomy_eligible": eligible,
                    "taxonomy_aligned": aligned,
                    "eligible_ratio": eligible / amount if amount > 0 else 0.0,
                    "aligned_ratio": aligned / amount if amount > 0 else 0.0
                }

            return {
                "reporting_date": self.config.reporting_date,
                "consolidation_scope": self.config.consolidation_scope,
                "total_on_balance_sheet": total_on_balance,
                "total_taxonomy_eligible": total_taxonomy_eligible,
                "total_taxonomy_aligned": total_taxonomy_aligned,
                "gar_numerator": total_taxonomy_aligned,
                "gar_denominator": total_on_balance,
                "exposures_by_class": exposures_by_class,
                "provenance_hash": self._calculate_hash(exposures_by_class),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Exposure aggregation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_epc_ratings(
        self,
        property_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get EPC (Energy Performance Certificate) ratings for real estate exposures.

        Args:
            property_ids: List of property identifiers

        Returns:
            EPC ratings with taxonomy alignment indicators
        """
        try:
            if self._service and hasattr(self._service, "get_epc_ratings"):
                return await self._service.get_epc_ratings(property_ids)

            ratings = []
            for prop_id in property_ids:
                # Fallback - no real EPC data
                ratings.append({
                    "property_id": prop_id,
                    "epc_rating": None,
                    "epc_source": self.config.epc_source,
                    "alignment": None,
                    "data_available": False
                })

            return {
                "total_properties": len(property_ids),
                "ratings_available": 0,
                "ratings": ratings,
                "epc_source": self.config.epc_source,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"EPC rating retrieval failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def calculate_coverage(
        self,
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate taxonomy data coverage across portfolio.

        Determines what percentage of the portfolio has counterparty
        taxonomy data available for GAR calculation.

        Args:
            portfolio: Portfolio data

        Returns:
            Coverage analysis with data quality indicators
        """
        try:
            if self._service and hasattr(self._service, "calculate_coverage"):
                return await self._service.calculate_coverage(portfolio)

            total_exposures = portfolio.get("total_exposures", 0)
            data_available = portfolio.get("data_available_count", 0)

            coverage_ratio = data_available / total_exposures if total_exposures > 0 else 0.0

            coverage_by_class = {}
            for exp_type in self.config.exposure_types:
                class_data = portfolio.get(exp_type, {})
                class_total = class_data.get("count", 0)
                class_covered = class_data.get("data_available", 0)
                coverage_by_class[exp_type] = {
                    "total": class_total,
                    "covered": class_covered,
                    "coverage_ratio": class_covered / class_total if class_total > 0 else 0.0
                }

            return {
                "overall_coverage": round(coverage_ratio * 100, 1),
                "total_exposures": total_exposures,
                "data_available": data_available,
                "coverage_by_class": coverage_by_class,
                "de_minimis_applied": self.config.de_minimis_threshold > 0,
                "provenance_hash": self._calculate_hash(coverage_by_class),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Coverage calculation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_epc_alignment(self, epc_rating: str) -> Dict[str, Any]:
        """Get taxonomy alignment for an EPC rating."""
        return self.EPC_ALIGNMENT_MAP.get(
            epc_rating.upper(),
            {"aligned": False, "top_15_percent": False, "nzeb_compliant": False}
        )

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
