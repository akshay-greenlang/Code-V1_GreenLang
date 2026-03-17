"""
CSRD Cross-Framework Bridge - PACK-008 EU Taxonomy Alignment

This module maps EU Taxonomy KPIs and alignment data to other EU sustainability
disclosure frameworks: CSRD/ESRS E1, SFDR Article 8/9 PAI indicators, and
TCFD metrics. It enables consolidated reporting across regulatory frameworks.

Cross-framework mappings:
- Taxonomy -> ESRS E1 (Climate Change disclosures)
- Taxonomy -> SFDR (Sustainable Finance Disclosure Regulation)
- Taxonomy -> TCFD (Task Force on Climate-related Financial Disclosures)
- Consolidated multi-framework output

Example:
    >>> config = CrossFrameworkConfig(
    ...     enable_esrs=True,
    ...     enable_sfdr=True,
    ...     enable_tcfd=True
    ... )
    >>> bridge = CSRDCrossFrameworkBridge(config)
    >>> esrs_data = await bridge.map_to_esrs(taxonomy_data)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class CrossFrameworkConfig(BaseModel):
    """Configuration for cross-framework bridge."""

    enable_esrs: bool = Field(
        default=True,
        description="Enable CSRD ESRS E1 Climate mapping"
    )
    enable_sfdr: bool = Field(
        default=True,
        description="Enable SFDR PAI indicator mapping"
    )
    enable_tcfd: bool = Field(
        default=True,
        description="Enable TCFD metric mapping"
    )
    enable_cdp: bool = Field(
        default=False,
        description="Enable CDP questionnaire mapping"
    )
    esrs_version: str = Field(
        default="2023",
        description="ESRS standard version"
    )
    sfdr_level: int = Field(
        default=1,
        ge=1,
        le=2,
        description="SFDR disclosure level (1=entity, 2=product)"
    )


# EU Taxonomy KPI -> ESRS E1 disclosure mapping
TAXONOMY_TO_ESRS_MAP: Dict[str, Dict[str, Any]] = {
    "turnover_ratio": {
        "esrs_ref": "E1-6",
        "disclosure": "Anticipated financial effects from material physical and transition risks",
        "paragraph": "E1-6.69",
        "datapoint": "Proportion of taxonomy-aligned turnover",
        "category": "transition_risk"
    },
    "capex_ratio": {
        "esrs_ref": "E1-6",
        "disclosure": "Anticipated financial effects from material physical and transition risks",
        "paragraph": "E1-6.69",
        "datapoint": "Proportion of taxonomy-aligned CapEx",
        "category": "transition_risk"
    },
    "opex_ratio": {
        "esrs_ref": "E1-6",
        "disclosure": "Anticipated financial effects from material physical and transition risks",
        "paragraph": "E1-6.69",
        "datapoint": "Proportion of taxonomy-aligned OpEx",
        "category": "transition_risk"
    },
    "ghg_emissions_scope1": {
        "esrs_ref": "E1-4",
        "disclosure": "GHG emissions",
        "paragraph": "E1-4.34",
        "datapoint": "Gross Scope 1 GHG emissions",
        "category": "emissions"
    },
    "ghg_emissions_scope2": {
        "esrs_ref": "E1-4",
        "disclosure": "GHG emissions",
        "paragraph": "E1-4.36",
        "datapoint": "Gross Scope 2 GHG emissions (location-based)",
        "category": "emissions"
    },
    "ghg_emissions_scope3": {
        "esrs_ref": "E1-4",
        "disclosure": "GHG emissions",
        "paragraph": "E1-4.37",
        "datapoint": "Gross Scope 3 GHG emissions",
        "category": "emissions"
    },
    "ghg_reduction_target": {
        "esrs_ref": "E1-3",
        "disclosure": "Actions and resources in relation to climate change policies",
        "paragraph": "E1-3.29",
        "datapoint": "GHG emission reduction targets",
        "category": "targets"
    },
    "energy_intensity": {
        "esrs_ref": "E1-5",
        "disclosure": "Energy consumption and mix",
        "paragraph": "E1-5.40",
        "datapoint": "Energy consumption and mix",
        "category": "energy"
    },
    "transition_plan": {
        "esrs_ref": "E1-1",
        "disclosure": "Transition plan for climate change mitigation",
        "paragraph": "E1-1.14",
        "datapoint": "Transition plan aligned with 1.5C target",
        "category": "strategy"
    },
    "climate_risk_assessment": {
        "esrs_ref": "E1-2",
        "disclosure": "Policies related to climate change mitigation and adaptation",
        "paragraph": "E1-2.22",
        "datapoint": "Climate-related policies and due diligence",
        "category": "governance"
    }
}

# EU Taxonomy -> SFDR PAI indicator mapping
TAXONOMY_TO_SFDR_MAP: Dict[str, Dict[str, Any]] = {
    "turnover_ratio": {
        "pai_indicator": "PAI 4",
        "name": "Exposure to companies active in the fossil fuel sector",
        "article": "Article 8(1)",
        "level": "entity",
        "table": "Table 1"
    },
    "ghg_emissions_scope1": {
        "pai_indicator": "PAI 1",
        "name": "GHG emissions (Scope 1)",
        "article": "Article 4",
        "level": "entity",
        "table": "Table 1"
    },
    "ghg_emissions_scope2": {
        "pai_indicator": "PAI 1",
        "name": "GHG emissions (Scope 2)",
        "article": "Article 4",
        "level": "entity",
        "table": "Table 1"
    },
    "ghg_emissions_scope3": {
        "pai_indicator": "PAI 1",
        "name": "GHG emissions (Scope 3)",
        "article": "Article 4",
        "level": "entity",
        "table": "Table 1"
    },
    "carbon_footprint": {
        "pai_indicator": "PAI 2",
        "name": "Carbon footprint",
        "article": "Article 4",
        "level": "product",
        "table": "Table 1"
    },
    "ghg_intensity": {
        "pai_indicator": "PAI 3",
        "name": "GHG intensity of investee companies",
        "article": "Article 4",
        "level": "product",
        "table": "Table 1"
    },
    "energy_from_nonrenewable": {
        "pai_indicator": "PAI 5",
        "name": "Share of non-renewable energy consumption and production",
        "article": "Article 4",
        "level": "entity",
        "table": "Table 1"
    },
    "energy_intensity_per_sector": {
        "pai_indicator": "PAI 6",
        "name": "Energy consumption intensity per high impact climate sector",
        "article": "Article 4",
        "level": "entity",
        "table": "Table 1"
    }
}

# EU Taxonomy -> TCFD metric mapping
TAXONOMY_TO_TCFD_MAP: Dict[str, Dict[str, Any]] = {
    "turnover_ratio": {
        "tcfd_pillar": "Metrics and Targets",
        "recommendation": "Disclose metrics used to assess climate-related risks and opportunities",
        "metric_type": "transition_risk",
        "disclosure_element": "Proportion of revenue from taxonomy-aligned activities"
    },
    "capex_ratio": {
        "tcfd_pillar": "Metrics and Targets",
        "recommendation": "Disclose metrics used to assess climate-related risks and opportunities",
        "metric_type": "transition_opportunity",
        "disclosure_element": "Capital deployment towards climate solutions"
    },
    "ghg_emissions_scope1": {
        "tcfd_pillar": "Metrics and Targets",
        "recommendation": "Disclose Scope 1 GHG emissions",
        "metric_type": "emissions",
        "disclosure_element": "Absolute Scope 1 emissions"
    },
    "ghg_emissions_scope2": {
        "tcfd_pillar": "Metrics and Targets",
        "recommendation": "Disclose Scope 2 GHG emissions",
        "metric_type": "emissions",
        "disclosure_element": "Absolute Scope 2 emissions"
    },
    "climate_risk_assessment": {
        "tcfd_pillar": "Risk Management",
        "recommendation": "Describe processes for identifying climate-related risks",
        "metric_type": "risk_management",
        "disclosure_element": "Climate risk identification and assessment process"
    },
    "transition_plan": {
        "tcfd_pillar": "Strategy",
        "recommendation": "Describe impact of climate-related risks and opportunities on strategy",
        "metric_type": "strategy",
        "disclosure_element": "Transition plan and scenario analysis"
    },
    "physical_risk_exposure": {
        "tcfd_pillar": "Risk Management",
        "recommendation": "Describe processes for managing climate-related risks",
        "metric_type": "physical_risk",
        "disclosure_element": "Physical risk exposure and adaptation measures"
    }
}


class CSRDCrossFrameworkBridge:
    """
    Cross-framework bridge mapping EU Taxonomy data to ESRS E1, SFDR, and TCFD.

    Enables consolidated disclosure across EU sustainability regulations by mapping
    taxonomy KPIs, alignment results, and emissions data to corresponding disclosure
    requirements in CSRD, SFDR, and TCFD frameworks.

    Example:
        >>> config = CrossFrameworkConfig(enable_esrs=True, enable_sfdr=True)
        >>> bridge = CSRDCrossFrameworkBridge(config)
        >>> bridge.inject_service(csrd_service)
        >>> esrs = await bridge.map_to_esrs(taxonomy_data)
    """

    def __init__(self, config: CrossFrameworkConfig):
        """Initialize cross-framework bridge."""
        self.config = config
        self._service: Any = None
        logger.info("CSRDCrossFrameworkBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real cross-framework mapping service."""
        self._service = service
        logger.info("Injected cross-framework service")

    async def map_to_esrs(
        self,
        taxonomy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map taxonomy KPIs to CSRD ESRS E1 disclosure requirements.

        Args:
            taxonomy_data: Taxonomy alignment and KPI data

        Returns:
            ESRS E1 disclosure mappings with populated values
        """
        try:
            if not self.config.enable_esrs:
                return {
                    "status": "disabled",
                    "message": "ESRS E1 mapping not enabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if self._service and hasattr(self._service, "map_to_esrs"):
                return await self._service.map_to_esrs(taxonomy_data)

            # Build ESRS E1 mappings from taxonomy data
            esrs_disclosures = {}
            for kpi_name, esrs_ref in TAXONOMY_TO_ESRS_MAP.items():
                value = taxonomy_data.get(kpi_name)
                esrs_disclosures[kpi_name] = {
                    "esrs_reference": esrs_ref["esrs_ref"],
                    "disclosure": esrs_ref["disclosure"],
                    "paragraph": esrs_ref["paragraph"],
                    "datapoint": esrs_ref["datapoint"],
                    "category": esrs_ref["category"],
                    "value": value,
                    "data_available": value is not None
                }

            return {
                "framework": "CSRD_ESRS_E1",
                "version": self.config.esrs_version,
                "disclosures": esrs_disclosures,
                "total_mapped": len(esrs_disclosures),
                "data_populated": sum(
                    1 for d in esrs_disclosures.values() if d["data_available"]
                ),
                "provenance_hash": self._calculate_hash(esrs_disclosures),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"ESRS mapping failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def map_to_sfdr(
        self,
        taxonomy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map taxonomy KPIs to SFDR PAI indicators.

        Args:
            taxonomy_data: Taxonomy alignment and emissions data

        Returns:
            SFDR PAI indicator mappings with populated values
        """
        try:
            if not self.config.enable_sfdr:
                return {
                    "status": "disabled",
                    "message": "SFDR mapping not enabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if self._service and hasattr(self._service, "map_to_sfdr"):
                return await self._service.map_to_sfdr(taxonomy_data)

            # Build SFDR PAI mappings
            sfdr_indicators = {}
            for kpi_name, pai_ref in TAXONOMY_TO_SFDR_MAP.items():
                value = taxonomy_data.get(kpi_name)
                sfdr_indicators[kpi_name] = {
                    "pai_indicator": pai_ref["pai_indicator"],
                    "indicator_name": pai_ref["name"],
                    "article": pai_ref["article"],
                    "level": pai_ref["level"],
                    "table": pai_ref["table"],
                    "value": value,
                    "data_available": value is not None
                }

            return {
                "framework": "SFDR",
                "disclosure_level": self.config.sfdr_level,
                "indicators": sfdr_indicators,
                "total_mapped": len(sfdr_indicators),
                "data_populated": sum(
                    1 for i in sfdr_indicators.values() if i["data_available"]
                ),
                "provenance_hash": self._calculate_hash(sfdr_indicators),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"SFDR mapping failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def map_to_tcfd(
        self,
        taxonomy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map taxonomy KPIs to TCFD metrics and recommendations.

        Args:
            taxonomy_data: Taxonomy alignment and risk data

        Returns:
            TCFD metric mappings with populated values
        """
        try:
            if not self.config.enable_tcfd:
                return {
                    "status": "disabled",
                    "message": "TCFD mapping not enabled",
                    "timestamp": datetime.utcnow().isoformat()
                }

            if self._service and hasattr(self._service, "map_to_tcfd"):
                return await self._service.map_to_tcfd(taxonomy_data)

            # Build TCFD mappings
            tcfd_metrics = {}
            for kpi_name, tcfd_ref in TAXONOMY_TO_TCFD_MAP.items():
                value = taxonomy_data.get(kpi_name)
                tcfd_metrics[kpi_name] = {
                    "tcfd_pillar": tcfd_ref["tcfd_pillar"],
                    "recommendation": tcfd_ref["recommendation"],
                    "metric_type": tcfd_ref["metric_type"],
                    "disclosure_element": tcfd_ref["disclosure_element"],
                    "value": value,
                    "data_available": value is not None
                }

            return {
                "framework": "TCFD",
                "metrics": tcfd_metrics,
                "total_mapped": len(tcfd_metrics),
                "data_populated": sum(
                    1 for m in tcfd_metrics.values() if m["data_available"]
                ),
                "provenance_hash": self._calculate_hash(tcfd_metrics),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"TCFD mapping failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def generate_consolidated(
        self,
        taxonomy_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate consolidated cross-framework disclosure package.

        Maps taxonomy data to all enabled frameworks and produces a unified
        disclosure view suitable for multi-framework reporting.

        Args:
            taxonomy_data: Complete taxonomy alignment data

        Returns:
            Consolidated disclosure package across all frameworks
        """
        try:
            if self._service and hasattr(self._service, "generate_consolidated"):
                return await self._service.generate_consolidated(taxonomy_data)

            consolidated = {
                "source_framework": "EU_Taxonomy",
                "reporting_period": taxonomy_data.get("reporting_period", ""),
                "frameworks": {}
            }

            # Map to each enabled framework
            if self.config.enable_esrs:
                esrs_result = await self.map_to_esrs(taxonomy_data)
                consolidated["frameworks"]["ESRS_E1"] = esrs_result

            if self.config.enable_sfdr:
                sfdr_result = await self.map_to_sfdr(taxonomy_data)
                consolidated["frameworks"]["SFDR"] = sfdr_result

            if self.config.enable_tcfd:
                tcfd_result = await self.map_to_tcfd(taxonomy_data)
                consolidated["frameworks"]["TCFD"] = tcfd_result

            # Cross-framework consistency check
            consolidated["consistency_check"] = self._check_consistency(
                consolidated["frameworks"]
            )

            consolidated["total_frameworks"] = len(consolidated["frameworks"])
            consolidated["provenance_hash"] = self._calculate_hash(consolidated)
            consolidated["timestamp"] = datetime.utcnow().isoformat()

            return consolidated

        except Exception as e:
            logger.error(f"Consolidated generation failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _check_consistency(
        self,
        frameworks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check data consistency across frameworks."""
        consistency = {
            "consistent": True,
            "warnings": [],
            "checked_datapoints": 0
        }

        # Verify that shared KPIs have consistent values across frameworks
        shared_kpis = ["turnover_ratio", "ghg_emissions_scope1", "ghg_emissions_scope2"]
        for kpi in shared_kpis:
            values = set()
            for framework_data in frameworks.values():
                disclosures = (
                    framework_data.get("disclosures", {}) or
                    framework_data.get("indicators", {}) or
                    framework_data.get("metrics", {})
                )
                if kpi in disclosures and disclosures[kpi].get("data_available"):
                    values.add(str(disclosures[kpi].get("value")))

            consistency["checked_datapoints"] += 1
            if len(values) > 1:
                consistency["consistent"] = False
                consistency["warnings"].append(
                    f"Inconsistent values for {kpi}: {values}"
                )

        return consistency

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
