"""
Activity Registry Bridge - PACK-008 EU Taxonomy Alignment

This module manages the ~240 EU Taxonomy economic activity catalog with NACE
code mappings and Delegated Act version tracking. It provides lookup, search,
and sector-based filtering for taxonomy activity identification.

Activity catalog coverage:
- NACE Rev. 2 code mappings to taxonomy activities
- Delegated Act version tracking (Climate DA, Environmental DA, Complementary DA)
- Sector-based grouping (Energy, Transport, Buildings, Manufacturing, etc.)
- Activity search by keyword, NACE code, or sector
- DA version change tracking

Example:
    >>> config = ActivityRegistryConfig(da_version="2023")
    >>> bridge = ActivityRegistryBridge(config)
    >>> activity = await bridge.lookup_activity("35.11")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class ActivityRegistryConfig(BaseModel):
    """Configuration for Activity Registry Bridge."""

    da_version: str = Field(
        default="2023",
        description="Active Delegated Act version"
    )
    include_complementary_da: bool = Field(
        default=False,
        description="Include Complementary Climate DA (nuclear/gas)"
    )
    include_environmental_da: bool = Field(
        default=True,
        description="Include Environmental DA (WTR, CE, PPC, BIO)"
    )
    language: str = Field(
        default="en",
        description="Activity description language"
    )


# Representative activity catalog (~30+ entries representing key sectors)
ACTIVITY_CATALOG: Dict[str, Dict[str, Any]] = {
    # Forestry (1.x)
    "1.1": {"name": "Afforestation", "nace": ["A02.10"], "sector": "forestry", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 1.1"},
    "1.2": {"name": "Rehabilitation and restoration of forests", "nace": ["A02.10"], "sector": "forestry", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 1.2"},
    "1.3": {"name": "Forest management", "nace": ["A02.10"], "sector": "forestry", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 1.3"},
    "1.4": {"name": "Conservation forestry", "nace": ["A02.10"], "sector": "forestry", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 1.4"},
    # Environmental protection and restoration (2.x)
    "2.1": {"name": "Restoration of wetlands", "nace": ["F42.99"], "sector": "environment", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 2.1"},
    # Manufacturing (3.x)
    "3.1": {"name": "Manufacture of renewable energy technologies", "nace": ["C25.11", "C27.11", "C28.11"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.1"},
    "3.4": {"name": "Manufacture of batteries", "nace": ["C27.20"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.4"},
    "3.5": {"name": "Manufacture of energy efficiency equipment for buildings", "nace": ["C23.11", "C23.20", "C23.31", "C25.21", "C27.51", "C28.21"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.5"},
    "3.6": {"name": "Manufacture of other low carbon technologies", "nace": ["C22.11", "C22.19", "C25.99", "C27.90"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.6"},
    "3.7": {"name": "Manufacture of cement", "nace": ["C23.51"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.7"},
    "3.9": {"name": "Manufacture of iron and steel", "nace": ["C24.10", "C24.20", "C24.31", "C24.32", "C24.33", "C24.34", "C24.51", "C24.52"], "sector": "manufacturing", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 3.9"},
    # Energy (4.x)
    "4.1": {"name": "Electricity generation using solar photovoltaic technology", "nace": ["D35.11"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.1"},
    "4.3": {"name": "Electricity generation from wind power", "nace": ["D35.11"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.3"},
    "4.5": {"name": "Electricity generation from hydropower", "nace": ["D35.11"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.5"},
    "4.8": {"name": "Electricity generation from bioenergy", "nace": ["D35.11"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.8"},
    "4.9": {"name": "Transmission and distribution of electricity", "nace": ["D35.12", "D35.13"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.9"},
    "4.15": {"name": "District heating/cooling distribution", "nace": ["D35.30"], "sector": "energy", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 4.15"},
    # Water supply and waste (5.x)
    "5.1": {"name": "Construction, extension and operation of water collection, treatment and supply systems", "nace": ["E36.00", "F42.99"], "sector": "water", "objectives": ["CCM", "CCA", "WTR"], "da": "climate", "da_ref": "Annex I, Section 5.1"},
    "5.3": {"name": "Construction, extension and operation of waste water collection and treatment", "nace": ["E37.00", "F42.99"], "sector": "water", "objectives": ["CCM", "CCA", "WTR"], "da": "climate", "da_ref": "Annex I, Section 5.3"},
    "5.9": {"name": "Material recovery from non-hazardous waste", "nace": ["E38.32"], "sector": "waste", "objectives": ["CCM", "CE"], "da": "climate", "da_ref": "Annex I, Section 5.9"},
    # Transport (6.x)
    "6.1": {"name": "Passenger interurban rail transport", "nace": ["H49.10"], "sector": "transport", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 6.1"},
    "6.3": {"name": "Urban and suburban transport, road passenger transport", "nace": ["H49.31", "H49.39", "N77.39"], "sector": "transport", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 6.3"},
    "6.5": {"name": "Transport by motorbikes, passenger cars and light commercial vehicles", "nace": ["H49.32", "H49.39", "N77.11"], "sector": "transport", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 6.5"},
    "6.6": {"name": "Freight transport services by road", "nace": ["H49.41", "N77.12"], "sector": "transport", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 6.6"},
    "6.10": {"name": "Sea and coastal freight water transport, vessels for port operations and auxiliary activities", "nace": ["H50.20", "N77.34"], "sector": "transport", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 6.10"},
    # Buildings (7.x)
    "7.1": {"name": "Construction of new buildings", "nace": ["F41.10", "F41.20"], "sector": "buildings", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 7.1"},
    "7.2": {"name": "Renovation of existing buildings", "nace": ["F41.10", "F41.20", "F43.21", "F43.22", "F43.29", "F43.31", "F43.32", "F43.33", "F43.39", "F43.91"], "sector": "buildings", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 7.2"},
    "7.3": {"name": "Installation, maintenance and repair of energy efficiency equipment", "nace": ["F43.21", "F43.22", "F43.29"], "sector": "buildings", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 7.3"},
    "7.6": {"name": "Installation, maintenance and repair of renewable energy technologies", "nace": ["F43.21", "F43.22", "F43.29", "D35.30"], "sector": "buildings", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 7.6"},
    "7.7": {"name": "Acquisition and ownership of buildings", "nace": ["L68.10", "L68.20"], "sector": "buildings", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 7.7"},
    # ICT (8.x)
    "8.1": {"name": "Data processing, hosting and related activities", "nace": ["J63.11"], "sector": "ict", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 8.1"},
    "8.2": {"name": "Data-driven solutions for GHG emissions reductions", "nace": ["J61", "J62", "J63.11"], "sector": "ict", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 8.2"},
    # Professional and scientific (9.x)
    "9.1": {"name": "Close to market research, development and innovation", "nace": ["M71.12", "M72.19"], "sector": "professional", "objectives": ["CCM", "CCA"], "da": "climate", "da_ref": "Annex I, Section 9.1"},
    "9.3": {"name": "Professional services related to energy performance of buildings", "nace": ["M71.12"], "sector": "professional", "objectives": ["CCM"], "da": "climate", "da_ref": "Annex I, Section 9.3"},
}


# NACE Rev. 2 to taxonomy activity reverse lookup
_NACE_TO_ACTIVITY: Dict[str, List[str]] = {}
for act_code, act_data in ACTIVITY_CATALOG.items():
    for nace in act_data.get("nace", []):
        if nace not in _NACE_TO_ACTIVITY:
            _NACE_TO_ACTIVITY[nace] = []
        _NACE_TO_ACTIVITY[nace].append(act_code)


class ActivityRegistryBridge:
    """
    Bridge to EU Taxonomy economic activity registry (~240 activities).

    Manages NACE code mappings, DA version tracking, and activity lookup
    for taxonomy eligibility screening and alignment assessment.

    Example:
        >>> config = ActivityRegistryConfig()
        >>> bridge = ActivityRegistryBridge(config)
        >>> activity = await bridge.lookup_activity("35.11")
        >>> activities = await bridge.get_activities_by_sector("energy")
    """

    def __init__(self, config: ActivityRegistryConfig):
        """Initialize activity registry bridge."""
        self.config = config
        self._service: Any = None
        logger.info(
            f"ActivityRegistryBridge initialized "
            f"(DA version={config.da_version}, "
            f"catalog size={len(ACTIVITY_CATALOG)})"
        )

    def inject_service(self, service: Any) -> None:
        """Inject real activity registry service."""
        self._service = service
        logger.info("Injected activity registry service")

    async def lookup_activity(
        self,
        nace_code: str
    ) -> Dict[str, Any]:
        """
        Look up taxonomy activities for a given NACE code.

        Args:
            nace_code: NACE Rev. 2 code (e.g., "D35.11", "F41.20")

        Returns:
            Matching taxonomy activities with details
        """
        try:
            if self._service and hasattr(self._service, "lookup_activity"):
                return await self._service.lookup_activity(nace_code)

            # Normalize NACE code format
            normalized = nace_code.strip().upper()

            # Direct lookup
            matching_codes = _NACE_TO_ACTIVITY.get(normalized, [])

            if not matching_codes:
                # Try with letter prefix
                for prefix in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]:
                    prefixed = f"{prefix}{normalized}"
                    matching_codes = _NACE_TO_ACTIVITY.get(prefixed, [])
                    if matching_codes:
                        break

            activities = []
            for code in matching_codes:
                act_data = ACTIVITY_CATALOG.get(code, {})
                activities.append({
                    "activity_code": code,
                    "name": act_data.get("name", ""),
                    "sector": act_data.get("sector", ""),
                    "objectives": act_data.get("objectives", []),
                    "da": act_data.get("da", ""),
                    "da_ref": act_data.get("da_ref", "")
                })

            return {
                "nace_code": nace_code,
                "matching_activities": activities,
                "total_matches": len(activities),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Activity lookup failed for {nace_code}: {str(e)}")
            return {"nace_code": nace_code, "error": str(e)}

    async def get_activities_by_sector(
        self,
        sector: str
    ) -> Dict[str, Any]:
        """
        Get all taxonomy activities in a given sector.

        Args:
            sector: Sector name (energy, transport, buildings, manufacturing, etc.)

        Returns:
            List of activities in the specified sector
        """
        try:
            if self._service and hasattr(self._service, "get_activities_by_sector"):
                return await self._service.get_activities_by_sector(sector)

            sector_lower = sector.lower()
            activities = []

            for code, data in ACTIVITY_CATALOG.items():
                if data.get("sector", "").lower() == sector_lower:
                    activities.append({
                        "activity_code": code,
                        "name": data.get("name", ""),
                        "nace_codes": data.get("nace", []),
                        "objectives": data.get("objectives", []),
                        "da": data.get("da", ""),
                        "da_ref": data.get("da_ref", "")
                    })

            return {
                "sector": sector,
                "activities": activities,
                "total_activities": len(activities),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Sector lookup failed for {sector}: {str(e)}")
            return {"sector": sector, "error": str(e)}

    async def get_da_version(
        self,
        activity_code: str
    ) -> Dict[str, Any]:
        """
        Get Delegated Act version information for a taxonomy activity.

        Args:
            activity_code: EU Taxonomy activity code (e.g., "4.1")

        Returns:
            DA version details for the activity
        """
        try:
            if self._service and hasattr(self._service, "get_da_version"):
                return await self._service.get_da_version(activity_code)

            act_data = ACTIVITY_CATALOG.get(activity_code)
            if not act_data:
                return {
                    "activity_code": activity_code,
                    "found": False,
                    "message": f"Activity {activity_code} not found in catalog"
                }

            da_type = act_data.get("da", "climate")

            da_versions = {
                "climate": {
                    "regulation": "(EU) 2021/2139",
                    "type": "Climate Delegated Act",
                    "effective_date": "2022-01-01",
                    "last_amendment": "2023-06-27",
                    "objectives": ["CCM", "CCA"]
                },
                "environmental": {
                    "regulation": "(EU) 2023/2486",
                    "type": "Environmental Delegated Act",
                    "effective_date": "2024-01-01",
                    "last_amendment": "2023-11-21",
                    "objectives": ["WTR", "CE", "PPC", "BIO"]
                },
                "complementary": {
                    "regulation": "(EU) 2022/1214",
                    "type": "Complementary Climate Delegated Act",
                    "effective_date": "2023-01-01",
                    "last_amendment": "2022-07-15",
                    "objectives": ["CCM"]
                },
                "disclosures": {
                    "regulation": "(EU) 2021/2178",
                    "type": "Disclosures Delegated Act",
                    "effective_date": "2022-01-01",
                    "last_amendment": "2023-06-27",
                    "objectives": []
                }
            }

            da_info = da_versions.get(da_type, da_versions["climate"])

            return {
                "activity_code": activity_code,
                "activity_name": act_data.get("name", ""),
                "da_type": da_type,
                "da_info": da_info,
                "da_ref": act_data.get("da_ref", ""),
                "current_version": self.config.da_version,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"DA version lookup failed: {str(e)}")
            return {"activity_code": activity_code, "error": str(e)}

    async def search_activities(
        self,
        query: str,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Search activities by keyword in name or NACE codes.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            Matching activities sorted by relevance
        """
        try:
            if self._service and hasattr(self._service, "search_activities"):
                return await self._service.search_activities(query, max_results)

            query_lower = query.lower()
            results = []

            for code, data in ACTIVITY_CATALOG.items():
                name = data.get("name", "").lower()
                nace_codes = [n.lower() for n in data.get("nace", [])]
                sector = data.get("sector", "").lower()

                # Score based on match quality
                score = 0.0
                if query_lower in name:
                    score = 1.0
                elif query_lower in sector:
                    score = 0.7
                elif any(query_lower in n for n in nace_codes):
                    score = 0.9
                elif any(word in name for word in query_lower.split()):
                    score = 0.5

                if score > 0:
                    results.append({
                        "activity_code": code,
                        "name": data.get("name", ""),
                        "sector": data.get("sector", ""),
                        "nace_codes": data.get("nace", []),
                        "objectives": data.get("objectives", []),
                        "relevance_score": score
                    })

            # Sort by relevance
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = results[:max_results]

            return {
                "query": query,
                "results": results,
                "total_matches": len(results),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Activity search failed: {str(e)}")
            return {"query": query, "error": str(e)}

    def get_catalog_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the activity catalog."""
        sectors = {}
        objectives_coverage = {}

        for data in ACTIVITY_CATALOG.values():
            sector = data.get("sector", "unknown")
            sectors[sector] = sectors.get(sector, 0) + 1

            for obj in data.get("objectives", []):
                objectives_coverage[obj] = objectives_coverage.get(obj, 0) + 1

        return {
            "total_activities": len(ACTIVITY_CATALOG),
            "sectors": sectors,
            "objectives_coverage": objectives_coverage,
            "da_version": self.config.da_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
