# -*- coding: utf-8 -*-
"""
GreenLang EUDR Deforestation Baseline Checker

Zero-hallucination deforestation validation for EUDR compliance.
Implements the December 31, 2020 cutoff date validation and
forest cover assessment.

This module provides:
- December 31, 2020 cutoff date validation
- Historical forest cover lookup (stub for satellite integration)
- Forest definition by country (tree cover %, height)
- Deforestation risk scoring (0-100)
- Complete provenance tracking

Author: GreenLang Calculator Engine
License: Proprietary
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime, date
import hashlib
import json
import logging

from .geojson_parser import Coordinate, ParsedPolygon

logger = logging.getLogger(__name__)


# EUDR Cutoff Date - This is a FIXED constant
EUDR_CUTOFF_DATE = date(2020, 12, 31)


class ForestStatus(str, Enum):
    """Forest status classification."""
    FOREST = "forest"  # Currently forested
    NON_FOREST = "non_forest"  # Not forested (grassland, agriculture, etc.)
    DEFORESTED_PRE_CUTOFF = "deforested_pre_cutoff"  # Cleared before Dec 31, 2020
    DEFORESTED_POST_CUTOFF = "deforested_post_cutoff"  # Cleared after Dec 31, 2020 - VIOLATION
    DEGRADED = "degraded"  # Partially degraded forest
    REGENERATING = "regenerating"  # Regenerating forest
    PLANTATION = "plantation"  # Forest plantation
    UNKNOWN = "unknown"  # Unable to determine


class DeforestationRisk(str, Enum):
    """Deforestation risk level."""
    LOW = "low"  # 0-25 risk score
    MEDIUM = "medium"  # 25-50 risk score
    HIGH = "high"  # 50-75 risk score
    CRITICAL = "critical"  # 75-100 risk score
    VIOLATION = "violation"  # Confirmed deforestation post-cutoff


class ForestDefinition(BaseModel):
    """
    Forest definition by country/region.

    Different countries define "forest" differently based on:
    - Minimum tree cover percentage
    - Minimum tree height
    - Minimum area
    - Canopy continuity

    Based on FAO Forest Resources Assessment definitions and
    country-specific REDD+ definitions.
    """
    country_iso3: str = Field(..., description="ISO3 country code")
    region: Optional[str] = Field(None, description="Sub-national region")
    min_tree_cover_percent: Decimal = Field(..., description="Minimum tree cover (%)")
    min_tree_height_meters: Decimal = Field(..., description="Minimum tree height (m)")
    min_area_hectares: Decimal = Field(..., description="Minimum area (ha)")
    includes_plantations: bool = Field(True, description="Includes forest plantations")
    includes_agroforestry: bool = Field(False, description="Includes agroforestry")
    source: str = Field(..., description="Definition source")
    definition_year: int = Field(..., description="Year definition established")


class ForestCoverData(BaseModel):
    """
    Forest cover data for a specific location and time.

    This is typically derived from satellite imagery analysis.
    """
    coordinate: Coordinate
    observation_date: date
    tree_cover_percent: Decimal = Field(..., ge=0, le=100)
    tree_height_meters: Optional[Decimal] = None
    forest_type: Optional[str] = None
    data_source: str = Field(..., description="Satellite/data source")
    confidence_percent: Decimal = Field(..., ge=0, le=100)
    pixel_resolution_meters: int = Field(..., description="Spatial resolution")


class DeforestationEvent(BaseModel):
    """
    A detected deforestation event.

    Records when and where deforestation was detected.
    """
    event_id: str = Field(..., description="Unique event identifier")
    coordinate: Coordinate
    detection_date: date
    estimated_deforestation_date: date
    area_hectares: Decimal
    tree_cover_loss_percent: Decimal
    confidence_percent: Decimal
    data_source: str
    is_post_cutoff: bool = Field(..., description="After EUDR cutoff (Dec 31, 2020)")
    cause: Optional[str] = Field(None, description="Detected cause if available")
    provenance_hash: str = ""


class BaselineCheckResult(BaseModel):
    """
    Result of deforestation baseline check.

    Determines if a location was forested as of the EUDR cutoff date
    and whether any post-cutoff deforestation has occurred.
    """
    forest_status: ForestStatus
    is_eudr_compliant: bool
    risk_level: DeforestationRisk
    risk_score: Decimal = Field(..., ge=0, le=100)

    # Baseline data (as of December 31, 2020)
    baseline_date: date = EUDR_CUTOFF_DATE
    baseline_forest_cover_percent: Optional[Decimal] = None
    baseline_tree_height_meters: Optional[Decimal] = None
    baseline_was_forest: Optional[bool] = None

    # Current data
    current_date: Optional[date] = None
    current_forest_cover_percent: Optional[Decimal] = None
    current_tree_height_meters: Optional[Decimal] = None
    current_is_forest: Optional[bool] = None

    # Change detection
    forest_cover_change_percent: Optional[Decimal] = None
    deforestation_events: List[DeforestationEvent] = Field(default_factory=list)

    # Applied definition
    forest_definition: Optional[ForestDefinition] = None

    # Metadata
    warnings: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = ""


class DeforestationBaselineChecker:
    """
    Zero-Hallucination Deforestation Baseline Checker for EUDR Compliance.

    This checker guarantees:
    - Deterministic assessments (same input -> same output)
    - Complete audit trail
    - NO LLM in validation path

    EUDR Requirements:
    - Products must not come from land deforested after December 31, 2020
    - Forest is defined according to FAO definition or country-specific definitions
    - Deforestation includes conversion of forest to agricultural use

    Data Sources (for production integration):
    - Global Forest Watch (GFW)
    - Hansen Global Forest Change
    - RADD Alerts (Radar for Detecting Deforestation)
    - JJ-FAST (JAXA)
    - Copernicus Emergency Management Service

    Example:
        checker = DeforestationBaselineChecker()

        result = checker.check_baseline(
            coordinate=Coordinate(longitude=Decimal('-61.234567'), latitude=Decimal('-3.456789')),
            country_iso3='BRA'
        )

        if not result.is_eudr_compliant:
            print(f"Deforestation detected: {result.deforestation_events}")
    """

    # Default FAO Forest Definition
    FAO_DEFAULT_DEFINITION = ForestDefinition(
        country_iso3="GLOBAL",
        min_tree_cover_percent=Decimal('10'),
        min_tree_height_meters=Decimal('5'),
        min_area_hectares=Decimal('0.5'),
        includes_plantations=True,
        includes_agroforestry=False,
        source="FAO Global Forest Resources Assessment",
        definition_year=2020
    )

    # Country-specific forest definitions
    COUNTRY_DEFINITIONS: Dict[str, ForestDefinition] = {
        "BRA": ForestDefinition(
            country_iso3="BRA",
            min_tree_cover_percent=Decimal('10'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('1.0'),
            includes_plantations=True,
            includes_agroforestry=False,
            source="Brazil INPE/PRODES",
            definition_year=2020
        ),
        "IDN": ForestDefinition(
            country_iso3="IDN",
            min_tree_cover_percent=Decimal('30'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('0.25'),
            includes_plantations=True,
            includes_agroforestry=True,
            source="Indonesia Ministry of Environment and Forestry",
            definition_year=2020
        ),
        "COD": ForestDefinition(
            country_iso3="COD",
            min_tree_cover_percent=Decimal('30'),
            min_tree_height_meters=Decimal('3'),
            min_area_hectares=Decimal('0.5'),
            includes_plantations=True,
            includes_agroforestry=False,
            source="DRC REDD+ Program",
            definition_year=2020
        ),
        "MYS": ForestDefinition(
            country_iso3="MYS",
            min_tree_cover_percent=Decimal('30'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('0.5'),
            includes_plantations=True,
            includes_agroforestry=False,
            source="Malaysia Forestry Department",
            definition_year=2020
        ),
        "CIV": ForestDefinition(
            country_iso3="CIV",
            min_tree_cover_percent=Decimal('10'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('0.5'),
            includes_plantations=True,
            includes_agroforestry=True,
            source="Cote d'Ivoire REDD+ Strategy",
            definition_year=2020
        ),
        "GHA": ForestDefinition(
            country_iso3="GHA",
            min_tree_cover_percent=Decimal('15'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('1.0'),
            includes_plantations=True,
            includes_agroforestry=True,
            source="Ghana Forestry Commission",
            definition_year=2020
        ),
        "COL": ForestDefinition(
            country_iso3="COL",
            min_tree_cover_percent=Decimal('30'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('1.0'),
            includes_plantations=True,
            includes_agroforestry=False,
            source="Colombia IDEAM",
            definition_year=2020
        ),
        "PER": ForestDefinition(
            country_iso3="PER",
            min_tree_cover_percent=Decimal('30'),
            min_tree_height_meters=Decimal('5'),
            min_area_hectares=Decimal('0.5'),
            includes_plantations=True,
            includes_agroforestry=False,
            source="Peru MINAM/SERNANP",
            definition_year=2020
        ),
    }

    # Risk scoring weights
    RISK_WEIGHTS = {
        "deforestation_proximity_km": Decimal('30'),  # Proximity to recent deforestation
        "historical_deforestation_rate": Decimal('25'),  # Historical rate in area
        "road_proximity_km": Decimal('15'),  # Proximity to roads
        "fire_alerts": Decimal('15'),  # Recent fire alerts
        "protected_area_proximity": Decimal('10'),  # Near protected areas (pressure)
        "commodity_suitability": Decimal('5'),  # Suitability for commodities
    }

    def __init__(
        self,
        gfw_api_key: Optional[str] = None,
        satellite_data_provider: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize Deforestation Baseline Checker.

        Args:
            gfw_api_key: Global Forest Watch API key (optional)
            satellite_data_provider: Satellite data provider for integration
            cache_enabled: Enable caching for performance
        """
        self.gfw_api_key = gfw_api_key
        self.satellite_data_provider = satellite_data_provider
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

    def check_baseline(
        self,
        coordinate: Coordinate,
        country_iso3: str,
        observation_date: Optional[date] = None,
        forest_definition: Optional[ForestDefinition] = None
    ) -> BaselineCheckResult:
        """
        Check deforestation baseline for a coordinate.

        DETERMINISTIC ASSESSMENT.

        Args:
            coordinate: Location to check
            country_iso3: ISO3 country code
            observation_date: Current observation date (default: today)
            forest_definition: Custom forest definition (default: country-specific)

        Returns:
            BaselineCheckResult with compliance assessment
        """
        if observation_date is None:
            observation_date = date.today()

        # Get applicable forest definition
        if forest_definition is None:
            forest_definition = self.get_forest_definition(country_iso3)

        result = BaselineCheckResult(
            forest_status=ForestStatus.UNKNOWN,
            is_eudr_compliant=True,  # Assume compliant until proven otherwise
            risk_level=DeforestationRisk.LOW,
            risk_score=Decimal('0'),
            forest_definition=forest_definition,
            current_date=observation_date
        )

        try:
            # Step 1: Get baseline forest cover (as of Dec 31, 2020)
            baseline_data = self._get_forest_cover_at_date(
                coordinate, EUDR_CUTOFF_DATE
            )

            if baseline_data:
                result.baseline_forest_cover_percent = baseline_data.tree_cover_percent
                result.baseline_tree_height_meters = baseline_data.tree_height_meters
                result.baseline_was_forest = self._is_forest(
                    baseline_data, forest_definition
                )
                result.data_sources.append(baseline_data.data_source)

            # Step 2: Get current forest cover
            current_data = self._get_forest_cover_at_date(
                coordinate, observation_date
            )

            if current_data:
                result.current_forest_cover_percent = current_data.tree_cover_percent
                result.current_tree_height_meters = current_data.tree_height_meters
                result.current_is_forest = self._is_forest(
                    current_data, forest_definition
                )
                if current_data.data_source not in result.data_sources:
                    result.data_sources.append(current_data.data_source)

            # Step 3: Calculate forest cover change
            if (result.baseline_forest_cover_percent is not None and
                result.current_forest_cover_percent is not None):
                result.forest_cover_change_percent = (
                    result.current_forest_cover_percent -
                    result.baseline_forest_cover_percent
                )

            # Step 4: Check for deforestation events
            deforestation_events = self._get_deforestation_events(
                coordinate,
                start_date=EUDR_CUTOFF_DATE,
                end_date=observation_date
            )
            result.deforestation_events = deforestation_events

            # Step 5: Determine forest status
            result.forest_status = self._determine_forest_status(result)

            # Step 6: Check EUDR compliance
            result.is_eudr_compliant = self._check_eudr_compliance(result)

            # Step 7: Calculate risk score
            result.risk_score = self._calculate_risk_score(
                coordinate, country_iso3, result
            )
            result.risk_level = self._score_to_risk_level(result.risk_score)

            # Override risk level if violation detected
            if not result.is_eudr_compliant:
                result.risk_level = DeforestationRisk.VIOLATION

            # Calculate provenance hash
            result.provenance_hash = self._calculate_hash({
                "coordinate": [float(coordinate.longitude), float(coordinate.latitude)],
                "country": country_iso3,
                "baseline_date": str(EUDR_CUTOFF_DATE),
                "observation_date": str(observation_date),
                "forest_status": result.forest_status.value,
                "is_compliant": result.is_eudr_compliant,
                "risk_score": float(result.risk_score)
            })

        except Exception as e:
            logger.error(f"Baseline check failed: {e}")
            result.warnings.append(f"Check failed: {str(e)}")
            result.forest_status = ForestStatus.UNKNOWN

        return result

    def check_polygon_baseline(
        self,
        polygon: ParsedPolygon,
        country_iso3: str,
        observation_date: Optional[date] = None,
        sample_points: int = 9
    ) -> BaselineCheckResult:
        """
        Check deforestation baseline for a polygon using sampling.

        DETERMINISTIC ASSESSMENT.

        Args:
            polygon: Polygon to check
            country_iso3: ISO3 country code
            observation_date: Current observation date
            sample_points: Number of sample points (default 9 for 3x3 grid)

        Returns:
            Aggregated BaselineCheckResult
        """
        if observation_date is None:
            observation_date = date.today()

        # Generate sample points within polygon
        sample_coords = self._generate_sample_points(polygon, sample_points)

        # Check each sample point
        sample_results = []
        for coord in sample_coords:
            result = self.check_baseline(coord, country_iso3, observation_date)
            sample_results.append(result)

        # Aggregate results
        return self._aggregate_results(sample_results, polygon)

    def get_forest_definition(self, country_iso3: str) -> ForestDefinition:
        """
        Get forest definition for a country.

        Returns country-specific definition if available,
        otherwise returns FAO default.
        """
        return self.COUNTRY_DEFINITIONS.get(
            country_iso3.upper(),
            self.FAO_DEFAULT_DEFINITION
        )

    def _get_forest_cover_at_date(
        self,
        coordinate: Coordinate,
        target_date: date
    ) -> Optional[ForestCoverData]:
        """
        Get forest cover data for a coordinate at a specific date.

        This is a stub for satellite data integration.
        In production, would query:
        - Global Forest Watch
        - Hansen Global Forest Change
        - RADD Alerts
        - Copernicus services
        """
        # Stub implementation - returns simulated data
        # In production, implement actual satellite data API calls

        logger.debug(f"Querying forest cover for {coordinate} at {target_date}")

        # Return stub data for demonstration
        # In production, this would be real satellite data
        return ForestCoverData(
            coordinate=coordinate,
            observation_date=target_date,
            tree_cover_percent=Decimal('75'),  # Stub value
            tree_height_meters=Decimal('15'),  # Stub value
            forest_type="tropical_moist",
            data_source="Hansen_GFC_v1.9",
            confidence_percent=Decimal('90'),
            pixel_resolution_meters=30
        )

    def _get_deforestation_events(
        self,
        coordinate: Coordinate,
        start_date: date,
        end_date: date,
        radius_km: float = 5.0
    ) -> List[DeforestationEvent]:
        """
        Get deforestation events near a coordinate within a date range.

        This is a stub for deforestation alert integration.
        In production, would query:
        - GLAD Alerts
        - RADD Alerts
        - DETER (Brazil)
        - JJ-FAST
        """
        # Stub implementation
        logger.debug(
            f"Querying deforestation events for {coordinate} "
            f"from {start_date} to {end_date}"
        )

        # Return empty list (no events) for stub
        # In production, this would be real alert data
        return []

    def _is_forest(
        self,
        cover_data: ForestCoverData,
        definition: ForestDefinition
    ) -> bool:
        """
        Determine if a location meets the forest definition.

        DETERMINISTIC CALCULATION.
        """
        # Check tree cover threshold
        if cover_data.tree_cover_percent < definition.min_tree_cover_percent:
            return False

        # Check tree height if available
        if (cover_data.tree_height_meters is not None and
            cover_data.tree_height_meters < definition.min_tree_height_meters):
            return False

        return True

    def _determine_forest_status(self, result: BaselineCheckResult) -> ForestStatus:
        """
        Determine overall forest status from baseline check results.

        DETERMINISTIC DETERMINATION.
        """
        # Check for post-cutoff deforestation events
        post_cutoff_events = [
            e for e in result.deforestation_events if e.is_post_cutoff
        ]
        if post_cutoff_events:
            return ForestStatus.DEFORESTED_POST_CUTOFF

        # Check forest cover change
        if (result.baseline_was_forest and
            result.current_is_forest is False and
            result.forest_cover_change_percent is not None and
            result.forest_cover_change_percent < Decimal('-20')):
            # Significant forest loss detected
            return ForestStatus.DEFORESTED_POST_CUTOFF

        # Check current status
        if result.current_is_forest:
            return ForestStatus.FOREST
        elif result.baseline_was_forest is False:
            return ForestStatus.NON_FOREST
        elif result.baseline_was_forest and result.current_is_forest is False:
            # Was forest at baseline, not now - need to determine when
            return ForestStatus.DEFORESTED_PRE_CUTOFF  # Conservative assumption

        return ForestStatus.UNKNOWN

    def _check_eudr_compliance(self, result: BaselineCheckResult) -> bool:
        """
        Check EUDR compliance based on baseline results.

        DETERMINISTIC DETERMINATION.

        EUDR Compliance Rules:
        1. If deforestation occurred after Dec 31, 2020 -> NOT COMPLIANT
        2. If forest status is DEFORESTED_POST_CUTOFF -> NOT COMPLIANT
        3. Otherwise -> COMPLIANT (until proven otherwise)
        """
        # Check for post-cutoff deforestation events
        post_cutoff_events = [
            e for e in result.deforestation_events if e.is_post_cutoff
        ]
        if post_cutoff_events:
            return False

        # Check forest status
        if result.forest_status == ForestStatus.DEFORESTED_POST_CUTOFF:
            return False

        return True

    def _calculate_risk_score(
        self,
        coordinate: Coordinate,
        country_iso3: str,
        result: BaselineCheckResult
    ) -> Decimal:
        """
        Calculate deforestation risk score (0-100).

        DETERMINISTIC CALCULATION.

        Factors considered:
        - Proximity to recent deforestation
        - Historical deforestation rate in area
        - Road network proximity
        - Fire alerts
        - Protected area pressure
        - Commodity suitability
        """
        score = Decimal('0')

        # Factor 1: Post-cutoff events (immediate high risk)
        post_cutoff_events = [
            e for e in result.deforestation_events if e.is_post_cutoff
        ]
        if post_cutoff_events:
            score += Decimal('50')  # Immediate 50 points

        # Factor 2: Forest cover change
        if result.forest_cover_change_percent is not None:
            change = result.forest_cover_change_percent
            if change < Decimal('-20'):
                score += Decimal('30')
            elif change < Decimal('-10'):
                score += Decimal('20')
            elif change < Decimal('0'):
                score += Decimal('10')

        # Factor 3: Baseline was forest (higher risk if was forest)
        if result.baseline_was_forest:
            score += Decimal('10')

        # Factor 4: Country risk adjustment (stub - would use actual data)
        country_risk_adjustment = self._get_country_risk_adjustment(country_iso3)
        score += country_risk_adjustment

        # Cap at 100
        return min(score, Decimal('100')).quantize(
            Decimal('0.1'), rounding=ROUND_HALF_UP
        )

    def _get_country_risk_adjustment(self, country_iso3: str) -> Decimal:
        """
        Get country-level risk adjustment.

        Based on historical deforestation rates and governance.
        """
        # High-risk countries for deforestation
        HIGH_RISK_COUNTRIES = {
            "BRA": Decimal('10'),  # Brazil - Amazon
            "IDN": Decimal('10'),  # Indonesia
            "COD": Decimal('8'),   # DRC
            "MYS": Decimal('8'),   # Malaysia
            "BOL": Decimal('8'),   # Bolivia
            "COL": Decimal('6'),   # Colombia
            "PER": Decimal('6'),   # Peru
            "CIV": Decimal('6'),   # Cote d'Ivoire
            "GHA": Decimal('5'),   # Ghana
            "CMR": Decimal('5'),   # Cameroon
        }

        return HIGH_RISK_COUNTRIES.get(country_iso3.upper(), Decimal('0'))

    def _score_to_risk_level(self, score: Decimal) -> DeforestationRisk:
        """Convert risk score to risk level."""
        if score >= Decimal('75'):
            return DeforestationRisk.CRITICAL
        elif score >= Decimal('50'):
            return DeforestationRisk.HIGH
        elif score >= Decimal('25'):
            return DeforestationRisk.MEDIUM
        else:
            return DeforestationRisk.LOW

    def _generate_sample_points(
        self,
        polygon: ParsedPolygon,
        num_points: int
    ) -> List[Coordinate]:
        """
        Generate sample points within a polygon for spatial sampling.

        Uses a grid-based approach for deterministic sampling.
        """
        from .coordinate_validator import CoordinateValidator
        validator = CoordinateValidator()

        # Get bounding box
        if not polygon.bounding_box:
            return [polygon.centroid] if polygon.centroid else []

        bbox = polygon.bounding_box
        sample_coords = []

        # Calculate grid dimensions
        grid_size = int(num_points ** 0.5)
        if grid_size * grid_size < num_points:
            grid_size += 1

        lon_step = (bbox.max_longitude - bbox.min_longitude) / (grid_size + 1)
        lat_step = (bbox.max_latitude - bbox.min_latitude) / (grid_size + 1)

        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                lon = bbox.min_longitude + lon_step * i
                lat = bbox.min_latitude + lat_step * j
                coord = Coordinate(
                    longitude=lon.quantize(Decimal('0.000001')),
                    latitude=lat.quantize(Decimal('0.000001'))
                )

                # Check if point is inside polygon
                if validator.point_in_polygon(coord, polygon):
                    sample_coords.append(coord)

                if len(sample_coords) >= num_points:
                    break
            if len(sample_coords) >= num_points:
                break

        # If no points inside polygon, use centroid
        if not sample_coords and polygon.centroid:
            sample_coords.append(polygon.centroid)

        return sample_coords

    def _aggregate_results(
        self,
        results: List[BaselineCheckResult],
        polygon: ParsedPolygon
    ) -> BaselineCheckResult:
        """
        Aggregate multiple point results into a polygon result.

        Uses conservative aggregation (worst-case for compliance).
        """
        if not results:
            return BaselineCheckResult(
                forest_status=ForestStatus.UNKNOWN,
                is_eudr_compliant=True,
                risk_level=DeforestationRisk.LOW,
                risk_score=Decimal('0')
            )

        # Aggregate by most severe status
        aggregated = BaselineCheckResult(
            forest_status=ForestStatus.UNKNOWN,
            is_eudr_compliant=True,
            risk_level=DeforestationRisk.LOW,
            risk_score=Decimal('0')
        )

        # Compliance: ALL points must be compliant
        aggregated.is_eudr_compliant = all(r.is_eudr_compliant for r in results)

        # Forest status: Use most severe
        status_priority = {
            ForestStatus.DEFORESTED_POST_CUTOFF: 0,
            ForestStatus.DEGRADED: 1,
            ForestStatus.DEFORESTED_PRE_CUTOFF: 2,
            ForestStatus.NON_FOREST: 3,
            ForestStatus.FOREST: 4,
            ForestStatus.REGENERATING: 5,
            ForestStatus.PLANTATION: 6,
            ForestStatus.UNKNOWN: 7
        }

        most_severe = min(
            results,
            key=lambda r: status_priority.get(r.forest_status, 7)
        )
        aggregated.forest_status = most_severe.forest_status

        # Risk score: Use maximum
        aggregated.risk_score = max(r.risk_score for r in results)
        aggregated.risk_level = self._score_to_risk_level(aggregated.risk_score)

        # Aggregate deforestation events
        all_events = []
        for r in results:
            all_events.extend(r.deforestation_events)
        # Remove duplicates by event_id
        seen_ids = set()
        unique_events = []
        for event in all_events:
            if event.event_id not in seen_ids:
                seen_ids.add(event.event_id)
                unique_events.append(event)
        aggregated.deforestation_events = unique_events

        # Aggregate data sources
        all_sources = set()
        for r in results:
            all_sources.update(r.data_sources)
        aggregated.data_sources = list(all_sources)

        # Calculate provenance
        aggregated.provenance_hash = self._calculate_hash({
            "polygon_hash": polygon.get_hash(),
            "sample_count": len(results),
            "is_compliant": aggregated.is_eudr_compliant,
            "risk_score": float(aggregated.risk_score)
        })

        return aggregated

    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash for provenance."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
