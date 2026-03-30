"""
ProtectedAreaEngine - WDPA and Key Biodiversity Area overlay analysis for EUDR

This module implements protected area overlay analysis for PACK-007 EUDR Professional Pack.
Provides World Database on Protected Areas (WDPA), Key Biodiversity Areas (KBA), indigenous
territories, Ramsar sites, and UNESCO natural heritage site checking per EU Regulation 2023/1115.

Example:
    >>> config = ProtectedAreaConfig(wdpa_enabled=True, buffer_km=5)
    >>> engine = ProtectedAreaEngine(config)
    >>> result = engine.full_overlay_analysis(lat=3.5, lon=101.5)
    >>> print(f"Risk amplification: {result.risk_amplification}")
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime
import math
import json

logger = logging.getLogger(__name__)


class IUCNCategory(str):
    """IUCN Protected Area Categories."""
    IA = "Ia"  # Strict Nature Reserve
    IB = "Ib"  # Wilderness Area
    II = "II"  # National Park
    III = "III"  # Natural Monument
    IV = "IV"  # Habitat/Species Management Area
    V = "V"  # Protected Landscape/Seascape
    VI = "VI"  # Managed Resource Protected Area


class ProtectedAreaConfig(BaseModel):
    """Configuration for protected area engine."""

    wdpa_enabled: bool = Field(True, description="Enable WDPA checking")
    kba_enabled: bool = Field(True, description="Enable Key Biodiversity Area checking")
    indigenous_check: bool = Field(True, description="Check indigenous/community territories")
    buffer_km: float = Field(5.0, ge=0.0, le=50.0, description="Buffer zone radius in kilometers")
    ramsar_check: bool = Field(True, description="Check Ramsar wetland sites")
    unesco_check: bool = Field(True, description="Check UNESCO natural heritage sites")
    strict_mode: bool = Field(False, description="Fail on any proximity to protected areas")
    risk_amplification_factor: float = Field(2.0, ge=1.0, le=5.0, description="Risk score multiplier")


class ProtectedArea(BaseModel):
    """Protected area from WDPA database."""

    wdpa_id: str = Field(..., description="WDPA identifier")
    name: str = Field(..., description="Protected area name")
    type: str = Field(..., description="Protected area type")
    country: str = Field(..., description="ISO3 country code")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometers")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (centroid)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (centroid)")
    designation: str = Field(..., description="Legal designation")
    iucn_category: Optional[str] = Field(None, description="IUCN category (Ia-VI)")
    established_year: Optional[int] = Field(None, description="Year established")


class KeyBiodiversityArea(BaseModel):
    """Key Biodiversity Area (KBA)."""

    kba_id: str = Field(..., description="KBA identifier")
    name: str = Field(..., description="KBA name")
    country: str = Field(..., description="ISO3 country code")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometers")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (centroid)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (centroid)")
    criteria: List[str] = Field(..., description="KBA criteria met (e.g., A1, B1, D1)")
    trigger_species: List[str] = Field(default_factory=list, description="Species triggering KBA status")


class IndigenousTerritory(BaseModel):
    """Indigenous or community territory."""

    territory_id: str = Field(..., description="Territory identifier")
    name: str = Field(..., description="Territory name")
    country: str = Field(..., description="ISO3 country code")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometers")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (centroid)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (centroid)")
    community_name: str = Field(..., description="Indigenous community name")
    legal_status: str = Field(..., description="Legal recognition status")
    fpic_required: bool = Field(True, description="Free Prior Informed Consent required")


class RamsarSite(BaseModel):
    """Ramsar Convention wetland site."""

    ramsar_id: str = Field(..., description="Ramsar site identifier")
    name: str = Field(..., description="Site name")
    country: str = Field(..., description="ISO3 country code")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometers")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (centroid)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (centroid)")
    designation_date: str = Field(..., description="Date designated as Ramsar site")
    wetland_type: str = Field(..., description="Wetland type classification")


class UNESCOSite(BaseModel):
    """UNESCO World Heritage natural site."""

    unesco_id: str = Field(..., description="UNESCO site identifier")
    name: str = Field(..., description="Site name")
    country: str = Field(..., description="ISO3 country code")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometers")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (centroid)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (centroid)")
    inscription_year: int = Field(..., description="Year inscribed")
    criteria: List[str] = Field(..., description="UNESCO criteria (e.g., vii, viii, ix, x)")


class ProximityScore(BaseModel):
    """Proximity scoring for protected area."""

    area_id: str = Field(..., description="Protected area identifier")
    area_name: str = Field(..., description="Protected area name")
    distance_km: float = Field(..., ge=0.0, description="Distance to area boundary (km)")
    proximity_score: float = Field(..., ge=0.0, le=1.0, description="Proximity score (0=far, 1=inside)")
    within_buffer: bool = Field(..., description="Whether within configured buffer zone")


class OverlayResult(BaseModel):
    """Result of protected area overlay analysis."""

    plot_lat: float = Field(..., description="Plot latitude")
    plot_lon: float = Field(..., description="Plot longitude")
    protected_areas_found: List[ProtectedArea] = Field(..., description="WDPA areas within buffer")
    kba_areas_found: List[KeyBiodiversityArea] = Field(default_factory=list, description="KBAs within buffer")
    indigenous_territories_found: List[IndigenousTerritory] = Field(
        default_factory=list,
        description="Indigenous territories within buffer"
    )
    ramsar_sites_found: List[RamsarSite] = Field(default_factory=list, description="Ramsar sites within buffer")
    unesco_sites_found: List[UNESCOSite] = Field(default_factory=list, description="UNESCO sites within buffer")
    proximity_scores: List[ProximityScore] = Field(..., description="Proximity scores for all areas")
    risk_amplification: float = Field(..., ge=1.0, description="Risk score amplification factor")
    exclusion_zone: bool = Field(..., description="Whether plot is in exclusion zone")
    fpic_required: bool = Field(False, description="Whether FPIC is required")
    analysis_timestamp: datetime = Field(..., description="Analysis timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class ExclusionZone(BaseModel):
    """Geographic exclusion zone definition."""

    zone_id: str = Field(..., description="Zone identifier")
    name: str = Field(..., description="Zone name")
    type: str = Field(..., description="Zone type (PROTECTED_AREA, INDIGENOUS, UNESCO, etc.)")
    lat: float = Field(..., description="Latitude (centroid)")
    lon: float = Field(..., description="Longitude (centroid)")
    radius_km: float = Field(..., ge=0.0, description="Exclusion radius in km")
    reason: str = Field(..., description="Reason for exclusion")


class ProtectedAreaEngine:
    """
    Protected area overlay analysis engine for EUDR compliance.

    Implements WDPA, KBA, indigenous territories, Ramsar, and UNESCO site checking
    with proximity scoring and risk amplification.

    Attributes:
        config: Engine configuration
        wdpa_database: WDPA protected areas database
        kba_database: Key Biodiversity Areas database
        indigenous_database: Indigenous territories database
        ramsar_database: Ramsar sites database
        unesco_database: UNESCO sites database

    Example:
        >>> config = ProtectedAreaConfig(buffer_km=10)
        >>> engine = ProtectedAreaEngine(config)
        >>> result = engine.full_overlay_analysis(lat=3.5, lon=101.5)
        >>> print(f"Found {len(result.protected_areas_found)} protected areas")
    """

    def __init__(self, config: ProtectedAreaConfig):
        """Initialize protected area engine with reference databases."""
        self.config = config
        self._initialize_databases()
        logger.info(f"ProtectedAreaEngine initialized with buffer={config.buffer_km}km")

    def _initialize_databases(self):
        """Initialize reference databases with representative data."""
        # WDPA Database (30 representative entries)
        self.wdpa_database = [
            ProtectedArea(wdpa_id="WDPA-555552889", name="Taman Negara National Park", type="National Park",
                         country="MYS", area_km2=4343.0, lat=4.5, lon=102.4, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1938),
            ProtectedArea(wdpa_id="WDPA-555552890", name="Royal Belum State Park", type="State Park",
                         country="MYS", area_km2=1175.0, lat=5.5, lon=101.3, designation="State Park",
                         iucn_category=IUCNCategory.IV, established_year=2007),
            ProtectedArea(wdpa_id="WDPA-555552891", name="Endau-Rompin National Park", type="National Park",
                         country="MYS", area_km2=870.0, lat=2.4, lon=103.3, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1993),
            ProtectedArea(wdpa_id="WDPA-555552892", name="Gunung Leuser National Park", type="National Park",
                         country="IDN", area_km2=7927.0, lat=3.5, lon=97.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1980),
            ProtectedArea(wdpa_id="WDPA-555552893", name="Kerinci Seblat National Park", type="National Park",
                         country="IDN", area_km2=13791.0, lat=-2.0, lon=101.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1982),
            ProtectedArea(wdpa_id="WDPA-555552894", name="Bukit Barisan Selatan National Park", type="National Park",
                         country="IDN", area_km2=3568.0, lat=-5.0, lon=104.0, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1982),
            ProtectedArea(wdpa_id="WDPA-555552895", name="Tesso Nilo National Park", type="National Park",
                         country="IDN", area_km2=835.0, lat=0.2, lon=101.6, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2004),
            ProtectedArea(wdpa_id="WDPA-555552896", name="Sebangau National Park", type="National Park",
                         country="IDN", area_km2=568.0, lat=-2.2, lon=113.8, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2004),
            ProtectedArea(wdpa_id="WDPA-555552897", name="Dja Faunal Reserve", type="Faunal Reserve",
                         country="CMR", area_km2=5260.0, lat=3.0, lon=12.8, designation="Faunal Reserve",
                         iucn_category=IUCNCategory.IV, established_year=1950),
            ProtectedArea(wdpa_id="WDPA-555552898", name="Dzanga-Sangha Special Reserve", type="Special Reserve",
                         country="CAF", area_km2=3159.0, lat=2.5, lon=16.2, designation="Special Reserve",
                         iucn_category=IUCNCategory.IV, established_year=1990),
            ProtectedArea(wdpa_id="WDPA-555552899", name="Lope National Park", type="National Park",
                         country="GAB", area_km2=4913.0, lat=-0.5, lon=11.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2002),
            ProtectedArea(wdpa_id="WDPA-555552900", name="Nouabale-Ndoki National Park", type="National Park",
                         country="COG", area_km2=3921.0, lat=2.3, lon=16.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1993),
            ProtectedArea(wdpa_id="WDPA-555552901", name="Salonga National Park", type="National Park",
                         country="COD", area_km2=33350.0, lat=-2.3, lon=21.2, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1970),
            ProtectedArea(wdpa_id="WDPA-555552902", name="Virunga National Park", type="National Park",
                         country="COD", area_km2=7800.0, lat=-0.9, lon=29.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1925),
            ProtectedArea(wdpa_id="WDPA-555552903", name="Kahuzi-Biega National Park", type="National Park",
                         country="COD", area_km2=6000.0, lat=-2.5, lon=28.7, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1970),
            ProtectedArea(wdpa_id="WDPA-555552904", name="Pacaya-Samiria National Reserve", type="National Reserve",
                         country="PER", area_km2=20800.0, lat=-5.0, lon=-74.5, designation="National Reserve",
                         iucn_category=IUCNCategory.VI, established_year=1982),
            ProtectedArea(wdpa_id="WDPA-555552905", name="Manu National Park", type="National Park",
                         country="PER", area_km2=15328.0, lat=-12.0, lon=-71.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1973),
            ProtectedArea(wdpa_id="WDPA-555552906", name="Yasuni National Park", type="National Park",
                         country="ECU", area_km2=9820.0, lat=-0.9, lon=-75.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1979),
            ProtectedArea(wdpa_id="WDPA-555552907", name="Madidi National Park", type="National Park",
                         country="BOL", area_km2=18958.0, lat=-14.0, lon=-68.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1995),
            ProtectedArea(wdpa_id="WDPA-555552908", name="Tumucumaque National Park", type="National Park",
                         country="BRA", area_km2=38874.0, lat=1.5, lon=-53.0, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2002),
            ProtectedArea(wdpa_id="WDPA-555552909", name="Amazonia National Park", type="National Park",
                         country="BRA", area_km2=10700.0, lat=-4.5, lon=-56.0, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1974),
            ProtectedArea(wdpa_id="WDPA-555552910", name="Juruena National Park", type="National Park",
                         country="BRA", area_km2=19598.0, lat=-10.0, lon=-58.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2006),
            ProtectedArea(wdpa_id="WDPA-555552911", name="Iguazu National Park", type="National Park",
                         country="ARG", area_km2=677.0, lat=-25.7, lon=-54.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1934),
            ProtectedArea(wdpa_id="WDPA-555552912", name="Canaima National Park", type="National Park",
                         country="VEN", area_km2=30000.0, lat=5.5, lon=-61.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1962),
            ProtectedArea(wdpa_id="WDPA-555552913", name="Central Suriname Nature Reserve", type="Nature Reserve",
                         country="SUR", area_km2=16000.0, lat=4.0, lon=-56.0, designation="Nature Reserve",
                         iucn_category=IUCNCategory.IA, established_year=1998),
            ProtectedArea(wdpa_id="WDPA-555552914", name="Darien National Park", type="National Park",
                         country="PAN", area_km2=5970.0, lat=8.0, lon=-77.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1980),
            ProtectedArea(wdpa_id="WDPA-555552915", name="La Amistad International Park", type="International Park",
                         country="CRI", area_km2=4000.0, lat=9.0, lon=-83.0, designation="International Park",
                         iucn_category=IUCNCategory.II, established_year=1982),
            ProtectedArea(wdpa_id="WDPA-555552916", name="Corcovado National Park", type="National Park",
                         country="CRI", area_km2=424.0, lat=8.5, lon=-83.5, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=1975),
            ProtectedArea(wdpa_id="WDPA-555552917", name="Sierra del Divisor National Park", type="National Park",
                         country="PER", area_km2=13547.0, lat=-7.5, lon=-73.8, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2015),
            ProtectedArea(wdpa_id="WDPA-555552918", name="Alto Purus National Park", type="National Park",
                         country="PER", area_km2=25104.0, lat=-10.5, lon=-71.0, designation="National Park",
                         iucn_category=IUCNCategory.II, established_year=2004),
        ]

        # KBA Database (15 representative entries)
        self.kba_database = [
            KeyBiodiversityArea(kba_id="KBA-MYS-001", name="Belum-Temengor Forest Complex", country="MYS",
                               area_km2=2900.0, lat=5.4, lon=101.4, criteria=["A1a", "B1", "D1"],
                               trigger_species=["Panthera tigris", "Elephas maximus", "Helarctos malayanus"]),
            KeyBiodiversityArea(kba_id="KBA-IDN-001", name="Leuser Ecosystem", country="IDN",
                               area_km2=26000.0, lat=3.8, lon=97.3, criteria=["A1a", "A1c", "B1"],
                               trigger_species=["Pongo abelii", "Panthera tigris sumatrae"]),
            KeyBiodiversityArea(kba_id="KBA-IDN-002", name="Sebangau Peat Swamp", country="IDN",
                               area_km2=6000.0, lat=-2.3, lon=113.9, criteria=["A1c", "D1"],
                               trigger_species=["Pongo pygmaeus", "Nasalis larvatus"]),
            KeyBiodiversityArea(kba_id="KBA-CMR-001", name="Dja-Odzala-Minkebe Forest Complex", country="CMR",
                               area_km2=85000.0, lat=2.5, lon=13.5, criteria=["A1a", "B1"],
                               trigger_species=["Gorilla gorilla", "Pan troglodytes", "Loxodonta cyclotis"]),
            KeyBiodiversityArea(kba_id="KBA-GAB-001", name="Gamba Complex", country="GAB",
                               area_km2=11000.0, lat=-2.0, lon=10.0, criteria=["A1a", "D1"],
                               trigger_species=["Loxodonta cyclotis", "Gorilla gorilla"]),
            KeyBiodiversityArea(kba_id="KBA-COD-001", name="Salonga-Lukenie-Sankuru Landscape", country="COD",
                               area_km2=125000.0, lat=-2.0, lon=22.0, criteria=["A1a", "B1"],
                               trigger_species=["Pan paniscus", "Loxodonta cyclotis"]),
            KeyBiodiversityArea(kba_id="KBA-PER-001", name="Tambopata-Candamo", country="PER",
                               area_km2=3700.0, lat=-13.0, lon=-69.5, criteria=["A1b", "B1"],
                               trigger_species=["Ara glaucogularis", "Pteronura brasiliensis"]),
            KeyBiodiversityArea(kba_id="KBA-BRA-001", name="Xingu Indigenous Park", country="BRA",
                               area_km2=27000.0, lat=-11.5, lon=-53.0, criteria=["A1a", "D1"],
                               trigger_species=["Ateles marginatus", "Priodontes maximus"]),
            KeyBiodiversityArea(kba_id="KBA-BRA-002", name="Cristalino State Park", country="BRA",
                               area_km2=1840.0, lat=-9.6, lon=-55.9, criteria=["A1b", "B1"],
                               trigger_species=["Harpia harpyja", "Panthera onca"]),
            KeyBiodiversityArea(kba_id="KBA-ECU-001", name="Yasuni-Napo Watershed", country="ECU",
                               area_km2=15000.0, lat=-1.0, lon=-76.0, criteria=["A1a", "B1", "D1"],
                               trigger_species=["Ateles belzebuth", "Trichechus inunguis"]),
            KeyBiodiversityArea(kba_id="KBA-BOL-001", name="Madidi-Tambopata Landscape", country="BOL",
                               area_km2=37000.0, lat=-13.5, lon=-68.0, criteria=["A1a", "B1"],
                               trigger_species=["Tremarctos ornatus", "Panthera onca"]),
            KeyBiodiversityArea(kba_id="KBA-COL-001", name="Chiribiquete National Park", country="COL",
                               area_km2=27820.0, lat=0.5, lon=-72.5, criteria=["A1a", "D1"],
                               trigger_species=["Ateles belzebuth", "Panthera onca"]),
            KeyBiodiversityArea(kba_id="KBA-VEN-001", name="Canaima Tepui Complex", country="VEN",
                               area_km2=35000.0, lat=5.0, lon=-61.0, criteria=["A1b", "B1"],
                               trigger_species=["Harpia harpyja", "Myrmecophaga tridactyla"]),
            KeyBiodiversityArea(kba_id="KBA-PNG-001", name="Kikori River Basin", country="PNG",
                               area_km2=12000.0, lat=-7.5, lon=144.0, criteria=["A1a", "D1"],
                               trigger_species=["Paradisaea apoda", "Casuarius casuarius"]),
            KeyBiodiversityArea(kba_id="KBA-MYS-002", name="Kinabatangan Floodplain", country="MYS",
                               area_km2=260.0, lat=5.5, lon=118.0, criteria=["A1a", "D1"],
                               trigger_species=["Elephas maximus borneensis", "Pongo pygmaeus"]),
        ]

        # Indigenous Territories Database (10 entries)
        self.indigenous_database = [
            IndigenousTerritory(territory_id="IND-BRA-001", name="Kayapo Territory", country="BRA",
                               area_km2=32800.0, lat=-7.5, lon=-51.5, community_name="Kayapo People",
                               legal_status="Legally Recognized", fpic_required=True),
            IndigenousTerritory(territory_id="IND-BRA-002", name="Yanomami Territory", country="BRA",
                               area_km2=96650.0, lat=2.0, lon=-63.5, community_name="Yanomami People",
                               legal_status="Legally Recognized", fpic_required=True),
            IndigenousTerritory(territory_id="IND-PER-001", name="Amarakaeri Communal Reserve", country="PER",
                               area_km2=4020.0, lat=-13.0, lon=-70.5, community_name="Harakmbut People",
                               legal_status="Legally Recognized", fpic_required=True),
            IndigenousTerritory(territory_id="IND-ECU-001", name="Waorani Territory", country="ECU",
                               area_km2=8000.0, lat=-1.0, lon=-76.5, community_name="Waorani People",
                               legal_status="Legally Recognized", fpic_required=True),
            IndigenousTerritory(territory_id="IND-COL-001", name="Nuquí Indigenous Reserve", country="COL",
                               area_km2=650.0, lat=5.7, lon=-77.3, community_name="Embera People",
                               legal_status="Legally Recognized", fpic_required=True),
            IndigenousTerritory(territory_id="IND-IDN-001", name="Dayak Customary Forest", country="IDN",
                               area_km2=1200.0, lat=-0.5, lon=110.5, community_name="Dayak People",
                               legal_status="Customary Rights", fpic_required=True),
            IndigenousTerritory(territory_id="IND-MYS-001", name="Penan Community Territory", country="MYS",
                               area_km2=450.0, lat=3.8, lon=115.0, community_name="Penan People",
                               legal_status="Customary Rights", fpic_required=True),
            IndigenousTerritory(territory_id="IND-PNG-001", name="Huli Territory", country="PNG",
                               area_km2=2500.0, lat=-6.0, lon=142.5, community_name="Huli People",
                               legal_status="Customary Rights", fpic_required=True),
            IndigenousTerritory(territory_id="IND-COD-001", name="Mbuti Forest Territory", country="COD",
                               area_km2=3500.0, lat=1.5, lon=28.0, community_name="Mbuti People",
                               legal_status="Customary Rights", fpic_required=True),
            IndigenousTerritory(territory_id="IND-CMR-001", name="Baka Territory", country="CMR",
                               area_km2=800.0, lat=3.5, lon=13.5, community_name="Baka People",
                               legal_status="Customary Rights", fpic_required=True),
        ]

        # Ramsar Sites Database (10 entries)
        self.ramsar_database = [
            RamsarSite(ramsar_id="RAM-MYS-001", name="Tasek Bera", country="MYS",
                      area_km2=310.0, lat=3.2, lon=102.6, designation_date="1994-11-10",
                      wetland_type="Freshwater lake and marsh"),
            RamsarSite(ramsar_id="RAM-IDN-001", name="Berbak National Park", country="IDN",
                      area_km2=1900.0, lat=-1.5, lon=104.5, designation_date="1992-10-06",
                      wetland_type="Peat swamp forest"),
            RamsarSite(ramsar_id="RAM-BRA-001", name="Pantanal Matogrossense", country="BRA",
                      area_km2=1350.0, lat=-17.5, lon=-57.0, designation_date="1993-05-24",
                      wetland_type="Seasonal floodplain"),
            RamsarSite(ramsar_id="RAM-PER-001", name="Pacaya-Samiria", country="PER",
                      area_km2=20800.0, lat=-5.0, lon=-74.5, designation_date="1986-01-30",
                      wetland_type="Amazonian floodplain"),
            RamsarSite(ramsar_id="RAM-COD-001", name="Ngiri-Tumba-Maindombe", country="COD",
                      area_km2=65696.0, lat=-1.8, lon=18.3, designation_date="2008-10-27",
                      wetland_type="Freshwater swamp forest"),
            RamsarSite(ramsar_id="RAM-CMR-001", name="Lac Ossa", country="CMR",
                      area_km2=40.0, lat=3.8, lon=10.5, designation_date="2008-01-02",
                      wetland_type="Freshwater lake"),
            RamsarSite(ramsar_id="RAM-ECU-001", name="Manglares Churute", country="ECU",
                      area_km2=350.0, lat=-2.5, lon=-79.7, designation_date="1990-06-07",
                      wetland_type="Mangrove and estuarine"),
            RamsarSite(ramsar_id="RAM-COL-001", name="Delta del Rio Baudó", country="COL",
                      area_km2=1040.0, lat=5.2, lon=-77.4, designation_date="2004-11-05",
                      wetland_type="Riverine delta"),
            RamsarSite(ramsar_id="RAM-VEN-001", name="Los Olivitos", country="VEN",
                      area_km2=260.0, lat=10.8, lon=-71.3, designation_date="1996-06-04",
                      wetland_type="Coastal lagoon"),
            RamsarSite(ramsar_id="RAM-PNG-001", name="Tonda Wildlife Management Area", country="PNG",
                      area_km2=5900.0, lat=-8.5, lon=141.5, designation_date="1993-11-16",
                      wetland_type="Coastal wetlands"),
        ]

        # UNESCO Natural Heritage Sites Database (10 entries)
        self.unesco_database = [
            UNESCOSite(unesco_id="UNESCO-352", name="Tropical Rainforest Heritage of Sumatra", country="IDN",
                      area_km2=25000.0, lat=3.5, lon=97.5, inscription_year=2004,
                      criteria=["vii", "ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-200", name="Gunung Mulu National Park", country="MYS",
                      area_km2=528.0, lat=4.0, lon=114.9, inscription_year=2000,
                      criteria=["vii", "viii", "ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-368", name="Kinabalu Park", country="MYS",
                      area_km2=754.0, lat=6.1, lon=116.6, inscription_year=2000,
                      criteria=["ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-180", name="Salonga National Park", country="COD",
                      area_km2=36000.0, lat=-2.3, lon=21.2, inscription_year=1984,
                      criteria=["vii", "ix"]),
            UNESCOSite(unesco_id="UNESCO-63", name="Virunga National Park", country="COD",
                      area_km2=7900.0, lat=-0.9, lon=29.5, inscription_year=1979,
                      criteria=["vii", "viii", "x"]),
            UNESCOSite(unesco_id="UNESCO-137", name="Manu National Park", country="PER",
                      area_km2=15328.0, lat=-12.0, lon=-71.5, inscription_year=1987,
                      criteria=["ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-981", name="Central Amazon Conservation Complex", country="BRA",
                      area_km2=60000.0, lat=-2.5, lon=-61.5, inscription_year=2000,
                      criteria=["ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-998", name="Atlantic Forest Reserves", country="BRA",
                      area_km2=4700.0, lat=-24.5, lon=-48.5, inscription_year=1999,
                      criteria=["vii", "ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-658", name="Sangay National Park", country="ECU",
                      area_km2=5178.0, lat=-2.0, lon=-78.4, inscription_year=1983,
                      criteria=["vii", "viii", "ix", "x"]),
            UNESCOSite(unesco_id="UNESCO-701", name="Darien National Park", country="PAN",
                      area_km2=5970.0, lat=8.0, lon=-77.5, inscription_year=1981,
                      criteria=["vii", "ix", "x"]),
        ]

        logger.info(f"Initialized databases: WDPA={len(self.wdpa_database)}, KBA={len(self.kba_database)}, "
                   f"Indigenous={len(self.indigenous_database)}, Ramsar={len(self.ramsar_database)}, "
                   f"UNESCO={len(self.unesco_database)}")

    def check_wdpa(self, lat: float, lon: float, buffer_km: float) -> List[ProtectedArea]:
        """
        Check World Database on Protected Areas within buffer.

        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer radius in kilometers

        Returns:
            List of protected areas within buffer
        """
        try:
            if not self.config.wdpa_enabled:
                return []

            areas_found = []
            for area in self.wdpa_database:
                distance = self._calculate_distance(lat, lon, area.lat, area.lon)
                if distance <= buffer_km:
                    areas_found.append(area)

            logger.info(f"WDPA check at ({lat}, {lon}): found {len(areas_found)} areas within {buffer_km}km")
            return areas_found

        except Exception as e:
            logger.error(f"WDPA check failed: {str(e)}", exc_info=True)
            return []

    def check_kba(self, lat: float, lon: float, buffer_km: float) -> List[KeyBiodiversityArea]:
        """
        Check Key Biodiversity Areas within buffer.

        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer radius in kilometers

        Returns:
            List of KBAs within buffer
        """
        try:
            if not self.config.kba_enabled:
                return []

            kbas_found = []
            for kba in self.kba_database:
                distance = self._calculate_distance(lat, lon, kba.lat, kba.lon)
                if distance <= buffer_km:
                    kbas_found.append(kba)

            logger.info(f"KBA check at ({lat}, {lon}): found {len(kbas_found)} areas within {buffer_km}km")
            return kbas_found

        except Exception as e:
            logger.error(f"KBA check failed: {str(e)}", exc_info=True)
            return []

    def check_indigenous_lands(self, lat: float, lon: float) -> List[IndigenousTerritory]:
        """
        Check indigenous and community territories.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            List of indigenous territories at or near location
        """
        try:
            if not self.config.indigenous_check:
                return []

            territories_found = []
            for territory in self.indigenous_database:
                distance = self._calculate_distance(lat, lon, territory.lat, territory.lon)
                # Use buffer for indigenous check
                if distance <= self.config.buffer_km:
                    territories_found.append(territory)

            logger.info(f"Indigenous check at ({lat}, {lon}): found {len(territories_found)} territories")
            return territories_found

        except Exception as e:
            logger.error(f"Indigenous lands check failed: {str(e)}", exc_info=True)
            return []

    def check_ramsar(self, lat: float, lon: float, buffer_km: float) -> List[RamsarSite]:
        """
        Check Ramsar Convention wetland sites within buffer.

        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer radius in kilometers

        Returns:
            List of Ramsar sites within buffer
        """
        try:
            if not self.config.ramsar_check:
                return []

            sites_found = []
            for site in self.ramsar_database:
                distance = self._calculate_distance(lat, lon, site.lat, site.lon)
                if distance <= buffer_km:
                    sites_found.append(site)

            logger.info(f"Ramsar check at ({lat}, {lon}): found {len(sites_found)} sites within {buffer_km}km")
            return sites_found

        except Exception as e:
            logger.error(f"Ramsar check failed: {str(e)}", exc_info=True)
            return []

    def check_unesco(self, lat: float, lon: float, buffer_km: float) -> List[UNESCOSite]:
        """
        Check UNESCO World Heritage natural sites within buffer.

        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer radius in kilometers

        Returns:
            List of UNESCO sites within buffer
        """
        try:
            if not self.config.unesco_check:
                return []

            sites_found = []
            for site in self.unesco_database:
                distance = self._calculate_distance(lat, lon, site.lat, site.lon)
                if distance <= buffer_km:
                    sites_found.append(site)

            logger.info(f"UNESCO check at ({lat}, {lon}): found {len(sites_found)} sites within {buffer_km}km")
            return sites_found

        except Exception as e:
            logger.error(f"UNESCO check failed: {str(e)}", exc_info=True)
            return []

    def calculate_proximity_score(self, distance_km: float) -> float:
        """
        Calculate proximity score based on distance.

        Score ranges from 0 (far) to 1 (inside/very close).
        Uses exponential decay: score = exp(-distance/buffer).

        Args:
            distance_km: Distance in kilometers

        Returns:
            Proximity score (0-1)
        """
        try:
            if distance_km <= 0:
                return 1.0  # Inside protected area

            # Exponential decay based on buffer distance
            decay_factor = self.config.buffer_km
            score = math.exp(-distance_km / decay_factor)

            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Proximity score calculation failed: {str(e)}", exc_info=True)
            return 0.0

    def amplify_risk_score(self, base_score: float, overlay_result: OverlayResult) -> float:
        """
        Amplify risk score based on protected area proximity.

        Args:
            base_score: Base EUDR risk score (0-1)
            overlay_result: Protected area overlay result

        Returns:
            Amplified risk score
        """
        try:
            # If in exclusion zone, maximum risk
            if overlay_result.exclusion_zone:
                return 1.0

            # Calculate amplification based on proximity scores
            max_proximity = 0.0
            for prox in overlay_result.proximity_scores:
                if prox.proximity_score > max_proximity:
                    max_proximity = prox.proximity_score

            # Amplify risk: new_score = base_score * (1 + amplification_factor * max_proximity)
            amplification = 1.0 + (self.config.risk_amplification_factor - 1.0) * max_proximity
            amplified_score = base_score * amplification

            logger.info(f"Risk amplification: base={base_score:.3f}, amplified={amplified_score:.3f}, "
                       f"factor={amplification:.2f}")
            return min(amplified_score, 1.0)

        except Exception as e:
            logger.error(f"Risk amplification failed: {str(e)}", exc_info=True)
            return base_score

    def full_overlay_analysis(self, lat: float, lon: float) -> OverlayResult:
        """
        Perform full protected area overlay analysis.

        Args:
            lat: Plot latitude
            lon: Plot longitude

        Returns:
            Complete overlay analysis result
        """
        try:
            # Run all checks
            protected_areas = self.check_wdpa(lat, lon, self.config.buffer_km)
            kba_areas = self.check_kba(lat, lon, self.config.buffer_km)
            indigenous_territories = self.check_indigenous_lands(lat, lon)
            ramsar_sites = self.check_ramsar(lat, lon, self.config.buffer_km)
            unesco_sites = self.check_unesco(lat, lon, self.config.buffer_km)

            # Calculate proximity scores
            proximity_scores = []
            for area in protected_areas:
                distance = self._calculate_distance(lat, lon, area.lat, area.lon)
                score = self.calculate_proximity_score(distance)
                proximity_scores.append(ProximityScore(
                    area_id=area.wdpa_id,
                    area_name=area.name,
                    distance_km=distance,
                    proximity_score=score,
                    within_buffer=distance <= self.config.buffer_km
                ))

            # Check exclusion zone
            exclusion_zone = self._check_exclusion_zone(lat, lon, protected_areas, unesco_sites)

            # Check FPIC requirement
            fpic_required = any(t.fpic_required for t in indigenous_territories)

            # Calculate risk amplification
            max_proximity = max([ps.proximity_score for ps in proximity_scores], default=0.0)
            risk_amplification = 1.0 + (self.config.risk_amplification_factor - 1.0) * max_proximity

            # Calculate provenance hash
            provenance_data = {
                "lat": lat,
                "lon": lon,
                "buffer_km": self.config.buffer_km,
                "protected_areas_count": len(protected_areas),
                "kba_count": len(kba_areas),
                "indigenous_count": len(indigenous_territories),
                "timestamp": datetime.utcnow().isoformat()
            }
            provenance_hash = self._calculate_hash(provenance_data)

            result = OverlayResult(
                plot_lat=lat,
                plot_lon=lon,
                protected_areas_found=protected_areas,
                kba_areas_found=kba_areas,
                indigenous_territories_found=indigenous_territories,
                ramsar_sites_found=ramsar_sites,
                unesco_sites_found=unesco_sites,
                proximity_scores=proximity_scores,
                risk_amplification=risk_amplification,
                exclusion_zone=exclusion_zone,
                fpic_required=fpic_required,
                analysis_timestamp=datetime.utcnow(),
                provenance_hash=provenance_hash
            )

            logger.info(f"Full overlay analysis at ({lat}, {lon}): "
                       f"WDPA={len(protected_areas)}, KBA={len(kba_areas)}, "
                       f"Indigenous={len(indigenous_territories)}, exclusion={exclusion_zone}")
            return result

        except Exception as e:
            logger.error(f"Full overlay analysis failed: {str(e)}", exc_info=True)
            raise

    def batch_overlay(self, plots: List[Dict[str, float]]) -> List[OverlayResult]:
        """
        Perform batch overlay analysis for multiple plots.

        Args:
            plots: List of plot dictionaries with 'lat' and 'lon' keys

        Returns:
            List of overlay results
        """
        try:
            results = []
            for i, plot in enumerate(plots):
                result = self.full_overlay_analysis(plot['lat'], plot['lon'])
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"Batch overlay progress: {i + 1}/{len(plots)} plots analyzed")

            logger.info(f"Batch overlay complete: {len(results)} plots analyzed")
            return results

        except Exception as e:
            logger.error(f"Batch overlay failed: {str(e)}", exc_info=True)
            raise

    def get_exclusion_zones(self) -> List[ExclusionZone]:
        """
        Get all defined exclusion zones.

        Returns:
            List of exclusion zones
        """
        try:
            exclusion_zones = []

            # Add UNESCO sites as exclusion zones
            for site in self.unesco_database:
                exclusion_zones.append(ExclusionZone(
                    zone_id=site.unesco_id,
                    name=site.name,
                    type="UNESCO_NATURAL_HERITAGE",
                    lat=site.lat,
                    lon=site.lon,
                    radius_km=10.0,  # 10km exclusion radius
                    reason="UNESCO World Heritage Site - production prohibited within 10km"
                ))

            # Add Strict Nature Reserves (IUCN Ia) as exclusion zones
            for area in self.wdpa_database:
                if area.iucn_category == IUCNCategory.IA:
                    exclusion_zones.append(ExclusionZone(
                        zone_id=area.wdpa_id,
                        name=area.name,
                        type="STRICT_NATURE_RESERVE",
                        lat=area.lat,
                        lon=area.lon,
                        radius_km=5.0,  # 5km exclusion radius
                        reason="IUCN Category Ia Strict Nature Reserve - production prohibited"
                    ))

            logger.info(f"Retrieved {len(exclusion_zones)} exclusion zones")
            return exclusion_zones

        except Exception as e:
            logger.error(f"Failed to get exclusion zones: {str(e)}", exc_info=True)
            return []

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.

        Args:
            lat1: Latitude of point 1
            lon1: Longitude of point 1
            lat2: Latitude of point 2
            lon2: Longitude of point 2

        Returns:
            Distance in kilometers
        """
        # Earth radius in kilometers
        R = 6371.0

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def _check_exclusion_zone(
        self,
        lat: float,
        lon: float,
        protected_areas: List[ProtectedArea],
        unesco_sites: List[UNESCOSite]
    ) -> bool:
        """
        Check if plot is in an exclusion zone.

        Args:
            lat: Plot latitude
            lon: Plot longitude
            protected_areas: Protected areas found
            unesco_sites: UNESCO sites found

        Returns:
            True if in exclusion zone
        """
        # Any UNESCO site within 10km is exclusion zone
        for site in unesco_sites:
            distance = self._calculate_distance(lat, lon, site.lat, site.lon)
            if distance <= 10.0:
                return True

        # Any IUCN Category Ia within 5km is exclusion zone
        for area in protected_areas:
            if area.iucn_category == IUCNCategory.IA:
                distance = self._calculate_distance(lat, lon, area.lat, area.lon)
                if distance <= 5.0:
                    return True

        return False

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash for provenance tracking.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
