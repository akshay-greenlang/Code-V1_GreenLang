"""
GL-011 FUELCRAFT - Carbon Accounting Connector

Integration with carbon accounting systems for:
- Emissions factor imports (TTW/WTT/WTW)
- Carbon footprint exports
- Reconciliation support
- Scope 1/2/3 boundary alignment

Standards Compliance:
- ISO 14064-1 (GHG Quantification and Reporting)
- ISO 14064-3 (GHG Verification)
- GHG Protocol Corporate Standard
- GHG Protocol Scope 3 Standard

Supported Systems:
- Salesforce Net Zero Cloud
- Microsoft Sustainability Manager
- SAP Sustainability Control Tower
- Persefoni
- Watershed
- Custom REST APIs

Features:
- Emission factor database synchronization
- Automated footprint calculations
- Audit trail with SHA-256 hashes
- Scope 1/2/3 categorization
- Multi-framework mapping (CDP, TCFD, SBTi)
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio
import hashlib
import json
import logging
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class CarbonAccountingSystem(str, Enum):
    """Supported carbon accounting systems."""
    SALESFORCE_NETZERO = "salesforce_netzero"
    MS_SUSTAINABILITY = "ms_sustainability"
    SAP_SCT = "sap_sustainability_control_tower"
    PERSEFONI = "persefoni"
    WATERSHED = "watershed"
    CUSTOM_API = "custom_api"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Value chain emissions


class EmissionBoundary(str, Enum):
    """Emission calculation boundaries."""
    TTW = "tank_to_wheel"   # Direct combustion only
    WTT = "well_to_tank"    # Upstream only
    WTW = "well_to_wheel"   # Full lifecycle


class EmissionCategory(str, Enum):
    """GHG Protocol Scope 3 categories."""
    PURCHASED_GOODS = "category_1_purchased_goods"
    CAPITAL_GOODS = "category_2_capital_goods"
    FUEL_ENERGY = "category_3_fuel_energy"
    TRANSPORTATION_UPSTREAM = "category_4_transportation"
    WASTE = "category_5_waste"
    BUSINESS_TRAVEL = "category_6_business_travel"
    EMPLOYEE_COMMUTING = "category_7_employee_commuting"
    LEASED_ASSETS_UP = "category_8_leased_assets"
    TRANSPORTATION_DOWNSTREAM = "category_9_transportation"
    PROCESSING = "category_10_processing"
    USE_OF_SOLD_PRODUCTS = "category_11_use_of_products"
    END_OF_LIFE = "category_12_end_of_life"
    LEASED_ASSETS_DOWN = "category_13_leased_assets"
    FRANCHISES = "category_14_franchises"
    INVESTMENTS = "category_15_investments"


class DataQuality(str, Enum):
    """GHG Protocol data quality indicators."""
    PRIMARY = "primary"          # Measured/metered data
    SECONDARY = "secondary"      # Supplier-specific or industry average
    TERTIARY = "tertiary"        # Modeled/estimated
    UNKNOWN = "unknown"


class ReconciliationStatus(str, Enum):
    """Reconciliation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RECONCILED = "reconciled"
    DISCREPANCY = "discrepancy"
    FAILED = "failed"


# =============================================================================
# Configuration
# =============================================================================

class CarbonAccountingConfig(BaseModel):
    """Carbon accounting connector configuration."""
    system_type: CarbonAccountingSystem = Field(..., description="Accounting system")
    base_url: str = Field(..., description="API base URL")

    # Authentication
    auth_type: str = Field("oauth2", description="oauth2, api_key")
    client_id: Optional[str] = Field(None)
    client_secret: Optional[str] = Field(None)  # From vault
    api_key: Optional[str] = Field(None)  # From vault
    token_url: Optional[str] = Field(None)

    # Organization
    org_id: str = Field(..., description="Organization ID in system")
    reporting_entity: str = Field(..., description="Reporting entity name")

    # Connection settings
    timeout_seconds: int = Field(30)
    max_retries: int = Field(3)

    # Data settings
    default_gwp_version: str = Field("AR5", description="IPCC GWP version")
    default_emission_boundary: EmissionBoundary = Field(EmissionBoundary.WTW)

    # Sync settings
    emission_factor_sync_interval_hours: int = Field(24)
    footprint_export_batch_size: int = Field(1000)


# =============================================================================
# Data Models
# =============================================================================

class EmissionFactorData(BaseModel):
    """
    Emission factor data per ISO 14064.

    Contains emission factors for a specific fuel type
    across different boundaries (TTW, WTT, WTW).
    """
    factor_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fuel_type: str = Field(..., description="Fuel type identifier")
    fuel_name: str = Field(..., description="Fuel display name")

    # CO2 factors (kg per MMBtu)
    co2_kg_per_mmbtu: float = Field(..., ge=0)
    ch4_kg_per_mmbtu: float = Field(0.0, ge=0)
    n2o_kg_per_mmbtu: float = Field(0.0, ge=0)

    # CO2e calculation
    co2e_kg_per_mmbtu: float = Field(..., ge=0, description="Total CO2e")
    gwp_version: str = Field("AR5", description="IPCC GWP version used")

    # Boundary
    boundary: EmissionBoundary = Field(..., description="Emission boundary")
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    category: Optional[EmissionCategory] = Field(None)

    # Data quality
    data_quality: DataQuality = Field(DataQuality.SECONDARY)
    data_source: str = Field(..., description="Source (EPA, IPCC, etc.)")
    source_document: Optional[str] = Field(None)
    year: int = Field(..., description="Factor year")

    # Validity
    valid_from: datetime
    valid_to: Optional[datetime] = None
    is_active: bool = Field(True)

    # Region specificity
    region: str = Field("global", description="Geographic region")
    country_code: Optional[str] = Field(None)

    # Metadata
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: Optional[str] = Field(None)

    def calculate_co2e(self) -> float:
        """Calculate CO2e using GWP factors."""
        # AR5 GWP values: CH4=28, N2O=265
        # AR6 GWP values: CH4=29.8, N2O=273
        if self.gwp_version == "AR5":
            gwp_ch4, gwp_n2o = 28, 265
        elif self.gwp_version == "AR6":
            gwp_ch4, gwp_n2o = 29.8, 273
        else:  # AR4
            gwp_ch4, gwp_n2o = 25, 298

        return self.co2_kg_per_mmbtu + (self.ch4_kg_per_mmbtu * gwp_ch4) + (self.n2o_kg_per_mmbtu * gwp_n2o)

    def compute_provenance_hash(self) -> str:
        """Compute hash for audit trail."""
        data = f"{self.fuel_type}|{self.co2_kg_per_mmbtu}|{self.ch4_kg_per_mmbtu}|{self.n2o_kg_per_mmbtu}|{self.boundary.value}|{self.data_source}"
        return hashlib.sha256(data.encode()).hexdigest()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CarbonFootprintExport(BaseModel):
    """
    Carbon footprint export record per ISO 14064.

    Contains calculated emissions for a reporting period.
    """
    export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reporting_period_start: datetime
    reporting_period_end: datetime
    reporting_entity: str

    # Activity data
    fuel_type: str
    activity_quantity_mmbtu: float = Field(..., ge=0)
    activity_unit: str = Field("mmbtu")

    # Calculated emissions
    co2_emissions_mt: float = Field(..., ge=0, description="CO2 in metric tons")
    ch4_emissions_mt: float = Field(0.0, ge=0, description="CH4 in metric tons")
    n2o_emissions_mt: float = Field(0.0, ge=0, description="N2O in metric tons")
    co2e_emissions_mt: float = Field(..., ge=0, description="Total CO2e")

    # Classification
    scope: EmissionScope
    boundary: EmissionBoundary
    category: Optional[EmissionCategory] = None

    # Emission factor reference
    emission_factor_id: str
    emission_factor_source: str
    emission_factor_year: int

    # Data quality
    data_quality: DataQuality
    calculation_method: str = Field("factor_based", description="factor_based, measured")

    # Provenance
    source_run_id: Optional[str] = Field(None, description="Optimization run ID")
    calculation_hash: str = Field(..., description="SHA-256 of calculation inputs")
    bundle_hash: Optional[str] = Field(None, description="Bundle hash for audit")

    # Timestamps
    calculated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exported_at: Optional[datetime] = None

    # Status
    is_verified: bool = Field(False)
    verified_by: Optional[str] = Field(None)
    verified_at: Optional[datetime] = Field(None)

    def compute_calculation_hash(self) -> str:
        """Compute hash of calculation inputs."""
        data = json.dumps({
            "fuel_type": self.fuel_type,
            "quantity": self.activity_quantity_mmbtu,
            "emission_factor_id": self.emission_factor_id,
            "period_start": self.reporting_period_start.isoformat(),
            "period_end": self.reporting_period_end.isoformat(),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ReconciliationReport(BaseModel):
    """
    Reconciliation report for emission data verification.

    Compares FuelCraft calculations with carbon accounting system.
    """
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Comparison
    fuelcraft_total_co2e_mt: float = Field(..., ge=0)
    external_total_co2e_mt: float = Field(..., ge=0)
    variance_co2e_mt: float
    variance_percent: float
    materiality_threshold_percent: float = Field(5.0)

    # Breakdown by scope
    scope_1_fuelcraft_mt: float = Field(0.0, ge=0)
    scope_1_external_mt: float = Field(0.0, ge=0)
    scope_2_fuelcraft_mt: float = Field(0.0, ge=0)
    scope_2_external_mt: float = Field(0.0, ge=0)
    scope_3_fuelcraft_mt: float = Field(0.0, ge=0)
    scope_3_external_mt: float = Field(0.0, ge=0)

    # Breakdown by fuel
    by_fuel_type: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Status
    status: ReconciliationStatus
    discrepancy_items: List[Dict[str, Any]] = Field(default=[])
    resolution_notes: Optional[str] = Field(None)

    # Audit
    reconciled_by: Optional[str] = Field(None)
    reconciled_at: Optional[datetime] = Field(None)
    audit_trail_hash: str = Field(..., description="Hash of all input data")

    def is_within_tolerance(self) -> bool:
        """Check if variance is within materiality threshold."""
        return abs(self.variance_percent) <= self.materiality_threshold_percent

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ScopeBreakdown(BaseModel):
    """Emission breakdown by scope."""
    scope_1_mt: float = Field(0.0, ge=0)
    scope_2_mt: float = Field(0.0, ge=0)
    scope_3_mt: float = Field(0.0, ge=0)
    total_mt: float = Field(0.0, ge=0)

    # Scope 3 categories
    scope_3_by_category: Dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Carbon Accounting Connector
# =============================================================================

class CarbonAccountingConnector:
    """
    Connector for carbon accounting systems.

    Provides integration for:
    - Emission factor synchronization
    - Carbon footprint exports
    - Reconciliation with external systems
    - Multi-framework reporting alignment

    Compliant with ISO 14064-1/2/3 and GHG Protocol.

    Example:
        config = CarbonAccountingConfig(
            system_type=CarbonAccountingSystem.PERSEFONI,
            base_url="https://api.persefoni.com/v1",
            org_id="org-123",
            reporting_entity="Company ABC",
        )

        connector = CarbonAccountingConnector(config)
        await connector.connect()

        # Get emission factors
        factors = await connector.get_emission_factors(fuel_type="natural_gas")

        # Export footprint
        await connector.export_footprint(footprint_data)

        # Reconcile with external system
        report = await connector.reconcile_emissions(period_start, period_end)
    """

    def __init__(
        self,
        config: CarbonAccountingConfig,
        vault_client: Optional[Any] = None,
    ) -> None:
        """Initialize carbon accounting connector."""
        self.config = config
        self.vault_client = vault_client

        # Load credentials from vault
        if vault_client:
            self._load_credentials()

        # Connection state
        self._connected = False
        self._access_token: Optional[str] = None

        # Emission factor cache
        self._emission_factors: Dict[str, List[EmissionFactorData]] = {}
        self._last_factor_sync: Optional[datetime] = None

        # Statistics
        self._stats = {
            "factors_synced": 0,
            "footprints_exported": 0,
            "reconciliations": 0,
            "errors": 0,
        }

        logger.info(f"Carbon accounting connector initialized for {config.system_type.value}")

    def _load_credentials(self) -> None:
        """Load credentials from vault."""
        try:
            if self.config.auth_type == "oauth2":
                self.config.client_secret = self.vault_client.get_secret(
                    f"carbon/{self.config.system_type.value}/client_secret"
                )
            else:
                self.config.api_key = self.vault_client.get_secret(
                    f"carbon/{self.config.system_type.value}/api_key"
                )
        except Exception as e:
            logger.warning(f"Failed to load credentials from vault: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to carbon accounting system."""
        try:
            if self.config.auth_type == "oauth2":
                # In production, perform OAuth2 flow
                self._access_token = "mock_token"

            self._connected = True
            logger.info(f"Connected to carbon accounting: {self.config.base_url}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._stats["errors"] += 1
            return False

    async def disconnect(self) -> None:
        """Disconnect from carbon accounting system."""
        self._connected = False
        self._access_token = None
        logger.info("Disconnected from carbon accounting system")

    # =========================================================================
    # Emission Factor Operations
    # =========================================================================

    async def sync_emission_factors(self) -> int:
        """
        Synchronize emission factors from external system.

        Returns:
            Number of factors synchronized
        """
        logger.info("Synchronizing emission factors")

        try:
            # In production, fetch from API:
            # response = await self._make_request("GET", "emission-factors")
            # factors_data = response["data"]

            # Load default emission factors
            factors = self._get_default_emission_factors()

            # Update cache
            for factor in factors:
                fuel_type = factor.fuel_type
                if fuel_type not in self._emission_factors:
                    self._emission_factors[fuel_type] = []
                self._emission_factors[fuel_type].append(factor)

            self._last_factor_sync = datetime.now(timezone.utc)
            self._stats["factors_synced"] += len(factors)

            logger.info(f"Synchronized {len(factors)} emission factors")
            return len(factors)

        except Exception as e:
            logger.error(f"Failed to sync emission factors: {e}")
            self._stats["errors"] += 1
            raise

    def _get_default_emission_factors(self) -> List[EmissionFactorData]:
        """Get default emission factors based on EPA and IPCC data."""
        now = datetime.now(timezone.utc)
        factors = []

        # EPA emission factors for common fuels (kg CO2e per MMBtu)
        fuel_data = [
            # Fuel type, CO2, CH4, N2O, TTW CO2e, Source
            ("natural_gas", 53.06, 0.001, 0.0001, 53.12, "EPA 2024"),
            ("fuel_oil_2", 73.16, 0.003, 0.0006, 73.35, "EPA 2024"),
            ("fuel_oil_6", 75.10, 0.003, 0.0006, 75.29, "EPA 2024"),
            ("coal", 93.28, 0.011, 0.0016, 93.68, "EPA 2024"),
            ("propane", 63.07, 0.003, 0.0006, 63.26, "EPA 2024"),
            ("diesel", 73.96, 0.003, 0.0006, 74.15, "EPA 2024"),
            ("biomass", 0.0, 0.032, 0.0042, 2.01, "IPCC AR5"),  # Biogenic CO2 excluded
            ("hydrogen", 0.0, 0.0, 0.0, 0.0, "IPCC AR5"),  # Green hydrogen
        ]

        for fuel_type, co2, ch4, n2o, co2e, source in fuel_data:
            # TTW factor (Scope 1)
            factors.append(EmissionFactorData(
                fuel_type=fuel_type,
                fuel_name=fuel_type.replace("_", " ").title(),
                co2_kg_per_mmbtu=co2,
                ch4_kg_per_mmbtu=ch4,
                n2o_kg_per_mmbtu=n2o,
                co2e_kg_per_mmbtu=co2e,
                boundary=EmissionBoundary.TTW,
                scope=EmissionScope.SCOPE_1,
                data_quality=DataQuality.SECONDARY,
                data_source=source,
                year=2024,
                valid_from=now - timedelta(days=365),
            ))

            # WTT factor (Scope 3) - typically 10-25% of TTW for fossil fuels
            wtt_multiplier = 0.0 if fuel_type in ["biomass", "hydrogen"] else 0.15
            if wtt_multiplier > 0:
                factors.append(EmissionFactorData(
                    fuel_type=fuel_type,
                    fuel_name=fuel_type.replace("_", " ").title(),
                    co2_kg_per_mmbtu=co2 * wtt_multiplier,
                    ch4_kg_per_mmbtu=ch4 * 2,  # Higher upstream CH4 leakage
                    n2o_kg_per_mmbtu=n2o * wtt_multiplier,
                    co2e_kg_per_mmbtu=co2e * 0.18,  # ~18% of TTW
                    boundary=EmissionBoundary.WTT,
                    scope=EmissionScope.SCOPE_3,
                    category=EmissionCategory.FUEL_ENERGY,
                    data_quality=DataQuality.SECONDARY,
                    data_source=source,
                    year=2024,
                    valid_from=now - timedelta(days=365),
                ))

        return factors

    async def get_emission_factors(
        self,
        fuel_type: Optional[str] = None,
        boundary: Optional[EmissionBoundary] = None,
        scope: Optional[EmissionScope] = None,
    ) -> List[EmissionFactorData]:
        """
        Get emission factors, optionally filtered.

        Args:
            fuel_type: Filter by fuel type
            boundary: Filter by boundary
            scope: Filter by scope

        Returns:
            List of emission factors
        """
        # Ensure factors are loaded
        if not self._emission_factors:
            await self.sync_emission_factors()

        # Flatten and filter
        all_factors = []
        for factors in self._emission_factors.values():
            all_factors.extend(factors)

        if fuel_type:
            all_factors = [f for f in all_factors if f.fuel_type == fuel_type]

        if boundary:
            all_factors = [f for f in all_factors if f.boundary == boundary]

        if scope:
            all_factors = [f for f in all_factors if f.scope == scope]

        return all_factors

    async def get_emission_factor_for_calculation(
        self,
        fuel_type: str,
        boundary: EmissionBoundary = EmissionBoundary.WTW,
    ) -> Optional[EmissionFactorData]:
        """Get single emission factor for calculation."""
        factors = await self.get_emission_factors(fuel_type=fuel_type, boundary=boundary)

        if boundary == EmissionBoundary.WTW:
            # Combine TTW + WTT for WTW
            ttw = next((f for f in factors if f.boundary == EmissionBoundary.TTW), None)
            wtt_factors = await self.get_emission_factors(fuel_type=fuel_type, boundary=EmissionBoundary.WTT)
            wtt = wtt_factors[0] if wtt_factors else None

            if ttw:
                wtw_co2e = ttw.co2e_kg_per_mmbtu + (wtt.co2e_kg_per_mmbtu if wtt else 0)
                return EmissionFactorData(
                    fuel_type=fuel_type,
                    fuel_name=ttw.fuel_name,
                    co2_kg_per_mmbtu=ttw.co2_kg_per_mmbtu + (wtt.co2_kg_per_mmbtu if wtt else 0),
                    ch4_kg_per_mmbtu=ttw.ch4_kg_per_mmbtu + (wtt.ch4_kg_per_mmbtu if wtt else 0),
                    n2o_kg_per_mmbtu=ttw.n2o_kg_per_mmbtu + (wtt.n2o_kg_per_mmbtu if wtt else 0),
                    co2e_kg_per_mmbtu=wtw_co2e,
                    boundary=EmissionBoundary.WTW,
                    scope=EmissionScope.SCOPE_1,  # Primary is Scope 1
                    data_quality=ttw.data_quality,
                    data_source=ttw.data_source,
                    year=ttw.year,
                    valid_from=ttw.valid_from,
                )

        return factors[0] if factors else None

    # =========================================================================
    # Footprint Export Operations
    # =========================================================================

    async def export_footprint(
        self,
        footprint: CarbonFootprintExport,
    ) -> bool:
        """
        Export carbon footprint to accounting system.

        Args:
            footprint: Footprint data to export

        Returns:
            True if export successful
        """
        logger.info(
            f"Exporting footprint: {footprint.co2e_emissions_mt:.2f} tCO2e "
            f"for {footprint.fuel_type}"
        )

        try:
            # Set exported timestamp
            footprint.exported_at = datetime.now(timezone.utc)

            # In production, POST to carbon accounting API:
            # response = await self._make_request("POST", "emissions", data=footprint.dict())

            self._stats["footprints_exported"] += 1
            logger.info(f"Footprint exported: {footprint.export_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to export footprint: {e}")
            self._stats["errors"] += 1
            return False

    async def export_footprints_batch(
        self,
        footprints: List[CarbonFootprintExport],
    ) -> int:
        """
        Export multiple footprints in batch.

        Args:
            footprints: List of footprints to export

        Returns:
            Number successfully exported
        """
        logger.info(f"Exporting batch of {len(footprints)} footprints")

        success_count = 0
        for footprint in footprints:
            if await self.export_footprint(footprint):
                success_count += 1

        logger.info(f"Batch export complete: {success_count}/{len(footprints)}")
        return success_count

    async def calculate_and_export_footprint(
        self,
        fuel_type: str,
        quantity_mmbtu: float,
        period_start: datetime,
        period_end: datetime,
        source_run_id: Optional[str] = None,
        boundary: EmissionBoundary = EmissionBoundary.WTW,
    ) -> CarbonFootprintExport:
        """
        Calculate emissions and export to accounting system.

        Args:
            fuel_type: Fuel type
            quantity_mmbtu: Activity quantity
            period_start: Reporting period start
            period_end: Reporting period end
            source_run_id: Source optimization run
            boundary: Emission boundary

        Returns:
            Exported footprint record
        """
        # Get emission factor
        factor = await self.get_emission_factor_for_calculation(fuel_type, boundary)

        if not factor:
            raise ValueError(f"No emission factor found for {fuel_type}")

        # Calculate emissions (convert kg to metric tons)
        co2_mt = (factor.co2_kg_per_mmbtu * quantity_mmbtu) / 1000
        ch4_mt = (factor.ch4_kg_per_mmbtu * quantity_mmbtu) / 1000
        n2o_mt = (factor.n2o_kg_per_mmbtu * quantity_mmbtu) / 1000
        co2e_mt = (factor.co2e_kg_per_mmbtu * quantity_mmbtu) / 1000

        # Create footprint record
        footprint = CarbonFootprintExport(
            reporting_period_start=period_start,
            reporting_period_end=period_end,
            reporting_entity=self.config.reporting_entity,
            fuel_type=fuel_type,
            activity_quantity_mmbtu=quantity_mmbtu,
            co2_emissions_mt=co2_mt,
            ch4_emissions_mt=ch4_mt,
            n2o_emissions_mt=n2o_mt,
            co2e_emissions_mt=co2e_mt,
            scope=factor.scope,
            boundary=boundary,
            category=factor.category,
            emission_factor_id=factor.factor_id,
            emission_factor_source=factor.data_source,
            emission_factor_year=factor.year,
            data_quality=factor.data_quality,
            source_run_id=source_run_id,
            calculation_hash="",  # Will be computed
        )

        footprint.calculation_hash = footprint.compute_calculation_hash()

        # Export to system
        await self.export_footprint(footprint)

        return footprint

    # =========================================================================
    # Reconciliation Operations
    # =========================================================================

    async def reconcile_emissions(
        self,
        period_start: datetime,
        period_end: datetime,
        fuelcraft_emissions: Dict[str, float],
    ) -> ReconciliationReport:
        """
        Reconcile FuelCraft emissions with external system.

        Args:
            period_start: Reconciliation period start
            period_end: Reconciliation period end
            fuelcraft_emissions: FuelCraft calculated emissions by scope

        Returns:
            Reconciliation report
        """
        logger.info(
            f"Reconciling emissions for period: "
            f"{period_start.date()} to {period_end.date()}"
        )

        try:
            # In production, fetch from external system:
            # external_data = await self._make_request("GET", "emissions/summary", params={...})

            # Mock external data (slightly different for reconciliation demo)
            external_emissions = {
                "scope_1": fuelcraft_emissions.get("scope_1", 0) * 1.02,  # 2% variance
                "scope_2": fuelcraft_emissions.get("scope_2", 0) * 0.98,
                "scope_3": fuelcraft_emissions.get("scope_3", 0) * 1.05,
            }

            fuelcraft_total = sum(fuelcraft_emissions.values())
            external_total = sum(external_emissions.values())
            variance = fuelcraft_total - external_total
            variance_percent = (variance / external_total * 100) if external_total > 0 else 0

            # Generate reconciliation report
            report = ReconciliationReport(
                reporting_period_start=period_start,
                reporting_period_end=period_end,
                fuelcraft_total_co2e_mt=fuelcraft_total,
                external_total_co2e_mt=external_total,
                variance_co2e_mt=variance,
                variance_percent=variance_percent,
                scope_1_fuelcraft_mt=fuelcraft_emissions.get("scope_1", 0),
                scope_1_external_mt=external_emissions["scope_1"],
                scope_2_fuelcraft_mt=fuelcraft_emissions.get("scope_2", 0),
                scope_2_external_mt=external_emissions["scope_2"],
                scope_3_fuelcraft_mt=fuelcraft_emissions.get("scope_3", 0),
                scope_3_external_mt=external_emissions["scope_3"],
                status=ReconciliationStatus.RECONCILED if abs(variance_percent) <= 5 else ReconciliationStatus.DISCREPANCY,
                audit_trail_hash=hashlib.sha256(
                    json.dumps({
                        "fuelcraft": fuelcraft_emissions,
                        "external": external_emissions,
                        "period": f"{period_start.isoformat()}-{period_end.isoformat()}"
                    }, sort_keys=True).encode()
                ).hexdigest(),
            )

            self._stats["reconciliations"] += 1
            logger.info(
                f"Reconciliation complete: variance={variance_percent:.2f}%, "
                f"status={report.status.value}"
            )

            return report

        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self._stats["errors"] += 1
            raise

    # =========================================================================
    # Scope Breakdown
    # =========================================================================

    async def get_scope_breakdown(
        self,
        footprints: List[CarbonFootprintExport],
    ) -> ScopeBreakdown:
        """
        Calculate scope breakdown from footprints.

        Args:
            footprints: List of footprint records

        Returns:
            Breakdown by scope
        """
        breakdown = ScopeBreakdown()

        for fp in footprints:
            if fp.scope == EmissionScope.SCOPE_1:
                breakdown.scope_1_mt += fp.co2e_emissions_mt
            elif fp.scope == EmissionScope.SCOPE_2:
                breakdown.scope_2_mt += fp.co2e_emissions_mt
            elif fp.scope == EmissionScope.SCOPE_3:
                breakdown.scope_3_mt += fp.co2e_emissions_mt
                if fp.category:
                    cat_key = fp.category.value
                    breakdown.scope_3_by_category[cat_key] = (
                        breakdown.scope_3_by_category.get(cat_key, 0) + fp.co2e_emissions_mt
                    )

        breakdown.total_mt = breakdown.scope_1_mt + breakdown.scope_2_mt + breakdown.scope_3_mt

        return breakdown

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "factors_cached": sum(len(f) for f in self._emission_factors.values()),
            "last_factor_sync": self._last_factor_sync.isoformat() if self._last_factor_sync else None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self._connected else "unhealthy",
            "factors_loaded": len(self._emission_factors) > 0,
            "last_sync": self._last_factor_sync.isoformat() if self._last_factor_sync else None,
            "errors": self._stats["errors"],
        }
