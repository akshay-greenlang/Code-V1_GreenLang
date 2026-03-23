"""
Purchased Goods & Services Agent - Provenance Tracking Module

This module provides SHA-256 provenance chain tracking for GL-MRV-S3-001 (AGENT-MRV-014).
Thread-safe singleton pattern ensures audit trail integrity across all procurement emissions operations.

Agent: GL-MRV-S3-001 (AGENT-MRV-014)
Purpose: Track spend-based, average-data, and supplier-specific calculation lineage
Regulatory: GHG Protocol Scope 3, ISO 14064-1, CSRD E1, CDP Supply Chain, SBTi
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal


class ProvenanceStage(str, Enum):
    """17 provenance stages for purchased goods & services emissions."""

    INPUT_VALIDATION = "input_validation"
    CLASSIFICATION_LOOKUP = "classification_lookup"
    CURRENCY_CONVERSION = "currency_conversion"
    INFLATION_ADJUSTMENT = "inflation_adjustment"
    MARGIN_REMOVAL = "margin_removal"
    EEIO_FACTOR_LOOKUP = "eeio_factor_lookup"
    SPEND_BASED_CALCULATION = "spend_based_calculation"
    PHYSICAL_EF_LOOKUP = "physical_ef_lookup"
    AVERAGE_DATA_CALCULATION = "average_data_calculation"
    SUPPLIER_DATA_COLLECTION = "supplier_data_collection"
    SUPPLIER_ALLOCATION = "supplier_allocation"
    HYBRID_AGGREGATION = "hybrid_aggregation"
    DQI_SCORING = "dqi_scoring"
    COVERAGE_ANALYSIS = "coverage_analysis"
    HOTSPOT_ANALYSIS = "hotspot_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    RESULT_FINALIZATION = "result_finalization"


@dataclass
class ProvenanceEntry:
    """Single provenance chain entry."""

    stage: ProvenanceStage
    timestamp: datetime
    input_hash: str
    output_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    chain_hash: Optional[str] = None


@dataclass
class ProvenanceChain:
    """Complete provenance chain for procurement emissions calculation."""

    calculation_id: str
    organization_id: str
    reporting_period: str
    created_at: datetime
    entries: List[ProvenanceEntry] = field(default_factory=list)
    sealed: bool = False
    seal_hash: Optional[str] = None
    seal_timestamp: Optional[datetime] = None


class PurchasedGoodsProvenanceTracker:
    """
    Thread-safe singleton provenance tracker for purchased goods & services emissions.

    Tracks 17-stage procurement emissions lineage with SHA-256 chain integrity.
    Supports multi-method calculations (spend-based, average-data, supplier-specific).

    Thread Safety:
        - Singleton with double-checked locking
        - Per-chain locks for concurrent operations
        - Atomic chain sealing

    Example:
        >>> tracker = PurchasedGoodsProvenanceTracker.get_instance()
        >>> chain_id = tracker.create_chain("CALC-001", "ORG-123", "2024")
        >>> tracker.add_stage(
        ...     chain_id,
        ...     ProvenanceStage.INPUT_VALIDATION,
        ...     {"item_count": 1500},
        ...     validation_result
        ... )
        >>> tracker.seal_chain(chain_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize provenance tracker (use get_instance instead)."""
        if PurchasedGoodsProvenanceTracker._instance is not None:
            raise RuntimeError("Use get_instance() to access singleton")

        self._chains: Dict[str, ProvenanceChain] = {}
        self._chain_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "PurchasedGoodsProvenanceTracker":
        """Get singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_chain(
        self,
        calculation_id: str,
        organization_id: str,
        reporting_period: str
    ) -> str:
        """
        Create new provenance chain.

        Args:
            calculation_id: Unique calculation identifier
            organization_id: Organization identifier
            reporting_period: Reporting period (YYYY or YYYY-QN)

        Returns:
            Chain ID (same as calculation_id)

        Raises:
            ValueError: If chain already exists
        """
        with self._global_lock:
            if calculation_id in self._chains:
                raise ValueError(f"Chain {calculation_id} already exists")

            chain = ProvenanceChain(
                calculation_id=calculation_id,
                organization_id=organization_id,
                reporting_period=reporting_period,
                created_at=datetime.utcnow()
            )

            self._chains[calculation_id] = chain
            self._chain_locks[calculation_id] = threading.Lock()

        return calculation_id

    def add_stage(
        self,
        chain_id: str,
        stage: ProvenanceStage,
        metadata: Dict[str, Any],
        output_data: Any
    ) -> str:
        """
        Add stage to provenance chain.

        Args:
            chain_id: Chain identifier
            stage: Provenance stage
            metadata: Stage metadata
            output_data: Output data to hash

        Returns:
            Chain hash after adding stage

        Raises:
            ValueError: If chain not found or sealed
        """
        if chain_id not in self._chains:
            raise ValueError(f"Chain {chain_id} not found")

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if chain.sealed:
                raise ValueError(f"Chain {chain_id} is sealed")

            # Get previous hash
            previous_hash = None
            if chain.entries:
                previous_hash = chain.entries[-1].chain_hash

            # Calculate input hash (previous chain hash + stage + metadata)
            input_data = {
                "previous_hash": previous_hash,
                "stage": stage.value,
                "metadata": metadata
            }
            input_hash = self._hash_dict(input_data)

            # Calculate output hash
            output_hash = self._hash_data(output_data)

            # Calculate chain hash (input + output)
            chain_data = {
                "input_hash": input_hash,
                "output_hash": output_hash
            }
            chain_hash = self._hash_dict(chain_data)

            # Create entry
            entry = ProvenanceEntry(
                stage=stage,
                timestamp=datetime.utcnow(),
                input_hash=input_hash,
                output_hash=output_hash,
                metadata=metadata,
                previous_hash=previous_hash,
                chain_hash=chain_hash
            )

            chain.entries.append(entry)

        return chain_hash

    def seal_chain(self, chain_id: str) -> str:
        """
        Seal provenance chain (immutable after sealing).

        Args:
            chain_id: Chain identifier

        Returns:
            Final seal hash

        Raises:
            ValueError: If chain not found or already sealed
        """
        if chain_id not in self._chains:
            raise ValueError(f"Chain {chain_id} not found")

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if chain.sealed:
                raise ValueError(f"Chain {chain_id} already sealed")

            if not chain.entries:
                raise ValueError(f"Cannot seal empty chain {chain_id}")

            # Calculate seal hash (all entry hashes + metadata)
            seal_data = {
                "calculation_id": chain.calculation_id,
                "organization_id": chain.organization_id,
                "reporting_period": chain.reporting_period,
                "created_at": chain.created_at.isoformat(),
                "entry_count": len(chain.entries),
                "entry_hashes": [e.chain_hash for e in chain.entries]
            }

            seal_hash = self._hash_dict(seal_data)

            chain.sealed = True
            chain.seal_hash = seal_hash
            chain.seal_timestamp = datetime.utcnow()

        return seal_hash

    def verify_chain(self, chain_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify provenance chain integrity.

        Args:
            chain_id: Chain identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        if chain_id not in self._chains:
            return False, f"Chain {chain_id} not found"

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]

            if not chain.entries:
                return False, "Chain has no entries"

            # Verify each entry's chain hash
            previous_hash = None
            for i, entry in enumerate(chain.entries):
                # Verify previous hash linkage
                if entry.previous_hash != previous_hash:
                    return False, f"Entry {i} previous hash mismatch"

                # Recalculate input hash
                input_data = {
                    "previous_hash": previous_hash,
                    "stage": entry.stage.value,
                    "metadata": entry.metadata
                }
                expected_input_hash = self._hash_dict(input_data)

                if entry.input_hash != expected_input_hash:
                    return False, f"Entry {i} input hash mismatch"

                # Recalculate chain hash
                chain_data = {
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash
                }
                expected_chain_hash = self._hash_dict(chain_data)

                if entry.chain_hash != expected_chain_hash:
                    return False, f"Entry {i} chain hash mismatch"

                previous_hash = entry.chain_hash

            # Verify seal hash if sealed
            if chain.sealed:
                seal_data = {
                    "calculation_id": chain.calculation_id,
                    "organization_id": chain.organization_id,
                    "reporting_period": chain.reporting_period,
                    "created_at": chain.created_at.isoformat(),
                    "entry_count": len(chain.entries),
                    "entry_hashes": [e.chain_hash for e in chain.entries]
                }
                expected_seal_hash = self._hash_dict(seal_data)

                if chain.seal_hash != expected_seal_hash:
                    return False, "Seal hash mismatch"

        return True, None

    def get_chain(self, chain_id: str) -> Optional[ProvenanceChain]:
        """Get provenance chain (returns copy for thread safety)."""
        if chain_id not in self._chains:
            return None

        with self._chain_locks[chain_id]:
            chain = self._chains[chain_id]
            # Return deep copy to prevent external modification
            return ProvenanceChain(
                calculation_id=chain.calculation_id,
                organization_id=chain.organization_id,
                reporting_period=chain.reporting_period,
                created_at=chain.created_at,
                entries=list(chain.entries),
                sealed=chain.sealed,
                seal_hash=chain.seal_hash,
                seal_timestamp=chain.seal_timestamp
            )

    def export_chain(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Export provenance chain to JSON-serializable dict."""
        chain = self.get_chain(chain_id)
        if not chain:
            return None

        return {
            "calculation_id": chain.calculation_id,
            "organization_id": chain.organization_id,
            "reporting_period": chain.reporting_period,
            "created_at": chain.created_at.isoformat(),
            "sealed": chain.sealed,
            "seal_hash": chain.seal_hash,
            "seal_timestamp": chain.seal_timestamp.isoformat() if chain.seal_timestamp else None,
            "entries": [
                {
                    "stage": entry.stage.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                    "metadata": entry.metadata,
                    "previous_hash": entry.previous_hash,
                    "chain_hash": entry.chain_hash
                }
                for entry in chain.entries
            ]
        }

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Hash dictionary using SHA-256."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _hash_data(self, data: Any) -> str:
        """Hash arbitrary data using SHA-256."""
        if isinstance(data, dict):
            return self._hash_dict(data)
        elif isinstance(data, (list, tuple)):
            return self._hash_dict({"items": data})
        elif isinstance(data, Decimal):
            return hashlib.sha256(str(data).encode()).hexdigest()
        elif isinstance(data, datetime):
            return hashlib.sha256(data.isoformat().encode()).hexdigest()
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()


# ============================================================================
# Hash Helper Functions
# ============================================================================


def hash_procurement_item(
    item_id: str,
    description: str,
    category: str,
    spend_amount: Decimal,
    currency: str,
    quantity: Optional[Decimal],
    unit: Optional[str]
) -> str:
    """
    Hash procurement item input.

    Args:
        item_id: Unique item identifier
        description: Item description
        category: Procurement category
        spend_amount: Spend amount in original currency
        currency: Currency code (ISO 4217)
        quantity: Physical quantity (optional)
        unit: Physical unit (optional)

    Returns:
        SHA-256 hash of procurement item
    """
    data = {
        "item_id": item_id,
        "description": description,
        "category": category,
        "spend_amount": str(spend_amount),
        "currency": currency,
        "quantity": str(quantity) if quantity else None,
        "unit": unit
    }
    return _hash_helper(data)


def hash_spend_record(
    item_id: str,
    spend_amount_base: Decimal,
    currency_base: str,
    fx_rate: Decimal,
    cpi_factor: Decimal,
    margin_factor: Decimal
) -> str:
    """
    Hash spend-based calculation record.

    Args:
        item_id: Item identifier
        spend_amount_base: Spend amount in base currency
        currency_base: Base currency (e.g., USD)
        fx_rate: FX rate applied
        cpi_factor: CPI deflation factor
        margin_factor: Producer price margin factor

    Returns:
        SHA-256 hash of spend record
    """
    data = {
        "item_id": item_id,
        "spend_amount_base": str(spend_amount_base),
        "currency_base": currency_base,
        "fx_rate": str(fx_rate),
        "cpi_factor": str(cpi_factor),
        "margin_factor": str(margin_factor)
    }
    return _hash_helper(data)


def hash_physical_record(
    item_id: str,
    quantity: Decimal,
    unit: str,
    product_type: str,
    region: str
) -> str:
    """
    Hash physical activity data record.

    Args:
        item_id: Item identifier
        quantity: Physical quantity
        unit: Physical unit
        product_type: Product classification
        region: Geographic region

    Returns:
        SHA-256 hash of physical record
    """
    data = {
        "item_id": item_id,
        "quantity": str(quantity),
        "unit": unit,
        "product_type": product_type,
        "region": region
    }
    return _hash_helper(data)


def hash_supplier_record(
    supplier_id: str,
    supplier_name: str,
    total_emissions: Decimal,
    allocation_method: str,
    allocation_denominator: Decimal
) -> str:
    """
    Hash supplier-specific emissions record.

    Args:
        supplier_id: Supplier identifier
        supplier_name: Supplier name
        total_emissions: Total supplier emissions (tCO2e)
        allocation_method: Allocation method (revenue/mass/economic)
        allocation_denominator: Allocation denominator value

    Returns:
        SHA-256 hash of supplier record
    """
    data = {
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "total_emissions": str(total_emissions),
        "allocation_method": allocation_method,
        "allocation_denominator": str(allocation_denominator)
    }
    return _hash_helper(data)


def hash_eeio_factor(
    sector_code: str,
    classification_system: str,
    emission_factor: Decimal,
    unit: str,
    source: str,
    version: str,
    region: str
) -> str:
    """
    Hash EEIO emission factor.

    Args:
        sector_code: Sector classification code (NAICS/NACE/UNSPSC)
        classification_system: Classification system name
        emission_factor: Emission factor (kgCO2e/currency)
        unit: Factor unit
        source: Data source (USEEIO/EXIOBASE/DEFRA)
        version: Source version
        region: Geographic region

    Returns:
        SHA-256 hash of EEIO factor
    """
    data = {
        "sector_code": sector_code,
        "classification_system": classification_system,
        "emission_factor": str(emission_factor),
        "unit": unit,
        "source": source,
        "version": version,
        "region": region
    }
    return _hash_helper(data)


def hash_physical_ef(
    product_code: str,
    emission_factor: Decimal,
    unit: str,
    scope: str,
    source: str,
    version: str,
    region: str
) -> str:
    """
    Hash physical emission factor.

    Args:
        product_code: Product classification code
        emission_factor: Emission factor (kgCO2e/unit)
        unit: Physical unit (kg/tonne/m3/etc.)
        scope: Scope coverage (upstream/cradle-to-gate)
        source: Data source (GaBi/Ecoinvent/DEFRA)
        version: Source version
        region: Geographic region

    Returns:
        SHA-256 hash of physical emission factor
    """
    data = {
        "product_code": product_code,
        "emission_factor": str(emission_factor),
        "unit": unit,
        "scope": scope,
        "source": source,
        "version": version,
        "region": region
    }
    return _hash_helper(data)


def hash_supplier_ef(
    supplier_id: str,
    allocation_factor: Decimal,
    allocation_method: str,
    numerator: Decimal,
    denominator: Decimal,
    year: int
) -> str:
    """
    Hash supplier-specific emission factor.

    Args:
        supplier_id: Supplier identifier
        allocation_factor: Calculated allocation factor (0-1)
        allocation_method: Allocation method used
        numerator: Allocation numerator (e.g., revenue from org)
        denominator: Allocation denominator (e.g., total revenue)
        year: Data year

    Returns:
        SHA-256 hash of supplier emission factor
    """
    data = {
        "supplier_id": supplier_id,
        "allocation_factor": str(allocation_factor),
        "allocation_method": allocation_method,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "year": year
    }
    return _hash_helper(data)


def hash_currency_rate(
    from_currency: str,
    to_currency: str,
    rate: Decimal,
    date: str,
    source: str
) -> str:
    """
    Hash currency exchange rate.

    Args:
        from_currency: Source currency (ISO 4217)
        to_currency: Target currency (ISO 4217)
        rate: Exchange rate
        date: Rate date (YYYY-MM-DD)
        source: Rate source (e.g., "ECB", "Federal Reserve")

    Returns:
        SHA-256 hash of currency rate
    """
    data = {
        "from_currency": from_currency,
        "to_currency": to_currency,
        "rate": str(rate),
        "date": date,
        "source": source
    }
    return _hash_helper(data)


def hash_margin_factor(
    sector_code: str,
    margin_percentage: Decimal,
    source: str,
    year: int
) -> str:
    """
    Hash producer price margin factor.

    Args:
        sector_code: Sector classification code
        margin_percentage: Margin percentage (%)
        source: Data source
        year: Data year

    Returns:
        SHA-256 hash of margin factor
    """
    data = {
        "sector_code": sector_code,
        "margin_percentage": str(margin_percentage),
        "source": source,
        "year": year
    }
    return _hash_helper(data)


def hash_spend_result(
    item_id: str,
    spend_adjusted: Decimal,
    eeio_factor: Decimal,
    emissions: Decimal,
    co2: Decimal,
    ch4: Decimal,
    n2o: Decimal
) -> str:
    """
    Hash spend-based calculation result.

    Args:
        item_id: Item identifier
        spend_adjusted: Adjusted spend amount (base currency, deflated, margin-removed)
        eeio_factor: EEIO emission factor applied
        emissions: Total emissions (tCO2e)
        co2: CO2 emissions (tCO2)
        ch4: CH4 emissions (tCO2e)
        n2o: N2O emissions (tCO2e)

    Returns:
        SHA-256 hash of spend-based result
    """
    data = {
        "item_id": item_id,
        "spend_adjusted": str(spend_adjusted),
        "eeio_factor": str(eeio_factor),
        "emissions": str(emissions),
        "co2": str(co2),
        "ch4": str(ch4),
        "n2o": str(n2o)
    }
    return _hash_helper(data)


def hash_average_data_result(
    item_id: str,
    quantity: Decimal,
    unit: str,
    ef: Decimal,
    emissions: Decimal,
    co2: Decimal,
    ch4: Decimal,
    n2o: Decimal
) -> str:
    """
    Hash average-data calculation result.

    Args:
        item_id: Item identifier
        quantity: Physical quantity
        unit: Physical unit
        ef: Physical emission factor applied
        emissions: Total emissions (tCO2e)
        co2: CO2 emissions (tCO2)
        ch4: CH4 emissions (tCO2e)
        n2o: N2O emissions (tCO2e)

    Returns:
        SHA-256 hash of average-data result
    """
    data = {
        "item_id": item_id,
        "quantity": str(quantity),
        "unit": unit,
        "ef": str(ef),
        "emissions": str(emissions),
        "co2": str(co2),
        "ch4": str(ch4),
        "n2o": str(n2o)
    }
    return _hash_helper(data)


def hash_supplier_result(
    supplier_id: str,
    total_emissions: Decimal,
    allocation_factor: Decimal,
    allocated_emissions: Decimal,
    co2: Decimal,
    ch4: Decimal,
    n2o: Decimal
) -> str:
    """
    Hash supplier-specific calculation result.

    Args:
        supplier_id: Supplier identifier
        total_emissions: Total supplier emissions (tCO2e)
        allocation_factor: Allocation factor applied
        allocated_emissions: Allocated emissions (tCO2e)
        co2: CO2 emissions (tCO2)
        ch4: CH4 emissions (tCO2e)
        n2o: N2O emissions (tCO2e)

    Returns:
        SHA-256 hash of supplier-specific result
    """
    data = {
        "supplier_id": supplier_id,
        "total_emissions": str(total_emissions),
        "allocation_factor": str(allocation_factor),
        "allocated_emissions": str(allocated_emissions),
        "co2": str(co2),
        "ch4": str(ch4),
        "n2o": str(n2o)
    }
    return _hash_helper(data)


def hash_hybrid_result(
    item_id: str,
    method_used: str,
    emissions: Decimal,
    data_quality_score: Decimal,
    coverage_percentage: Decimal
) -> str:
    """
    Hash hybrid method aggregation result.

    Args:
        item_id: Item identifier
        method_used: Primary method selected (spend/average/supplier)
        emissions: Total emissions (tCO2e)
        data_quality_score: Data quality score (0-5)
        coverage_percentage: Coverage percentage (%)

    Returns:
        SHA-256 hash of hybrid result
    """
    data = {
        "item_id": item_id,
        "method_used": method_used,
        "emissions": str(emissions),
        "data_quality_score": str(data_quality_score),
        "coverage_percentage": str(coverage_percentage)
    }
    return _hash_helper(data)


def hash_dqi_assessment(
    item_id: str,
    method: str,
    technology_score: int,
    temporal_score: int,
    geography_score: int,
    completeness_score: int,
    reliability_score: int,
    composite_score: Decimal
) -> str:
    """
    Hash data quality indicator assessment.

    Args:
        item_id: Item identifier
        method: Calculation method
        technology_score: Technology representativeness (1-5)
        temporal_score: Temporal representativeness (1-5)
        geography_score: Geographic representativeness (1-5)
        completeness_score: Completeness (1-5)
        reliability_score: Reliability (1-5)
        composite_score: Composite DQI score (1-5)

    Returns:
        SHA-256 hash of DQI assessment
    """
    data = {
        "item_id": item_id,
        "method": method,
        "technology_score": technology_score,
        "temporal_score": temporal_score,
        "geography_score": geography_score,
        "completeness_score": completeness_score,
        "reliability_score": reliability_score,
        "composite_score": str(composite_score)
    }
    return _hash_helper(data)


def hash_coverage_report(
    total_items: int,
    spend_based_count: int,
    average_data_count: int,
    supplier_specific_count: int,
    spend_coverage_pct: Decimal,
    emissions_coverage_pct: Decimal
) -> str:
    """
    Hash coverage analysis report.

    Args:
        total_items: Total procurement items
        spend_based_count: Items using spend-based method
        average_data_count: Items using average-data method
        supplier_specific_count: Items using supplier-specific method
        spend_coverage_pct: Spend coverage percentage (%)
        emissions_coverage_pct: Emissions coverage percentage (%)

    Returns:
        SHA-256 hash of coverage report
    """
    data = {
        "total_items": total_items,
        "spend_based_count": spend_based_count,
        "average_data_count": average_data_count,
        "supplier_specific_count": supplier_specific_count,
        "spend_coverage_pct": str(spend_coverage_pct),
        "emissions_coverage_pct": str(emissions_coverage_pct)
    }
    return _hash_helper(data)


def hash_hotspot_analysis(
    ranking_method: str,
    top_n: int,
    pareto_threshold: Decimal,
    hotspot_count: int,
    hotspot_emissions_pct: Decimal
) -> str:
    """
    Hash hotspot analysis result.

    Args:
        ranking_method: Ranking method (emissions/spend/intensity)
        top_n: Top N items analyzed
        pareto_threshold: Pareto threshold percentage (%)
        hotspot_count: Number of hotspot items identified
        hotspot_emissions_pct: Hotspot emissions percentage (%)

    Returns:
        SHA-256 hash of hotspot analysis
    """
    data = {
        "ranking_method": ranking_method,
        "top_n": top_n,
        "pareto_threshold": str(pareto_threshold),
        "hotspot_count": hotspot_count,
        "hotspot_emissions_pct": str(hotspot_emissions_pct)
    }
    return _hash_helper(data)


def hash_compliance_result(
    framework: str,
    status: str,
    requirements_met: int,
    requirements_total: int,
    findings: List[str]
) -> str:
    """
    Hash compliance check result.

    Args:
        framework: Regulatory framework (GHG_PROTOCOL/ISO_14064/CSRD/CDP)
        status: Compliance status (PASS/FAIL/PARTIAL)
        requirements_met: Number of requirements met
        requirements_total: Total number of requirements
        findings: Compliance findings

    Returns:
        SHA-256 hash of compliance result
    """
    data = {
        "framework": framework,
        "status": status,
        "requirements_met": requirements_met,
        "requirements_total": requirements_total,
        "findings": findings
    }
    return _hash_helper(data)


def hash_calculation_result(
    calculation_id: str,
    total_emissions: Decimal,
    co2_total: Decimal,
    ch4_total: Decimal,
    n2o_total: Decimal,
    item_count: int,
    spend_total: Decimal,
    dqi_average: Decimal
) -> str:
    """
    Hash final calculation result.

    Args:
        calculation_id: Calculation identifier
        total_emissions: Total emissions (tCO2e)
        co2_total: Total CO2 emissions (tCO2)
        ch4_total: Total CH4 emissions (tCO2e)
        n2o_total: Total N2O emissions (tCO2e)
        item_count: Total items calculated
        spend_total: Total spend amount
        dqi_average: Average data quality score

    Returns:
        SHA-256 hash of calculation result
    """
    data = {
        "calculation_id": calculation_id,
        "total_emissions": str(total_emissions),
        "co2_total": str(co2_total),
        "ch4_total": str(ch4_total),
        "n2o_total": str(n2o_total),
        "item_count": item_count,
        "spend_total": str(spend_total),
        "dqi_average": str(dqi_average)
    }
    return _hash_helper(data)


def hash_batch_result(
    batch_id: str,
    total_calculations: int,
    completed: int,
    failed: int,
    total_emissions: Decimal
) -> str:
    """
    Hash batch calculation result.

    Args:
        batch_id: Batch identifier
        total_calculations: Total number of calculations
        completed: Number of completed calculations
        failed: Number of failed calculations
        total_emissions: Total emissions across batch (tCO2e)

    Returns:
        SHA-256 hash of batch result
    """
    data = {
        "batch_id": batch_id,
        "total_calculations": total_calculations,
        "completed": completed,
        "failed": failed,
        "total_emissions": str(total_emissions)
    }
    return _hash_helper(data)


def hash_aggregation_result(
    group_by: str,
    group_count: int,
    total_emissions: Decimal,
    aggregation_method: str
) -> str:
    """
    Hash aggregation result.

    Args:
        group_by: Grouping dimension (category/supplier/region)
        group_count: Number of groups
        total_emissions: Total aggregated emissions (tCO2e)
        aggregation_method: Aggregation method (sum/average/weighted)

    Returns:
        SHA-256 hash of aggregation result
    """
    data = {
        "group_by": group_by,
        "group_count": group_count,
        "total_emissions": str(total_emissions),
        "aggregation_method": aggregation_method
    }
    return _hash_helper(data)


def hash_pipeline_context(
    calculation_id: str,
    organization_id: str,
    reporting_period: str,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: int
) -> str:
    """
    Hash pipeline execution context.

    Args:
        calculation_id: Calculation identifier
        organization_id: Organization identifier
        reporting_period: Reporting period
        started_at: Pipeline start timestamp
        completed_at: Pipeline completion timestamp
        duration_ms: Pipeline duration (milliseconds)

    Returns:
        SHA-256 hash of pipeline context
    """
    data = {
        "calculation_id": calculation_id,
        "organization_id": organization_id,
        "reporting_period": reporting_period,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": duration_ms
    }
    return _hash_helper(data)


def _hash_helper(data: Dict[str, Any]) -> str:
    """Internal helper for SHA-256 hashing."""
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# ============================================================================
# Batch Provenance Tracking
# ============================================================================


class BatchProvenanceTracker:
    """
    Track provenance for multi-period batch calculations.

    Manages per-period chains with batch-level aggregation.
    """

    def __init__(self):
        """Initialize batch provenance tracker."""
        self.tracker = PurchasedGoodsProvenanceTracker.get_instance()
        self._batch_chains: Dict[str, List[str]] = {}
        self._batch_lock = threading.Lock()

    def create_batch(self, batch_id: str) -> str:
        """
        Create batch provenance container.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch ID
        """
        with self._batch_lock:
            if batch_id in self._batch_chains:
                raise ValueError(f"Batch {batch_id} already exists")

            self._batch_chains[batch_id] = []

        return batch_id

    def add_period_chain(self, batch_id: str, chain_id: str) -> None:
        """
        Add period chain to batch.

        Args:
            batch_id: Batch identifier
            chain_id: Period chain identifier
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                raise ValueError(f"Batch {batch_id} not found")

            self._batch_chains[batch_id].append(chain_id)

    def seal_batch(self, batch_id: str) -> str:
        """
        Seal batch provenance (seals all period chains).

        Args:
            batch_id: Batch identifier

        Returns:
            Batch seal hash
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                raise ValueError(f"Batch {batch_id} not found")

            chain_ids = self._batch_chains[batch_id]

            # Seal all period chains
            seal_hashes = []
            for chain_id in chain_ids:
                try:
                    seal_hash = self.tracker.seal_chain(chain_id)
                    seal_hashes.append(seal_hash)
                except ValueError:
                    # Chain already sealed
                    chain = self.tracker.get_chain(chain_id)
                    if chain and chain.seal_hash:
                        seal_hashes.append(chain.seal_hash)

            # Calculate batch seal hash
            batch_data = {
                "batch_id": batch_id,
                "period_count": len(chain_ids),
                "period_seal_hashes": seal_hashes
            }

            batch_seal_hash = _hash_helper(batch_data)

        return batch_seal_hash

    def verify_batch(self, batch_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verify batch provenance integrity.

        Args:
            batch_id: Batch identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                return False, f"Batch {batch_id} not found"

            chain_ids = self._batch_chains[batch_id]

            # Verify each period chain
            for chain_id in chain_ids:
                is_valid, error = self.tracker.verify_chain(chain_id)
                if not is_valid:
                    return False, f"Period {chain_id}: {error}"

        return True, None

    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch provenance summary.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch summary dict or None if not found
        """
        with self._batch_lock:
            if batch_id not in self._batch_chains:
                return None

            chain_ids = self._batch_chains[batch_id]

            # Collect period summaries
            period_summaries = []
            for chain_id in chain_ids:
                chain = self.tracker.get_chain(chain_id)
                if chain:
                    period_summaries.append({
                        "calculation_id": chain.calculation_id,
                        "reporting_period": chain.reporting_period,
                        "stage_count": len(chain.entries),
                        "sealed": chain.sealed,
                        "seal_hash": chain.seal_hash
                    })

            return {
                "batch_id": batch_id,
                "period_count": len(chain_ids),
                "periods": period_summaries
            }


# ============================================================================
# Provenance Query Utilities
# ============================================================================


def get_stage_hash(chain_id: str, stage: ProvenanceStage) -> Optional[str]:
    """Get hash for specific stage in chain."""
    tracker = PurchasedGoodsProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    for entry in chain.entries:
        if entry.stage == stage:
            return entry.chain_hash

    return None


def get_stage_metadata(chain_id: str, stage: ProvenanceStage) -> Optional[Dict[str, Any]]:
    """Get metadata for specific stage in chain."""
    tracker = PurchasedGoodsProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    for entry in chain.entries:
        if entry.stage == stage:
            return entry.metadata

    return None


def get_chain_summary(chain_id: str) -> Optional[Dict[str, Any]]:
    """Get chain summary with stage progression."""
    tracker = PurchasedGoodsProvenanceTracker.get_instance()
    chain = tracker.get_chain(chain_id)

    if not chain:
        return None

    return {
        "calculation_id": chain.calculation_id,
        "organization_id": chain.organization_id,
        "reporting_period": chain.reporting_period,
        "created_at": chain.created_at.isoformat(),
        "stage_count": len(chain.entries),
        "stages_completed": [entry.stage.value for entry in chain.entries],
        "sealed": chain.sealed,
        "seal_hash": chain.seal_hash,
        "seal_timestamp": chain.seal_timestamp.isoformat() if chain.seal_timestamp else None
    }


def verify_upstream_hash(
    chain_id: str,
    stage: ProvenanceStage,
    expected_hash: str
) -> bool:
    """Verify upstream hash matches chain entry."""
    actual_hash = get_stage_hash(chain_id, stage)
    return actual_hash == expected_hash if actual_hash else False


# ============================================================================
# Classification Provenance Helpers
# ============================================================================


def hash_naics_classification(
    item_description: str,
    naics_code: str,
    naics_title: str,
    confidence_score: Decimal,
    classification_method: str
) -> str:
    """
    Hash NAICS classification result.

    Args:
        item_description: Item description text
        naics_code: NAICS code assigned
        naics_title: NAICS code title
        confidence_score: Classification confidence (0-1)
        classification_method: Method used (rule/ml/manual)

    Returns:
        SHA-256 hash of NAICS classification
    """
    data = {
        "item_description": item_description,
        "naics_code": naics_code,
        "naics_title": naics_title,
        "confidence_score": str(confidence_score),
        "classification_method": classification_method
    }
    return _hash_helper(data)


def hash_nace_classification(
    item_description: str,
    nace_code: str,
    nace_title: str,
    confidence_score: Decimal,
    classification_method: str
) -> str:
    """
    Hash NACE classification result.

    Args:
        item_description: Item description text
        nace_code: NACE code assigned
        nace_title: NACE code title
        confidence_score: Classification confidence (0-1)
        classification_method: Method used (rule/ml/manual)

    Returns:
        SHA-256 hash of NACE classification
    """
    data = {
        "item_description": item_description,
        "nace_code": nace_code,
        "nace_title": nace_title,
        "confidence_score": str(confidence_score),
        "classification_method": classification_method
    }
    return _hash_helper(data)


def hash_unspsc_classification(
    item_description: str,
    unspsc_code: str,
    unspsc_title: str,
    confidence_score: Decimal,
    classification_method: str
) -> str:
    """
    Hash UNSPSC classification result.

    Args:
        item_description: Item description text
        unspsc_code: UNSPSC code assigned
        unspsc_title: UNSPSC code title
        confidence_score: Classification confidence (0-1)
        classification_method: Method used (rule/ml/manual)

    Returns:
        SHA-256 hash of UNSPSC classification
    """
    data = {
        "item_description": item_description,
        "unspsc_code": unspsc_code,
        "unspsc_title": unspsc_title,
        "confidence_score": str(confidence_score),
        "classification_method": classification_method
    }
    return _hash_helper(data)


def hash_cpi_adjustment(
    base_year: int,
    current_year: int,
    cpi_base: Decimal,
    cpi_current: Decimal,
    deflation_factor: Decimal,
    source: str
) -> str:
    """
    Hash CPI inflation adjustment.

    Args:
        base_year: Base year for deflation
        current_year: Current data year
        cpi_base: CPI for base year
        cpi_current: CPI for current year
        deflation_factor: Calculated deflation factor
        source: CPI data source

    Returns:
        SHA-256 hash of CPI adjustment
    """
    data = {
        "base_year": base_year,
        "current_year": current_year,
        "cpi_base": str(cpi_base),
        "cpi_current": str(cpi_current),
        "deflation_factor": str(deflation_factor),
        "source": source
    }
    return _hash_helper(data)


def hash_method_selection(
    item_id: str,
    available_methods: List[str],
    selected_method: str,
    selection_criteria: str,
    dqi_comparison: Dict[str, Decimal]
) -> str:
    """
    Hash hybrid method selection.

    Args:
        item_id: Item identifier
        available_methods: Methods available for item
        selected_method: Method selected
        selection_criteria: Selection criteria used
        dqi_comparison: DQI scores by method

    Returns:
        SHA-256 hash of method selection
    """
    data = {
        "item_id": item_id,
        "available_methods": available_methods,
        "selected_method": selected_method,
        "selection_criteria": selection_criteria,
        "dqi_comparison": {k: str(v) for k, v in dqi_comparison.items()}
    }
    return _hash_helper(data)


def hash_uncertainty_analysis(
    item_id: str,
    emissions_mean: Decimal,
    emissions_std_dev: Decimal,
    confidence_interval_95: Tuple[Decimal, Decimal],
    uncertainty_sources: List[str]
) -> str:
    """
    Hash uncertainty analysis result.

    Args:
        item_id: Item identifier
        emissions_mean: Mean emissions estimate (tCO2e)
        emissions_std_dev: Standard deviation (tCO2e)
        confidence_interval_95: 95% confidence interval (lower, upper)
        uncertainty_sources: Sources of uncertainty

    Returns:
        SHA-256 hash of uncertainty analysis
    """
    data = {
        "item_id": item_id,
        "emissions_mean": str(emissions_mean),
        "emissions_std_dev": str(emissions_std_dev),
        "confidence_interval_95_lower": str(confidence_interval_95[0]),
        "confidence_interval_95_upper": str(confidence_interval_95[1]),
        "uncertainty_sources": uncertainty_sources
    }
    return _hash_helper(data)


def hash_validation_result(
    validation_type: str,
    passed: bool,
    errors: List[str],
    warnings: List[str],
    item_count_checked: int
) -> str:
    """
    Hash input validation result.

    Args:
        validation_type: Type of validation (schema/business/data)
        passed: Whether validation passed
        errors: Validation errors
        warnings: Validation warnings
        item_count_checked: Number of items checked

    Returns:
        SHA-256 hash of validation result
    """
    data = {
        "validation_type": validation_type,
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "item_count_checked": item_count_checked
    }
    return _hash_helper(data)


# ============================================================================
# Alias for backward compatibility
# ============================================================================

#: Alias used by the pipeline engine, setup.py, and __init__.py.
PurchasedGoodsServicesProvenance = PurchasedGoodsProvenanceTracker
