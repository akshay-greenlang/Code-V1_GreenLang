# Data Engineering Team Charter

**Version:** 1.0
**Date:** 2025-12-03
**Team:** Data Engineering
**Tech Lead:** TBD
**Headcount:** 3-4 engineers

---

## Team Mission

Build robust data pipelines, contracts, and quality frameworks that ensure agents ingest, process, and output data with 100% accuracy, traceability, and regulatory compliance across all climate regulations.

**Core Principle:** Data quality is non-negotiable - garbage in, garbage out prevention through rigorous validation.

---

## Team Mandate

The Data Engineering Team owns the entire data lifecycle:

1. **Data Contracts:** Standardized schemas for all agent inputs and outputs
2. **Data Pipelines:** ETL/ELT pipelines for agent data flows
3. **Data Quality:** Validation, profiling, and monitoring frameworks
4. **Data Provenance:** Complete lineage tracking from source to output

**Non-Goals:**
- Agent business logic (AI/Agent Team owns this)
- ML model data (ML Platform Team owns this)
- Production infrastructure (DevOps Team owns this)

---

## Team Composition

### Roles & Responsibilities

**Tech Lead (1):**
- Data architecture and contracts
- Pipeline orchestration strategy
- Cross-team coordination (all teams)
- Data quality standards

**Data Engineers (2-3):**
- Pipeline development (Airflow, Prefect)
- Data contract implementation
- ETL/ELT development
- Data warehouse design

**Data Quality Engineer (1):**
- Quality framework development
- Data profiling and validation
- Anomaly detection
- Data observability

---

## Core Responsibilities

### 1. Data Contracts (Schema Standardization)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **CBAM Data Contracts** | Schemas for shipment data, emissions, reports | Phase 1 |
| **EUDR Data Contracts** | Schemas for geolocation, due diligence, risk | Phase 2 |
| **CSRD Data Contracts** | Schemas for ESG metrics, disclosures | Phase 2 |
| **Contract Validator** | Tool to validate data against contracts | Phase 1 |
| **Contract Registry** | Centralized repository for all contracts | Phase 2 |
| **Contract Versioning** | Semantic versioning with backward compatibility | Phase 2 |

**Technical Specifications:**

**Data Contract Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                  Data Contract Registry                  │
│  • Schema definitions (JSON Schema, Avro)               │
│  • Versioning and backward compatibility                │
│  • Contract validation and enforcement                  │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────▼────────┐ ┌───▼────────┐ ┌───▼────────┐
│  CBAM Contracts  │ │   EUDR     │ │   CSRD     │
│                  │ │ Contracts  │ │ Contracts  │
│ • Shipments      │ │ • Products │ │ • ESG Data │
│ • Emissions      │ │ • Geo Data │ │ • Reports  │
│ • Reports        │ │ • Due Dil. │ │ • KPIs     │
└──────────────────┘ └────────────┘ └────────────┘
```

**CBAM Data Contract Example:**
```yaml
# contracts/cbam/shipment_v1.0.0.yaml

contract:
  name: "cbam_shipment"
  version: "1.0.0"
  description: "CBAM shipment data for embedded emissions calculation"
  effective_date: "2025-12-03"
  owner: "data_engineering_team"

schema:
  type: "object"
  required:
    - shipment_id
    - cn_code
    - origin_country
    - weight_kg
    - import_date

  properties:
    shipment_id:
      type: "string"
      description: "Unique shipment identifier"
      pattern: "^SHP-[A-Z0-9-]+$"
      examples: ["SHP-2024-001"]

    cn_code:
      type: "string"
      description: "8-digit EU Combined Nomenclature code"
      pattern: "^[0-9]{8}$"
      examples: ["72081000"]
      validation:
        - rule: "exists_in_taric_database"
          message: "CN code must exist in EU TARIC database"

    origin_country:
      type: "string"
      description: "ISO 3166-1 alpha-2 country code"
      pattern: "^[A-Z]{2}$"
      examples: ["CN", "US", "IN"]
      validation:
        - rule: "valid_iso_country"
          message: "Must be valid ISO 3166-1 alpha-2 code"

    weight_kg:
      type: "number"
      description: "Gross weight in kilograms"
      minimum: 0.01
      maximum: 1000000
      examples: [10000, 25000.5]

    import_date:
      type: "string"
      format: "date"
      description: "Date of import (YYYY-MM-DD)"
      examples: ["2024-12-01"]
      validation:
        - rule: "not_future_date"
          message: "Import date cannot be in the future"

    production_route:
      type: "string"
      description: "Production process (optional, for steel/cement)"
      enum:
        - "blast_furnace_basic_oxygen_furnace"
        - "electric_arc_furnace"
        - "dry_kiln"
        - "wet_kiln"
      examples: ["blast_furnace_basic_oxygen_furnace"]

  metadata:
    data_sources: ["Customs declarations", "ERP systems", "Freight forwarders"]
    update_frequency: "real-time"
    retention_period: "7 years"  # CBAM record retention requirement

validation_rules:
  - rule: "weight_reasonable_for_product"
    description: "Weight should be reasonable for product category"
    severity: "warning"

  - rule: "origin_country_produces_product"
    description: "Origin country should be known producer of product"
    severity: "warning"

quality_metrics:
  completeness:
    target: 100%
    measurement: "% of required fields populated"

  accuracy:
    target: 99.9%
    measurement: "% of records passing validation"

  timeliness:
    target: "<24 hours"
    measurement: "Time from import to data availability"
```

**Data Contract Validator:**
```python
from pydantic import BaseModel, Field, validator
from datetime import date

class CBAMShipment(BaseModel):
    """CBAM shipment data contract (Pydantic model)."""

    shipment_id: str = Field(..., regex=r"^SHP-[A-Z0-9-]+$")
    cn_code: str = Field(..., regex=r"^[0-9]{8}$")
    origin_country: str = Field(..., regex=r"^[A-Z]{2}$")
    weight_kg: float = Field(..., ge=0.01, le=1000000)
    import_date: date
    production_route: Optional[str] = Field(None, regex=r"^(blast_furnace|electric_arc|dry_kiln|wet_kiln).*$")

    @validator("cn_code")
    def validate_cn_code(cls, v):
        """Validate CN code against TARIC database."""
        if not TARICDatabase.exists(v):
            raise ValueError(f"CN code {v} not found in TARIC database")
        return v

    @validator("origin_country")
    def validate_country(cls, v):
        """Validate ISO country code."""
        if v not in ISO_COUNTRIES:
            raise ValueError(f"Invalid ISO country code: {v}")
        return v

    @validator("import_date")
    def validate_import_date(cls, v):
        """Ensure import date is not in the future."""
        if v > date.today():
            raise ValueError("Import date cannot be in the future")
        return v

    class Config:
        schema_extra = {
            "example": {
                "shipment_id": "SHP-2024-001",
                "cn_code": "72081000",
                "origin_country": "CN",
                "weight_kg": 10000,
                "import_date": "2024-12-01",
                "production_route": "blast_furnace_basic_oxygen_furnace"
            }
        }
```

**Success Metrics:**
- Data contract coverage: 100% of agent inputs/outputs
- Contract validation accuracy: 100%
- Backward compatibility: 100% (no breaking changes without major version)

---

### 2. Data Pipelines (ETL/ELT)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Ingestion Pipelines** | CSV, Excel, JSON, API ingestion | Phase 1 |
| **Transformation Pipelines** | Data cleaning, normalization, enrichment | Phase 1 |
| **Output Pipelines** | Generate reports (JSON, XML, PDF) | Phase 1 |
| **ERP Connectors** | SAP, Oracle, Workday integrations | Phase 2 |
| **Real-Time Streaming** | Kafka/Kinesis for real-time data | Phase 2 |
| **Data Warehouse** | Snowflake/BigQuery for analytics | Phase 3 |

**Technical Specifications:**

**Pipeline Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                  Data Sources                           │
│  • CSV/Excel files (upload)                             │
│  • ERP systems (SAP, Oracle, Workday)                   │
│  • APIs (REST, GraphQL)                                 │
│  • Streaming (Kafka, Kinesis)                           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Ingestion Layer                         │
│  • File parsing (pandas, openpyxl)                      │
│  • API polling (REST clients)                           │
│  • Stream consumption (Kafka consumer)                  │
│  • Data validation (Pydantic, Great Expectations)       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Transformation Layer                       │
│  • Data cleaning (dedupe, trim, normalize)              │
│  • Enrichment (lookup emission factors, CN codes)       │
│  • Calculation (embedded emissions)                     │
│  • Aggregation (rollup by product, country)             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Storage Layer                          │
│  • Transactional DB (PostgreSQL)                        │
│  • Object storage (S3)                                  │
│  • Data warehouse (Snowflake, BigQuery)                 │
│  • Cache (Redis)                                        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Output Layer                          │
│  • Reports (JSON, XML, PDF)                             │
│  • Dashboards (BI tools)                                │
│  • APIs (REST, GraphQL)                                 │
│  • Notifications (email, Slack)                         │
└─────────────────────────────────────────────────────────┘
```

**Pipeline Example (Airflow DAG):**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from datetime import datetime, timedelta

# DAG: Process CBAM shipment data daily
dag = DAG(
    dag_id="cbam_shipment_processing",
    description="Daily processing of CBAM shipment data",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=datetime(2025, 12, 1),
    catchup=False,
    default_args={
        "owner": "data_engineering",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    },
)

def ingest_shipment_data(**context):
    """Ingest shipment data from S3."""
    import pandas as pd
    from greenlang_sdk.data import CBAMShipment

    # Read CSV from S3
    df = pd.read_csv("s3://greenlang-data/shipments/daily/{{ ds }}.csv")

    # Validate against data contract
    validated_records = []
    errors = []

    for _, row in df.iterrows():
        try:
            record = CBAMShipment(**row.to_dict())
            validated_records.append(record.dict())
        except ValidationError as e:
            errors.append({"row": row.to_dict(), "error": str(e)})

    # Store validated data
    validated_df = pd.DataFrame(validated_records)
    validated_df.to_parquet("s3://greenlang-data/validated/{{ ds }}.parquet")

    # Log errors
    if errors:
        pd.DataFrame(errors).to_csv("s3://greenlang-data/errors/{{ ds }}.csv")

    return {
        "total_records": len(df),
        "validated_records": len(validated_records),
        "error_records": len(errors),
    }

def enrich_emission_factors(**context):
    """Enrich shipments with emission factors."""
    import pandas as pd
    from greenlang_sdk.emissions import EmissionFactorDB

    # Read validated data
    df = pd.read_parquet("s3://greenlang-data/validated/{{ ds }}.parquet")

    # Lookup emission factors
    ef_db = EmissionFactorDB()
    df["emission_factor"] = df.apply(
        lambda row: ef_db.lookup(
            cn_code=row["cn_code"],
            origin_country=row["origin_country"],
            production_route=row.get("production_route")
        ),
        axis=1
    )

    # Calculate embedded emissions
    df["embedded_emissions_tco2e"] = (
        df["weight_kg"] / 1000 * df["emission_factor"]
    )

    # Store enriched data
    df.to_parquet("s3://greenlang-data/enriched/{{ ds }}.parquet")

def generate_cbam_report(**context):
    """Generate CBAM JSON report."""
    import pandas as pd
    from greenlang_sdk.reporting import CBAMReportGenerator

    # Read enriched data
    df = pd.read_parquet("s3://greenlang-data/enriched/{{ ds }}.parquet")

    # Generate report
    generator = CBAMReportGenerator()
    report_json = generator.generate(df)

    # Store report
    with open("s3://greenlang-data/reports/{{ ds }}.json", "w") as f:
        f.write(report_json)

# Define tasks
ingest = PythonOperator(
    task_id="ingest_shipment_data",
    python_callable=ingest_shipment_data,
    dag=dag,
)

enrich = PythonOperator(
    task_id="enrich_emission_factors",
    python_callable=enrich_emission_factors,
    dag=dag,
)

report = PythonOperator(
    task_id="generate_cbam_report",
    python_callable=generate_cbam_report,
    dag=dag,
)

# Define dependencies
ingest >> enrich >> report
```

**Success Metrics:**
- Pipeline uptime: 99.9%
- Data latency: <1 hour (ingestion to output)
- Pipeline success rate: >99%
- Data volume: 1M+ records/day by Phase 3

---

### 3. Data Quality (Validation & Monitoring)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Quality Framework** | Great Expectations-based quality checks | Phase 1 |
| **Data Profiling** | Automated profiling of all datasets | Phase 1 |
| **Anomaly Detection** | Detect outliers and data drift | Phase 2 |
| **Quality Dashboard** | Real-time quality metrics | Phase 2 |
| **Data Observability** | Lineage, freshness, volume monitoring | Phase 3 |

**Technical Specifications:**

**Quality Framework (Great Expectations):**
```python
import great_expectations as ge

# Define expectations for CBAM shipment data
def create_cbam_shipment_expectations():
    """Create quality expectations for CBAM shipments."""

    expectations = [
        # Completeness
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "shipment_id"},
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "cn_code"},
        },

        # Validity
        {
            "expectation_type": "expect_column_values_to_match_regex",
            "kwargs": {
                "column": "cn_code",
                "regex": r"^[0-9]{8}$",
            },
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "origin_country",
                "value_set": ISO_COUNTRIES,
            },
        },

        # Accuracy
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "weight_kg",
                "min_value": 0.01,
                "max_value": 1000000,
            },
        },

        # Uniqueness
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {"column": "shipment_id"},
        },

        # Consistency
        {
            "expectation_type": "expect_column_pair_values_to_be_in_set",
            "kwargs": {
                "column_A": "cn_code",
                "column_B": "origin_country",
                "value_pairs_set": VALID_CN_COUNTRY_PAIRS,  # e.g., steel from China
            },
        },
    ]

    return expectations

# Run quality checks
def validate_data_quality(df: pd.DataFrame) -> dict:
    """Run quality checks on dataframe."""

    # Convert to Great Expectations dataset
    ge_df = ge.from_pandas(df)

    # Add expectations
    for expectation in create_cbam_shipment_expectations():
        ge_df.expect(**expectation)

    # Validate
    validation_result = ge_df.validate()

    # Calculate quality score
    total_checks = validation_result.statistics["evaluated_expectations"]
    successful_checks = validation_result.statistics["successful_expectations"]
    quality_score = (successful_checks / total_checks) * 100

    return {
        "quality_score": quality_score,
        "total_checks": total_checks,
        "successful_checks": successful_checks,
        "failed_checks": total_checks - successful_checks,
        "validation_result": validation_result,
    }
```

**Data Quality Metrics:**
```yaml
quality_metrics:
  completeness:
    name: "Data Completeness"
    description: "Percentage of required fields populated"
    target: 100%
    measurement: "COUNT(non_null) / COUNT(total)"
    severity: "critical"

  accuracy:
    name: "Data Accuracy"
    description: "Percentage of records passing validation"
    target: 99.9%
    measurement: "COUNT(valid) / COUNT(total)"
    severity: "critical"

  uniqueness:
    name: "Data Uniqueness"
    description: "Percentage of unique records"
    target: 100%
    measurement: "COUNT(DISTINCT id) / COUNT(id)"
    severity: "high"

  timeliness:
    name: "Data Timeliness"
    description: "Time from creation to availability"
    target: "<1 hour"
    measurement: "AVG(ingestion_time - creation_time)"
    severity: "medium"

  consistency:
    name: "Data Consistency"
    description: "Percentage of records consistent with business rules"
    target: 99%
    measurement: "COUNT(consistent) / COUNT(total)"
    severity: "medium"
```

**Success Metrics:**
- Data quality score: >99.9%
- Quality check coverage: 100% of pipelines
- Anomaly detection accuracy: >95%
- False positive rate: <5%

---

### 4. Data Provenance (Lineage Tracking)

**Deliverables:**

| Component | Description | Phase |
|-----------|-------------|-------|
| **Lineage Tracking** | Track data from source to output | Phase 1 |
| **SHA-256 Hashing** | Cryptographic hashing for audit trails | Phase 1 |
| **Metadata Store** | Centralized metadata repository | Phase 2 |
| **Lineage Visualization** | UI for exploring data lineage | Phase 3 |

**Technical Specifications:**

**Provenance Tracking:**
```python
from hashlib import sha256
from datetime import datetime

class ProvenanceTracker:
    """Track data provenance for audit trails."""

    def __init__(self):
        self.lineage = []

    def track(
        self,
        operation: str,
        input_data: dict,
        output_data: dict,
        metadata: dict = None
    ):
        """
        Track data transformation.

        Args:
            operation: Name of operation (e.g., "enrich_emission_factors")
            input_data: Input data
            output_data: Output data
            metadata: Additional metadata
        """
        provenance_record = {
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": self._hash_data(input_data),
            "output_hash": self._hash_data(output_data),
            "metadata": metadata or {},
        }

        self.lineage.append(provenance_record)

    def _hash_data(self, data: dict) -> str:
        """Generate SHA-256 hash of data."""
        data_str = json.dumps(data, sort_keys=True)
        return sha256(data_str.encode()).hexdigest()

    def get_lineage(self) -> list:
        """Get complete lineage."""
        return self.lineage

# Usage
tracker = ProvenanceTracker()

# Track ingestion
tracker.track(
    operation="ingest_shipment_data",
    input_data={"source": "s3://raw/shipments.csv"},
    output_data={"records": validated_records},
    metadata={"pipeline": "cbam_daily", "version": "1.0.0"}
)

# Track enrichment
tracker.track(
    operation="enrich_emission_factors",
    input_data={"records": validated_records},
    output_data={"records": enriched_records},
    metadata={"emission_factor_db": "IEA-2024"}
)
```

**Success Metrics:**
- Provenance coverage: 100% of data transformations
- Hash collision rate: 0%
- Audit trail completeness: 100%

---

## Deliverables by Phase

### Phase 1: Foundation (Weeks 1-16)

**Week 1-4: Data Contracts**
- [ ] CBAM data contracts (5+ schemas)
- [ ] Contract validator (Pydantic)
- [ ] Contract documentation
- [ ] Contract registry setup

**Week 5-8: Data Pipelines**
- [ ] Ingestion pipelines (CSV, Excel, JSON)
- [ ] Transformation pipelines
- [ ] Output pipelines (JSON reports)
- [ ] Airflow DAG orchestration

**Week 9-12: Data Quality**
- [ ] Quality framework (Great Expectations)
- [ ] Data profiling
- [ ] Quality dashboard
- [ ] Automated quality checks

**Week 13-16: Provenance**
- [ ] Provenance tracker
- [ ] SHA-256 hashing
- [ ] Lineage storage
- [ ] Audit trail documentation

**Phase 1 Exit Criteria:**
- [ ] 5+ data contracts defined
- [ ] 3+ pipelines operational
- [ ] Data quality score: >99%
- [ ] Provenance tracking: 100%

---

### Phase 2: Production Scale (Weeks 17-28)

**Week 17-20: ERP Connectors**
- [ ] SAP connector
- [ ] Oracle connector
- [ ] Workday connector
- [ ] Real-time sync

**Week 21-24: Streaming**
- [ ] Kafka/Kinesis setup
- [ ] Real-time pipelines
- [ ] Stream processing
- [ ] Low-latency ingestion (<1 min)

**Week 25-28: Advanced Quality**
- [ ] Anomaly detection (ML-based)
- [ ] Data drift monitoring
- [ ] Quality alerting
- [ ] Root cause analysis

**Phase 2 Exit Criteria:**
- [ ] ERP connectors operational
- [ ] Streaming pipelines live
- [ ] Data latency: <1 minute
- [ ] Anomaly detection: >95% accuracy

---

### Phase 3: Enterprise Ready (Weeks 29-40)

**Week 29-32: Data Warehouse**
- [ ] Snowflake/BigQuery setup
- [ ] ELT pipelines
- [ ] BI tool integration (Tableau, Looker)
- [ ] Analytics dashboards

**Week 33-36: Data Observability**
- [ ] Lineage visualization
- [ ] Freshness monitoring
- [ ] Volume monitoring
- [ ] Quality trends

**Week 37-40: Scale & Optimization**
- [ ] Multi-region data replication
- [ ] Cost optimization (<$0.05/GB)
- [ ] Performance tuning
- [ ] Enterprise SLAs

**Phase 3 Exit Criteria:**
- [ ] Data warehouse operational
- [ ] Lineage visualization live
- [ ] Data volume: 10M+ records/day
- [ ] Cost per GB: <$0.05

---

## Success Metrics & KPIs

### North Star Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target | Measurement |
|--------|---------------|---------------|---------------|-------------|
| **Data Quality Score** | >99% | >99.5% | >99.9% | % of records passing validation |
| **Pipeline Uptime** | 99.9% | 99.95% | 99.99% | Availability over 30 days |
| **Data Latency** | <1 hour | <10 min | <1 min | Ingestion to availability |
| **Provenance Coverage** | 100% | 100% | 100% | % of transformations tracked |

---

## Interfaces with Other Teams

### AI/Agent Team
- Provides: Data contracts, pipelines
- Receives: Agent data requirements

### Platform Team
- Provides: Data APIs, schemas
- Receives: Storage infrastructure

### Climate Science Team
- Provides: Emission factor data, validation rules
- Receives: Data quality feedback

---

## Technical Stack

- **Orchestration:** Airflow, Prefect
- **Storage:** PostgreSQL, S3, Snowflake
- **Streaming:** Kafka, Kinesis
- **Quality:** Great Expectations, dbt
- **Lineage:** Apache Atlas, OpenLineage

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | GL Product Manager | Initial Data Engineering Team charter |

---

**Approvals:**

- Data Engineering Tech Lead: ___________________
- Engineering Lead: ___________________
- Product Manager: ___________________
