# GreenLang Feature Flags Reference

This document provides comprehensive documentation for all feature flags used in the GreenLang platform and its applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Flag Categories](#feature-flag-categories)
3. [Core Platform Features](#core-platform-features)
4. [AI/ML Features](#aiml-features)
5. [Regulatory Module Features](#regulatory-module-features)
6. [Integration Features](#integration-features)
7. [Application-Specific Features](#application-specific-features)
8. [Infrastructure Features](#infrastructure-features)
9. [Development Features](#development-features)
10. [Feature Flag Dependencies](#feature-flag-dependencies)
11. [Configuration Guide](#configuration-guide)
12. [Best Practices](#best-practices)

---

## Overview

Feature flags in GreenLang enable:

- **Gradual Rollout**: Deploy new features to a subset of users
- **Kill Switches**: Quickly disable problematic features
- **A/B Testing**: Test different feature implementations
- **Environment Separation**: Different features for dev/staging/production
- **License Control**: Enable features based on customer tier

### Flag Naming Convention

```
FEATURE_{CATEGORY}_{NAME}
ENABLE_{FUNCTIONALITY}
```

Examples:
- `FEATURE_CBAM_MODULE` - Regulatory module flag
- `FEATURE_LLM_CLASSIFICATION` - AI feature flag
- `ENABLE_METRICS` - Infrastructure flag
- `ENABLE_AUDIT_LOG` - Compliance flag

### Default Behavior

- **Production**: Most features enabled by default (except experimental)
- **Development**: Core features enabled, AI features may require explicit enablement
- **All environments**: Integration features disabled by default (require configuration)

---

## Feature Flag Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| Core Features | `FEATURE_*` | Core platform capabilities |
| AI Features | `FEATURE_LLM_*`, `FEATURE_*_ML` | Machine learning and AI features |
| Regulatory | `FEATURE_*_MODULE` | Regulatory compliance modules |
| Integrations | `FEATURE_*_INTEGRATION` | Third-party integrations |
| Infrastructure | `ENABLE_*` | Infrastructure and operational features |
| Development | `DEBUG`, `ENABLE_*_FEATURES` | Development and debugging features |

---

## Core Platform Features

### FEATURE_ZERO_HALLUCINATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_ZERO_HALLUCINATION` |
| **Default** | `true` |
| **Category** | Core |
| **Purpose** | Enables zero-hallucination mode for calculations and reports |

**Description:**
Ensures all outputs are fully traceable to source data with no AI-generated content presented as factual data. When enabled:
- All calculations include provenance metadata
- Reports include source citations
- AI-generated content is clearly marked
- Audit trails are comprehensive

**Impact when enabled:**
- Increased traceability
- Longer processing times (provenance tracking overhead)
- Larger metadata storage requirements
- Full regulatory compliance support

**Impact when disabled:**
- Faster processing
- Reduced storage requirements
- May not meet regulatory requirements

**Dependencies:** None

**Used by:** Core platform, all regulatory modules

---

### FEATURE_PROVENANCE_TRACKING

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_PROVENANCE_TRACKING` |
| **Default** | `true` |
| **Category** | Core |
| **Purpose** | Enables comprehensive data provenance tracking |

**Description:**
Tracks the complete lineage of data from source to output, including:
- Data source identification
- Transformation history
- Calculation steps
- User actions and timestamps

**Impact when enabled:**
- Full data lineage available
- Audit-ready reports
- Increased storage for metadata
- Slightly longer processing times

**Impact when disabled:**
- No data lineage tracking
- Faster processing
- May fail regulatory audits

**Dependencies:**
- Required for `FEATURE_ZERO_HALLUCINATION` to be effective

**Used by:** Core platform, provenance module

---

### FEATURE_ASYNC_PROCESSING

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_ASYNC_PROCESSING` |
| **Default** | `true` |
| **Category** | Core |
| **Purpose** | Enables asynchronous job processing |

**Description:**
Allows long-running tasks to be processed asynchronously:
- File uploads processed in background
- Report generation queued
- Bulk operations batched
- Status polling available

**Impact when enabled:**
- Non-blocking API operations
- Better user experience for large datasets
- Requires message queue (RabbitMQ/Redis)
- WebSocket support for real-time updates

**Impact when disabled:**
- Synchronous processing only
- API calls block until completion
- Simpler infrastructure requirements
- Timeout issues for large datasets

**Dependencies:**
- Requires Redis or RabbitMQ for queue management

**Used by:** Core platform, all applications

---

### FEATURE_BATCH_PROCESSING

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_BATCH_PROCESSING` |
| **Default** | `true` |
| **Category** | Core |
| **Purpose** | Enables batch processing of large datasets |

**Description:**
Optimizes processing of large datasets by:
- Chunking data into configurable batches
- Parallel processing of batches
- Progress tracking
- Automatic retry on failure

**Configuration:**
- `BATCH_SIZE_DEFAULT`: Default batch size (default: 1000)
- `BATCH_MAX_PARALLEL_JOBS`: Maximum parallel jobs (default: 5)
- `BATCH_PROCESSING_TIMEOUT`: Timeout in seconds (default: 600)

**Impact when enabled:**
- Efficient large dataset processing
- Memory-optimized operations
- Higher throughput
- Progress visibility

**Impact when disabled:**
- Single-record processing
- Higher memory usage for large datasets
- Slower overall throughput

**Dependencies:**
- `FEATURE_ASYNC_PROCESSING` recommended

**Used by:** Core platform, data pipeline

---

### FEATURE_REAL_TIME_VALIDATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_REAL_TIME_VALIDATION` |
| **Default** | `true` |
| **Category** | Core |
| **Purpose** | Enables real-time data validation |

**Description:**
Validates data as it is received:
- Schema validation
- Business rule validation
- Cross-reference checks
- Data quality scoring

**Impact when enabled:**
- Immediate feedback on data issues
- Early error detection
- Slightly higher processing overhead
- Better data quality

**Impact when disabled:**
- Post-processing validation only
- Faster initial upload
- Later error discovery

**Dependencies:** None

**Used by:** Core platform, data intake

---

## AI/ML Features

### FEATURE_LLM_CLASSIFICATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_LLM_CLASSIFICATION` |
| **Default** | `true` (production), `false` (development) |
| **Category** | AI |
| **Purpose** | Enables LLM-powered data classification |

**Description:**
Uses large language models to automatically classify:
- Product categories
- Emission categories
- Regulatory classifications
- Supplier categories

**Requirements:**
- At least one LLM API key configured:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `AZURE_OPENAI_API_KEY`

**Impact when enabled:**
- Automated classification
- Higher accuracy for complex cases
- API costs for LLM usage
- Requires network access to LLM providers

**Impact when disabled:**
- Manual classification only
- Rule-based classification
- No API costs
- Offline operation possible

**Dependencies:**
- Requires LLM API configuration
- Budget controls recommended (`MAX_LLM_TOKENS_PER_DAY`, `MAX_LLM_COST_PER_DAY_USD`)

**Used by:** Core platform, GL-VCCI-Carbon-APP, GL-CBAM-APP

---

### FEATURE_LLM_NARRATIVE_GEN

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_LLM_NARRATIVE_GEN` |
| **Default** | `true` (production), `false` (development) |
| **Category** | AI |
| **Purpose** | Enables LLM-generated narrative content |

**Description:**
Generates human-readable narratives for:
- Report sections
- Explanatory text
- Summary descriptions
- Recommendation explanations

**Requirements:**
- LLM API key configured
- `FEATURE_ZERO_HALLUCINATION` recommended to mark AI content

**Impact when enabled:**
- Automated narrative generation
- Consistent writing style
- LLM API costs
- Requires review for accuracy

**Impact when disabled:**
- Manual narrative writing
- Template-based content only
- No API costs

**Dependencies:**
- Requires LLM API configuration
- `FEATURE_ZERO_HALLUCINATION` ensures proper labeling

**Used by:** GL-CSRD-APP, GL-VCCI-Carbon-APP

---

### FEATURE_SATELLITE_ML

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_SATELLITE_ML` |
| **Default** | `false` |
| **Category** | AI |
| **Purpose** | Enables satellite imagery machine learning |

**Description:**
Analyzes satellite imagery for:
- Deforestation detection
- Land use classification
- Supply chain verification
- Environmental monitoring

**Requirements:**
- Satellite data provider configured:
  - `PLANET_API_KEY`
  - `SENTINEL_HUB_CLIENT_ID` + `SENTINEL_HUB_CLIENT_SECRET`
  - `NASA_EARTHDATA_USERNAME` + `NASA_EARTHDATA_PASSWORD`

**Impact when enabled:**
- Automated environmental monitoring
- Visual verification of supplier claims
- High computational requirements
- Significant API costs

**Impact when disabled:**
- No satellite analysis
- Manual verification required
- Lower infrastructure requirements

**Dependencies:**
- Requires satellite data API configuration
- GPU resources recommended for image processing

**Used by:** EUDR module, environmental monitoring

---

### FEATURE_ANOMALY_DETECTION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_ANOMALY_DETECTION` |
| **Default** | `true` |
| **Category** | AI |
| **Purpose** | Enables anomaly detection in data |

**Description:**
Detects anomalies and outliers in:
- Emission data
- Supplier data
- Time series data
- Cross-reference data

**Impact when enabled:**
- Automatic anomaly flagging
- Data quality alerts
- Investigation recommendations
- Additional processing overhead

**Impact when disabled:**
- Manual anomaly detection only
- No automatic alerts
- Faster processing

**Dependencies:** None (uses statistical methods, no LLM required)

**Used by:** Core platform, all applications

---

## Regulatory Module Features

### FEATURE_CBAM_MODULE

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_CBAM_MODULE` |
| **Default** | `true` |
| **Category** | Regulatory |
| **Purpose** | Enables EU CBAM (Carbon Border Adjustment Mechanism) module |

**Description:**
Provides complete CBAM compliance functionality:
- CN code classification
- Embedded emissions calculation
- CBAM certificate management
- EU registry integration
- Quarterly reporting

**Impact when enabled:**
- Full CBAM reporting capabilities
- EU registry connectivity
- CBAM-specific calculations
- Additional storage for CBAM data

**Impact when disabled:**
- No CBAM functionality
- CBAM endpoints return 404
- Reduced application footprint

**Dependencies:**
- `CBAM_REGISTRY_URL` for EU registry integration
- `EMISSION_FACTORS_PATH` for calculation data

**Used by:** GL-CBAM-APP

---

### FEATURE_CSRD_MODULE

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_CSRD_MODULE` |
| **Default** | `true` |
| **Category** | Regulatory |
| **Purpose** | Enables EU CSRD (Corporate Sustainability Reporting Directive) module |

**Description:**
Provides CSRD/ESRS compliance functionality:
- Double materiality assessment
- ESRS disclosure generation
- XBRL tagging and export
- Narrative generation
- Stakeholder management

**Impact when enabled:**
- Full CSRD reporting capabilities
- AI-powered materiality assessment (with LLM)
- XBRL export functionality
- ESRS-compliant disclosures

**Impact when disabled:**
- No CSRD functionality
- CSRD endpoints return 404

**Dependencies:**
- `ENABLE_XBRL_GENERATION` for XBRL output
- `ENABLE_AI_MATERIALITY` for AI-powered assessment
- `ENABLE_AI_NARRATIVES` for AI narrative generation

**Used by:** GL-CSRD-APP

---

### FEATURE_SB253_MODULE

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_SB253_MODULE` |
| **Default** | `true` |
| **Category** | Regulatory |
| **Purpose** | Enables California SB253 (Climate Corporate Data Accountability Act) module |

**Description:**
Provides SB253 compliance functionality:
- Scope 1, 2, 3 emissions reporting
- Third-party assurance preparation
- CARB registry integration
- California-specific requirements

**Impact when enabled:**
- Full SB253 reporting capabilities
- CARB integration support
- US regulatory compliance

**Impact when disabled:**
- No SB253 functionality
- SB253 endpoints return 404

**Dependencies:**
- `CARB_CLIENT_ID`, `CARB_CLIENT_SECRET` for CARB integration

**Used by:** GL-SB253-APP

---

### FEATURE_EUDR_MODULE

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_EUDR_MODULE` |
| **Default** | `false` |
| **Category** | Regulatory |
| **Purpose** | Enables EU Deforestation Regulation module |

**Description:**
Provides EUDR compliance functionality:
- Deforestation risk assessment
- Supply chain geolocation tracking
- Satellite imagery analysis
- Due diligence statements

**Impact when enabled:**
- Full EUDR functionality
- Satellite imagery integration
- Geolocation tracking
- Higher infrastructure requirements

**Impact when disabled:**
- No EUDR functionality
- EUDR endpoints return 404

**Dependencies:**
- `FEATURE_SATELLITE_ML` for satellite analysis
- Satellite data API configuration

**Used by:** EUDR module

---

### FEATURE_TAXONOMY_MODULE

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_TAXONOMY_MODULE` |
| **Default** | `true` |
| **Category** | Regulatory |
| **Purpose** | Enables EU Taxonomy module |

**Description:**
Provides EU Taxonomy alignment functionality:
- Activity classification
- Technical screening criteria
- DNSH assessment
- Minimum safeguards check
- Taxonomy-aligned revenue calculation

**Impact when enabled:**
- Full EU Taxonomy reporting
- Alignment calculations
- CapEx/OpEx/Revenue reporting

**Impact when disabled:**
- No EU Taxonomy functionality
- Taxonomy endpoints return 404

**Dependencies:** None

**Used by:** Core platform, CSRD module

---

## Integration Features

### FEATURE_SAP_INTEGRATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_SAP_INTEGRATION` |
| **Default** | `false` |
| **Category** | Integration |
| **Purpose** | Enables SAP S/4HANA integration |

**Description:**
Provides SAP ERP integration:
- Purchase order data extraction
- Supplier master data sync
- Material master data
- Financial postings

**Requirements:**
- `SAP_API_ENDPOINT` - SAP OData endpoint
- `SAP_OAUTH_CLIENT_ID` - OAuth client ID
- `SAP_OAUTH_CLIENT_SECRET` - OAuth client secret
- `SAP_OAUTH_TOKEN_URL` - OAuth token URL

**Impact when enabled:**
- Automated SAP data sync
- Real-time data availability
- SAP license requirements
- Network connectivity to SAP

**Impact when disabled:**
- Manual data upload only
- CSV/Excel import for SAP data

**Dependencies:**
- SAP OAuth credentials required
- SAP connectivity

**Used by:** GL-VCCI-Carbon-APP

---

### FEATURE_ORACLE_INTEGRATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_ORACLE_INTEGRATION` |
| **Default** | `false` |
| **Category** | Integration |
| **Purpose** | Enables Oracle ERP Cloud integration |

**Description:**
Provides Oracle ERP integration:
- Purchase data extraction
- Supplier data sync
- Financial data access

**Requirements:**
- `ORACLE_API_ENDPOINT` - Oracle REST endpoint
- `ORACLE_OAUTH_CLIENT_ID` - OAuth client ID
- `ORACLE_OAUTH_CLIENT_SECRET` - OAuth client secret
- `ORACLE_OAUTH_TOKEN_URL` - OAuth token URL

**Impact when enabled:**
- Automated Oracle data sync
- Oracle license requirements
- Network connectivity to Oracle

**Impact when disabled:**
- Manual data upload only
- CSV/Excel import for Oracle data

**Dependencies:**
- Oracle OAuth credentials required

**Used by:** GL-VCCI-Carbon-APP

---

### FEATURE_SALESFORCE_INTEGRATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_SALESFORCE_INTEGRATION` |
| **Default** | `false` |
| **Category** | Integration |
| **Purpose** | Enables Salesforce integration |

**Description:**
Provides Salesforce integration:
- Customer data sync
- Opportunity tracking
- Custom object integration

**Impact when enabled:**
- Salesforce data sync
- CRM integration
- Salesforce API limits apply

**Impact when disabled:**
- No Salesforce integration
- Manual data entry

**Dependencies:**
- Salesforce OAuth credentials required

**Used by:** GL-VCCI-Carbon-APP

---

### FEATURE_AZURE_IOT_INTEGRATION

| Property | Value |
|----------|-------|
| **Name** | `FEATURE_AZURE_IOT_INTEGRATION` |
| **Default** | `false` |
| **Category** | Integration |
| **Purpose** | Enables Azure IoT Hub integration |

**Description:**
Provides Azure IoT integration:
- Real-time sensor data ingestion
- Device management
- Stream analytics

**Impact when enabled:**
- Real-time IoT data
- Azure IoT Hub connectivity
- Azure costs

**Impact when disabled:**
- No IoT integration
- Manual data entry

**Dependencies:**
- Azure IoT Hub configuration
- Azure credentials

**Used by:** GL-VCCI-Carbon-APP

---

## Application-Specific Features

### GL-CSRD-APP Features

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_AI_MATERIALITY` | `true` | AI-powered double materiality assessment |
| `ENABLE_AI_NARRATIVES` | `true` | AI-generated narrative content |
| `ENABLE_XBRL_GENERATION` | `true` | XBRL report generation |
| `ENABLE_EMAIL_NOTIFICATIONS` | `false` | Email notifications |

### GL-CBAM-APP Features

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_API_DOCS` | `true` | Enable Swagger/OpenAPI docs |
| `ENABLE_REDOC` | `true` | Enable ReDoc documentation |
| `ENABLE_CACHING` | `true` | Enable response caching |
| `ENABLE_RATE_LIMITING` | `true` | Enable API rate limiting |
| `ENABLE_BACKGROUND_TASKS` | `true` | Enable background task processing |
| `ENABLE_EXPERIMENTAL_FEATURES` | `false` | Enable experimental features |

### GL-VCCI-Carbon-APP Features

| Flag | Default | Description |
|------|---------|-------------|
| `FEATURE_ENTITY_RESOLUTION` | `true` | Automatic entity resolution |
| `FEATURE_LLM_CATEGORIZATION` | `true` | LLM-powered categorization |
| `FEATURE_SUPPLIER_ENGAGEMENT` | `true` | Supplier engagement workflows |
| `FEATURE_AUTOMATED_REPORTING` | `true` | Automated report generation |
| `FEATURE_SCENARIO_MODELING` | `true` | What-if scenario modeling |
| `FEATURE_REAL_TIME_MONITORING` | `true` | Real-time emission monitoring |
| `FEATURE_BLOCKCHAIN_PROVENANCE` | `false` | Blockchain-based provenance (beta) |
| `FEATURE_SATELLITE_MONITORING` | `false` | Satellite-based monitoring (beta) |
| `FEATURE_MOBILE_APP` | `false` | Mobile application support (beta) |

---

## Infrastructure Features

### Monitoring and Metrics

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `ENABLE_TRACING` | `false` | Enable distributed tracing |
| `PROMETHEUS_ENABLED` | `true` | Enable Prometheus endpoint |

### Security

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_CORS` | `true` | Enable CORS |
| `ENABLE_RATE_LIMITING` | `true` | Enable rate limiting |
| `API_KEY_ENABLED` | `false` | Enable API key authentication |
| `JWT_ENABLED` | `false` | Enable JWT authentication |

### Compliance

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_AUDIT_LOG` | `true` | Enable audit logging |
| `AUDIT_LOG_ENABLED` | `true` | Alternative audit log flag |
| `FEATURE_AUDIT_LOGGING` | `true` | Feature flag for audit |
| `LOG_ALL_CALCULATIONS` | `true` | Log all calculations |
| `LOG_ALL_API_CALLS` | `true` | Log all API calls |
| `LOG_ALL_DATA_ACCESS` | `true` | Log all data access |

### Backup

| Flag | Default | Description |
|------|---------|-------------|
| `BACKUP_ENABLED` | `true` | Enable backups |
| `ENABLE_DB_BACKUP` | `true` | Enable database backups |

---

## Development Features

| Flag | Default | Description |
|------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `TESTING` | `false` | Enable testing mode |
| `PROFILING` | `false` | Enable profiling |
| `ENABLE_PROFILING` | `false` | Alternative profiling flag |
| `HOT_RELOAD` | `false` | Enable hot reload |
| `RELOAD` | `false` | Alternative hot reload flag |
| `SQL_DEBUG` | `false` | Enable SQL query logging |
| `VERBOSE_LOGGING` | `false` | Enable verbose logging |
| `DETAILED_ERROR_TRACES` | `true` | Show detailed error traces |
| `LOG_SQL_QUERIES` | `false` | Log SQL queries |
| `MOCK_ERP_CONNECTIONS` | `false` | Mock ERP connections |
| `MOCK_LLM_RESPONSES` | `false` | Mock LLM responses |
| `DISABLE_AUTH` | `false` | Disable authentication (NEVER in production) |
| `ENABLE_EXPERIMENTAL_FEATURES` | `false` | Enable experimental features |
| `DEV_DB_SEED` | `true` | Seed development database |

**Warning:** Never enable `DEBUG`, `DISABLE_AUTH`, or `SQL_DEBUG` in production.

---

## Feature Flag Dependencies

The following diagram shows feature flag dependencies:

```
FEATURE_ZERO_HALLUCINATION
    |
    +-- FEATURE_PROVENANCE_TRACKING (recommended)

FEATURE_LLM_CLASSIFICATION
    |
    +-- Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY

FEATURE_LLM_NARRATIVE_GEN
    |
    +-- Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY
    +-- Recommended: FEATURE_ZERO_HALLUCINATION

FEATURE_SATELLITE_ML
    |
    +-- Requires: PLANET_API_KEY or SENTINEL_HUB credentials

FEATURE_EUDR_MODULE
    |
    +-- Requires: FEATURE_SATELLITE_ML (for satellite analysis)

FEATURE_CSRD_MODULE
    |
    +-- Optional: ENABLE_AI_MATERIALITY
    +-- Optional: ENABLE_AI_NARRATIVES
    +-- Optional: ENABLE_XBRL_GENERATION

FEATURE_SAP_INTEGRATION
    |
    +-- Requires: SAP_API_ENDPOINT
    +-- Requires: SAP_OAUTH_CLIENT_ID
    +-- Requires: SAP_OAUTH_CLIENT_SECRET

FEATURE_ASYNC_PROCESSING
    |
    +-- Requires: Redis or RabbitMQ
```

---

## Configuration Guide

### Production Configuration

```bash
# Core Features (all enabled)
FEATURE_ZERO_HALLUCINATION=true
FEATURE_PROVENANCE_TRACKING=true
FEATURE_ASYNC_PROCESSING=true
FEATURE_BATCH_PROCESSING=true
FEATURE_REAL_TIME_VALIDATION=true

# AI Features (enabled with API keys)
FEATURE_LLM_CLASSIFICATION=true
FEATURE_LLM_NARRATIVE_GEN=true
FEATURE_ANOMALY_DETECTION=true
FEATURE_SATELLITE_ML=false  # Enable if using EUDR

# Regulatory Modules (as licensed)
FEATURE_CBAM_MODULE=true
FEATURE_CSRD_MODULE=true
FEATURE_SB253_MODULE=true
FEATURE_EUDR_MODULE=false   # Enable when needed
FEATURE_TAXONOMY_MODULE=true

# Integrations (as configured)
FEATURE_SAP_INTEGRATION=false    # Enable if SAP configured
FEATURE_ORACLE_INTEGRATION=false # Enable if Oracle configured
FEATURE_SALESFORCE_INTEGRATION=false
FEATURE_AZURE_IOT_INTEGRATION=false

# Infrastructure
ENABLE_METRICS=true
ENABLE_TRACING=true
ENABLE_AUDIT_LOG=true
ENABLE_RATE_LIMITING=true
BACKUP_ENABLED=true

# Development (all disabled in production)
DEBUG=false
TESTING=false
PROFILING=false
SQL_DEBUG=false
DISABLE_AUTH=false
```

### Development Configuration

```bash
# Core Features
FEATURE_ZERO_HALLUCINATION=true
FEATURE_PROVENANCE_TRACKING=true
FEATURE_ASYNC_PROCESSING=true
FEATURE_BATCH_PROCESSING=true
FEATURE_REAL_TIME_VALIDATION=true

# AI Features (disabled unless testing)
FEATURE_LLM_CLASSIFICATION=false  # Enable for AI testing
FEATURE_LLM_NARRATIVE_GEN=false
FEATURE_ANOMALY_DETECTION=true
FEATURE_SATELLITE_ML=false

# Regulatory Modules
FEATURE_CBAM_MODULE=true
FEATURE_CSRD_MODULE=true
FEATURE_SB253_MODULE=true
FEATURE_EUDR_MODULE=false
FEATURE_TAXONOMY_MODULE=true

# Integrations (all disabled)
FEATURE_SAP_INTEGRATION=false
FEATURE_ORACLE_INTEGRATION=false
FEATURE_SALESFORCE_INTEGRATION=false
FEATURE_AZURE_IOT_INTEGRATION=false

# Development Features
DEBUG=true
TESTING=false
PROFILING=false
HOT_RELOAD=true
VERBOSE_LOGGING=true
MOCK_ERP_CONNECTIONS=true
MOCK_LLM_RESPONSES=true
DEV_DB_SEED=true
```

### Testing Configuration

```bash
# Core Features
FEATURE_ZERO_HALLUCINATION=true
FEATURE_PROVENANCE_TRACKING=true
FEATURE_ASYNC_PROCESSING=false  # Simpler testing
FEATURE_BATCH_PROCESSING=true
FEATURE_REAL_TIME_VALIDATION=true

# AI Features (mocked)
FEATURE_LLM_CLASSIFICATION=true
FEATURE_LLM_NARRATIVE_GEN=true
MOCK_LLM_RESPONSES=true  # Use mocks for testing

# All modules enabled for testing
FEATURE_CBAM_MODULE=true
FEATURE_CSRD_MODULE=true
FEATURE_SB253_MODULE=true
FEATURE_EUDR_MODULE=true
FEATURE_TAXONOMY_MODULE=true

# Testing mode
TESTING=true
DEBUG=true
DETAILED_ERROR_TRACES=true
DISABLE_AUTH=true  # Only for automated tests
```

---

## Best Practices

### 1. Feature Flag Hygiene

- **Document all flags**: Every flag should have clear documentation
- **Set sensible defaults**: Flags should work without explicit configuration
- **Clean up old flags**: Remove flags for fully released features
- **Version flags**: Consider flag versioning for major changes

### 2. Security

- **Never enable `DISABLE_AUTH` in production**
- **Never enable `DEBUG` in production**
- **Audit flag changes**: Log all feature flag modifications
- **Restrict flag access**: Only authorized users should modify flags

### 3. Testing

- **Test both states**: Test with flag enabled and disabled
- **Test flag combinations**: Test dependent flags together
- **Use mocks**: Use `MOCK_*` flags for isolated testing
- **Integration tests**: Test flag behavior in integration tests

### 4. Gradual Rollout

For new features, consider:

1. Deploy with flag disabled
2. Enable for internal users
3. Enable for beta users
4. Enable for all users
5. Remove flag and make permanent

### 5. Monitoring

- **Track flag state**: Include flag state in logs and metrics
- **Alert on changes**: Alert when critical flags change
- **Performance monitoring**: Monitor performance impact of flags

### 6. Dependencies

- **Document dependencies**: Clearly document flag dependencies
- **Validate at startup**: Check required flags at application start
- **Graceful degradation**: Handle missing dependencies gracefully

---

## Total Feature Flags Documented

| Category | Count |
|----------|-------|
| Core Platform | 5 |
| AI/ML | 5 |
| Regulatory | 5 |
| Integration | 4 |
| GL-CSRD-APP | 4 |
| GL-CBAM-APP | 6 |
| GL-VCCI-Carbon-APP | 9 |
| Infrastructure | 8 |
| Development | 14 |
| **Total** | **60** |

---

## See Also

- [ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md) - Complete environment variable reference
- [.env.template](../.env.template) - Environment template with all flags
- [config/env_validator.py](../config/env_validator.py) - Environment validation tool
