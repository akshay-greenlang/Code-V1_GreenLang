# PACK-030: Configuration Reference

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Pack Configuration](#pack-configuration)
3. [Framework Configuration](#framework-configuration)
4. [Branding Configuration](#branding-configuration)
5. [Data Source Configuration](#data-source-configuration)
6. [Output Configuration](#output-configuration)
7. [Notification Configuration](#notification-configuration)
8. [Assurance Configuration](#assurance-configuration)
9. [Performance Configuration](#performance-configuration)
10. [Configuration Presets](#configuration-presets)
11. [Environment Variables](#environment-variables)

---

## 1. Configuration Overview

PACK-030 uses a layered configuration system:

1. **Default values** - Built-in defaults for all settings
2. **Pack configuration file** - `config/pack_config.yaml` or `config/pack_config.py`
3. **Preset overrides** - Pre-built configurations for specific use cases
4. **Environment variables** - Runtime overrides for deployment-specific settings
5. **API parameters** - Per-request overrides for specific report generation

Priority: API parameters > Environment variables > Presets > Pack config > Defaults

---

## 2. Pack Configuration

### Python Configuration (Pydantic v2)

```python
from packs.net_zero.pack030.config import PACK030Config

config = PACK030Config(
    # Pack identification
    pack_id="PACK-030-net-zero-reporting",
    pack_version="1.0.0",

    # Frameworks to enable
    frameworks={
        "SBTi": ReportingFrameworkConfig(
            name="SBTi",
            version="v1.1",
            required_metrics=["scope1_total", "scope2_total", "scope3_total"],
            optional_metrics=["scope3_categories"],
            output_formats=["PDF", "JSON"],
            deadline_months=None,
        ),
        "CDP": ReportingFrameworkConfig(
            name="CDP",
            version="2025",
            required_metrics=["scope1_total", "scope2_location", "scope2_market"],
            optional_metrics=["scope3_all_categories"],
            output_formats=["Excel", "JSON"],
            deadline_months=7,
        ),
        # ... additional frameworks
    },

    # Branding
    branding=BrandingConfig(
        logo_path="/path/to/logo.png",
        primary_color="#1E3A8A",
        secondary_color="#3B82F6",
        font_family="Arial, sans-serif",
        style="corporate",
    ),

    # Data sources
    data_sources=[
        "PACK-021", "PACK-022", "PACK-028", "PACK-029",
        "GL-SBTi-APP", "GL-CDP-APP", "GL-TCFD-APP", "GL-GHG-APP",
    ],

    # Output formats
    output_formats=["PDF", "HTML", "Excel", "JSON", "XBRL", "iXBRL"],

    # Languages
    languages=["en", "de", "fr", "es"],

    # Features
    assurance_enabled=True,
    multi_framework_enabled=True,
)
```

### YAML Configuration

```yaml
# config/pack_config.yaml
pack_id: "PACK-030-net-zero-reporting"
pack_version: "1.0.0"

frameworks:
  SBTi:
    name: "SBTi"
    version: "v1.1"
    required_metrics:
      - scope1_total
      - scope2_total
      - scope3_total
    output_formats:
      - PDF
      - JSON

  CDP:
    name: "CDP"
    version: "2025"
    required_metrics:
      - scope1_total
      - scope2_location
      - scope2_market
    output_formats:
      - Excel
      - JSON
    deadline_months: 7

  TCFD:
    name: "TCFD"
    version: "2023"
    required_metrics:
      - scope1_total
      - scope2_total
      - scope3_total
    output_formats:
      - PDF
      - HTML

  GRI:
    name: "GRI 305"
    version: "2016"
    required_metrics:
      - scope1_by_gas
      - scope2_location
      - scope2_market
      - emissions_intensity
    output_formats:
      - PDF
      - HTML

  ISSB:
    name: "IFRS S2"
    version: "2023"
    required_metrics:
      - scope1_total
      - scope2_total
      - scope3_total
      - industry_metrics
    output_formats:
      - PDF
      - XBRL

  SEC:
    name: "SEC Climate"
    version: "2024"
    required_metrics:
      - scope1_total
      - scope2_total
    output_formats:
      - PDF
      - XBRL
      - iXBRL

  CSRD:
    name: "ESRS E1"
    version: "2024"
    required_metrics:
      - scope1_total
      - scope2_total
      - scope3_total
      - energy_consumption
      - energy_mix
    output_formats:
      - PDF

branding:
  logo_path: null
  primary_color: "#1E3A8A"
  secondary_color: "#3B82F6"
  font_family: "Arial, sans-serif"
  style: "corporate"

data_sources:
  - PACK-021
  - PACK-022
  - PACK-028
  - PACK-029
  - GL-SBTi-APP
  - GL-CDP-APP
  - GL-TCFD-APP
  - GL-GHG-APP

output_formats:
  - PDF
  - HTML
  - Excel
  - JSON
  - XBRL
  - iXBRL

languages:
  - en

assurance_enabled: true
multi_framework_enabled: true
```

---

## 3. Framework Configuration

### Per-Framework Settings

Each framework can be configured independently with the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | Required | Framework display name |
| `version` | string | Required | Framework version |
| `required_metrics` | list[str] | Required | Metrics that must be present |
| `optional_metrics` | list[str] | `[]` | Metrics that enhance the report |
| `output_formats` | list[str] | `["PDF"]` | Enabled output formats |
| `deadline_months` | int or null | null | Month number for annual deadline |
| `template_overrides` | dict | `{}` | Custom template section overrides |
| `narrative_style` | string | `"formal"` | Narrative tone (formal, concise, detailed) |
| `include_appendix` | bool | `true` | Include methodology appendix |

### Metric Names Reference

| Metric Name | Description |
|-------------|-------------|
| `scope1_total` | Total Scope 1 GHG emissions (tCO2e) |
| `scope1_by_gas` | Scope 1 broken down by GHG type |
| `scope2_location` | Scope 2 location-based emissions |
| `scope2_market` | Scope 2 market-based emissions |
| `scope2_total` | Scope 2 total (preferred approach) |
| `scope3_total` | Total Scope 3 emissions |
| `scope3_categories` | Scope 3 by category (1-15) |
| `scope3_all_categories` | All 15 Scope 3 categories |
| `emissions_intensity` | GHG intensity metric |
| `energy_consumption` | Total energy consumption |
| `energy_mix` | Renewable vs. non-renewable energy |
| `industry_metrics` | SASB industry-specific metrics |
| `reduction_targets` | GHG reduction targets |
| `reduction_progress` | Progress against targets |

---

## 4. Branding Configuration

### Branding Options

```yaml
branding:
  # Company logo (PNG/SVG, max 5MB)
  logo_path: "/path/to/company-logo.png"

  # Color scheme
  primary_color: "#1E3A8A"       # Headers, primary elements
  secondary_color: "#3B82F6"     # Accents, secondary elements
  accent_color: "#10B981"        # Highlights, success indicators
  warning_color: "#F59E0B"       # Warning indicators
  error_color: "#EF4444"         # Error indicators
  background_color: "#FFFFFF"    # Report background

  # Typography
  font_family: "Arial, sans-serif"
  heading_font: "Georgia, serif"
  font_size_base: 11             # Base font size in points

  # Style preset
  style: "corporate"              # corporate, executive, investor, minimal

  # Report elements
  include_cover_page: true
  include_page_numbers: true
  include_headers: true
  include_footers: true
  footer_text: "Confidential - For Internal Use Only"
  include_toc: true
  include_list_of_figures: false
  include_list_of_tables: true

  # Charts and visualization
  chart_color_palette:
    - "#1E3A8A"
    - "#3B82F6"
    - "#10B981"
    - "#F59E0B"
    - "#EF4444"
    - "#8B5CF6"
    - "#EC4899"
  chart_style: "modern"           # modern, classic, minimal
```

---

## 5. Data Source Configuration

### Pack Integration Settings

```yaml
data_sources:
  pack021:
    url: "http://pack-021:9021"
    api_key: "${PACK021_API_KEY}"
    timeout: 30
    retry_count: 3
    retry_delay: 1
    required: true
    data_points:
      - baseline_emissions
      - inventory_data
      - activity_data

  pack022:
    url: "http://pack-022:9022"
    api_key: "${PACK022_API_KEY}"
    timeout: 30
    retry_count: 3
    required: true
    data_points:
      - reduction_initiatives
      - macc_curves
      - abatement_costs

  pack028:
    url: "http://pack-028:9028"
    api_key: "${PACK028_API_KEY}"
    timeout: 30
    retry_count: 3
    required: true
    data_points:
      - sector_pathways
      - convergence_data
      - benchmarks

  pack029:
    url: "http://pack-029:9029"
    api_key: "${PACK029_API_KEY}"
    timeout: 30
    retry_count: 3
    required: true
    data_points:
      - interim_targets
      - progress_monitoring
      - variance_analysis
```

### Application Integration Settings

```yaml
applications:
  gl_sbti_app:
    url: "http://gl-sbti-app:8001"
    auth_type: "oauth2"
    client_id: "${SBTI_CLIENT_ID}"
    client_secret: "${SBTI_CLIENT_SECRET}"
    timeout: 30
    retry_count: 3

  gl_cdp_app:
    url: "http://gl-cdp-app:8002"
    auth_type: "oauth2"
    client_id: "${CDP_CLIENT_ID}"
    client_secret: "${CDP_CLIENT_SECRET}"
    timeout: 30

  gl_tcfd_app:
    url: "http://gl-tcfd-app:8003"
    auth_type: "oauth2"
    client_id: "${TCFD_CLIENT_ID}"
    client_secret: "${TCFD_CLIENT_SECRET}"
    timeout: 30

  gl_ghg_app:
    url: "http://gl-ghg-app:8004"
    auth_type: "oauth2"
    client_id: "${GHG_CLIENT_ID}"
    client_secret: "${GHG_CLIENT_SECRET}"
    timeout: 30
```

---

## 6. Output Configuration

### Format-Specific Settings

```yaml
output:
  pdf:
    engine: "weasyprint"
    page_size: "A4"
    orientation: "portrait"
    margin_top: "2cm"
    margin_bottom: "2cm"
    margin_left: "2.5cm"
    margin_right: "2.5cm"
    dpi: 300
    embed_fonts: true
    compress: true

  html:
    responsive: true
    include_javascript: true
    chart_library: "plotly"
    export_to_pdf: true
    minify: false

  excel:
    engine: "openpyxl"
    include_charts: true
    include_pivot_tables: false
    include_data_validation: true
    protect_formulas: true

  json:
    pretty_print: true
    include_metadata: true
    include_provenance: true
    api_version: "v1"

  xbrl:
    taxonomy: "sec-2024"
    validate: true
    include_context: true
    include_units: true

  ixbrl:
    base_template: "sec_ixbrl_template.html"
    highlight_tags: false
    viewer_enabled: true
```

---

## 7. Notification Configuration

```yaml
notifications:
  deadline_reminders:
    enabled: true
    days_before: [120, 90, 60, 30, 14, 7, 3, 1]
    channels:
      - type: "email"
        recipients:
          - "sustainability@company.com"
          - "cso@company.com"
      - type: "slack"
        webhook_url: "${SLACK_WEBHOOK_URL}"
        channel: "#climate-reporting"
      - type: "teams"
        webhook_url: "${TEAMS_WEBHOOK_URL}"

  report_completion:
    enabled: true
    channels:
      - type: "email"
        recipients:
          - "sustainability@company.com"

  validation_alerts:
    enabled: true
    severity_threshold: "high"
    channels:
      - type: "slack"
        webhook_url: "${SLACK_WEBHOOK_URL}"
```

---

## 8. Assurance Configuration

```yaml
assurance:
  enabled: true
  level: "limited"                    # none, limited, reasonable
  standard: "ISAE3410"               # ISAE3410, ISAE3000, AA1000

  evidence_bundle:
    include_provenance: true          # SHA-256 calculation hashes
    include_lineage_diagrams: true    # Visual data flow diagrams
    include_methodology_docs: true    # Calculation methodology
    include_control_matrix: true      # ISAE 3410 control requirements
    include_source_data: false        # Raw source data (large)
    include_change_log: true          # Data modification history

  provenance:
    algorithm: "SHA-256"
    include_timestamps: true
    include_user_ids: true
    immutable_log: true

  retention:
    years: 7                          # Evidence retention period
    storage: "s3"                     # s3, local, azure_blob
    encryption: true
```

---

## 9. Performance Configuration

```yaml
performance:
  # Caching
  cache:
    enabled: true
    backend: "redis"
    ttl_seconds: 3600
    max_memory: "512mb"
    eviction_policy: "lru"

  # Database connection pool
  database:
    pool_size: 20
    max_overflow: 10
    pool_timeout: 30
    pool_recycle: 1800

  # Parallelization
  parallel:
    max_workers: 8
    framework_parallel: true
    data_fetch_parallel: true

  # Rate limiting
  rate_limit:
    api_requests_per_second: 100
    report_generations_per_minute: 60
    external_api_calls_per_second: 10

  # Timeouts
  timeouts:
    data_aggregation: 30
    report_generation: 60
    pdf_rendering: 30
    xbrl_generation: 30
    translation: 30
```

---

## 10. Configuration Presets

PACK-030 includes 8 pre-built configuration presets:

| Preset | File | Primary Framework | Use Case |
|--------|------|-------------------|----------|
| `csrd_focus` | `config/presets/csrd_focus.yaml` | CSRD | EU ESRS E1 compliance |
| `cdp_alist` | `config/presets/cdp_alist.yaml` | CDP | CDP A-list scoring |
| `tcfd_investor` | `config/presets/tcfd_investor.yaml` | TCFD | Investor-grade disclosure |
| `sbti_validation` | `config/presets/sbti_validation.yaml` | SBTi | SBTi target validation |
| `sec_10k` | `config/presets/sec_10k.yaml` | SEC | SEC 10-K filing |
| `multi_framework` | `config/presets/multi_framework.yaml` | All 7 | Comprehensive reporting |
| `investor_relations` | `config/presets/investor_relations.yaml` | TCFD/ISSB | Investor package |
| `assurance_ready` | `config/presets/assurance_ready.yaml` | All 7 | ISAE 3410 audit |

### Loading a Preset

```python
from packs.net_zero.pack030.config import load_preset

# Load preset
config = load_preset("cdp_alist")

# Override specific settings
config.languages = ["en", "de"]
config.branding.logo_path = "/path/to/logo.png"
```

---

## 11. Environment Variables

All configuration settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | Required | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `PACK030_PORT` | `8030` | Service port |
| `PACK030_WORKERS` | `4` | Uvicorn workers |
| `PACK030_LOG_LEVEL` | `INFO` | Logging level |
| `DATA_AGGREGATION_TIMEOUT` | `30` | Data fetch timeout (seconds) |
| `REPORT_GENERATION_TIMEOUT` | `60` | Report generation timeout |
| `PDF_RENDERING_TIMEOUT` | `30` | PDF rendering timeout |
| `CACHE_TTL` | `3600` | Cache TTL (seconds) |
| `CACHE_MAX_MEMORY` | `512mb` | Redis max memory |
| `JWT_SECRET_KEY` | Required | JWT signing key |
| `DEEPL_API_KEY` | Optional | DeepL translation API key |
| `S3_BUCKET` | `greenlang-reports` | Report archive bucket |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
