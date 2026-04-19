# PRD: AGENT-DATA-008 - Supplier Questionnaire Processor

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-008 |
| **Agent ID** | GL-DATA-SUP-001 |
| **Component** | Supplier Questionnaire Processor Agent (Template Management, Multi-Channel Distribution, Response Collection, Validation Engine, Framework-Based Scoring, Follow-Up Automation, Analytics & Reporting) |
| **Category** | Data Intake Agent (Supplier Engagement / ESG Data Collection) |
| **Priority** | P0 - Critical (required for Scope 3 supplier data collection, CSRD/CSDDD compliance, CDP Supply Chain program) |
| **Status** | Layer 1 Partial (~3 files in agents/data/ and agents/procurement/), SDK Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS requires automated supplier sustainability questionnaire processing for comprehensive Scope 3 emissions data collection, ESG due diligence, and regulatory compliance. Without a production-grade Supplier Questionnaire Processor:

- **No standardized questionnaire templates**: Organizations cannot create framework-aligned sustainability questionnaires (CDP, SBTi, TCFD, GRI, EcoVadis, DJSI, CSRD, CSDDD)
- **No automated distribution**: Manual email-based distribution to thousands of suppliers is error-prone and unscalable
- **No response collection pipeline**: Supplier responses via email, portal, API, Excel uploads lack unified ingestion
- **No validation engine**: No automated completeness checking, consistency validation, or cross-reference verification of supplier responses
- **No framework-based scoring**: Cannot score supplier sustainability performance against CDP Climate Change (A-D), EcoVadis (0-100), or custom frameworks
- **No follow-up automation**: No automated reminders, escalation workflows, or deadline management for non-responsive suppliers
- **No analytics & benchmarking**: Cannot track response rates, benchmark supplier performance, identify laggards, or generate compliance reports
- **No questionnaire versioning**: Cannot track template evolution or compare responses across questionnaire versions
- **No multi-language support**: Cannot distribute questionnaires in supplier's preferred language
- **No audit trail**: Questionnaire lifecycle events not tracked for regulatory compliance

## 3. Existing Implementation

### 3.1 Layer 1: Supplier Data Exchange Agent
**File**: `greenlang/agents/data/supplier_data_exchange_agent.py` (~500 lines)

- GL-DATA-X-012 agent with PCF submission handling
- PCFStandard enum (PACT, TfS, GHG Protocol, ISO 14067, PEF, Custom)
- SubmissionStatus lifecycle (PENDING -> VALIDATED/REJECTED/NEEDS_REVISION -> ACCEPTED/EXPIRED)
- DataQualityRating (PRIMARY, SECONDARY, TERTIARY, UNKNOWN)
- SupplierInfo, ProductInfo, PCFDataPoint, PCFSubmission models
- PACT validation with minimum primary data share (20%)
- Supplier-to-product mapping
- Scope 3 Category 1 emissions calculation from PCF data
- SHA-256 provenance hashing
- In-memory storage only

### 3.2 Layer 1: Supplier Engagement Agent
**File**: `greenlang/agents/procurement/supplier_engagement_agent.py` (~400 lines)

- GL-PROC-X-003 agent with engagement program management
- EngagementPriority (CRITICAL, HIGH, MEDIUM, LOW, OPPORTUNISTIC)
- EngagementStatus lifecycle (NOT_STARTED -> INITIATED -> IN_PROGRESS -> ON_TRACK/AT_RISK/DELAYED -> COMPLETED/ESCALATED)
- ActionType enum (CDP_DISCLOSURE, SBTI_COMMITMENT, EMISSIONS_REPORTING, RENEWABLE_ENERGY, SUPPLIER_AUDIT, CAPACITY_BUILDING, JOINT_PROJECT, POLICY_IMPROVEMENT, CERTIFICATION, DATA_SHARING)
- Priority-based action planning
- Progress tracking and monitoring
- Engagement escalation management

### 3.3 Layer 1: Supplier Sustainability Scorer
**File**: `greenlang/agents/procurement/supplier_sustainability_scorer.py` (~400 lines)

- GL-PROC-X-001 agent with multi-dimensional scoring
- ScoreCategory (ENVIRONMENTAL, SOCIAL, GOVERNANCE, CLIMATE, OVERALL)
- DataQuality (HIGH, MEDIUM, LOW, ESTIMATED)
- RiskLevel (LOW, MEDIUM, HIGH, CRITICAL)
- PerformanceTier (LEADER, ADVANCED, DEVELOPING, LAGGING, NON_COMPLIANT)
- Default weights: Environmental 35%, Social 25%, Governance 20%, Climate 20%
- Data quality multipliers: HIGH 1.0, MEDIUM 0.9, LOW 0.75, ESTIMATED 0.5

### 3.4 Layer 1 Tests
None found in production test suite.

## 4. Identified Gaps

### Gap 1: No Supplier Questionnaire SDK Package
No `greenlang/supplier_questionnaire/` package providing a clean SDK wrapping Layer 1 capabilities into a production-grade questionnaire processing pipeline.

### Gap 2: No Questionnaire Template Engine
Layer 1 has no template management. Need versioned, framework-aligned questionnaire templates with section/question/option hierarchies, conditional logic, and multi-language support.

### Gap 3: No Distribution Engine
Layer 1 has no distribution capabilities. Need multi-channel distribution (email, portal link, API, bulk upload), scheduling, batch sending, and delivery tracking.

### Gap 4: No Response Collection Pipeline
Layer 1 has basic PCF submission handling. Need unified response ingestion from multiple channels with parsing, normalization, deduplication, and partial response support.

### Gap 5: No Production Validation Engine
Layer 1 has PACT-specific validation only. Need comprehensive validation: completeness checking, consistency validation, cross-field rules, framework-specific requirements, and data quality scoring.

### Gap 6: No Framework-Based Scoring Engine
Layer 1 has basic sustainability scoring. Need production scoring aligned to CDP Climate Change (A-D-), EcoVadis (0-100), DJSI (0-100), custom weighted frameworks, and benchmarking.

### Gap 7: No Follow-Up Automation Engine
Layer 1 has engagement tracking. Need automated reminder scheduling, escalation workflows, deadline management, and stakeholder notification.

### Gap 8: No Analytics & Reporting Engine
Layer 1 has no analytics. Need response rate tracking, supplier benchmarking, trend analysis, compliance gap identification, and multi-format report generation.

### Gap 9: No Prometheus Metrics
No 12-metric pattern for supplier questionnaire monitoring.

### Gap 10: No REST API
No FastAPI endpoints for questionnaire management operations.

### Gap 11: No Database Migration
No persistent storage for questionnaires, distributions, responses, scores, and analytics.

### Gap 12: No K8s/CI/CD
No deployment manifests or CI/CD pipeline.

## 5. Architecture (Final State)

### 5.1 SDK Package Structure

```
greenlang/supplier_questionnaire/
+-- __init__.py                  # Public API, agent metadata (GL-DATA-SUP-001)
+-- config.py                    # SupplierQuestionnaireConfig with GL_SUPPLIER_QUEST_ env prefix
+-- models.py                    # Pydantic v2 models for all data structures
+-- questionnaire_template.py    # QuestionnaireTemplateEngine - template CRUD + versioning
+-- distribution.py              # DistributionEngine - multi-channel distribution + tracking
+-- response_collector.py        # ResponseCollectorEngine - response ingestion + normalization
+-- validation_engine.py         # ValidationEngine - completeness + consistency + framework rules
+-- scoring_engine.py            # ScoringEngine - framework-based scoring + benchmarking
+-- follow_up.py                 # FollowUpEngine - reminders + escalation + deadline management
+-- analytics.py                 # AnalyticsEngine - response rates + benchmarking + reporting
+-- provenance.py                # ProvenanceTracker - SHA-256 chain-hashed audit trails
+-- metrics.py                   # 12 Prometheus metrics
+-- setup.py                     # SupplierQuestionnaireService facade
+-- api/
    +-- __init__.py
    +-- router.py                # FastAPI HTTP service with 20 endpoints
```

### 5.2 Seven Core Engines

#### Engine 1: QuestionnaireTemplateEngine
- Template CRUD with UUID identification and semantic versioning (SemVer)
- Framework-aligned templates: CDP Climate Change (11 modules), CDP Water Security, CDP Forests, EcoVadis (4 themes), DJSI (3 dimensions), GRI Standards, TCFD (4 pillars), CSRD/ESRS, CSDDD, SBTi, Custom
- Hierarchical structure: Template -> Sections -> Questions -> Options
- 8 question types: TEXT, NUMERIC, SINGLE_CHOICE, MULTI_CHOICE, DATE, FILE_UPLOAD, SCALE, BOOLEAN
- Conditional logic: show/hide questions based on previous answers (skip patterns)
- Multi-language support: 12 languages (EN, DE, FR, ES, IT, PT, NL, ZH, JA, KO, AR, HI)
- Template cloning and customization
- Version history with diff tracking
- Template validation (completeness, consistency, framework alignment)
- Template export/import (JSON, YAML)
- SHA-256 provenance on all template operations

#### Engine 2: DistributionEngine
- 5 distribution channels: EMAIL, PORTAL_LINK, API_PUSH, BULK_UPLOAD, MANUAL
- Batch distribution to supplier lists with personalization tokens
- Distribution scheduling: immediate, scheduled, recurring (annual, semi-annual, quarterly)
- Campaign management: group distributions into campaigns with tracking
- Supplier list management with inclusion/exclusion filters
- Delivery tracking: QUEUED -> SENT -> DELIVERED -> OPENED -> STARTED -> COMPLETED/EXPIRED
- Access token generation for portal-based questionnaires (secure, time-limited)
- Deadline assignment per distribution with configurable windows (30/60/90 days)
- Re-distribution for bounced or expired distributions
- Distribution summary statistics
- SHA-256 provenance on all distribution operations

#### Engine 3: ResponseCollectorEngine
- Multi-source response ingestion: portal submission, email attachment, API upload, Excel/CSV import
- Response parsing from structured (JSON/API) and semi-structured (Excel) sources
- Question-level response extraction with answer normalization
- Partial response support with progress tracking (% complete)
- Response deduplication by supplier + questionnaire + period
- Version conflict resolution (latest wins, with history)
- File attachment handling for supporting documents (PDF, images)
- Response acknowledgment with confirmation token
- Bulk response import with error reporting
- Re-opened response support for corrections
- SHA-256 provenance on all response operations

#### Engine 4: ValidationEngine
- 5 validation levels: STRUCTURAL, COMPLETENESS, CONSISTENCY, FRAMEWORK, DATA_QUALITY
- Structural validation: required fields, data types, allowed values
- Completeness validation: mandatory question coverage, section minimums
- Consistency validation: cross-field rules (e.g., Scope 1+2+3 = Total), year-over-year plausibility
- Framework-specific validation rules per questionnaire framework
- 6 CDP-specific checks: boundary completeness, methodology disclosure, target alignment, verification status, data quality flags, sector classification
- 4 EcoVadis-specific checks: evidence requirements, policy documentation, action plan specifics, certification verification
- Data quality scoring per response (0-100) based on completeness, consistency, evidence, verification
- Validation result classification: PASS, FAIL, WARNING, INFO
- Auto-fix suggestions for common validation failures
- Batch validation with summary reporting
- SHA-256 provenance on all validation operations

#### Engine 5: ScoringEngine
- 8 scoring frameworks with configurable weights and thresholds:
  - CDP Climate Change: A/A-, B/B-, C/C-, D/D- (4 tiers, 8 sub-grades, 11 module scores)
  - CDP Water Security: A-D scoring aligned with CDP methodology
  - CDP Forests: A-D scoring aligned with CDP methodology
  - EcoVadis: 0-100 composite (Environment, Labor, Ethics, Sustainable Procurement)
  - DJSI: 0-100 composite (Environmental, Social, Governance dimensions)
  - GRI: Compliance scoring per disclosure requirement
  - CSRD/ESRS: Double materiality scoring per ESRS standard
  - Custom: User-defined weights and thresholds
- Section-level and question-level scoring with weight propagation
- Benchmarking against industry averages and peer groups
- Year-over-year trend analysis with improvement/regression detection
- Performance tier assignment: LEADER (>80), ADVANCED (60-80), DEVELOPING (40-60), LAGGING (20-40), NON_COMPLIANT (<20)
- Score normalization for cross-framework comparison
- Confidence scoring based on response data quality
- Score export for integration with AGENT-DATA-003 ERP connector
- SHA-256 provenance on all scoring operations

#### Engine 6: FollowUpEngine
- 4 reminder types: GENTLE (7 days before deadline), FIRM (3 days before), URGENT (1 day before), FINAL (on deadline)
- Automated reminder scheduling based on distribution deadlines
- Escalation workflows: supplier contact -> procurement manager -> category lead -> VP supply chain
- 5 escalation levels with configurable thresholds and stakeholders
- Stakeholder notification on key events (response received, validation complete, score published)
- Non-response tracking with risk flagging
- Bulk reminder operations for campaign-wide follow-ups
- Follow-up effectiveness analytics (response rate by reminder count)
- Do-not-contact and opt-out respect
- Follow-up history with full audit trail
- SHA-256 provenance on all follow-up operations

#### Engine 7: AnalyticsEngine
- Campaign-level response rate tracking (sent, opened, started, completed, expired)
- Supplier-level response history and performance trends
- Framework compliance gap analysis (which questions/sections have lowest scores)
- Industry/sector benchmarking with percentile rankings
- Geographic heatmaps of supplier sustainability performance
- Year-over-year improvement tracking per supplier
- Data quality distribution analysis
- Top/bottom performer identification
- Risk scoring aggregation (combining questionnaire scores with other risk signals)
- 4 report formats: TEXT, JSON, MARKDOWN, HTML
- Dashboard data export for Grafana integration
- SHA-256 provenance on all analytics operations

### 5.3 Database Schema

**Schema**: `supplier_questionnaire_service`

| Table | Purpose | Type |
|-------|---------|------|
| `questionnaire_templates` | Template definitions with versioning | Regular |
| `template_sections` | Sections within questionnaire templates | Regular |
| `template_questions` | Questions within sections with options | Regular |
| `distributions` | Questionnaire distribution records | Regular |
| `distribution_events` | Distribution lifecycle event tracking | Hypertable |
| `responses` | Collected supplier responses | Regular |
| `response_answers` | Individual question-level answers | Regular |
| `validation_results` | Validation check results per response | Hypertable |
| `scores` | Framework-based scoring results | Regular |
| `follow_up_actions` | Follow-up reminders and escalations | Hypertable |

### 5.4 Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_supplier_quest_templates_total` | Counter | `framework`, `status` |
| 2 | `gl_supplier_quest_distributions_total` | Counter | `channel`, `status` |
| 3 | `gl_supplier_quest_responses_total` | Counter | `channel`, `status` |
| 4 | `gl_supplier_quest_validations_total` | Counter | `level`, `result` |
| 5 | `gl_supplier_quest_scores_total` | Counter | `framework`, `tier` |
| 6 | `gl_supplier_quest_followups_total` | Counter | `type`, `status` |
| 7 | `gl_supplier_quest_response_rate` | Gauge | `campaign_id` |
| 8 | `gl_supplier_quest_processing_duration_seconds` | Histogram | `operation` |
| 9 | `gl_supplier_quest_active_campaigns` | Gauge | - |
| 10 | `gl_supplier_quest_pending_responses` | Gauge | - |
| 11 | `gl_supplier_quest_processing_errors_total` | Counter | `engine`, `error_type` |
| 12 | `gl_supplier_quest_data_quality_score` | Histogram | `framework` |

### 5.5 REST API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/v1/questionnaires/templates` | Create questionnaire template |
| 2 | GET | `/v1/questionnaires/templates` | List questionnaire templates |
| 3 | GET | `/v1/questionnaires/templates/{template_id}` | Get template details |
| 4 | PUT | `/v1/questionnaires/templates/{template_id}` | Update template (new version) |
| 5 | POST | `/v1/questionnaires/templates/{template_id}/clone` | Clone template |
| 6 | POST | `/v1/questionnaires/distribute` | Distribute questionnaire to suppliers |
| 7 | GET | `/v1/questionnaires/distributions` | List distributions |
| 8 | GET | `/v1/questionnaires/distributions/{dist_id}` | Get distribution status |
| 9 | POST | `/v1/questionnaires/responses` | Submit questionnaire response |
| 10 | GET | `/v1/questionnaires/responses` | List responses with filters |
| 11 | GET | `/v1/questionnaires/responses/{response_id}` | Get response details |
| 12 | POST | `/v1/questionnaires/responses/{response_id}/validate` | Validate a response |
| 13 | POST | `/v1/questionnaires/score` | Score a validated response |
| 14 | GET | `/v1/questionnaires/scores/{score_id}` | Get score details |
| 15 | GET | `/v1/questionnaires/scores/supplier/{supplier_id}` | Get supplier score history |
| 16 | POST | `/v1/questionnaires/followup` | Trigger follow-up actions |
| 17 | GET | `/v1/questionnaires/followup/{campaign_id}` | Get follow-up status |
| 18 | GET | `/v1/questionnaires/analytics/{campaign_id}` | Get campaign analytics |
| 19 | GET | `/v1/questionnaires/health` | Health check |
| 20 | GET | `/v1/questionnaires/statistics` | Service statistics |

### 5.6 Configuration

**Environment Variable Prefix**: `GL_SUPPLIER_QUEST_`

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_SUPPLIER_QUEST_DATABASE_URL` | `""` | PostgreSQL connection string |
| `GL_SUPPLIER_QUEST_REDIS_URL` | `""` | Redis connection string |
| `GL_SUPPLIER_QUEST_LOG_LEVEL` | `"INFO"` | Logging level |
| `GL_SUPPLIER_QUEST_DEFAULT_FRAMEWORK` | `"custom"` | Default scoring framework |
| `GL_SUPPLIER_QUEST_DEFAULT_DEADLINE_DAYS` | `60` | Default response deadline |
| `GL_SUPPLIER_QUEST_MAX_REMINDERS` | `4` | Maximum reminder count per distribution |
| `GL_SUPPLIER_QUEST_REMINDER_GENTLE_DAYS` | `7` | Days before deadline for gentle reminder |
| `GL_SUPPLIER_QUEST_REMINDER_FIRM_DAYS` | `3` | Days before deadline for firm reminder |
| `GL_SUPPLIER_QUEST_REMINDER_URGENT_DAYS` | `1` | Days before deadline for urgent reminder |
| `GL_SUPPLIER_QUEST_MIN_COMPLETION_PCT` | `80.0` | Minimum completion % to accept response |
| `GL_SUPPLIER_QUEST_SCORE_LEADER_THRESHOLD` | `80` | Score threshold for LEADER tier |
| `GL_SUPPLIER_QUEST_SCORE_ADVANCED_THRESHOLD` | `60` | Score threshold for ADVANCED tier |
| `GL_SUPPLIER_QUEST_SCORE_DEVELOPING_THRESHOLD` | `40` | Score threshold for DEVELOPING tier |
| `GL_SUPPLIER_QUEST_SCORE_LAGGING_THRESHOLD` | `20` | Score threshold for LAGGING tier |
| `GL_SUPPLIER_QUEST_BATCH_SIZE` | `100` | Batch distribution size |
| `GL_SUPPLIER_QUEST_WORKER_COUNT` | `4` | Parallel processing workers |
| `GL_SUPPLIER_QUEST_CACHE_TTL_SECONDS` | `1800` | Score cache TTL (30 min) |
| `GL_SUPPLIER_QUEST_POOL_MIN_SIZE` | `2` | DB pool minimum |
| `GL_SUPPLIER_QUEST_POOL_MAX_SIZE` | `10` | DB pool maximum |
| `GL_SUPPLIER_QUEST_RETENTION_DAYS` | `1095` | Response retention (3 years) |
| `GL_SUPPLIER_QUEST_PORTAL_BASE_URL` | `""` | Base URL for portal-based questionnaires |
| `GL_SUPPLIER_QUEST_SMTP_HOST` | `""` | SMTP host for email distribution |
| `GL_SUPPLIER_QUEST_DEFAULT_LANGUAGE` | `"en"` | Default questionnaire language |

## 6. Completion Plan

### Phase 1: SDK Core
1. Build config.py, models.py, __init__.py
2. Build 7 core engines
3. Build provenance.py, metrics.py, setup.py
4. Build api/router.py

### Phase 2: Infrastructure
5. Build V038 database migration
6. Build K8s manifests (8 files)
7. Build CI/CD pipeline
8. Build Grafana dashboard + alerts

### Phase 3: Testing
9. Build 600+ unit tests across 14 test files

## 7. Success Criteria

- [ ] 7 engines with deterministic questionnaire processing
- [ ] 8+ questionnaire frameworks supported (CDP, EcoVadis, DJSI, GRI, TCFD, CSRD, CSDDD, SBTi)
- [ ] 5 distribution channels (email, portal, API, bulk upload, manual)
- [ ] 8 question types (text, numeric, single choice, multi choice, date, file upload, scale, boolean)
- [ ] 5 validation levels (structural, completeness, consistency, framework, data quality)
- [ ] 5 performance tiers (leader, advanced, developing, lagging, non_compliant)
- [ ] 4 reminder types (gentle, firm, urgent, final)
- [ ] 5 escalation levels with configurable workflows
- [ ] 12 languages supported for questionnaire templates
- [ ] 4 report formats (text, json, markdown, html)
- [ ] 20 REST API endpoints operational
- [ ] 12 Prometheus metrics instrumented
- [ ] SHA-256 provenance on all operations
- [ ] V038 database migration with 10 tables
- [ ] 600+ tests passing
- [ ] K8s manifests with full security hardening

## 8. Integration Points

### Upstream Dependencies
- AGENT-DATA-003 ERP/Finance Connector (supplier master data, spend data)
- AGENT-DATA-005 EUDR Traceability (supplier plot geolocation for EUDR questionnaires)
- AGENT-FOUND-002 Schema Compiler (questionnaire schema validation)
- AGENT-FOUND-006 Access Guard (authorization, tenant isolation)
- AGENT-FOUND-010 Observability Agent (metrics/tracing)

### Downstream Consumers
- GL-CSRD-APP (CSRD double materiality supplier data)
- GL-CSDDD-APP (supply chain due diligence questionnaire data)
- GL-SB253-APP (Scope 3 supplier emissions data)
- AGENT-DATA-002 Excel/CSV Normalizer (response import processing)
- AGENT-FOUND-005 Citations & Evidence (questionnaire response evidence packaging)
- Supplier Sustainability Scorer (GL-PROC-X-001, scoring input)
- Supplier Engagement Agent (GL-PROC-X-003, engagement tracking)
- Supplier Data Exchange Agent (GL-DATA-X-012, PCF data from questionnaires)

### Layer 1 Foundation Integration
- `greenlang/agents/data/supplier_data_exchange_agent.py` - PCF submission models, validation
- `greenlang/agents/procurement/supplier_engagement_agent.py` - Engagement status, action types
- `greenlang/agents/procurement/supplier_sustainability_scorer.py` - Score categories, performance tiers, weights
