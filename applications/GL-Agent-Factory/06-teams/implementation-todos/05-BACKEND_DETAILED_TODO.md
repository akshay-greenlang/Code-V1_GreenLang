# Backend Services & API Development - Detailed Implementation To-Do List

**Version:** 1.0.0
**Date:** 2025-12-04
**Lead:** GL-BackendDeveloper
**Priority:** P1 HIGH PRIORITY
**Total Tasks:** 565 tasks across 7 major sections
**Path:** `C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory`

---

## Executive Summary

This document provides a comprehensive, task-by-task implementation plan for GreenLang Agent Factory backend services. It covers:

- **Core Agent Services** (5 services, 180 tasks)
- **API Gateway Layer** (50 tasks)
- **Agent Implementation** (7 agents, 140 tasks)
- **Database Layer** (75 tasks)
- **Message Queue Integration** (60 tasks)
- **External Integrations** (60 tasks)

---

## SECTION 1: CORE AGENT SERVICES (180 Tasks)

### 1.1 Agent Execution Service (45 Tasks)

The Agent Execution Service is responsible for running agents with full lifecycle management, provenance tracking, and zero-hallucination enforcement.

#### 1.1.1 Execution Engine Core

- [ ] Create `AgentExecutionService` class in `services/execution/agent_execution_service.py`
- [ ] Implement `execute_agent(agent_id: str, input_data: Dict, context: ExecutionContext) -> ExecutionResult`
- [ ] Create `ExecutionContext` Pydantic model with tenant_id, user_id, correlation_id, timeout
- [ ] Implement `ExecutionResult` Pydantic model with result, provenance_hash, metrics, cost
- [ ] Add execution timeout handling with configurable limits (default: 300s, max: 3600s)
- [ ] Implement execution cancellation via `cancel_execution(execution_id: str) -> bool`
- [ ] Create execution state machine (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
- [ ] Add execution state transition validation
- [ ] Implement execution progress tracking (0-100%)
- [ ] Create execution checkpointing for long-running agents

#### 1.1.2 Input/Output Processing

- [ ] Implement `validate_input(agent_id: str, input_data: Dict) -> ValidationResult`
- [ ] Create input schema loading from agent pack.yaml
- [ ] Add input sanitization (remove sensitive data, normalize formats)
- [ ] Implement input size validation (max 10MB per request)
- [ ] Create `validate_output(agent_id: str, output_data: Any) -> ValidationResult`
- [ ] Add output schema validation against pack.yaml outputs
- [ ] Implement output provenance hash calculation (SHA-256)
- [ ] Create output serialization (JSON, MessagePack)

#### 1.1.3 Provenance Tracking

- [ ] Create `ProvenanceTracker` class in `services/execution/provenance_tracker.py`
- [ ] Implement `track_input(input_data: Dict) -> str` returning SHA-256 hash
- [ ] Add `track_output(output_data: Any) -> str` returning SHA-256 hash
- [ ] Create `track_calculation(formula_id: str, inputs: Dict, result: float) -> str`
- [ ] Implement `build_provenance_chain(steps: List[ProvenanceStep]) -> str`
- [ ] Add provenance storage to PostgreSQL `execution_provenance` table
- [ ] Create provenance verification `verify_provenance(execution_id: str) -> bool`
- [ ] Implement provenance export for audit (JSON, CSV)

#### 1.1.4 Cost Tracking

- [ ] Create `CostTracker` class in `services/execution/cost_tracker.py`
- [ ] Implement LLM token counting (input, output, cached)
- [ ] Add LLM cost calculation per model (Claude, GPT-4)
- [ ] Create compute cost estimation (CPU seconds * rate)
- [ ] Implement storage cost tracking (S3 bytes * rate)
- [ ] Add `calculate_execution_cost(execution_id: str) -> CostBreakdown`
- [ ] Create cost aggregation by agent, tenant, period
- [ ] Implement cost budget enforcement per tenant

#### 1.1.5 Execution Metrics

- [ ] Create execution latency metric (p50, p95, p99)
- [ ] Implement execution success rate metric
- [ ] Add execution throughput metric (executions/minute)
- [ ] Create execution queue depth metric
- [ ] Implement execution cost per agent metric
- [ ] Add execution error rate by error type
- [ ] Create execution resource utilization metrics

### 1.2 Agent Registry Service (40 Tasks)

The Agent Registry Service manages agent lifecycle, versioning, and discovery.

#### 1.2.1 Registry Core

- [ ] Create `AgentRegistryService` class in `services/registry/agent_registry_service.py`
- [ ] Implement `register_agent(agent_spec: AgentSpec) -> AgentRegistration`
- [ ] Add `get_agent(agent_id: str) -> Agent` with caching
- [ ] Create `list_agents(filters: AgentFilters, pagination: PaginationParams) -> List[Agent]`
- [ ] Implement `update_agent(agent_id: str, updates: AgentUpdate) -> Agent`
- [ ] Add `delete_agent(agent_id: str) -> bool` (soft delete)
- [ ] Create agent state machine (DRAFT, EXPERIMENTAL, CERTIFIED, DEPRECATED, RETIRED)
- [ ] Implement state transition validation and hooks

#### 1.2.2 Version Management

- [ ] Create `VersionManager` class in `services/registry/version_manager.py`
- [ ] Implement `create_version(agent_id: str, version: str, artifact: bytes) -> AgentVersion`
- [ ] Add semantic version validation (X.Y.Z format)
- [ ] Create version uniqueness enforcement per agent
- [ ] Implement `get_version(agent_id: str, version: str) -> AgentVersion`
- [ ] Add `get_latest_version(agent_id: str) -> AgentVersion`
- [ ] Create `list_versions(agent_id: str) -> List[AgentVersion]`
- [ ] Implement version comparison (is_compatible, is_upgrade)
- [ ] Add version deprecation with sunset date
- [ ] Create version artifact storage to S3

#### 1.2.3 Agent Discovery

- [ ] Create `AgentDiscoveryService` class in `services/registry/agent_discovery_service.py`
- [ ] Implement `search_agents(query: str, filters: Dict) -> List[AgentMatch]`
- [ ] Add keyword search on name, description, tags
- [ ] Create vector similarity search using embeddings
- [ ] Implement hybrid search (keyword + vector)
- [ ] Add `get_similar_agents(agent_id: str, limit: int) -> List[Agent]`
- [ ] Create `get_trending_agents(period: str, limit: int) -> List[Agent]`
- [ ] Implement `get_recommended_agents(user_id: str) -> List[Agent]`
- [ ] Add search result ranking and scoring

#### 1.2.4 Capability Registry

- [ ] Create `CapabilityRegistry` class
- [ ] Implement capability indexing from pack.yaml
- [ ] Add `get_agents_by_capability(capability: str) -> List[Agent]`
- [ ] Create capability taxonomy (emissions, cbam, csrd, eudr, etc.)
- [ ] Implement capability compatibility checking

#### 1.2.5 Registry Metrics

- [ ] Create agent count metric by status
- [ ] Implement version count metric by agent
- [ ] Add search latency metric
- [ ] Create discovery success rate metric
- [ ] Implement registry API latency metrics

### 1.3 Calculation Engine Service (35 Tasks)

The Calculation Engine Service provides zero-hallucination deterministic calculations.

#### 1.3.1 Calculation Engine Core

- [ ] Create `CalculationEngineService` class in `services/calculation/calculation_engine_service.py`
- [ ] Implement `calculate(formula_id: str, inputs: Dict) -> CalculationResult`
- [ ] Add formula registry loading from YAML database
- [ ] Create formula validation (syntax, units, bounds)
- [ ] Implement formula execution with Python eval sandbox
- [ ] Add formula caching for repeated calculations
- [ ] Create calculation provenance tracking

#### 1.3.2 Emission Factor Database

- [ ] Create `EmissionFactorDatabase` class in `services/calculation/emission_factor_db.py`
- [ ] Implement `get_emission_factor(material_id: str, region: str, year: int) -> EmissionFactor`
- [ ] Add emission factor versioning (quarterly updates)
- [ ] Create emission factor source tracking (EPA, IPCC, DEFRA, IEA)
- [ ] Implement emission factor uncertainty ranges
- [ ] Add `list_emission_factors(filters: Dict) -> List[EmissionFactor]`
- [ ] Create emission factor caching (LRU, 10000 entries)
- [ ] Implement emission factor validation on load

#### 1.3.3 Unit Conversion

- [ ] Create `UnitConverter` class in `services/calculation/unit_converter.py`
- [ ] Implement `convert(value: float, from_unit: str, to_unit: str) -> float`
- [ ] Add unit compatibility validation
- [ ] Create unit dimension tracking (mass, energy, volume, emissions)
- [ ] Implement unit alias resolution (kg, kilogram, KG -> kg)
- [ ] Add compound unit handling (kgCO2e/kWh)
- [ ] Create unit conversion provenance

#### 1.3.4 Emissions Calculations

- [ ] Implement `calculate_scope1_emissions(fuel_data: List[FuelConsumption]) -> Emissions`
- [ ] Add `calculate_scope2_emissions(energy_data: List[EnergyConsumption]) -> Emissions`
- [ ] Create `calculate_scope3_emissions(category: int, activity_data: Dict) -> Emissions`
- [ ] Implement all 15 Scope 3 categories
- [ ] Add emissions aggregation by scope, source, geography
- [ ] Create emissions uncertainty calculation
- [ ] Implement GWP set selection (AR5, AR6)

#### 1.3.5 Calculation Metrics

- [ ] Create calculation latency metric
- [ ] Implement calculation accuracy metric
- [ ] Add emission factor cache hit rate
- [ ] Create calculation throughput metric

### 1.4 Workflow Orchestration Service (35 Tasks)

The Workflow Orchestration Service manages multi-agent pipelines and complex workflows.

#### 1.4.1 Workflow Engine Core

- [ ] Create `WorkflowOrchestrationService` class in `services/workflow/workflow_orchestration_service.py`
- [ ] Implement `create_workflow(definition: WorkflowDefinition) -> Workflow`
- [ ] Add `execute_workflow(workflow_id: str, input: Dict) -> WorkflowExecution`
- [ ] Create workflow state machine (PENDING, RUNNING, PAUSED, COMPLETED, FAILED)
- [ ] Implement workflow state persistence to PostgreSQL
- [ ] Add workflow cancellation and cleanup
- [ ] Create workflow timeout handling

#### 1.4.2 Workflow Definition

- [ ] Create `WorkflowDefinition` Pydantic model
- [ ] Implement YAML/JSON workflow definition parsing
- [ ] Add workflow validation (DAG validation, no cycles)
- [ ] Create step dependency resolution
- [ ] Implement workflow versioning
- [ ] Add workflow templates for common patterns

#### 1.4.3 Workflow Steps

- [ ] Create `AgentExecutionStep` for running agents
- [ ] Implement `DataTransformationStep` for data processing
- [ ] Add `ValidationStep` for data validation
- [ ] Create `ConditionalStep` for branching logic
- [ ] Implement `ParallelStep` for concurrent execution
- [ ] Add `WaitStep` for delays and scheduling
- [ ] Create `NotificationStep` for alerts
- [ ] Implement `LoopStep` for iteration

#### 1.4.4 Workflow State Management

- [ ] Create `WorkflowStateStore` for persisting workflow state
- [ ] Implement checkpoint and recovery mechanism
- [ ] Add step output caching for retry
- [ ] Create workflow history tracking
- [ ] Implement workflow replay for debugging

#### 1.4.5 Workflow Metrics

- [ ] Create workflow execution time metric
- [ ] Implement workflow success rate metric
- [ ] Add step execution time metric
- [ ] Create workflow queue depth metric
- [ ] Implement workflow cost tracking

### 1.5 Audit Logging Service (25 Tasks)

The Audit Logging Service provides comprehensive audit trails for compliance.

#### 1.5.1 Audit Logger Core

- [ ] Create `AuditLoggingService` class in `services/audit/audit_logging_service.py`
- [ ] Implement `log_event(event: AuditEvent) -> str`
- [ ] Add `AuditEvent` Pydantic model (actor, action, resource, timestamp, context)
- [ ] Create audit event categories (AGENT, EXECUTION, USER, SYSTEM)
- [ ] Implement async audit logging (non-blocking)
- [ ] Add audit log batching for performance

#### 1.5.2 Audit Storage

- [ ] Create `audit_logs` table in PostgreSQL
- [ ] Implement audit log partitioning by month
- [ ] Add audit log indexing (actor, action, resource, timestamp)
- [ ] Create audit log retention policy (7 years)
- [ ] Implement audit log archival to S3

#### 1.5.3 Audit Query

- [ ] Create `query_audit_logs(filters: AuditFilters) -> List[AuditEvent]`
- [ ] Implement filtering by actor, action, resource, date range
- [ ] Add full-text search on audit context
- [ ] Create audit log aggregation queries
- [ ] Implement audit log export (CSV, JSON)

#### 1.5.4 Compliance Features

- [ ] Create tamper-evident audit log (hash chain)
- [ ] Implement audit log integrity verification
- [ ] Add regulatory compliance reports (SOC2, ISO27001)
- [ ] Create audit log access logging (who viewed what)

---

## SECTION 2: API GATEWAY LAYER (50 Tasks)

### 2.1 FastAPI Application Structure (10 Tasks)

- [ ] Create `app/main.py` FastAPI application entry point
- [ ] Implement application factory pattern `create_app() -> FastAPI`
- [ ] Add OpenAPI documentation configuration (title, version, description)
- [ ] Create API router structure (`routers/agents.py`, `routers/executions.py`, etc.)
- [ ] Implement lifespan events (startup, shutdown)
- [ ] Add database connection pool initialization on startup
- [ ] Create Redis connection initialization on startup
- [ ] Implement graceful shutdown with request draining
- [ ] Add health check endpoint `GET /health`
- [ ] Create readiness check endpoint `GET /ready`

### 2.2 Authentication Middleware (JWT/OAuth2) (10 Tasks)

- [ ] Create `middleware/auth.py` authentication middleware
- [ ] Implement `JWTAuthMiddleware` class
- [ ] Add JWT token extraction from Authorization header
- [ ] Create JWT token validation (signature, expiration, issuer)
- [ ] Implement JWT token refresh mechanism
- [ ] Add OAuth2 authorization code flow support
- [ ] Create API key authentication fallback
- [ ] Implement user context injection into request state
- [ ] Add authentication bypass for public endpoints
- [ ] Create authentication error handling (401, 403)

### 2.3 Rate Limiting Middleware (8 Tasks)

- [ ] Create `middleware/rate_limit.py` rate limiting middleware
- [ ] Implement token bucket algorithm
- [ ] Add Redis-based distributed rate limiting
- [ ] Create per-user rate limits (configurable, default: 100 req/min)
- [ ] Implement per-endpoint rate limits (sensitive endpoints: 10 req/min)
- [ ] Add rate limit headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)
- [ ] Create rate limit bypass for admin users
- [ ] Implement rate limit exceeded response (HTTP 429)

### 2.4 Request Validation (Pydantic) (8 Tasks)

- [ ] Create comprehensive Pydantic models for all request bodies
- [ ] Implement `AgentCreateRequest` model with validation
- [ ] Add `AgentUpdateRequest` model with partial update support
- [ ] Create `ExecutionRequest` model with input validation
- [ ] Implement `SearchRequest` model with filter validation
- [ ] Add custom validators for domain-specific fields (CN codes, ISO dates)
- [ ] Create nested model validation
- [ ] Implement request size validation middleware (max 10MB)

### 2.5 Response Serialization (7 Tasks)

- [ ] Create `ResponseEnvelope` standard response wrapper
- [ ] Implement success response format `{"data": ..., "meta": {...}}`
- [ ] Add error response format `{"error": {"code": ..., "message": ..., "details": [...]}}`
- [ ] Create pagination response format with next/prev links
- [ ] Implement content negotiation (JSON, YAML)
- [ ] Add response compression (gzip) middleware
- [ ] Create response timing headers (X-Response-Time)

### 2.6 API Versioning (7 Tasks)

- [ ] Implement URL-based versioning (`/v1/`, `/v2/`)
- [ ] Create version routing middleware
- [ ] Add Accept header version negotiation (application/vnd.greenlang.v1+json)
- [ ] Implement version deprecation headers (Deprecation, Sunset)
- [ ] Create version-specific OpenAPI documentation
- [ ] Add version compatibility checking
- [ ] Implement graceful version migration support

---

## SECTION 3: AGENT IMPLEMENTATION (140 Tasks - 20 per Agent)

### 3.1 GL-001: Carbon Emissions Calculator Agent (20 Tasks)

#### 3.1.1 Agent Core Implementation

- [ ] Create `agents/gl_001_carbon_emissions/agent.py`
- [ ] Implement `CarbonEmissionsAgent` class extending `CalculatorAgentBase`
- [ ] Create `CarbonEmissionsInput` Pydantic model (fuel_type, quantity, unit, region)
- [ ] Add `CarbonEmissionsOutput` Pydantic model (emissions_kgco2e, emission_factor_used, provenance_hash)
- [ ] Implement `get_calculation_parameters()` method
- [ ] Create `validate_calculation_result()` method

#### 3.1.2 Calculation Logic

- [ ] Implement Scope 1 emissions calculation (stationary combustion)
- [ ] Add Scope 1 emissions calculation (mobile combustion)
- [ ] Create Scope 2 location-based emissions calculation
- [ ] Implement Scope 2 market-based emissions calculation
- [ ] Add emissions unit conversion (kg, t, MT)
- [ ] Create emissions rounding per regulatory requirements

#### 3.1.3 Input Validation

- [ ] Validate fuel_type against supported fuel list
- [ ] Add quantity validation (positive number)
- [ ] Create unit validation against allowed units
- [ ] Implement region validation (ISO 3166 country codes)
- [ ] Add cross-field validation (fuel_type + region compatibility)

#### 3.1.4 Testing & Documentation

- [ ] Create 100 golden test cases
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with emission factor database
- [ ] Create agent pack.yaml specification

### 3.2 GL-002: CBAM Compliance Agent (20 Tasks)

#### 3.2.1 Agent Core Implementation

- [ ] Create `agents/gl_002_cbam_compliance/agent.py`
- [ ] Implement `CBAMComplianceAgent` class extending `RegulatoryAgentBase`
- [ ] Create `CBAMInput` Pydantic model (importer_info, shipment_data, cn_code, origin_country)
- [ ] Add `CBAMOutput` Pydantic model (embedded_emissions, cbam_liability, report_data, provenance_hash)
- [ ] Implement `map_to_framework()` method for CBAM schema
- [ ] Create `_check_compliance()` method

#### 3.2.2 CBAM Calculation Logic

- [ ] Implement embedded emissions calculation
- [ ] Add precursor emissions calculation
- [ ] Create direct emissions from production
- [ ] Implement indirect emissions from electricity
- [ ] Add CBAM liability calculation (emissions * carbon_price)
- [ ] Create carbon price paid deduction

#### 3.2.3 CBAM Validation

- [ ] Validate CN code format (8-digit)
- [ ] Add CN code lookup against CBAM goods list
- [ ] Create origin country validation
- [ ] Implement importer registration validation
- [ ] Add shipment data completeness check

#### 3.2.4 CBAM Reporting

- [ ] Generate CBAM quarterly report JSON
- [ ] Create CBAM declaration XML
- [ ] Implement CBAM audit trail generation
- [ ] Add CBAM report PDF export

#### 3.2.5 Testing & Documentation

- [ ] Create 150 golden test cases (all CN codes)
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with EU CBAM schema
- [ ] Create agent pack.yaml specification

### 3.3 GL-003: CSRD Reporting Agent (20 Tasks)

#### 3.3.1 Agent Core Implementation

- [ ] Create `agents/gl_003_csrd_reporting/agent.py`
- [ ] Implement `CSRDReportingAgent` class extending `ReportingAgentBase`
- [ ] Create `CSRDInput` Pydantic model (company_profile, esrs_data, materiality_assessment)
- [ ] Add `CSRDOutput` Pydantic model (esrs_report, datapoint_coverage, compliance_status)
- [ ] Implement `prepare_report_data()` method
- [ ] Create `generate_charts()` method for visualizations

#### 3.3.2 ESRS Data Processing

- [ ] Implement E1 Climate metrics processing
- [ ] Add E2 Pollution metrics processing
- [ ] Create E3 Water metrics processing
- [ ] Implement E4 Biodiversity metrics processing
- [ ] Add E5 Circular Economy metrics processing
- [ ] Create S1-S4 Social metrics processing
- [ ] Implement G1 Governance metrics processing

#### 3.3.3 CSRD Validation

- [ ] Validate double materiality assessment completeness
- [ ] Add ESRS datapoint coverage validation (>80% required)
- [ ] Create ESRS topic compliance check
- [ ] Implement transition plan validation

#### 3.3.4 CSRD Report Generation

- [ ] Generate ESRS-compliant iXBRL report
- [ ] Create PDF management report
- [ ] Implement Excel datapoint inventory
- [ ] Add CSRD gap analysis report

#### 3.3.5 Testing & Documentation

- [ ] Create 200 golden test cases (all ESRS standards)
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with ESRS taxonomy
- [ ] Create agent pack.yaml specification

### 3.4 GL-004: EUDR Compliance Agent (20 Tasks)

#### 3.4.1 Agent Core Implementation

- [ ] Create `agents/gl_004_eudr_compliance/agent.py`
- [ ] Implement `EUDRComplianceAgent` class extending `RegulatoryAgentBase`
- [ ] Create `EUDRInput` Pydantic model (commodity_type, geolocation, supply_chain_docs)
- [ ] Add `EUDROutput` Pydantic model (risk_assessment, due_diligence_statement, compliance_status)
- [ ] Implement `_identify_gaps()` method
- [ ] Create `_generate_recommendations()` method

#### 3.4.2 EUDR Risk Assessment

- [ ] Implement country risk scoring
- [ ] Add region risk scoring
- [ ] Create supplier history risk assessment
- [ ] Implement deforestation risk calculation
- [ ] Add land cover change detection integration
- [ ] Create composite risk score calculation

#### 3.4.3 EUDR Validation

- [ ] Validate commodity type against EUDR scope (cattle, cocoa, coffee, palm oil, rubber, soya, wood)
- [ ] Add geolocation data validation (GPS coordinates, polygons)
- [ ] Create supply chain traceability validation
- [ ] Implement production date validation (post-2020 cutoff)

#### 3.4.4 EUDR Due Diligence

- [ ] Generate EU DDS-compliant JSON statement
- [ ] Create risk mitigation measures documentation
- [ ] Implement traceability map (GeoJSON)
- [ ] Add non-compliance alert generation

#### 3.4.5 Testing & Documentation

- [ ] Create 150 golden test cases (all commodities)
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with EU DDS schema
- [ ] Create agent pack.yaml specification

### 3.5 GL-005: Building Energy Agent (20 Tasks)

#### 3.5.1 Agent Core Implementation

- [ ] Create `agents/gl_005_building_energy/agent.py`
- [ ] Implement `BuildingEnergyAgent` class extending `CalculatorAgentBase`
- [ ] Create `BuildingEnergyInput` Pydantic model (building_info, meter_data, hvac_data)
- [ ] Add `BuildingEnergyOutput` Pydantic model (energy_consumption, efficiency_rating, recommendations)
- [ ] Implement `get_calculation_parameters()` method
- [ ] Create `validate_calculation_result()` method

#### 3.5.2 Energy Calculation Logic

- [ ] Implement total energy consumption calculation
- [ ] Add energy use intensity (EUI) calculation
- [ ] Create energy efficiency rating
- [ ] Implement renewable energy percentage
- [ ] Add energy cost calculation
- [ ] Create energy benchmark comparison

#### 3.5.3 Building Energy Validation

- [ ] Validate building type against categories
- [ ] Add meter data completeness validation
- [ ] Create meter reading range validation
- [ ] Implement HVAC system validation

#### 3.5.4 Recommendations Engine

- [ ] Implement energy efficiency recommendations
- [ ] Add cost-benefit analysis for improvements
- [ ] Create payback period calculation
- [ ] Implement carbon reduction potential

#### 3.5.5 Testing & Documentation

- [ ] Create 100 golden test cases
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with energy databases
- [ ] Create agent pack.yaml specification

### 3.6 GL-006: Scope 3 Emissions Agent (20 Tasks)

#### 3.6.1 Agent Core Implementation

- [ ] Create `agents/gl_006_scope3_emissions/agent.py`
- [ ] Implement `Scope3EmissionsAgent` class extending `CalculatorAgentBase`
- [ ] Create `Scope3Input` Pydantic model (category, activity_data, supplier_data)
- [ ] Add `Scope3Output` Pydantic model (emissions_by_category, data_quality_scores, methodology)
- [ ] Implement `get_calculation_parameters()` method
- [ ] Create `validate_calculation_result()` method

#### 3.6.2 Scope 3 Category Calculations

- [ ] Implement Category 1: Purchased goods and services
- [ ] Add Category 2: Capital goods
- [ ] Create Category 3: Fuel and energy activities
- [ ] Implement Category 4: Upstream transportation
- [ ] Add Category 5: Waste generated
- [ ] Create Category 6: Business travel
- [ ] Implement Category 7: Employee commuting
- [ ] Add Categories 8-15 (remaining categories)

#### 3.6.3 Data Quality Assessment

- [ ] Implement GHG Protocol data quality indicators
- [ ] Add supplier-specific vs. spend-based data scoring
- [ ] Create uncertainty quantification

#### 3.6.4 Reporting Features

- [ ] Generate Scope 3 inventory report
- [ ] Create CDP response format export
- [ ] Implement SBTi submission format

#### 3.6.5 Testing & Documentation

- [ ] Create 200 golden test cases (all 15 categories)
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests
- [ ] Create agent pack.yaml specification

### 3.7 GL-007: EU Taxonomy Agent (20 Tasks)

#### 3.7.1 Agent Core Implementation

- [ ] Create `agents/gl_007_eu_taxonomy/agent.py`
- [ ] Implement `EUTaxonomyAgent` class extending `RegulatoryAgentBase`
- [ ] Create `EUTaxonomyInput` Pydantic model (economic_activities, revenue_data, capex_data, opex_data)
- [ ] Add `EUTaxonomyOutput` Pydantic model (taxonomy_alignment, kpis, dnsh_assessment)
- [ ] Implement `map_to_framework()` method
- [ ] Create `_check_compliance()` method

#### 3.7.2 Taxonomy Assessment Logic

- [ ] Implement NACE code to Taxonomy activity mapping
- [ ] Add eligibility assessment (activity in taxonomy)
- [ ] Create substantial contribution criteria evaluation
- [ ] Implement DNSH (Do No Significant Harm) assessment
- [ ] Add minimum safeguards verification

#### 3.7.3 Taxonomy KPI Calculation

- [ ] Implement Taxonomy-aligned revenue KPI
- [ ] Add Taxonomy-aligned CapEx KPI
- [ ] Create Taxonomy-aligned OpEx KPI
- [ ] Implement eligibility vs. alignment breakdown

#### 3.7.4 Taxonomy Reporting

- [ ] Generate Taxonomy disclosure (ESRS E1 format)
- [ ] Create activity assessment Excel export
- [ ] Implement investor screening JSON API

#### 3.7.5 Testing & Documentation

- [ ] Create 150 golden test cases (all activities)
- [ ] Implement unit tests with 90% coverage
- [ ] Add integration tests with EU Delegated Acts
- [ ] Create agent pack.yaml specification

---

## SECTION 4: DATABASE LAYER (75 Tasks)

### 4.1 SQLAlchemy Models (25 Tasks)

#### 4.1.1 Core Models

- [ ] Create `models/agent.py` with `Agent` SQLAlchemy model
- [ ] Add Agent columns: id, agent_id, name, description, category, status, tenant_id
- [ ] Create `models/agent_version.py` with `AgentVersion` model
- [ ] Add AgentVersion columns: id, agent_id, version, artifact_path, checksum, created_at
- [ ] Create `models/execution.py` with `Execution` model
- [ ] Add Execution columns: id, execution_id, agent_id, status, input_hash, output_hash
- [ ] Create `models/audit_log.py` with `AuditLog` model
- [ ] Add AuditLog columns: id, actor, action, resource, timestamp, context

#### 4.1.2 Tenant Models

- [ ] Create `models/tenant.py` with `Tenant` model
- [ ] Add Tenant columns: id, tenant_id, name, slug, config, quotas, created_at
- [ ] Create `models/user.py` with `User` model
- [ ] Add User columns: id, user_id, email, tenant_id, roles, created_at
- [ ] Create `models/tenant_quota.py` with `TenantQuota` model

#### 4.1.3 Metrics Models

- [ ] Create `models/agent_metrics.py` with `AgentMetrics` model
- [ ] Add metrics columns: agent_id, timestamp, invocations, latency_p50, latency_p95
- [ ] Create `models/execution_cost.py` with `ExecutionCost` model

#### 4.1.4 Provenance Models

- [ ] Create `models/provenance.py` with `ProvenanceRecord` model
- [ ] Add provenance columns: execution_id, step_type, input_hash, output_hash, formula_id

#### 4.1.5 Emission Factor Models

- [ ] Create `models/emission_factor.py` with `EmissionFactor` model
- [ ] Add EF columns: id, material_id, region, year, value, unit, source, uncertainty

#### 4.1.6 Workflow Models

- [ ] Create `models/workflow.py` with `Workflow` model
- [ ] Add workflow columns: id, workflow_id, definition, status, tenant_id
- [ ] Create `models/workflow_execution.py` with `WorkflowExecution` model

### 4.2 Alembic Migrations (15 Tasks)

#### 4.2.1 Initial Migration

- [ ] Create initial migration `001_initial_schema.py`
- [ ] Add agents table with all columns and constraints
- [ ] Create agent_versions table with foreign key to agents
- [ ] Add executions table with indexes
- [ ] Create audit_logs table with partitioning setup

#### 4.2.2 Tenant Migration

- [ ] Create migration `002_add_tenants.py`
- [ ] Add tenants table
- [ ] Create users table with tenant_id foreign key
- [ ] Add tenant_id column to agents, executions, audit_logs
- [ ] Create Row-Level Security policies

#### 4.2.3 Metrics Migration

- [ ] Create migration `003_add_metrics.py`
- [ ] Add agent_metrics table with time-series indexes
- [ ] Create execution_costs table
- [ ] Add materialized views for aggregations

#### 4.2.4 Additional Migrations

- [ ] Create migration `004_add_workflows.py`
- [ ] Create migration `005_add_emission_factors.py`
- [ ] Create migration `006_add_provenance.py`

### 4.3 Repository Pattern Implementation (20 Tasks)

#### 4.3.1 Base Repository

- [ ] Create `repositories/base.py` with `BaseRepository` abstract class
- [ ] Implement `create(entity: T) -> T` method
- [ ] Add `get(id: str) -> Optional[T]` method
- [ ] Create `update(id: str, updates: Dict) -> T` method
- [ ] Implement `delete(id: str) -> bool` method
- [ ] Add `list(filters: Dict, pagination: Pagination) -> List[T]` method

#### 4.3.2 Agent Repository

- [ ] Create `repositories/agent_repository.py`
- [ ] Implement `AgentRepository` extending `BaseRepository`
- [ ] Add `get_by_agent_id(agent_id: str) -> Agent`
- [ ] Create `search(query: str, filters: Dict) -> List[Agent]`
- [ ] Implement `get_by_category(category: str) -> List[Agent]`

#### 4.3.3 Execution Repository

- [ ] Create `repositories/execution_repository.py`
- [ ] Implement `ExecutionRepository` extending `BaseRepository`
- [ ] Add `get_by_execution_id(execution_id: str) -> Execution`
- [ ] Create `get_by_agent(agent_id: str, pagination: Pagination) -> List[Execution]`

#### 4.3.4 Audit Repository

- [ ] Create `repositories/audit_repository.py`
- [ ] Implement `AuditRepository` extending `BaseRepository`
- [ ] Add `query(filters: AuditFilters) -> List[AuditLog]`
- [ ] Create `aggregate(group_by: str, period: str) -> Dict`

#### 4.3.5 Tenant Repository

- [ ] Create `repositories/tenant_repository.py`
- [ ] Implement `TenantRepository` extending `BaseRepository`
- [ ] Add tenant-specific query methods

### 4.4 Connection Pooling (10 Tasks)

#### 4.4.1 Pool Configuration

- [ ] Create `db/connection.py` with connection pool setup
- [ ] Configure SQLAlchemy engine with pool settings (min: 5, max: 20)
- [ ] Add pool timeout configuration (30 seconds)
- [ ] Create pool recycle setting (1 hour)
- [ ] Implement pool overflow (10 connections)

#### 4.4.2 Pool Management

- [ ] Create connection health check
- [ ] Implement connection retry on failure
- [ ] Add pool metrics collection (active, idle, waiting)
- [ ] Create pool warmup on startup
- [ ] Implement graceful pool shutdown

### 4.5 Row-Level Security (5 Tasks)

- [ ] Create RLS policy function `current_tenant_id()`
- [ ] Implement RLS policy for agents table
- [ ] Add RLS policy for executions table
- [ ] Create RLS policy for audit_logs table
- [ ] Test cross-tenant data isolation

---

## SECTION 5: MESSAGE QUEUE INTEGRATION (60 Tasks)

### 5.1 Redis Streams for Async Processing (20 Tasks)

#### 5.1.1 Stream Configuration

- [ ] Create `messaging/redis_streams.py` Redis Streams client
- [ ] Implement `create_stream(stream_name: str) -> bool`
- [ ] Add stream configuration (max length, trim strategy)
- [ ] Create consumer group setup
- [ ] Implement stream partitioning strategy

#### 5.1.2 Producer Implementation

- [ ] Create `StreamProducer` class
- [ ] Implement `publish(stream: str, message: Dict) -> str` returning message ID
- [ ] Add batch publishing `publish_batch(stream: str, messages: List[Dict]) -> List[str]`
- [ ] Create message serialization (JSON, MessagePack)
- [ ] Implement producer retry logic
- [ ] Add producer metrics (messages sent, latency)

#### 5.1.3 Consumer Implementation

- [ ] Create `StreamConsumer` class
- [ ] Implement `consume(stream: str, handler: Callable) -> None`
- [ ] Add consumer group management
- [ ] Create message acknowledgment `ack(stream: str, message_id: str)`
- [ ] Implement pending message handling (XPENDING, XCLAIM)
- [ ] Add consumer health check and heartbeat
- [ ] Create consumer concurrency configuration

#### 5.1.4 Stream Monitoring

- [ ] Create stream length monitoring
- [ ] Implement consumer lag tracking
- [ ] Add stream throughput metrics
- [ ] Create pending message alerts

### 5.2 Event Publishing (15 Tasks)

#### 5.2.1 Event Publisher Core

- [ ] Create `events/event_publisher.py` with `EventPublisher` class
- [ ] Implement `publish_event(event_type: str, payload: Dict) -> str`
- [ ] Add event schema validation
- [ ] Create event correlation ID tracking
- [ ] Implement event timestamp and metadata

#### 5.2.2 Domain Events

- [ ] Create `AgentCreatedEvent` event class
- [ ] Implement `AgentUpdatedEvent` event class
- [ ] Add `ExecutionStartedEvent` event class
- [ ] Create `ExecutionCompletedEvent` event class
- [ ] Implement `ExecutionFailedEvent` event class
- [ ] Add `AuditLogEvent` event class

#### 5.2.3 Event Publishing Patterns

- [ ] Implement outbox pattern for reliable publishing
- [ ] Add event batching for high-throughput
- [ ] Create event priority queuing
- [ ] Implement event deduplication

### 5.3 Event Consumption (15 Tasks)

#### 5.3.1 Event Consumer Core

- [ ] Create `events/event_consumer.py` with `EventConsumer` class
- [ ] Implement `subscribe(event_type: str, handler: Callable) -> None`
- [ ] Add event deserialization and validation
- [ ] Create error handling for failed handlers
- [ ] Implement consumer retry with backoff

#### 5.3.2 Event Handlers

- [ ] Create metrics update handler for execution events
- [ ] Implement audit log handler for all events
- [ ] Add notification handler for alert events
- [ ] Create cache invalidation handler
- [ ] Implement search index update handler

#### 5.3.3 Consumer Patterns

- [ ] Implement at-least-once delivery guarantee
- [ ] Add idempotent event handling
- [ ] Create event ordering within partition
- [ ] Implement consumer offset management

### 5.4 Dead Letter Handling (10 Tasks)

#### 5.4.1 DLQ Infrastructure

- [ ] Create `messaging/dead_letter_queue.py` with `DeadLetterQueue` class
- [ ] Implement `enqueue(message: Dict, error: str, context: Dict) -> str`
- [ ] Add DLQ storage to PostgreSQL table
- [ ] Create DLQ retention policy (30 days)

#### 5.4.2 DLQ Operations

- [ ] Implement `get_dlq_messages(limit: int) -> List[DLQMessage]`
- [ ] Add `reprocess_message(message_id: str) -> bool`
- [ ] Create `reprocess_all(limit: int) -> int` batch reprocessing
- [ ] Implement `purge_dlq(older_than: datetime) -> int`
- [ ] Add DLQ monitoring and alerting
- [ ] Create DLQ export for analysis

---

## SECTION 6: EXTERNAL INTEGRATIONS (60 Tasks)

### 6.1 ERP Connectors (SAP, Oracle) (25 Tasks)

#### 6.1.1 SAP Connector

- [ ] Create `integrations/erp/sap_connector.py` with `SAPConnector` class
- [ ] Implement OAuth2 authentication for SAP
- [ ] Add `get_purchase_orders(filters: Dict) -> List[PurchaseOrder]`
- [ ] Create `get_material_masters(filters: Dict) -> List[Material]`
- [ ] Implement `get_vendors(filters: Dict) -> List[Vendor]`
- [ ] Add `get_invoices(filters: Dict) -> List[Invoice]`
- [ ] Create `get_gl_postings(filters: Dict) -> List[GLPosting]`
- [ ] Implement pagination handling
- [ ] Add rate limiting (100 req/min)
- [ ] Create retry logic with exponential backoff
- [ ] Implement SAP RFC/BAPI support

#### 6.1.2 Oracle Connector

- [ ] Create `integrations/erp/oracle_connector.py` with `OracleConnector` class
- [ ] Implement REST API authentication for Oracle
- [ ] Add `get_ap_invoices(filters: Dict) -> List[Invoice]`
- [ ] Create `get_gl_accounts(filters: Dict) -> List[Account]`
- [ ] Implement `get_procurement_data(filters: Dict) -> List[Procurement]`
- [ ] Add pagination handling
- [ ] Create rate limiting

#### 6.1.3 ERP Data Transformation

- [ ] Create `UnifiedERPData` Pydantic model
- [ ] Implement `transform_sap_to_unified(sap_data: Dict) -> UnifiedERPData`
- [ ] Add `transform_oracle_to_unified(oracle_data: Dict) -> UnifiedERPData`
- [ ] Create field mapping configuration
- [ ] Implement data validation post-transformation

### 6.2 File Parsers (CSV, Excel, XML) (20 Tasks)

#### 6.2.1 CSV Parser

- [ ] Create `integrations/parsers/csv_parser.py` with `CSVParser` class
- [ ] Implement `parse(file_path: str, schema: Dict) -> List[Dict]`
- [ ] Add delimiter detection
- [ ] Create encoding detection
- [ ] Implement header validation
- [ ] Add data type inference
- [ ] Create row-level error handling

#### 6.2.2 Excel Parser

- [ ] Create `integrations/parsers/excel_parser.py` with `ExcelParser` class
- [ ] Implement `parse(file_path: str, sheet_name: str, schema: Dict) -> List[Dict]`
- [ ] Add multi-sheet handling
- [ ] Create merged cell handling
- [ ] Implement date/time parsing
- [ ] Add formula evaluation option

#### 6.2.3 XML Parser

- [ ] Create `integrations/parsers/xml_parser.py` with `XMLParser` class
- [ ] Implement `parse(file_path: str, schema: Dict) -> Dict`
- [ ] Add XPath query support
- [ ] Create namespace handling
- [ ] Implement XML validation against XSD

#### 6.2.4 Parser Utilities

- [ ] Create `validate_file_format(file_path: str, expected_format: str) -> bool`
- [ ] Implement `detect_file_format(file_path: str) -> str`
- [ ] Add file size validation
- [ ] Create streaming parser for large files

### 6.3 API Clients for Data Sources (15 Tasks)

#### 6.3.1 Emission Factor APIs

- [ ] Create `integrations/api/epa_client.py` for EPA eGRID data
- [ ] Implement `get_grid_emission_factor(region: str, year: int) -> EmissionFactor`
- [ ] Add `integrations/api/ipcc_client.py` for IPCC factors
- [ ] Create `integrations/api/defra_client.py` for DEFRA factors
- [ ] Implement caching for API responses

#### 6.3.2 Geolocation APIs

- [ ] Create `integrations/api/geolocation_client.py`
- [ ] Implement `validate_coordinates(lat: float, lon: float) -> bool`
- [ ] Add `get_country_from_coordinates(lat: float, lon: float) -> str`
- [ ] Create satellite imagery integration for EUDR

#### 6.3.3 Carbon Registry APIs

- [ ] Create `integrations/api/vcs_client.py` for Verified Carbon Standard
- [ ] Implement `verify_offset(project_id: str, vintage: int) -> OffsetVerification`
- [ ] Add `integrations/api/gold_standard_client.py`
- [ ] Create offset quality scoring

#### 6.3.4 API Client Utilities

- [ ] Create `APIClient` base class with common functionality
- [ ] Implement connection pooling
- [ ] Add circuit breaker pattern
- [ ] Create response caching layer

---

## APPENDIX A: Implementation Priorities

### Phase 1: Foundation (Weeks 1-4) - 180 Tasks

| Component | Tasks | Priority |
|-----------|-------|----------|
| Agent Execution Service | 45 | P0 |
| Agent Registry Service | 40 | P0 |
| FastAPI Application | 10 | P0 |
| Authentication Middleware | 10 | P0 |
| SQLAlchemy Models | 25 | P0 |
| Alembic Migrations | 15 | P0 |
| GL-001 Carbon Emissions | 20 | P0 |
| GL-002 CBAM Compliance | 20 | P0 |

### Phase 2: Core Features (Weeks 5-8) - 200 Tasks

| Component | Tasks | Priority |
|-----------|-------|----------|
| Calculation Engine Service | 35 | P1 |
| Workflow Orchestration | 35 | P1 |
| Rate Limiting Middleware | 8 | P1 |
| Request Validation | 8 | P1 |
| Response Serialization | 7 | P1 |
| Repository Pattern | 20 | P1 |
| Connection Pooling | 10 | P1 |
| Redis Streams | 20 | P1 |
| GL-003 CSRD Reporting | 20 | P1 |
| GL-004 EUDR Compliance | 20 | P1 |
| GL-005 Building Energy | 20 | P1 |

### Phase 3: Enterprise (Weeks 9-12) - 185 Tasks

| Component | Tasks | Priority |
|-----------|-------|----------|
| Audit Logging Service | 25 | P2 |
| API Versioning | 7 | P2 |
| Event Publishing | 15 | P2 |
| Event Consumption | 15 | P2 |
| Dead Letter Handling | 10 | P2 |
| Row-Level Security | 5 | P2 |
| ERP Connectors | 25 | P2 |
| File Parsers | 20 | P2 |
| API Clients | 15 | P2 |
| GL-006 Scope 3 | 20 | P2 |
| GL-007 EU Taxonomy | 20 | P2 |

---

## APPENDIX B: Summary Statistics

### Total Tasks by Section

| Section | Task Count |
|---------|------------|
| 1. Core Agent Services | 180 |
| 2. API Gateway Layer | 50 |
| 3. Agent Implementation (7 agents) | 140 |
| 4. Database Layer | 75 |
| 5. Message Queue Integration | 60 |
| 6. External Integrations | 60 |
| **TOTAL** | **565** |

### Task Distribution by Type

| Task Type | Count |
|-----------|-------|
| Service Implementation | 180 |
| API Endpoints | 50 |
| Agent Classes | 140 |
| Database Models/Migrations | 75 |
| Messaging/Events | 60 |
| External Integrations | 60 |

### Estimated Effort

| Phase | FTE-Weeks | Duration |
|-------|-----------|----------|
| Phase 1: Foundation | 36 | 4 weeks |
| Phase 2: Core Features | 40 | 4 weeks |
| Phase 3: Enterprise | 37 | 4 weeks |
| **TOTAL** | **113** | **12 weeks** |

---

## APPENDIX C: Code Quality Standards

### Required for All Code

1. **Type Safety**
   - 100% type hint coverage
   - Pydantic models for all data structures
   - MyPy strict mode passing

2. **Testing**
   - 85%+ unit test coverage
   - Integration tests for all services
   - Golden tests for all agents

3. **Documentation**
   - Docstrings for all public methods
   - API documentation in OpenAPI
   - Architecture decision records

4. **Zero-Hallucination**
   - No LLM calls in calculation paths
   - All numeric operations deterministic
   - Provenance tracking for all calculations

5. **Security**
   - Bandit security scanning
   - Secrets detection
   - Input sanitization

### Performance Targets

| Metric | Target |
|--------|--------|
| API Response Time (p95) | <200ms |
| Database Query Time (p95) | <50ms |
| Agent Execution Time (p95) | <5s |
| Cache Hit Rate | >80% |
| Message Processing Latency | <100ms |

---

## APPENDIX D: File Structure

```
greenlang-backend/
|-- app/
|   |-- main.py                    # FastAPI application entry
|   |-- config.py                  # Configuration management
|   |-- routers/
|   |   |-- agents.py              # Agent CRUD endpoints
|   |   |-- executions.py          # Execution endpoints
|   |   |-- search.py              # Search endpoints
|   |   |-- metrics.py             # Metrics endpoints
|   |   |-- tenants.py             # Tenant endpoints
|   |   |-- audit.py               # Audit endpoints
|   |-- middleware/
|   |   |-- auth.py                # JWT authentication
|   |   |-- rate_limit.py          # Rate limiting
|   |   |-- request_validation.py  # Request validation
|   |   |-- response_format.py     # Response formatting
|-- services/
|   |-- execution/
|   |   |-- agent_execution_service.py
|   |   |-- provenance_tracker.py
|   |   |-- cost_tracker.py
|   |-- registry/
|   |   |-- agent_registry_service.py
|   |   |-- version_manager.py
|   |   |-- agent_discovery_service.py
|   |-- calculation/
|   |   |-- calculation_engine_service.py
|   |   |-- emission_factor_db.py
|   |   |-- unit_converter.py
|   |-- workflow/
|   |   |-- workflow_orchestration_service.py
|   |   |-- workflow_steps.py
|   |-- audit/
|   |   |-- audit_logging_service.py
|-- agents/
|   |-- gl_001_carbon_emissions/
|   |   |-- agent.py
|   |   |-- pack.yaml
|   |   |-- tests/
|   |-- gl_002_cbam_compliance/
|   |   |-- agent.py
|   |   |-- pack.yaml
|   |   |-- tests/
|   |-- gl_003_csrd_reporting/
|   |-- gl_004_eudr_compliance/
|   |-- gl_005_building_energy/
|   |-- gl_006_scope3_emissions/
|   |-- gl_007_eu_taxonomy/
|-- models/
|   |-- agent.py
|   |-- agent_version.py
|   |-- execution.py
|   |-- audit_log.py
|   |-- tenant.py
|   |-- user.py
|   |-- emission_factor.py
|   |-- workflow.py
|-- repositories/
|   |-- base.py
|   |-- agent_repository.py
|   |-- execution_repository.py
|   |-- audit_repository.py
|   |-- tenant_repository.py
|-- db/
|   |-- connection.py
|   |-- migrations/
|   |   |-- versions/
|   |   |   |-- 001_initial_schema.py
|   |   |   |-- 002_add_tenants.py
|   |   |   |-- 003_add_metrics.py
|-- messaging/
|   |-- redis_streams.py
|   |-- dead_letter_queue.py
|-- events/
|   |-- event_publisher.py
|   |-- event_consumer.py
|   |-- domain_events.py
|-- integrations/
|   |-- erp/
|   |   |-- sap_connector.py
|   |   |-- oracle_connector.py
|   |-- parsers/
|   |   |-- csv_parser.py
|   |   |-- excel_parser.py
|   |   |-- xml_parser.py
|   |-- api/
|   |   |-- epa_client.py
|   |   |-- ipcc_client.py
|   |   |-- vcs_client.py
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- golden/
|-- requirements.txt
|-- pyproject.toml
|-- alembic.ini
```

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-BackendDeveloper | Initial comprehensive backend TODO |

**Approvals:**

- Engineering Lead: ___________________ Date: _______
- Architecture Lead: ___________________ Date: _______
- Product Manager: ___________________ Date: _______

---

**END OF BACKEND SERVICES & API DEVELOPMENT TODO LIST**