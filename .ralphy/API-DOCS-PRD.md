# API Documentation Technical Debt - Complete PRD

## Phase 1: Core Documentation Infrastructure

- [ ] Create docs/api/agents/foundation/ directory and index.md with overview of all 10 Foundation agents listing name, description, endpoint count, and links to individual docs
- [ ] Create docs/api/agents/data/ directory and index.md with overview of all 20 Data agents listing name, description, endpoint count, and links to individual docs
- [ ] Create docs/api/agents/mrv/ directory and index.md with overview of all 30 MRV agents organized by Scope 1/2/3, listing name, description, endpoint count
- [ ] Create docs/api/agents/eudr/ directory and index.md with overview of all 40 EUDR agents organized by category (Traceability/Risk/DueDiligence/Support/Workflow)
- [ ] Create docs/api/infrastructure/ directory and index.md listing all infrastructure services (JWT, RBAC, Agent Factory, Feature Flags, Audit)
- [ ] Create docs/api/applications/ directory and index.md listing all 10 applications with purpose and endpoint counts

## Phase 2: Foundation Agent API Documentation (AGENT-FOUND-001 to 010)

- [ ] Document Orchestrator API (AGENT-FOUND-001): Read greenlang/agents/foundation/orchestrator/ router files, extract all endpoints, write docs/api/agents/foundation/orchestrator.md with method, path, summary, request/response JSON examples, status codes
- [ ] Document Schema Compiler API (AGENT-FOUND-002): Read greenlang/agents/foundation/schema_compiler/ router files, write docs/api/agents/foundation/schema_compiler.md
- [ ] Document Unit Normalizer API (AGENT-FOUND-003): Read greenlang/agents/foundation/unit_normalizer/ router files, write docs/api/agents/foundation/unit_normalizer.md
- [ ] Document Assumptions Registry API (AGENT-FOUND-004): Read greenlang/agents/foundation/assumptions_registry/ router files, write docs/api/agents/foundation/assumptions_registry.md
- [ ] Document Citations Evidence API (AGENT-FOUND-005): Read greenlang/agents/foundation/citations_evidence/ router files, write docs/api/agents/foundation/citations_evidence.md
- [ ] Document Access Policy Guard API (AGENT-FOUND-006): Read greenlang/agents/foundation/access_policy/ router files, write docs/api/agents/foundation/access_policy.md
- [ ] Document Agent Registry API (AGENT-FOUND-007): Read greenlang/agents/foundation/agent_registry/ router files, write docs/api/agents/foundation/agent_registry.md
- [ ] Document Reproducibility API (AGENT-FOUND-008): Read greenlang/agents/foundation/reproducibility/ router files, write docs/api/agents/foundation/reproducibility.md
- [ ] Document QA Harness API (AGENT-FOUND-009): Read greenlang/agents/foundation/qa_harness/ router files, write docs/api/agents/foundation/qa_harness.md
- [ ] Document Observability API (AGENT-FOUND-010): Read greenlang/agents/foundation/observability/ router files, write docs/api/agents/foundation/observability.md

## Phase 3: Data Agent API Documentation (AGENT-DATA-001 to 020)

- [ ] Document PDF Extractor API (AGENT-DATA-001): Read greenlang/agents/data/pdf_extractor/ router files, write docs/api/agents/data/pdf_extractor.md with all endpoints, request/response examples
- [ ] Document Excel Normalizer API (AGENT-DATA-002): Read greenlang/agents/data/excel_normalizer/ router files, write docs/api/agents/data/excel_normalizer.md
- [ ] Document ERP Connector API (AGENT-DATA-003): Read greenlang/agents/data/erp_connector/ router files, write docs/api/agents/data/erp_connector.md
- [ ] Document API Gateway Agent (AGENT-DATA-004): Read greenlang/agents/data/api_gateway/ router files, write docs/api/agents/data/api_gateway.md
- [ ] Document EUDR Traceability API (AGENT-DATA-005): Read greenlang/agents/data/eudr_traceability/ router files, write docs/api/agents/data/eudr_traceability.md
- [ ] Document GIS Mapping API (AGENT-DATA-006): Read greenlang/agents/data/gis_mapping/ router files, write docs/api/agents/data/gis_mapping.md
- [ ] Document Satellite Connector API (AGENT-DATA-007): Read greenlang/agents/data/satellite_connector/ router files, write docs/api/agents/data/satellite_connector.md
- [ ] Document Questionnaire Processor API (AGENT-DATA-008): Read greenlang/agents/data/questionnaire_processor/ router files, write docs/api/agents/data/questionnaire_processor.md
- [ ] Document Spend Categorizer API (AGENT-DATA-009): Read greenlang/agents/data/spend_categorizer/ router files, write docs/api/agents/data/spend_categorizer.md
- [ ] Document Data Quality Profiler API (AGENT-DATA-010): Read greenlang/agents/data/data_quality_profiler/ router files, write docs/api/agents/data/data_quality_profiler.md
- [ ] Document Duplicate Detection API (AGENT-DATA-011): Read greenlang/agents/data/duplicate_detection/ router files, write docs/api/agents/data/duplicate_detection.md
- [ ] Document Missing Value Imputer API (AGENT-DATA-012): Read greenlang/agents/data/missing_value_imputer/ router files, write docs/api/agents/data/missing_value_imputer.md
- [ ] Document Outlier Detection API (AGENT-DATA-013): Read greenlang/agents/data/outlier_detection/ router files, write docs/api/agents/data/outlier_detection.md
- [ ] Document Time Series Gap Filler API (AGENT-DATA-014): Read greenlang/agents/data/time_series_gap_filler/ router files, write docs/api/agents/data/time_series_gap_filler.md
- [ ] Document Cross-Source Reconciliation API (AGENT-DATA-015): Read greenlang/agents/data/cross_source_reconciliation/ router files, write docs/api/agents/data/cross_source_reconciliation.md
- [ ] Document Data Freshness Monitor API (AGENT-DATA-016): Read greenlang/agents/data/data_freshness_monitor/ router files, write docs/api/agents/data/data_freshness_monitor.md
- [ ] Document Schema Migration API (AGENT-DATA-017): Read greenlang/agents/data/schema_migration/ router files, write docs/api/agents/data/schema_migration.md
- [ ] Document Data Lineage Tracker API (AGENT-DATA-018): Read greenlang/agents/data/data_lineage_tracker/ router files, write docs/api/agents/data/data_lineage_tracker.md
- [ ] Document Validation Rule Engine API (AGENT-DATA-019): Read greenlang/agents/data/validation_rule_engine/ router files, write docs/api/agents/data/validation_rule_engine.md
- [ ] Document Climate Hazard Connector API (AGENT-DATA-020): Read greenlang/agents/data/climate_hazard_connector/ router files, write docs/api/agents/data/climate_hazard_connector.md

## Phase 4: MRV Agent API Documentation (Scope 1: AGENT-MRV-001 to 008)

- [ ] Document Stationary Combustion API (AGENT-MRV-001): Read greenlang/agents/mrv/stationary_combustion/ router files, write docs/api/agents/mrv/stationary_combustion.md
- [ ] Document Refrigerant API (AGENT-MRV-002): Read greenlang/agents/mrv/refrigerant/ router files, write docs/api/agents/mrv/refrigerant.md
- [ ] Document Mobile Combustion API (AGENT-MRV-003): Read greenlang/agents/mrv/mobile_combustion/ router files, write docs/api/agents/mrv/mobile_combustion.md
- [ ] Document Process Emissions API (AGENT-MRV-004): Read greenlang/agents/mrv/process_emissions/ router files, write docs/api/agents/mrv/process_emissions.md
- [ ] Document Fugitive Emissions API (AGENT-MRV-005): Read greenlang/agents/mrv/fugitive_emissions/ router files, write docs/api/agents/mrv/fugitive_emissions.md
- [ ] Document Land Use API (AGENT-MRV-006): Read greenlang/agents/mrv/land_use/ router files, write docs/api/agents/mrv/land_use.md
- [ ] Document Waste Emissions API (AGENT-MRV-007): Read greenlang/agents/mrv/waste_emissions/ router files, write docs/api/agents/mrv/waste_emissions.md
- [ ] Document Agricultural Emissions API (AGENT-MRV-008): Read greenlang/agents/mrv/agricultural/ router files, write docs/api/agents/mrv/agricultural.md

## Phase 5: MRV Agent API Documentation (Scope 2: AGENT-MRV-009 to 013)

- [ ] Document Location-Based API (AGENT-MRV-009): Read greenlang/agents/mrv/location_based/ router files, write docs/api/agents/mrv/location_based.md
- [ ] Document Market-Based API (AGENT-MRV-010): Read greenlang/agents/mrv/market_based/ router files, write docs/api/agents/mrv/market_based.md
- [ ] Document Steam API (AGENT-MRV-011): Read greenlang/agents/mrv/steam/ router files, write docs/api/agents/mrv/steam.md
- [ ] Document Cooling API (AGENT-MRV-012): Read greenlang/agents/mrv/cooling/ router files, write docs/api/agents/mrv/cooling.md
- [ ] Document Dual Reporting API (AGENT-MRV-013): Read greenlang/agents/mrv/dual_reporting/ router files, write docs/api/agents/mrv/dual_reporting.md

## Phase 6: MRV Agent API Documentation (Scope 3: AGENT-MRV-014 to 030)

- [ ] Document Scope 3 Cat 1 Purchased Goods API (AGENT-MRV-014): Read router files, write docs/api/agents/mrv/scope3_cat01_purchased_goods.md
- [ ] Document Scope 3 Cat 2 Capital Goods API (AGENT-MRV-015): Read router files, write docs/api/agents/mrv/scope3_cat02_capital_goods.md
- [ ] Document Scope 3 Cat 3 Fuel & Energy API (AGENT-MRV-016): Read router files, write docs/api/agents/mrv/scope3_cat03_fuel_energy.md
- [ ] Document Scope 3 Cat 4 Upstream Transport API (AGENT-MRV-017): Read router files, write docs/api/agents/mrv/scope3_cat04_upstream_transport.md
- [ ] Document Scope 3 Cat 5 Waste API (AGENT-MRV-018): Read router files, write docs/api/agents/mrv/scope3_cat05_waste.md
- [ ] Document Scope 3 Cat 6 Business Travel API (AGENT-MRV-019): Read router files, write docs/api/agents/mrv/scope3_cat06_business_travel.md
- [ ] Document Scope 3 Cat 7 Employee Commuting API (AGENT-MRV-020): Read router files, write docs/api/agents/mrv/scope3_cat07_employee_commuting.md
- [ ] Document Scope 3 Cat 8 Upstream Leased API (AGENT-MRV-021): Read router files, write docs/api/agents/mrv/scope3_cat08_upstream_leased.md
- [ ] Document Scope 3 Cat 9 Downstream Transport API (AGENT-MRV-022): Read router files, write docs/api/agents/mrv/scope3_cat09_downstream_transport.md
- [ ] Document Scope 3 Cat 10 Processing API (AGENT-MRV-023): Read router files, write docs/api/agents/mrv/scope3_cat10_processing.md
- [ ] Document Scope 3 Cat 11 Use of Sold Products API (AGENT-MRV-024): Read router files, write docs/api/agents/mrv/scope3_cat11_use_sold_products.md
- [ ] Document Scope 3 Cat 12 End-of-Life API (AGENT-MRV-025): Read router files, write docs/api/agents/mrv/scope3_cat12_end_of_life.md
- [ ] Document Scope 3 Cat 13 Downstream Leased API (AGENT-MRV-026): Read router files, write docs/api/agents/mrv/scope3_cat13_downstream_leased.md
- [ ] Document Scope 3 Cat 14 Franchises API (AGENT-MRV-027): Read router files, write docs/api/agents/mrv/scope3_cat14_franchises.md
- [ ] Document Scope 3 Cat 15 Investments API (AGENT-MRV-028): Read router files, write docs/api/agents/mrv/scope3_cat15_investments.md
- [ ] Document Category Mapper API (AGENT-MRV-029): Read router files, write docs/api/agents/mrv/category_mapper.md
- [ ] Document Audit Trail API (AGENT-MRV-030): Read router files, write docs/api/agents/mrv/audit_trail.md

## Phase 7: Infrastructure Service API Documentation

- [ ] Document JWT Authentication Service API: Read greenlang/infrastructure/auth/ or security/auth/ router files, write docs/api/infrastructure/authentication.md with login, token refresh, token validation endpoints
- [ ] Document RBAC Authorization API: Read greenlang/infrastructure/rbac/ or security/rbac/ router files, write docs/api/infrastructure/authorization.md with role assignment, permission check endpoints
- [ ] Document Agent Factory API (INFRA-010): Read greenlang/infrastructure/agent_factory/ router files, write docs/api/infrastructure/agent_factory.md
- [ ] Document Feature Flags API (INFRA-008): Read greenlang/infrastructure/feature_flags/ router files, write docs/api/infrastructure/feature_flags.md
- [ ] Document Centralized Audit Logging API (SEC-005): Read greenlang/infrastructure/audit/ router files, write docs/api/infrastructure/audit_logging.md

## Phase 8: Application API Documentation

- [ ] Document GL-CSRD-APP API: Read applications/GL-CSRD-APP/ router files, write docs/api/applications/gl-csrd-app.md
- [ ] Document GL-CBAM-APP API: Read applications/GL-CBAM-APP/ router files, write docs/api/applications/gl-cbam-app.md
- [ ] Document GL-EUDR-APP API: Read applications/GL-EUDR-APP/ router files, write docs/api/applications/gl-eudr-app.md
- [ ] Document GL-GHG-APP API: Read applications/GL-GHG-APP/ router files, write docs/api/applications/gl-ghg-app.md
- [ ] Document GL-VCCI-Carbon-APP API: Read applications/GL-VCCI-Carbon-APP/ router files, write docs/api/applications/gl-vcci-app.md
- [ ] Document GL-ISO14064-APP API: Read applications/GL-ISO14064-APP/ router files, write docs/api/applications/gl-iso14064-app.md
- [ ] Document GL-CDP-APP API: Read applications/GL-CDP-APP/ router files, write docs/api/applications/gl-cdp-app.md
- [ ] Document GL-TCFD-APP API: Read applications/GL-TCFD-APP/ router files, write docs/api/applications/gl-tcfd-app.md
- [ ] Document GL-SBTi-APP API: Read applications/GL-SBTi-APP/ router files, write docs/api/applications/gl-sbti-app.md
- [ ] Document GL-Taxonomy-APP API: Read applications/GL-Taxonomy-APP/ router files, write docs/api/applications/gl-taxonomy-app.md
- [ ] Document CBAM Pack MVP API: Read cbam-pack-mvp/src/cbam_pack/web/app.py, write docs/api/cbam-pack/endpoints.md with all 48 endpoints

## Phase 9: Cross-Cutting Documentation

- [ ] Create docs/api/authentication.md - comprehensive authentication guide covering JWT tokens, API keys, OAuth flows, and per-endpoint auth requirements
- [ ] Create docs/api/error-codes.md - complete error code reference mapping HTTP status codes to GreenLang error types from greenlang/exceptions/ hierarchy
- [ ] Create docs/api/pagination.md - pagination patterns used across all endpoints (cursor-based, offset-based, keyset)
- [ ] Create docs/api/rate-limiting.md - rate limiting policies per endpoint category
- [ ] Create docs/api/versioning.md - API versioning strategy and migration guides
- [ ] Update docs/api/API_DOCUMENTATION.md to serve as the main landing page linking to all sub-sections

## Phase 10: Endpoint Docstring Enhancement

- [ ] Add comprehensive Google-style docstrings to all Foundation agent router endpoints (10 agents): Args, Returns, Raises, Examples
- [ ] Add comprehensive Google-style docstrings to all Data agent router endpoints (20 agents): Args, Returns, Raises, Examples
- [ ] Add comprehensive Google-style docstrings to all MRV agent router endpoints (30 agents): Args, Returns, Raises, Examples
- [ ] Add response_model, status_code, and response descriptions to all FastAPI endpoint decorators in Foundation agents
- [ ] Add response_model, status_code, and response descriptions to all FastAPI endpoint decorators in Data agents
- [ ] Add response_model, status_code, and response descriptions to all FastAPI endpoint decorators in MRV agents
