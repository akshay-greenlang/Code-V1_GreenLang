# GL-018 FLUEFLOW - Architecture Design Summary

**Document Status**: Architecture Design Complete
**Created**: December 2, 2025
**Target Release**: Q1 2026
**Priority**: P1 (High Priority)

---

## Executive Summary

GL-018 FLUEFLOW (FlueGasAnalyzer) architecture design is complete and production-ready for development. The application follows proven GreenLang patterns from GL-016 WATERGUARD with a 4-agent pipeline optimized for real-time combustion optimization.

### Key Deliverables

1. **Complete Architecture Specification** (GL-018_FLUEFLOW_ARCHITECTURE.md)
   - 60+ pages comprehensive technical design
   - Agent pipeline architecture (4 agents)
   - Data flow diagrams
   - Technology stack specifications
   - API endpoint design (REST + WebSocket)
   - Database schema (TimescaleDB)
   - Security architecture (Grade A target: 92+)
   - Deployment architecture (Kubernetes, Terraform)
   - Development estimates (8 weeks, 2-3 engineers)

2. **README Documentation** (README.md)
   - Pre-existing comprehensive user documentation
   - Combustion fundamentals and theory
   - Installation and configuration guides
   - API usage examples
   - Troubleshooting guides
   - Compliance standards (EPA, ASME, ISO)

---

## Architecture Highlights

### Agent Pipeline (4 Agents)

```
FlueGasIntakeAgent → CombustionCalculatorAgent → CombustionOptimizerAgent → CombustionReportingAgent
```

**Agent 1: FlueGasIntakeAgent** (400-500 LOC)
- Acquire flue gas data from SCADA/analyzers (O2, CO2, CO, NOx, stack temp)
- Validate data quality and unit conversion
- SHA-256 provenance hashing
- Support OPC-UA, Modbus TCP/RTU, Profinet protocols

**Agent 2: CombustionCalculatorAgent** (800-1000 LOC)
- Calculate combustion efficiency (ASME PTC 4.1 standard)
- Calculate excess air from O2 measurements
- Calculate air-fuel ratio (stoichiometric + actual)
- Calculate heat loss breakdown (stack, moisture, incomplete combustion)
- 100% deterministic - NO LLM in calculation path

**Agent 3: CombustionOptimizerAgent** (600-800 LOC)
- Optimize excess air for maximum efficiency
- Generate setpoint recommendations (O2 target, damper position)
- Multi-objective optimization (efficiency, emissions, safety)
- Constraint handling (min O2, max CO/NOx limits)
- LLM-generated narrative recommendations (NO calculations)

**Agent 4: CombustionReportingAgent** (500-600 LOC)
- Generate PDF/Excel reports
- Real-time dashboards (Grafana)
- Compliance reports (EPA, permit limits)
- Email/Slack notifications

---

## Technology Stack

**Core**: Python 3.11+, FastAPI 0.104.0+, Uvicorn (ASGI server)
**Data**: NumPy 1.24.0+, Pandas 2.1.0+, Pydantic 2.5.0+
**Database**: TimescaleDB (PostgreSQL 14+), Redis 7.0+ (caching)
**SCADA**: asyncua 1.0.0+ (OPC-UA), pymodbus 3.5.0+ (Modbus)
**AI**: Anthropic Claude Sonnet 4.5 (recommendations only - NO calculations)
**Deployment**: Docker 24.0+, Kubernetes 1.28+, Terraform 1.6.0+
**Monitoring**: Prometheus, Grafana, structlog

---

## Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| Analysis Latency (p95) | < 200 ms | Real-time combustion control |
| Availability | 99.9% | Critical infrastructure |
| Test Coverage | 85%+ | GreenLang standard |
| Security Score | Grade A (92+) | Industrial security requirements |
| SCADA Polling Rate | 1-5 seconds | Fast response to process changes |

---

## Key Design Decisions

### 1. Zero-Hallucination Architecture

**CRITICAL**: NO LLM in calculation path
- All combustion calculations use deterministic ASME PTC 4.1 formulas
- Complete provenance tracking with SHA-256 hashes
- Bit-perfect reproducibility guarantee
- LLM usage restricted to narrative generation only

### 2. Real-Time Architecture

- FastAPI async framework for non-blocking I/O
- Redis caching for sub-millisecond state access (66% cost reduction target)
- TimescaleDB continuous aggregates for fast historical queries
- WebSocket support for live dashboard updates

### 3. SCADA Integration

**Supported Protocols**:
- OPC-UA (recommended - secure, standards-based)
- Modbus TCP/RTU (legacy equipment)
- Profinet (Siemens PLCs)

**Integration Pattern**: Subscription-based (change-of-value) for low latency

### 4. Database Design

**TimescaleDB Hypertables**:
- `flue_gas_measurements` (raw data, 90-day retention)
- `combustion_analysis_results` (calculated results)
- `optimization_recommendations` (setpoint recommendations)

**Continuous Aggregates**:
- 1-minute averages (for charts)
- 1-hour averages (for reports)
- Automatic refresh for performance

---

## API Design

**Base URL**: `https://api.greenlang.io/v1/flueflow`

**Key Endpoints**:
- `POST /api/v1/fluegas/analyze` - Real-time flue gas analysis
- `POST /api/v1/optimize/combustion` - Get optimization recommendations
- `GET /api/v1/reports/performance/{combustor_id}` - Performance reports
- `WS /ws/realtime/{combustor_id}` - WebSocket live updates
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

---

## Security Architecture

**Defense-in-Depth (5 Layers)**:
1. Network Security: Firewall, VPN
2. Transport Security: TLS 1.3, certificate-based OPC-UA
3. Application Security: OAuth2 + JWT, RBAC
4. Data Security: AES-256 at rest, SHA-256 provenance
5. Audit Logging: Immutable audit trails

**RBAC Roles**:
- ADMIN: Full access
- ENGINEER: Read, submit, optimize, control SCADA
- OPERATOR: Read, submit data
- VIEWER: Read-only access

---

## Development Timeline: 8 Weeks

**Team Size**: 2-3 engineers (Senior Backend, Combustion/Process Engineer, DevOps)

### Phase 1: Core Agents (3 weeks)
- Week 1-2: FlueGasIntakeAgent + CombustionCalculatorAgent
- Week 3: CombustionOptimizerAgent + CombustionReportingAgent
- Deliverable: 4 working agents with 85%+ test coverage

### Phase 2: Integrations (2 weeks)
- Week 4: SCADA integration (OPC-UA, Modbus)
- Week 5: Database + API implementation
- Deliverable: Full integration stack operational

### Phase 3: Testing (2 weeks)
- Week 6: Integration + performance testing
- Week 7: Security testing + documentation
- Deliverable: Test coverage 85%+, security score Grade A

### Phase 4: Deployment (1 week)
- Week 8: Production deployment (Kubernetes, monitoring)
- Deliverable: Production-ready system

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| SCADA connectivity issues | High | High | Mock SCADA for testing, VPN support, early firewall coordination |
| Analyzer compatibility | Medium | Medium | Support 3 protocols (OPC-UA, Modbus, Profinet), CSV import fallback |
| Real-time performance | Medium | Medium | Redis caching, TimescaleDB optimization, load testing in Week 6 |
| Calculation accuracy disputes | Low | High | ASME PTC 4.1 formulas, formula references in reports, validation against manual calcs |

---

## Business Value

**For a 50 MMBtu/hr natural gas boiler @ $4.50/MMBtu**:

| Metric | Improvement | Annual Value |
|--------|-------------|--------------|
| Fuel Cost Reduction | 5-10% | $197,000 - $394,000 |
| Combustion Efficiency | +3-7% | $118,000 - $275,000 |
| NOx Emissions | -10-20% | Compliance + avoided penalties |
| CO2 Emissions | -5-10% | Carbon credit potential |

**Typical ROI**: 6-12 months payback period

---

## Next Steps

1. **Immediate**: Begin Phase 1 development (Core Agents)
   - Set up development environment
   - Implement FlueGasIntakeAgent
   - Implement CombustionCalculatorAgent with ASME PTC 4.1 formulas

2. **Week 2**: Formula validation
   - Validate all combustion formulas against ASME standards
   - Create unit tests with hand-calculated reference values
   - Zero-hallucination compliance certification

3. **Week 4**: SCADA integration testing
   - Coordinate with plant IT for firewall rules
   - Test OPC-UA connection to real analyzers
   - Develop SCADA client with auto-reconnect

4. **Week 6**: Performance testing
   - Load test to verify <200ms p95 latency
   - Stress test with 50 combustors
   - Database performance optimization

5. **Week 8**: Production deployment
   - Deploy to Kubernetes cluster
   - Configure Prometheus/Grafana monitoring
   - Beta customer onboarding

---

## Success Criteria

The architecture will be considered successful if:

1. **Zero-Hallucination Compliance**: All calculations use deterministic formulas with SHA-256 provenance
2. **Performance Targets Met**: <200ms latency (p95), 99.9% availability
3. **Test Coverage**: 85%+ unit test coverage, 95%+ pass rate
4. **Security Score**: Grade A (92+/100)
5. **SCADA Integration**: Successful connection to 3+ analyzer types
6. **Deployment Ready**: Docker + Kubernetes deployment functional
7. **Timeline Met**: 8-week development timeline achieved

---

## Files Created

1. **GL-018_FLUEFLOW_ARCHITECTURE.md** (60+ pages)
   - Complete technical architecture specification
   - Agent pipeline design
   - Technology stack
   - API design
   - Database schema
   - Security architecture
   - Performance targets
   - Testing strategy
   - Deployment architecture
   - Development estimates

2. **README.md** (Pre-existing, comprehensive)
   - User-facing documentation
   - Combustion theory and fundamentals
   - Installation guides
   - Configuration examples
   - API usage examples
   - Troubleshooting guides
   - Compliance standards

3. **ARCHITECTURE_SUMMARY.md** (This document)
   - Executive summary
   - Architecture highlights
   - Timeline and next steps

---

## Architecture Pattern Compliance

GL-018 FLUEFLOW follows proven GreenLang architecture patterns:

**Based on GL-016 WATERGUARD**:
- Modular, event-driven architecture
- 4-agent pipeline (simplified from GL-016's more complex workflow)
- Zero-hallucination calculation engine
- SCADA integration patterns (OPC-UA, Modbus)
- TimescaleDB for time-series data
- Redis caching for real-time state
- Prometheus/Grafana monitoring
- Kubernetes deployment
- FastAPI REST + WebSocket API

**Key Differentiators from GL-016**:
- Simpler agent pipeline (4 vs. GL-016's more complex workflow)
- Focus on real-time combustion optimization (<200ms vs. <100ms for GL-016)
- Different domain (combustion vs. water treatment)
- Different calculations (ASME PTC 4.1 vs. ASME/ABMA water chemistry)

---

## Conclusion

GL-018 FLUEFLOW architecture is production-ready and optimized for rapid development (8 weeks). The design leverages proven GreenLang patterns from GL-016 WATERGUARD while tailoring the architecture for real-time combustion optimization with zero-hallucination guarantees.

**Architecture Status**: APPROVED - Ready for Development
**Next Milestone**: Phase 1 Completion (3 weeks)
**Target Release**: Q1 2026

---

**Document Version**: 1.0.0
**Last Updated**: December 2, 2025
**Maintained By**: GreenLang Architecture Team (GL-AppArchitect)
