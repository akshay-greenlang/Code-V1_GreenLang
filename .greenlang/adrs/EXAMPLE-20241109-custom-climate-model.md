# ADR-001: Custom ClimateGPT Provider for Specialized Carbon Modeling

**Date:** 2024-11-09
**Status:** Accepted
**Deciders:** Engineering Team, Architecture Team
**Consulted:** Product, Security

---

## Context

### Problem Statement
We need to integrate with ClimateGPT, a specialized large language model fine-tuned for carbon accounting and climate calculations. This model is not available through standard OpenAI or Anthropic APIs that greenlang.intelligence supports.

### Current Situation
- **GreenLang Infrastructure:** `greenlang.intelligence.ChatSession` supports OpenAI (GPT-4) and Anthropic (Claude)
- **Our Need:** ClimateGPT is hosted on a custom API endpoint with proprietary authentication
- **Constraints:**
  - ClimateGPT provider cannot be added to greenlang core immediately (licensing discussions ongoing)
  - Business needs this integration within 2 weeks for customer demo
  - Standard GPT-4 accuracy is 72% vs ClimateGPT's 94% for carbon calculations

### Business Impact
- **Customer Demo:** Major client requiring ClimateGPT integration
- **Revenue Impact:** $2M deal depends on this capability
- **Timeline:** Cannot wait 3-6 months for greenlang core integration
- **Competitive Advantage:** First to market with ClimateGPT

---

## Decision

### What We're Implementing
Custom ClimateGPT client that maintains greenlang-compatible interfaces while connecting to specialized climate model API.

### Technology Stack
- **Language:** Python 3.11+
- **HTTP Client:** httpx (for async support)
- **Authentication:** Custom API key + OAuth2 flow
- **Dependencies:**
  - httpx >= 0.25.0
  - pydantic >= 2.0 (for response validation)

### Code Location
- `apps/GL-CBAM-APP/services/climate_llm/`
  - `client.py` - Main client implementation
  - `auth.py` - ClimateGPT authentication
  - `models.py` - Request/response models
  - `config.py` - Configuration
  - `tests/` - Test suite

---

## Rationale

### Why GreenLang Infrastructure Can't Support This

**Specific Limitations:**

1. **Custom Authentication:** ClimateGPT uses OAuth2 + API key, different from OpenAI/Anthropic
2. **Proprietary API:** ClimateGPT API schema is incompatible with OpenAI completion format
3. **Response Format:** Returns structured carbon calculation metadata not in standard LLM responses
4. **Rate Limiting:** ClimateGPT has different rate limit structure (per-calculation vs per-token)

**What Would Need to Change in GreenLang:**

1. **Add Provider Interface:** New provider in `greenlang.intelligence.providers.climate_gpt.py`
2. **Update ChatSession:** Support for custom response parsers
3. **Add Auth Flow:** OAuth2 support in greenlang.auth
4. **Estimated Effort:** 3-4 weeks engineering + 2 weeks testing
5. **Timeline:** Not feasible before customer demo

---

## Alternatives Considered

### Alternative 1: Use GPT-4 with Fine-tuning
**Pros:**
- Would use greenlang.intelligence directly
- No custom code needed

**Cons:**
- Accuracy only 72% vs 94% for ClimateGPT
- Fine-tuning cost: $50K+
- Still wouldn't match ClimateGPT's domain-specific training

**Why Rejected:** Insufficient accuracy for regulatory compliance requirements

### Alternative 2: Use Claude with Carbon Calculation Prompt Engineering
**Pros:**
- Compliant with greenlang infrastructure
- Immediate implementation

**Cons:**
- Accuracy 68% (worse than GPT-4)
- No structured carbon metadata
- Response time 3x slower than ClimateGPT

**Why Rejected:** Does not meet accuracy SLA (>90%)

### Alternative 3: Wait for GreenLang Support
**Pros:**
- Would be fully compliant
- Would benefit from infrastructure updates
- No technical debt

**Cons:**
- Timeline: 3-6 months minimum
- Would miss customer demo
- Risk losing $2M deal

**Why Rejected:** Business timeline cannot accommodate delay

---

## Consequences

### Positive
- **Accuracy:** 94% vs 72% with GPT-4
- **Speed:** 2x faster response times
- **Cost:** 60% lower cost per calculation
- **Compliance:** Better regulatory compliance due to accuracy
- **Customer Satisfaction:** Meets demo requirements

### Negative
- **Technical Debt:** Custom code to maintain (~500 LOC)
- **No Auto-Updates:** Won't benefit from greenlang.intelligence improvements automatically
- **Custom Monitoring:** Need to implement own retry logic, error handling
- **Security:** Need to maintain ClimateGPT credentials separately
- **Testing:** Additional test suite to maintain
- **Documentation:** Need separate docs for ClimateGPT integration

### Neutral
- **Team Knowledge:** Only 2 engineers familiar with ClimateGPT API
- **Dependencies:** One additional external dependency (httpx)

---

## Implementation Plan

### Phase 1: Development (Week 1)
1. Create client wrapper in `services/climate_llm/client.py`
2. Implement OAuth2 authentication flow
3. Add request/response models with Pydantic
4. Implement greenlang-compatible interface (same method signatures as ChatSession)
5. Add comprehensive error handling

### Phase 2: Testing (Week 2)
- **Unit Tests:** 90% coverage target
  - Mock ClimateGPT API responses
  - Test error scenarios
  - Test authentication flow
- **Integration Tests:**
  - Test against ClimateGPT sandbox
  - Validate response parsing
- **Security Review:**
  - Credentials management
  - API key rotation
  - OAuth token refresh
- **Performance Tests:**
  - Benchmark response times
  - Load testing (100 concurrent requests)

### Phase 3: Deployment (Week 2)
- Deploy to staging environment
- Configure monitoring (Datadog)
- Set up alerts (error rate, latency)
- Customer demo preparation

### Phase 4: Maintenance
- **Owner:** @carbon-team
- **On-call:** Rotation between @alice, @bob
- **Updates:** Monthly dependency updates
- **Security:** Quarterly security review

---

## Compliance & Security

### Security Considerations
- **Authentication:** OAuth2 tokens stored in greenlang.auth vault
- **API Keys:** Rotated monthly, stored in environment variables
- **Encryption:** All requests use TLS 1.3
- **Audit Logging:** All ClimateGPT calls logged with request/response
- **PII:** No customer PII sent to ClimateGPT (scrubbed before API call)
- **Compliance:**
  - SOC2: Audit trail maintained
  - GDPR: Data residency in EU region
  - ISO 27001: Encryption at rest and in transit

### Monitoring & Observability
- **Metrics:**
  - Request latency (p50, p95, p99)
  - Error rate
  - Token usage
  - Cost per request
- **Logs:**
  - Request/response payloads (sanitized)
  - Authentication events
  - Error traces
- **Alerts:**
  - Error rate > 5%
  - p95 latency > 2s
  - Authentication failures > 10/hour
- **Dashboard:** Datadog dashboard "ClimateGPT Integration"

### Testing Strategy
- **Unit Test Coverage:** 90% (enforced by CI)
- **Integration Tests:** 15 test scenarios
- **Performance Tests:**
  - Load: 100 concurrent users
  - Soak: 1000 requests over 1 hour
- **Security Tests:**
  - OWASP Top 10 scanning
  - Dependency vulnerability scanning (Snyk)

---

## Migration Plan

### Short-term (0-6 months)
Use custom ClimateGPT client in GL-CBAM-APP. Monitor usage, gather metrics for future greenlang integration.

### Medium-term (6-12 months)
**Q2 2025:** Contribute ClimateGPT provider to greenlang.intelligence
- Work with greenlang core team to add ClimateGPT provider
- Abstract common patterns (OAuth2, structured responses)
- Add to provider registry

### Long-term (12+ months)
**Q3 2025:** Migrate to greenlang.intelligence.ChatSession
- Remove custom client code
- Update to use `ChatSession(provider="climate_gpt")`
- Delete `services/climate_llm/` directory

**Migration Trigger:**
When ClimateGPT provider is available in greenlang.intelligence v0.8.0+, we will migrate.

**Estimated Migration Effort:**
2-3 person-days (mostly testing and validation)

---

## Documentation

### User Documentation
- [x] Usage guide: `docs/climate-gpt-integration.md`
- [x] Examples: `examples/climate_gpt_demo.py`
- [x] API documentation: Auto-generated from docstrings
- [x] Troubleshooting: `docs/climate-gpt-troubleshooting.md`

### Developer Documentation
- [x] Architecture: `docs/architecture/climate-gpt.md`
- [x] Code comments: Comprehensive docstrings
- [x] Deployment: `docs/deployment/climate-gpt.md`
- [x] Runbook: `runbooks/climate-gpt.md`

### Team Communication
- [x] Team notified (Slack #engineering, 2024-11-01)
- [x] Knowledge sharing session (2024-11-05, recording in Drive)
- [x] Wiki page: `wiki/ClimateGPT-Integration`

---

## Review & Approval

### Technical Review
- [x] Security team approval (@security-team, 2024-11-07)
  - Reviewed auth flow
  - Validated encryption
  - Approved credential management
- [x] Architecture team approval (@architecture, 2024-11-08)
  - Reviewed design
  - Approved interface compatibility
  - Endorsed migration plan
- [x] DevOps team approval (@devops, 2024-11-08)
  - Reviewed monitoring setup
  - Approved deployment plan

### Business Review
- [x] Product owner approval (@product, 2024-11-06)
- [x] Customer success sign-off (@cs-lead, 2024-11-07)

### Approvals
- **Engineering Lead:** Alice Chen - 2024-11-08
- **Security Lead:** Bob Martinez - 2024-11-07
- **Architecture Lead:** Carol Williams - 2024-11-08

---

## Links & References

- **GitHub Issue:** #1234 - ClimateGPT Integration
- **Implementation PR:** #1250 - Add ClimateGPT Client
- **ClimateGPT Docs:** https://docs.climategpt.ai/api/v1
- **Slack Discussion:** https://greenlang.slack.com/archives/C123/p1699450000
- **Demo Recording:** https://drive.google.com/file/d/demo-recording
- **Performance Benchmarks:** https://docs.google.com/spreadsheets/d/benchmarks

---

## Updates

### 2024-11-08 - Status: Accepted
ADR approved by all stakeholders. Implementation started.

### 2024-11-15 - Implementation Complete
ClimateGPT client deployed to production. Monitoring active. Demo successful.

### 2025-Q2 - Migration to GreenLang Core (Planned)
Will contribute ClimateGPT provider to greenlang.intelligence.

---

**Template Version:** 1.0
**Last Updated:** 2024-11-09
**ADR Author:** @alice
**Reviewers:** @bob, @carol, @security-team
