# SECURITY DEPLOYMENT CHECKLIST
## GreenLang Platform Production Deployment

**Version**: 1.0
**Date**: 2025-11-08
**Status**: Ready for Production

---

## PRE-DEPLOYMENT CHECKLIST

### 1. Environment Configuration

- [ ] **JWT_SECRET** - Set to strong 32+ character secret
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

- [ ] **ENCRYPTION_KEY** - Set to Fernet-generated key
  ```bash
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

- [ ] **CORS_ORIGINS** - Set to actual frontend URLs (NO WILDCARDS)
  ```bash
  CORS_ORIGINS=https://app.greenlang.io,https://admin.greenlang.io
  ```

- [ ] **APP_ENV** - Set to "production"
  ```bash
  APP_ENV=production
  ```

- [ ] **LOG_LEVEL** - Set to "INFO" or "WARNING"
  ```bash
  LOG_LEVEL=INFO
  ```

### 2. Dependency Installation

- [ ] VCCI: Install requirements
  ```bash
  cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
  pip install -r requirements.txt
  ```

- [ ] CBAM: Install requirements
  ```bash
  cd GL-CBAM-APP/CBAM-Importer-Copilot
  pip install -r requirements.txt
  ```

- [ ] CSRD: Install requirements
  ```bash
  cd GL-CSRD-APP/CSRD-Reporting-Platform
  pip install -r requirements.txt
  ```

- [ ] Verify security dependencies installed:
  - [ ] defusedxml>=0.7.1
  - [ ] slowapi>=0.1.9
  - [ ] python-jose[cryptography]>=3.3.0
  - [ ] passlib[bcrypt]>=1.7.4
  - [ ] cryptography>=41.0.0

### 3. Security Validation

- [ ] No `.env` files committed to git
- [ ] No hardcoded secrets in source code
- [ ] All API endpoints require authentication (except health checks)
- [ ] CORS configured with explicit origins (no wildcards)
- [ ] Rate limiting enabled on all applications
- [ ] SSL/TLS certificates valid and configured
- [ ] Security headers middleware enabled (VCCI)

### 4. Code Quality Checks

- [ ] Run security scanner (Bandit):
  ```bash
  bandit -r . -f json -o security-report.json
  ```

- [ ] Run dependency vulnerability scanner:
  ```bash
  pip-audit
  ```

- [ ] Run linter:
  ```bash
  ruff check .
  ```

- [ ] Run type checker:
  ```bash
  mypy .
  ```

### 5. Testing

- [ ] Unit tests passing
  ```bash
  pytest tests/
  ```

- [ ] Security tests passing
  ```bash
  pytest tests/security/
  ```

- [ ] Integration tests passing
- [ ] Load testing completed
- [ ] Authentication flow tested
- [ ] Rate limiting tested (429 responses)
- [ ] CORS policy tested

---

## DEPLOYMENT CHECKLIST

### 1. Pre-Deployment

- [ ] Backup current production database
- [ ] Tag release in git: `git tag v1.0.0-security-fixes`
- [ ] Document deployment window
- [ ] Notify stakeholders
- [ ] Prepare rollback plan

### 2. Staging Deployment

- [ ] Deploy to staging environment
- [ ] Verify health checks pass
- [ ] Test authentication with production-like config
- [ ] Test rate limiting behavior
- [ ] Run smoke tests
- [ ] Monitor error rates for 1 hour
- [ ] Get QA sign-off

### 3. Production Deployment

- [ ] Deploy to production
- [ ] Verify all pods/containers healthy
- [ ] Check health endpoints:
  - [ ] `/health/live`
  - [ ] `/health/ready`
  - [ ] `/health/startup`
- [ ] Verify metrics endpoint: `/metrics`
- [ ] Test authentication flow
- [ ] Monitor logs for errors
- [ ] Watch error rates and latency

### 4. Post-Deployment

- [ ] Run production smoke tests
- [ ] Verify authentication working
- [ ] Verify rate limiting active
- [ ] Check security logs
- [ ] Monitor for 2 hours
- [ ] Update deployment documentation
- [ ] Send deployment success notification

---

## MONITORING CHECKLIST

### 1. Security Metrics

- [ ] Monitor failed authentication attempts
  - Alert threshold: > 100/hour

- [ ] Monitor rate limit violations (429 responses)
  - Alert threshold: > 1% of requests

- [ ] Monitor CORS violations
  - Alert threshold: > 10/hour

- [ ] Monitor XXE attack attempts
  - Alert threshold: > 0 (should never happen)

### 2. Application Metrics

- [ ] Monitor API response times
  - Alert threshold: P95 > 1000ms

- [ ] Monitor error rates
  - Alert threshold: > 1%

- [ ] Monitor active connections
- [ ] Monitor memory usage
- [ ] Monitor CPU usage

### 3. Security Alerts Configured

- [ ] High authentication failure rate
- [ ] Rate limit violation spike
- [ ] Invalid token errors
- [ ] Suspicious IP activity
- [ ] Missing environment variables on startup

---

## ROLLBACK CHECKLIST

### Rollback Triggers

Rollback if ANY of these conditions occur:

- [ ] Authentication failure rate > 5%
- [ ] Error rate > 2%
- [ ] P95 latency increase > 100%
- [ ] Health checks failing
- [ ] Critical security vulnerability discovered

### Rollback Steps

1. [ ] Notify stakeholders of rollback
2. [ ] Execute rollback command:
   ```bash
   git revert <commit-sha>
   # OR
   kubectl rollout undo deployment/greenlang-api
   ```
3. [ ] Verify health checks pass
4. [ ] Monitor error rates
5. [ ] Investigate root cause
6. [ ] Document incident
7. [ ] Plan remediation

---

## SECURITY VERIFICATION

### Required Security Controls

- [ ] **Authentication**: JWT required on all API endpoints
- [ ] **Authorization**: Role-based access control (if applicable)
- [ ] **Encryption in Transit**: TLS 1.2+ required
- [ ] **Encryption at Rest**: Database encryption enabled
- [ ] **Rate Limiting**: Active on all endpoints
- [ ] **CORS**: Restricted to known origins
- [ ] **XXE Prevention**: defusedxml used for XML parsing
- [ ] **Secret Management**: All secrets from environment
- [ ] **Input Validation**: Pydantic models validate all inputs
- [ ] **Security Headers**: CSP, HSTS, X-Frame-Options, etc.

### Compliance Validation

- [ ] **GDPR**: Data protection controls in place
- [ ] **SOC 2**: Access controls documented
- [ ] **ISO 27001**: Security controls verified
- [ ] **Audit Logs**: Security events logged

---

## EMERGENCY CONTACTS

### Security Team
- **Lead**: [Your Name]
- **Email**: security@greenlang.io
- **Slack**: #security-incidents

### On-Call
- **Primary**: [Name]
- **Secondary**: [Name]
- **PagerDuty**: [Link]

---

## QUICK REFERENCE COMMANDS

### Check Application Status
```bash
# Health check
curl https://api.greenlang.io/health

# Readiness check
curl https://api.greenlang.io/health/ready

# Metrics
curl https://api.greenlang.io/metrics
```

### Test Authentication
```bash
# Without auth (should fail with 401)
curl -X GET https://api.greenlang.io/api/v1/intake

# With auth (should succeed)
curl -X GET https://api.greenlang.io/api/v1/intake \
  -H "Authorization: Bearer <JWT_TOKEN>"
```

### Test Rate Limiting
```bash
# Rapidly make requests (should eventually get 429)
for i in {1..100}; do
  curl https://api.greenlang.io/api/v1/validate
done
```

### View Logs
```bash
# Kubernetes
kubectl logs -f deployment/greenlang-api --tail=100

# Docker
docker logs -f greenlang-api

# Local
tail -f logs/app.log
```

### Generate Secrets
```bash
# JWT Secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Encryption Key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## SIGN-OFF

### Deployment Approval

- [ ] Security Team: _________________________ Date: _________
- [ ] DevOps Team: __________________________ Date: _________
- [ ] QA Team: ______________________________ Date: _________
- [ ] Product Owner: ________________________ Date: _________

### Post-Deployment Sign-Off

- [ ] All checks passed: _____________________ Date: _________
- [ ] Production stable for 24h: _____________ Date: _________

---

**Checklist Version**: 1.0
**Last Updated**: 2025-11-08
**Next Review**: 2025-12-08

---
*Complete all checklist items before production deployment. Document any deviations with justification.*
