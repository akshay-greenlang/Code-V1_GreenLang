# SOC 2 Type II — Pre-Audit Readiness Checklist

Run this list 30 days before the Type II observation window begins and
re-run 7 days before the audit kicks off. Every item must be either
checked or have a dated mitigation plan attached.

## 1. Governance & policy

- [ ] Information Security Policy published and signed by CEO + CISO.
- [ ] Acceptable Use Policy acknowledged by 100% of employees in HRIS.
- [ ] Vendor security review checklist reviewed and on file.
- [ ] Annual risk assessment completed and signed.
- [ ] Board / audit committee minutes available for the window.

## 2. Logical access (CC6)

- [ ] SSO enforced for all privileged accounts (no local-only auth).
- [ ] MFA enforced for all human accounts (SSO IdP config screenshot).
- [ ] API keys rotation policy verifiable from keyring metadata.
- [ ] SCIM deprovision tested end-to-end with an active IdP.
- [ ] Retention policy active; cron job running on schedule.
- [ ] Least-privilege review completed for every IAM role.
- [ ] Vault audit log retention >= 1y (enterprise) / 90d (non-enterprise).

## 3. System operations (CC7)

- [ ] Backup-restore drill passed within the last 90 days.
- [ ] DR drill (cold-region failover) passed within the last 12 months.
- [ ] Incident runbook reviewed within 90 days.
- [ ] Tabletop exercise completed within last 6 months.
- [ ] Dependency vulnerability scan: 0 critical, 0 high unremediated > 30 days.

## 4. Change management (CC8)

- [ ] CODEOWNERS file covers 100% of production-critical paths.
- [ ] Branch protection on production branches (no direct push).
- [ ] CI gates block merges when SAST/DAST/SBOM scans fail.
- [ ] 10-PR spot check completed (approvers + tests documented).

## 5. Availability (A1)

- [ ] 99.9% availability SLO met for each month in the window.
- [ ] Capacity load test executed monthly (p95 within thresholds).
- [ ] HPA tuning reviewed and documented.
- [ ] Cost + headroom dashboards current.

## 6. Data protection

- [ ] Encryption at rest verified (RDS, S3, pgvector).
- [ ] TLS 1.3 only; weak ciphers disabled at the ingress.
- [ ] Data flow diagram current (factors data lineage).
- [ ] DPA template Legal-reviewed and offered to all customers.

## 7. Monitoring (CC4)

- [ ] Grafana dashboards published; alert rules live.
- [ ] PagerDuty rotation filled for the entire audit window.
- [ ] Alert noise review completed within 30 days.

## 8. Evidence artifact bundle

- [ ] `python -m greenlang.factors.security.soc2_controls` run and JSON attached.
- [ ] Each `docs/security/soc2_evidence/*.md` checklist marked complete.
- [ ] Sample set of signed receipts (10 random, distinct tenants) archived.
- [ ] Sample set of SCIM provisioning/deprovisioning events archived.

## Sign-off

| Role | Name | Date |
|------|------|------|
| CISO | | |
| Eng Mgr | | |
| Lead SRE | | |
| Security Lead | | |
