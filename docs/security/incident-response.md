# GreenLang Security Incident Response Plan

## Purpose

This document provides a systematic approach to handling security incidents in GreenLang deployments.

## Incident Classification

### Severity Levels

#### P0 - Critical
- Active data breach
- Ransomware attack
- Complete system compromise
- Mass data exfiltration
**Response Time**: Immediate (< 15 minutes)

#### P1 - High
- Suspected data breach
- DDoS attack affecting availability
- Privilege escalation discovered
- Critical vulnerability in production
**Response Time**: < 1 hour

#### P2 - Medium
- Failed authentication attempts spike
- Rate limit violations
- Non-critical vulnerabilities
- Security configuration drift
**Response Time**: < 4 hours

#### P3 - Low
- Audit log anomalies
- Policy violations
- Minor misconfigurations
**Response Time**: < 24 hours

## Incident Response Team

### Roles

- **Incident Commander**: Overall incident coordination
- **Security Lead**: Technical security response
- **Operations Lead**: System operations and recovery
- **Communications Lead**: Internal/external communications
- **Legal/Compliance**: Legal and regulatory compliance

## Response Phases

### 1. Detection & Identification (0-15 minutes)

**Objectives**:
- Confirm incident is real (not false positive)
- Classify severity level
- Activate incident response team

**Actions**:
1. Review security alerts and audit logs
2. Verify incident using multiple sources
3. Document initial findings
4. Classify severity
5. Page incident response team
6. Create incident tracking ticket

**Tools**:
- Audit logs: `~/.greenlang/logs/audit.jsonl`
- System logs: Check application logs
- Bandit reports: `security/bandit-report.txt`
- pip-audit reports: `security/pip-audit-report.txt`

### 2. Containment (15-60 minutes)

**Objectives**:
- Stop incident from spreading
- Preserve evidence
- Minimize damage

**Short-term Containment**:
1. Isolate affected systems
2. Disable compromised accounts
3. Revoke compromised API keys
4. Block malicious IP addresses
5. Enable additional logging

**Long-term Containment**:
1. Apply temporary patches
2. Implement additional monitoring
3. Create backup of affected systems
4. Prepare for recovery

**Example Commands**:
```bash
# Review audit logs for suspicious activity
grep "security.violation" ~/.greenlang/logs/audit.jsonl

# Check for failed auth attempts
grep "auth.failure" ~/.greenlang/logs/audit.jsonl

# Review recent configuration changes
grep "config.changed" ~/.greenlang/logs/audit.jsonl
```

### 3. Eradication (1-4 hours)

**Objectives**:
- Remove threat from environment
- Fix vulnerabilities
- Strengthen defenses

**Actions**:
1. Identify root cause
2. Remove malware/backdoors
3. Patch vulnerabilities
4. Update configurations
5. Rotate all credentials
6. Update firewall rules

**Verification**:
```bash
# Run security scans
bandit -c .bandit -r greenlang core
pip-audit --desc

# Check for vulnerabilities
python scripts/run_security_checks.py
```

### 4. Recovery (4-24 hours)

**Objectives**:
- Restore normal operations
- Verify systems are clean
- Monitor for recurrence

**Actions**:
1. Restore from clean backups if needed
2. Rebuild compromised systems
3. Gradually restore services
4. Implement additional monitoring
5. Conduct security testing
6. Document changes made

**Validation Checklist**:
- [ ] All patches applied
- [ ] All credentials rotated
- [ ] Security scans passing
- [ ] Audit logs configured
- [ ] Monitoring alerts active
- [ ] Systems stable for 24+ hours

### 5. Post-Incident (1-7 days)

**Objectives**:
- Learn from incident
- Improve security posture
- Complete documentation

**Actions**:
1. Conduct post-mortem meeting
2. Document timeline of events
3. Identify lessons learned
4. Update security procedures
5. Implement preventive measures
6. Provide training if needed
7. Complete incident report

## Communication Plan

### Internal Communication

**Immediately**:
- Notify incident response team
- Notify management
- Notify affected teams

**Within 1 hour**:
- Send status update to stakeholders
- Schedule regular update calls (every 2-4 hours)

**Daily**:
- Send summary of progress
- Update incident tracker

### External Communication

**Customers** (if data breach):
- Within 72 hours (GDPR requirement)
- Clear, concise explanation
- Steps being taken
- Actions customers should take

**Regulators** (if required):
- Follow regulatory timelines
- Work with legal team
- Document all communications

**Media** (if public):
- Coordinate with PR team
- Prepared statement only
- Single point of contact

## Evidence Collection

### What to Collect

1. **Logs**:
   - Audit logs
   - Application logs
   - System logs
   - Network logs
   - Security tool logs

2. **System Information**:
   - Process lists
   - Network connections
   - File modifications
   - User sessions

3. **Artifacts**:
   - Malicious files
   - Modified configurations
   - Suspicious scripts
   - Network traffic captures

### Chain of Custody

- Document who collected evidence
- Document when collected
- Document from where
- Maintain chronological log
- Store securely
- Limit access

## Contact Information

### Internal Contacts

- **Security Team**: security@greenlang.io
- **On-Call**: Use PagerDuty/on-call system
- **Management**: executives@greenlang.io

### External Contacts

- **Cloud Provider Support**: [Provider-specific]
- **Security Vendor**: [Vendor-specific]
- **Legal Counsel**: [Law firm]
- **PR Firm**: [PR agency]
- **Law Enforcement**: [Local cyber crime unit]

## Post-Incident Report Template

```markdown
# Security Incident Report

**Incident ID**: IR-YYYY-MM-DD-NNN
**Severity**: [P0/P1/P2/P3]
**Date Detected**: YYYY-MM-DD HH:MM UTC
**Date Resolved**: YYYY-MM-DD HH:MM UTC

## Summary
[Brief description of incident]

## Timeline
- [Time]: Event 1
- [Time]: Event 2
- ...

## Impact
- Systems affected:
- Data affected:
- Users affected:
- Downtime:

## Root Cause
[Detailed analysis]

## Response Actions
[Actions taken]

## Lessons Learned
[What worked, what didn't]

## Preventive Measures
[Changes to prevent recurrence]

## Recommendations
[Suggestions for improvement]
```

## Regular Drills

Conduct incident response drills:
- **Tabletop exercises**: Quarterly
- **Simulated incidents**: Semi-annually
- **Full-scale tests**: Annually

## References

- NIST SP 800-61: Computer Security Incident Handling Guide
- ISO/IEC 27035: Information Security Incident Management
- SANS Incident Handler's Handbook

## Document Control

- **Version**: 1.0
- **Last Updated**: 2025-11-07
- **Review Frequency**: Quarterly
- **Owner**: Security Team
