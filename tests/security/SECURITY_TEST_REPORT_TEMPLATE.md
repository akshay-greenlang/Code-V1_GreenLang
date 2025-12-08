# GreenLang Security Test Report

**Report ID:** SEC-REPORT-{REPORT_ID}
**Generated:** {TIMESTAMP}
**Environment:** {ENVIRONMENT}
**Test Suite Version:** {VERSION}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests Executed** | {TOTAL_TESTS} |
| **Tests Passed** | {PASSED_TESTS} |
| **Tests Failed** | {FAILED_TESTS} |
| **Tests Skipped** | {SKIPPED_TESTS} |
| **Pass Rate** | {PASS_RATE}% |
| **Total Findings** | {TOTAL_FINDINGS} |

### Findings by Severity

| Severity | Count | Status |
|----------|-------|--------|
| Critical | {CRITICAL_COUNT} | {CRITICAL_STATUS} |
| High | {HIGH_COUNT} | {HIGH_STATUS} |
| Medium | {MEDIUM_COUNT} | {MEDIUM_STATUS} |
| Low | {LOW_COUNT} | {LOW_STATUS} |
| Informational | {INFO_COUNT} | {INFO_STATUS} |

### Risk Assessment

**Overall Risk Level:** {RISK_LEVEL}

{RISK_SUMMARY}

---

## Test Categories

### 1. Authentication & Authorization Tests

| Test Name | Status | Duration | Findings |
|-----------|--------|----------|----------|
{AUTH_TESTS_TABLE}

### 2. Injection Attack Tests

| Test Name | Status | Duration | Findings |
|-----------|--------|----------|----------|
{INJECTION_TESTS_TABLE}

### 3. Cross-Site Scripting (XSS) Tests

| Test Name | Status | Duration | Findings |
|-----------|--------|----------|----------|
{XSS_TESTS_TABLE}

### 4. API Security Tests

| Test Name | Status | Duration | Findings |
|-----------|--------|----------|----------|
{API_SECURITY_TESTS_TABLE}

### 5. Secrets Exposure Tests

| Test Name | Status | Duration | Findings |
|-----------|--------|----------|----------|
{SECRETS_TESTS_TABLE}

---

## Detailed Findings

{FINDINGS_SECTION}

### Finding Template

```
## Finding: {FINDING_ID}

**Title:** {FINDING_TITLE}

**Severity:** {SEVERITY} | **CVSS Score:** {CVSS_SCORE}

**Category:** {CATEGORY}

**CWE Reference:** {CWE_ID} - {CWE_NAME}

**Affected Component:** {COMPONENT}

### Description

{DESCRIPTION}

### Evidence

```
{EVIDENCE}
```

### Impact

{IMPACT}

### Remediation

{REMEDIATION}

### References

- {REFERENCE_1}
- {REFERENCE_2}
```

---

## Compliance Mapping

### OWASP Top 10 (2021)

| Category | Findings | Status |
|----------|----------|--------|
| A01:2021 - Broken Access Control | {A01_FINDINGS} | {A01_STATUS} |
| A02:2021 - Cryptographic Failures | {A02_FINDINGS} | {A02_STATUS} |
| A03:2021 - Injection | {A03_FINDINGS} | {A03_STATUS} |
| A04:2021 - Insecure Design | {A04_FINDINGS} | {A04_STATUS} |
| A05:2021 - Security Misconfiguration | {A05_FINDINGS} | {A05_STATUS} |
| A06:2021 - Vulnerable Components | {A06_FINDINGS} | {A06_STATUS} |
| A07:2021 - Auth Failures | {A07_FINDINGS} | {A07_STATUS} |
| A08:2021 - Software/Data Integrity | {A08_FINDINGS} | {A08_STATUS} |
| A09:2021 - Security Logging | {A09_FINDINGS} | {A09_STATUS} |
| A10:2021 - SSRF | {A10_FINDINGS} | {A10_STATUS} |

### CWE Coverage

| CWE ID | Description | Tests | Findings |
|--------|-------------|-------|----------|
{CWE_TABLE}

---

## Recommendations

### Immediate Actions (Critical/High Findings)

{IMMEDIATE_ACTIONS}

### Short-term Actions (Medium Findings)

{SHORT_TERM_ACTIONS}

### Long-term Actions (Low/Informational)

{LONG_TERM_ACTIONS}

---

## Test Environment Details

| Parameter | Value |
|-----------|-------|
| **Base URL** | {BASE_URL} |
| **API Version** | {API_VERSION} |
| **Test Framework** | pytest {PYTEST_VERSION} |
| **Python Version** | {PYTHON_VERSION} |
| **Test Duration** | {TEST_DURATION} |
| **Parallel Workers** | {WORKERS} |

### Test Configuration

```yaml
{TEST_CONFIG}
```

---

## Appendix A: Test Payload Summary

### SQL Injection Payloads Used

```
{SQL_PAYLOADS}
```

### XSS Payloads Used

```
{XSS_PAYLOADS}
```

### Command Injection Payloads Used

```
{CMD_PAYLOADS}
```

---

## Appendix B: Request/Response Samples

### Sample Vulnerable Request

```http
{SAMPLE_REQUEST}
```

### Sample Vulnerable Response

```http
{SAMPLE_RESPONSE}
```

---

## Appendix C: Remediation Resources

### Secure Coding Guidelines

1. **Input Validation**
   - Always validate input on the server side
   - Use allowlists over blocklists
   - Sanitize all user input before processing

2. **Parameterized Queries**
   - Never concatenate user input into SQL queries
   - Use ORM methods or prepared statements
   - Escape special characters when necessary

3. **Output Encoding**
   - Encode output based on context (HTML, JavaScript, URL)
   - Use templating engines with auto-escaping
   - Implement Content Security Policy

4. **Authentication & Session Management**
   - Use secure session tokens
   - Implement proper session expiration
   - Use HttpOnly and Secure cookie flags

5. **Error Handling**
   - Never expose stack traces to users
   - Use generic error messages
   - Log detailed errors server-side only

### Useful Resources

- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [CWE/SANS Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

## Report Metadata

| Field | Value |
|-------|-------|
| **Report Generated By** | GreenLang Security Test Suite |
| **Report Format Version** | 1.0 |
| **Classification** | {CLASSIFICATION} |
| **Distribution** | {DISTRIBUTION} |
| **Retention Period** | {RETENTION} |

---

**End of Report**

---

*This report was automatically generated by the GreenLang Security Test Suite.
For questions or concerns, contact the Security Engineering team.*
