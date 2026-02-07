# GreenLang Security Policy

## Vulnerability Disclosure Program

GreenLang maintains a Vulnerability Disclosure Program (VDP) to responsibly handle security vulnerabilities discovered in our platform. We appreciate the efforts of security researchers who help keep GreenLang and our users safe.

## Reporting a Vulnerability

### Submission Process

1. **Submit your report** via our secure submission form at https://greenlang.io/security/report or email security@greenlang.io
2. **Include the following information:**
   - Detailed description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact if exploited
   - Affected component or service
   - Proof of concept (if available)
   - Your contact email address

### What to Expect

| Timeline | Action |
|----------|--------|
| Within 24 hours | Acknowledgment of your report |
| Within 5 business days | Initial assessment and severity classification |
| Ongoing | Regular updates on remediation progress |
| After fix deployed | Notification and disclosure coordination |

## Scope

### In Scope

- **GreenLang Cloud Platform** (app.greenlang.io)
- **GreenLang API** (api.greenlang.io)
- **GreenLang CLI Tools**
- **GreenLang Open Source Projects** (github.com/greenlang/*)
- **GreenLang Mobile Applications**

### Out of Scope

- Social engineering attacks (phishing, vishing, etc.)
- Physical security issues
- Denial of service (DoS/DDoS) attacks
- Spam or social media hijacking
- Issues in third-party services we integrate with
- Vulnerabilities in outdated or unsupported versions
- Issues requiring physical access to a user's device

## Disclosure Timeline

We follow responsible disclosure practices with the following timelines:

| Severity | Disclosure Deadline |
|----------|---------------------|
| Critical (CVSS 9.0-10.0) | 7 days |
| High (CVSS 7.0-8.9) | 30 days |
| Medium (CVSS 4.0-6.9) | 60 days |
| Low (CVSS 0.1-3.9) | 90 days |

We may request deadline extensions for complex issues that require additional time to remediate. All extension requests will be communicated with the researcher.

## Bug Bounty Program

We offer monetary rewards for valid security vulnerabilities:

| Severity | Bounty Range |
|----------|--------------|
| Critical | $3,000 - $10,000 |
| High | $1,500 - $5,000 |
| Medium | $500 - $2,000 |
| Low | $100 - $500 |

### Bonus Multipliers

- **First reporter bonus**: +25% for being first to report a vulnerability type
- **Quality bonus**: Up to +50% for detailed proof of concept and suggested fix
- **Impact bonus**: Up to +100% for vulnerabilities affecting critical systems

### Payment Process

1. Bounty amount calculated after vulnerability is confirmed
2. Identity verification required for payments over $600 USD
3. Payment processed via PayPal, wire transfer, or cryptocurrency
4. Tax documentation required for US payments (W-9) and non-US payments (W-8BEN)

## Safe Harbor

GreenLang commits to not pursue legal action against researchers who:

- Act in good faith to avoid privacy violations, destruction of data, and service interruption
- Only interact with accounts they own or have explicit permission to test
- Do not exploit vulnerabilities beyond what is necessary for proof of concept
- Report vulnerabilities through our official channels
- Allow reasonable time for remediation before public disclosure

## Hall of Fame

Researchers who report valid vulnerabilities may be recognized in our public Hall of Fame at https://greenlang.io/security/hall-of-fame (opt-in only).

## Contact

- **Security Team Email**: security@greenlang.io
- **PGP Key**: https://greenlang.io/.well-known/pgp-key.txt
- **Security Portal**: https://greenlang.io/security
- **Bug Bounty Platform**: https://greenlang.io/security/report

## Security Best Practices for Users

### For API Users

- Store API keys securely using environment variables or secret managers
- Rotate API keys regularly (recommended: every 90 days)
- Use IP allowlisting for production API keys
- Monitor API usage for anomalies

### For Platform Users

- Enable multi-factor authentication (MFA)
- Use strong, unique passwords
- Review connected applications regularly
- Report suspicious activity immediately

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-06 | 1.0 | Initial security policy |

---

**Thank you for helping keep GreenLang secure.**

For general security questions, contact security@greenlang.io.
