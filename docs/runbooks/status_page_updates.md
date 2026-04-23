# Factors API - Status Page Update Playbook

**Who:** on-call engineer (SEV1/SEV2), with incident commander approval.
**Where:** https://status.greenlang.com (statuspage.io) or the self-host
Cachet instance when configured. Credentials live in Vault path
`kv/factors/ops/statuspage`.
**Config source of truth:** `deployment/status_page/statuspage.yaml`.

## Severity -> status page mapping

| Severity | Status page impact | Component state | Example |
|----------|--------------------|-----------------|---------|
| SEV1 | Major Outage | `major_outage` | Customer-visible outage > 5 min |
| SEV2 | Partial Outage | `partial_outage` | Degraded tier or sub-region |
| SEV3 | Degraded Performance | `degraded_performance` | Elevated latency under SLO |
| Planned maintenance | Under Maintenance | `under_maintenance` | Advertised >= 48h in advance |

## Cadence

1. **On alert fire (within 15 min):** publish initial incident. Use
   short factual summary, avoid speculation. Template: *"We're
   investigating elevated 5xx on the Factors API. Some customers may see
   intermittent errors."*
2. **Every 30 min until resolved:** post an update, even if nothing new.
   A silent status page is a broken status page.
3. **On mitigation:** update status to "Monitoring" with the change made.
4. **On resolution:** mark resolved with the root cause and link to the
   postmortem (published within 5 business days).

## Voice and tone

- First person plural (we).
- No jargon in customer-facing updates (say "database", not "Postgres
  primary IOPS throttled").
- No internal names, no personal names.
- Time-stamp every update in UTC.

## Planned maintenance

- Create the incident at least 48 hours ahead.
- Subscribe customers' IdP admins to maintenance notifications if
  configured per tenant.
- Status: `under_maintenance` for affected components during the window.

## Postmortem linkage

Once the postmortem is published, append a final status-page update:

> RCA and remediation plan published at
> https://greenlang.ai/postmortems/<date>-<slug>.

## Who updates when

- **Initial** and **every-30-min**: on-call engineer.
- **Mitigation**: on-call engineer, after incident commander confirms.
- **Resolution**: incident commander.
- **Postmortem link**: incident commander or communications lead.
