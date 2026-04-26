# Customer-Impact Memo — `<release-id or incident-id>`

> Template. Copy to
> `docs/factors/release-templates/instances/<release-id>/CUSTOMER_IMPACT.md`
> (or, for incidents not tied to a release, to
> `docs/factors/postmortems/<YYYY-MM-DD>-<slug>.md`).
>
> Required by RACI #6 Customer-Impact Communication for any event
> that touches a customer-visible boundary.

## Identity

| Field              | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Memo id            | `<release-id or incident-id>`                        |
| Author             | Partner Success Lead (R)                             |
| Approver           | CTO or Product Lead (A)                              |
| Memo type          | `<release \| incident \| value-correction \| deprecation \| source-decommission>` |
| Severity           | `<SEV-1 \| SEV-2 \| SEV-3 \| informational>`           |
| Drafted at         | `<ISO-8601>`                                         |
| Approved at        | `<ISO-8601>`                                         |
| Sent at            | `<ISO-8601>` (after approval)                        |
| Channels           | `<email \| Slack \| support portal \| status page>`   |

## Affected scope

### Affected tenants

| Tenant id (anonymized) | Segment                | Region   | Severity for this tenant |
| ---------------------- | ---------------------- | -------- | ------------------------ |
| `<tenant-id>`          | `<exporter \| ...>`     | `<...>`  | `<SEV-1 \| ...>`          |

### Affected factors

| Factor URN                                | Old value | New value | Reason                  |
| ----------------------------------------- | --------- | --------- | ----------------------- |
| `<urn:gl:factor:...>`                     |           |           |                         |

### Affected APIs / SDKs

| Surface                       | Affected since | Resolved at |
| ----------------------------- | -------------- | ----------- |
| `<endpoint or SDK method>`    | `<ISO-8601>`   | `<ISO-8601>` |

## What happened

Plain-language summary of the event. 1-2 short paragraphs.

## Why it happened

Root cause. If a post-mortem is in flight, link to it.

## What we did

Chronology, in plain language. What did the team do, when, in what
order? Who was paged?

## Customer message (verbatim)

This is the exact text that will be sent to affected customers.
Approver signs off on this section before send.

```
Subject: <subject line>

<body>
```

## Mitigation

What customers should do (if anything) and what they should NOT do.

## Rollback / hotfix status

| Action                     | Status      | Owner           | Done at      |
| -------------------------- | ----------- | --------------- | ------------ |
| Hotfix released            | `<state>`   | `<role>`        | `<timestamp>` |
| Rollback executed          | `<state>`   | `<role>`        | `<timestamp>` |
| Catalog correction shipped | `<state>`   | `<role>`        | `<timestamp>` |
| Customer follow-up call    | `<state>`   | `<role>`        | `<timestamp>` |

## Lessons / follow-up

* `<lesson 1 → action item ticket>`
* `<lesson 2 → ADR draft>`

## Sign-off

| Role                       | Name           | Decision     | Signed at (ISO-8601) |
| -------------------------- | -------------- | ------------ | -------------------- |
| Partner Success Lead (R)   | `<name>`       | `<approve>`  | `<timestamp>`        |
| CTO or Product Lead (A)    | `<name>`       | `<approve>`  | `<timestamp>`        |
| SRE Lead (C)               | `<name>`       | `<concur>`   | `<timestamp>`        |
| Backend/API Lead (C)       | `<name>`       | `<concur>`   | `<timestamp>`        |
| Climate Methodology (C)    | `<name>`       | `<concur>`   | `<timestamp>`        |
| Compliance/Security (C)    | `<name>`       | `<concur>` (when relevant) | `<timestamp>` |
