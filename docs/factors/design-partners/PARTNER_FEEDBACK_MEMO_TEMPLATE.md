# Partner Feedback Memo — `<partner-id>` `<release>`

> Template. Partner copies this to a private working doc, fills it
> in at the end of their pilot window, and returns to Partner
> Success. Partner Success captures a redacted summary in
> `DESIGN_PARTNER_TRACKER.md` and creates tickets under the
> matching epic.
>
> The existing `feedback-memo-template.md` in this directory is the
> v0.1-specific copy used by the two alpha partners. This template
> generalises that structure for use across v0.1 → v1.0+. When in
> doubt, use this one.

## 1. Identity

| Field                 | Value                                  |
| --------------------- | -------------------------------------- |
| Partner id            | `<partner-id>` (e.g. `IN-EXPORT-01`)    |
| Release window        | `<release>` (e.g. `v0.1 alpha FY27 Q1`) |
| Memo author           | `<partner contact name + role>`         |
| Submission date       | `<ISO-8601>`                           |
| Tenant id             | `<assigned by Operator>`                |
| Allow-listed sources  | `<list>`                               |

## 2. What we tried to calculate

In plain language: what calculation flow did you attempt? Use the
SDK or an HTTP example — be concrete.

```python
# example
client = FactorsClient(api_key="...")
factor = client.get_factor("urn:gl:factor:india-cea-co2-baseline:in:all_india:2025-26:cea-v22.0:v1")
emissions_kg = activity_kwh * factor.value
```

## 3. SDK experience

| Question                                           | Answer                                       |
| -------------------------------------------------- | -------------------------------------------- |
| Did `pip install greenlang-factors==<version>` succeed? | `<yes/no/issue>`                         |
| Did `client.health()` return `ok`?                 | `<yes/no/issue>`                             |
| Did the canonical SDK call you needed succeed first try? | `<yes/no/issue>`                       |
| If no, how many tries to succeed?                  | `<int>`                                       |
| Did you have to read source code to figure something out? | `<yes/no — describe>`                  |

## 4. API experience (if applicable)

| Question                                           | Answer                                       |
| -------------------------------------------------- | -------------------------------------------- |
| Endpoints used                                     | `<list>`                                     |
| Auth flow understandable?                          | `<yes/no/issue>`                             |
| Errors surfaced clearly?                           | `<yes/no/issue — list error envelope gaps>`  |
| Rate-limit experience                              | `<smooth / hit-limit / unclear>`             |

## 5. Where docs failed

List specific doc URLs / sections where you got stuck or wished for
more. Be ruthless — vague feedback is unactionable.

* `<URL or section>` — what was unclear
* `<URL or section>` — what was missing

## 6. Missing factors

List factor URNs you expected to find but didn't, or factor
slices that don't yet exist. Include enough detail for our
methodology lead to triage.

| Activity / context                                                                  | Expected factor (description)                  | What we used as a workaround |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------- |
|                                                                                     |                                                |                              |

## 7. Missing endpoints / SDK gaps

| Capability                                                                          | Why it matters                                 | Workaround                   |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------- |
|                                                                                     |                                                |                              |

## 8. Confidence / audit concerns

| Concern                                                                             | Why it matters                                 | Severity (low / med / high) |
| ----------------------------------------------------------------------------------- | ---------------------------------------------- | --------------------------- |
|                                                                                     |                                                |                             |

## 9. Must-fix before next release

Top items the partner needs us to fix before they will continue to
the next release.

1. `<must-fix 1>` — affected calculation, severity, suggested fix
2. `<must-fix 2>`
3. `<must-fix 3>`

## 10. Delight items (optional)

Anything that exceeded expectations and you want us to keep doing.

* `<delight 1>`

## 11. Continuation intent

| Question                                            | Answer                                       |
| --------------------------------------------------- | -------------------------------------------- |
| Will you continue to the next release?              | `<yes / hold / churn>`                       |
| If yes, what use-case will you focus on next?       | `<text>`                                     |
| Do you want to participate in v0.5 / v1.0 design?   | `<yes / no>`                                 |
| Are you willing to be a public reference partner?   | `<yes / not yet / no>`                       |

## 12. Signoff

| Role                       | Name           | Signed at (ISO-8601) |
| -------------------------- | -------------- | -------------------- |
| Partner technical contact  | `<name>`       | `<timestamp>`        |
| Partner Success Lead (GL)  | `<name>`       | `<timestamp>`        |
| CTO (GL) — receives summary | `<name>`      | `<timestamp>`        |
