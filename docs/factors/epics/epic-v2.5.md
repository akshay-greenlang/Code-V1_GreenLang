# Epic: v2.5 (FY30 Q2)

| Field            | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Release profile  | post-GA                                                  |
| Target quarter   | FY30 Q2                                                  |
| Owner            | Streaming/SRE Lead                                       |
| Status           | Planned                                                  |

## Scope

Agent-native ingestion. Per document:

* AI agents read upstream regulator publications (PDF, web, XBRL)
  and propose factor records into a staging gate.
* Methodology-lead review queue (human-in-the-loop).
* Auto-detect drift in published values vs prior edition.

## Out of scope

* Fully autonomous publishing without human review — out of scope
  through v3.0 per audit-grade requirements.

## Deliverables

* Ingestion-agent runtime (`greenlang/factors/agents/ingestion/`)
  with safe-doc/safe-shell sandboxing.
* Staging gate UI for methodology lead.
* Drift detector + alerting.

## Acceptance criteria

* Agent ingestion produces records that pass the v0.1 provenance
  gate without human edits in ≥ 80% of runs.
* Drift detector: 0 false-negatives on a known-drift fixture suite.
* Methodology-lead time-to-approve median < 30 min per record.

## Tickets

* [ ] Ingestion-agent runtime + sandboxing.
* [ ] Staging gate UI.
* [ ] Drift detector + alerts.

## Dependencies

* v2.0 marketplace stable.
* Internal LLM cost / latency budget for the ingestion path.

## Release risks

* Agent hallucination — must be zero into the published catalog;
  staging gate is the only safety net.
* Upstream regulator format churn — long-tail of PDF parsers.
