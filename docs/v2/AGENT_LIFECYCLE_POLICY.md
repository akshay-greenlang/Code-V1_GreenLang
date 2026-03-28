# GreenLang V2 Agent Lifecycle Policy

## Lifecycle States

1. `incubating`: internal experimentation only.
2. `qualified`: contract-compliant and test-verified.
3. `production`: approved for release-train deployment.
4. `deprecated`: active but scheduled for retirement.
5. `retired`: removed from release lanes.

## Required Metadata

- `agent_id`
- `owner_team`
- `support_channel`
- `current_version`
- `state`
- `deprecation_date` (required for deprecated state)
- `replacement_agent_id` (required for deprecated state)

## Enforcement Rules

- agents in `production` must have assigned owner and support channel.
- no `incubating` agent may run in regulated-critical pack workflows.
- deprecated agents must include migration replacement and retirement date.
- retired agents are blocked from active registry resolution.

## Promotion Gates

1. quality checks green.
2. security checks green.
3. determinism checks pass for regulated pathways.
4. runbook and failure-mode documentation complete.
