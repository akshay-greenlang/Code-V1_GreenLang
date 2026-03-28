# GreenLang V2 Migration Playbooks

## Playbook 1: V1 to V2 App Profile Migration

1. copy v1 contract files to app `v2` profile directory.
2. update `contract_version` to `2.0` and `runtime` to `greenlang-v2`.
3. validate command grammar and artifact contract fields.
4. run `gl v2 validate-contracts` and `gl v2 runtime-checks`.
5. add migration notes and rollback path.

## Playbook 2: Pack Tier Promotion Migration

1. classify current pack tier and target tier.
2. satisfy signature requirement for target tier.
3. pass policy/security/determinism gates.
4. update governance records and support ownership.

## Playbook 3: Agent Deprecation Migration

1. mark state as `deprecated` with replacement agent.
2. publish deprecation date and migration deadline.
3. update affected app/pack references.
4. retire agent after migration window and release-board approval.
