# GreenLang V2 Enterprise UX Shell

## UX Objective

Provide a single enterprise control surface for 6-8 apps with consistent behavior, policy visibility, and operational traceability.

## Required Product Surfaces

- global app switcher with role-aware visibility
- unified run center with cross-app filtering
- policy and compliance status rail
- artifact explorer and deterministic diff surface
- connector health and incident banner system

## Shared Interaction Contract

- consistent run lifecycle states (`queued`, `running`, `completed`, `failed`)
- consistent status chips (`PASS`, `WARN`, `FAIL`)
- consistent error envelope (`title`, `message`, `details`)
- consistent download semantics and export gating

## V2 UX Quality Gates

1. lint/type/test/build must pass for frontend packages.
2. e2e workflow must pass key operator journeys for `/apps` and app workspaces.
3. visual regression checks must pass baseline shell snapshots.
4. accessibility and keyboard navigation checks for critical workflows.

## V2.2 Implementation Status

- root React shell serves `/apps/*`, `/runs`, `/governance`, and `/admin` from FastAPI SPA fallback.
- command palette (`Ctrl/Cmd+K`) is enabled for app and platform route navigation.
- run center includes DAG stage progress, bundle/artifact evidence links, and artifact checksum diffing.
- governance center reads live pack tier, agent lifecycle, and policy bundle metadata from `/api/v1/governance/*`.
- role-aware routing is enforced in both navigation visibility and route guards.
- realtime status uses SSE heartbeat with reconnect/degraded state handling in workspace views.
