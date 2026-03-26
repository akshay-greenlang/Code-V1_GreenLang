# GreenLang v1 Dependency Graph

```mermaid
flowchart LR
    phase0[Phase0_Charter] --> phase1[Phase1_Contracts]
    phase1 --> phase2[Phase2_CLIRuntime]
    phase1 --> phase3[Phase3_SecurityPolicy]
    phase2 --> phase4[Phase4_DeterminismAuditObs]
    phase3 --> phase4
    phase4 --> phase5[Phase5_MultiApp_CI]
    phase5 --> phase6[Phase6_Docs_UAT]
    phase6 --> phase7[Phase7_ReleaseFreeze]
```

## Critical Path

Phase 0 -> Phase 1 -> Phase 2 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7

## Blocking Dependencies

- Phase 1 blocks all runtime and security conformance.
- Phase 3 policy baseline must pass before Phase 5 gate enforcement.
- Phase 6 UAT evidence is required for Phase 7 go/no-go.

