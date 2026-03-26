# GreenLang v1 Multi-App Capability Matrix

## Objective

Document what each app workspace can do in the multi-app frontend now, and what remains deferred.

## Matrix

| Capability | CBAM | CSRD | VCCI |
| --- | --- | --- | --- |
| Native backend run via web | Yes | Yes (adapter-driven) | Yes (adapter-driven) |
| Deterministic audit artifacts | Yes | Yes | Yes |
| Upload-driven input from web | Yes (`config + imports`) | Yes (`single input`) | Yes (`single input`) |
| Policy/validation summary cards | Yes | Yes (normalized) | Yes (normalized) |
| Individual artifact downloads | Yes | Yes | Yes |
| Zip bundle download | Yes (policy-gated) | Yes | Yes |
| Existing standalone app UI parity | Baseline preserved | Not applicable | Partial (frontend exists separately) |

## Readiness Notes

- `CBAM`: production-grade local web flow preserved as non-regression anchor.
- `CSRD`: web workspace uses adapter semantics to expose run/evidence flow with current backend maturity.
- `VCCI`: web workspace uses adapter semantics; backend depth can evolve independently without breaking shell contract.

## Deferred (Explicit)

- Deep per-app UX parity with each app's independent full product UI.
- Cross-app SSO and role orchestration beyond API-key/session baseline.
- Unified live job streaming with websocket telemetry.
