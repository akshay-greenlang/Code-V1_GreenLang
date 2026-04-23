# CC5 — Control Activities

**Owner:** Eng Mgr. **Review cadence:** continuous (changes) + annual (policy).

## Controls

### CC5.1 — Selection and development of controls

- Evidence: control matrix mapping risks -> controls.
- Collection: manual.

### CC5.2 — Technology General Controls (TGC)

- Evidence: SDLC pipeline (CI/CD, SAST/DAST, SBOM), segregation of duties.
- Collection: automated.
- Artifacts:
  - [ ] `.github/workflows/factors_ci.yml` snapshot.
  - [ ] `docs/security/SCANNING.md`.
  - [ ] CODEOWNERS file for production-critical paths.
  - [ ] Admin-access-approval queue export.

### CC5.3 — Policies and procedures

- Evidence: current security policy documents under `docs/security/`.
- Collection: manual.
