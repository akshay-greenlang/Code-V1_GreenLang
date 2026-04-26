# GreenLang Factors — Release Epics (FY27 → FY31)

The 8 epics below are derived 1:1 from the source-of-truth document
(`docs/factors/roadmap/SOURCE_OF_TRUTH_MANIFEST.md`). Each file is a
top-level epic that the engineering team turns into concrete tickets.

The repo does not currently bind to an external issue tracker; these
markdown files ARE the canonical scope-of-record until the team picks
one. Tickets MUST link back to their epic file.

| Epic file                             | Release       | FY/Quarter | Profile       | Owner                         | Status           |
| ------------------------------------- | ------------- | ---------- | ------------- | ----------------------------- | ---------------- |
| `epic-v0.1-alpha.md`                  | v0.1 Alpha    | FY27 Q1    | `alpha-v0.1`  | Platform/Data Lead            | In stabilization |
| `epic-v0.5-closed-beta.md`            | v0.5 Beta     | FY27 Q2    | `beta-v0.5`   | Backend/API Lead              | Planned          |
| `epic-v0.9-public-beta.md`            | v0.9 RC       | FY27 Q3    | `rc-v0.9`     | Developer Experience Lead     | Planned          |
| `epic-v1.0-ga.md`                     | v1.0 GA       | FY27 Q4    | `ga-v1.0`     | Engineering Manager, Factors  | Planned          |
| `epic-v1.5.md`                        | v1.5          | FY28 Q3    | (post-GA)     | ML and Community Lead         | Planned          |
| `epic-v2.0.md`                        | v2.0          | FY29 Q2    | (post-GA)     | Enterprise Platform Lead      | Planned          |
| `epic-v2.5.md`                        | v2.5          | FY30 Q2    | (post-GA)     | Streaming/SRE Lead            | Planned          |
| `epic-v3.0.md`                        | v3.0          | FY31 Q1    | (post-GA)     | CTO / Factors GM              | Planned          |

## Epic Template

Each epic file has the following sections (matches the CTO's Phase 0
checklist):

* **Scope** — what this release ships
* **Out of scope** — what is explicitly deferred
* **Deliverables** — concrete artifacts (code, data, docs, infra)
* **Acceptance criteria** — quantitative gates that close the epic
* **Source coverage** — emission-factor sources in scope
* **API / SDK expectations** — public surface for the release
* **Security / compliance gates** — must-pass items
* **Owner** — single accountable person
* **Target quarter** — FY/quarter the epic ships
* **Dependencies** — upstream epics or external blockers
* **Release risks** — known unknowns

Because the repo has no ticket tracker yet, "tickets" inside an epic
are bullet points under a `Tickets` heading; pin their status with a
checkbox. When the team adopts a tracker (Jira, Linear, GitHub
Projects), each bullet becomes one ticket and this file holds the
external IDs.
