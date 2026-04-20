# FY27 Scope Note — GL-EUDR-APP

**Status (2026-04-20):** Out of FY27 scope. See `docs/fy27-scope.md` §2.1.

## TL;DR

- The GreenLang Climate OS v3 Business Plan (FY27 launch) does **not** list EUDR as a commercial product.
- This application remains in the tree for code continuity, but it is **not sold in FY27**, has no battlecard, and is not wired into the Comply umbrella (`packs/eu-compliance/PACK-Comply-000-unified/`).
- No FY27-active product (CBAM, CSRD, GHG, SBTi, TCFD, ISO 14064, CDP, Taxonomy, SB 253, Scope Engine, Factors, Comply) imports from `greenlang.agents.eudr`.

## Who to route EUDR deals to

If a prospect asks for EUDR in FY27:

1. Confirm whether EUDR is on the **critical path** of the pilot or a "nice to have". Most Indian exporters are primarily driven by CBAM + CSRD.
2. If EUDR is genuinely blocking: route through a Big-4 partner for the EUDR work and keep GreenLang scope to CBAM / CSRD / Scope Engine. That's the FY27 "stay narrow" rule.
3. Log the request as a **design-partner signal** in the Phase 6 revisit tracker (`docs/fy27-scope.md` §2.1 "Revisit at FY28 Q1").

## When this changes

A FY28 scope-up trigger requires **≥ 3 independent design-partner requests** for EUDR alongside CBAM or CSRD, *with* willingness to pay. Until then, leave this app dormant.

## If you're maintaining this app

- Keep it compiling: any breaking change in `greenlang/agents/eudr/` that prevents Python import must still be fixed.
- Do not invest in new EUDR features during FY27.
- Do not link this app from top-level docs / README / sales collateral.

---

*Owner: Akshay. Last reviewed: 2026-04-20 (Phase 6).*
