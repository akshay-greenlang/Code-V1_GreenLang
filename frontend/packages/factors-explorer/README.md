# @greenlang/factors-explorer

Public Factor Explorer UI for GreenLang Factors FY27 — **Surface A** from
`docs/frontend/FACTORS-UI-SPEC.md`.

SEO-indexable, unauth-browseable, bundle < 200kB initial, LCP < 2s target.

## Stack

- Vite 7 + React 18 + TypeScript 5.9 (strict, `noUncheckedIndexedAccess`)
- TanStack Router (file-based routing under `src/routes/`)
- TanStack Query 5 for data fetching
- Tailwind 3.4 + shadcn/ui component patterns
- zod for runtime response validation
- vitest + @testing-library/react

Coexists with the root monorepo's MUI-based `@greenlang/v2-2-shell`. Tailwind
and MUI can live side-by-side because this package owns its own Vite root and
CSS layer.

## Scripts

```
npm install              # installs workspace deps from monorepo root
npm run dev --workspace=@greenlang/factors-explorer   # starts Vite on :5174
npm run build --workspace=@greenlang/factors-explorer
npm test --workspace=@greenlang/factors-explorer
```

Dev server runs on **port 5174** (shell-ui owns 5173). The `/api/v1` prefix
is proxied to `http://localhost:8080` (the Factors FastAPI service defined in
`greenlang/factors/api_endpoints.py`).

## Source layout

```
src/
  components/       FactorCard, FactorDetail, ExplainTrace, QualityMeter,
                    LicenseBadge, EditionPin, SignedReceipt, SourceTile,
                    SearchBar, ThreeLabelDashboard, Layout
  components/ui/    shadcn-style primitives (Button, Card, Badge, Input)
  lib/              api.ts (typed fetch + zod), query client, utils
  types/            factors.ts — TS + zod mirror of ResolvedFactor
  routes/           file-based TanStack Router tree
  __tests__/        vitest suites + fixtures/resolved_factor.json
  main.tsx          entry
  index.css         Tailwind + design tokens
```

## Non-negotiables (see UI spec §1)

1. Every factor view shows source + version + validity + license class + FQS.
2. Explain trace is always one click from a factor detail.
3. Every response's `_signed_receipt` is displayed with a Copy button.
4. Public pages are unauth, SEO-indexable. Preview / connector_only /
   deprecated factors carry `<meta name="robots" content="noindex,follow">`.

## Deploy

Static SPA — `npm run build` produces `dist/` consumable by any static host.
Target is Cloudflare Pages (`factors.greenlang.io`). A signed-URL CDN rule
should strip the `Authorization` header so Cloudflare can cache responses.

See `docs/frontend/FACTORS-UI-SPEC.md` for the full spec (Surface A = this app).
