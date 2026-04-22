# @greenlang/factors-ops-console

Internal **Operator Console** for GreenLang Factors FY27 — **Surface B** from
`docs/frontend/FACTORS-UI-SPEC.md`.

SSO/SAML-protected, RBAC-gated, NOT SEO-indexed. Companion to the public
`@greenlang/factors-explorer` (Surface A).

## Stack

- Vite 7 + React 18 + TypeScript 5.9 (strict, `noUncheckedIndexedAccess`)
- TanStack Router (file-based routing under `src/routes/`)
- TanStack Query 5 for data fetching + optimistic updates
- react-hook-form + zod resolvers for every mutation form
- Tailwind 3.4 + shadcn/ui primitives
- vitest + @testing-library/react

Tool versions match `@greenlang/factors-explorer` exactly.

## Scripts

```
npm install              # installs workspace deps from monorepo root
npm run dev --workspace=@greenlang/factors-ops-console     # Vite on :5175
npm run build --workspace=@greenlang/factors-ops-console
npm test --workspace=@greenlang/factors-ops-console
```

Dev server runs on **port 5175** (Explorer owns 5174, shell owns 5173). The
`/api/v1` prefix is proxied to `http://localhost:8080`. All ops-only endpoints
live under `/api/v1/ops/*`.

## Auth model (spec §3.3)

- **SSO**: SAML 2.0 via SEC-001 IdP. Login hits `/api/v1/auth/login`; receives
  JWT with `sub`, `tenant_id`, `roles[]`, `packs[]`. Session cookie
  `gl_ops_session` (HTTP-only, Secure, SameSite=Lax), 8-hour sliding TTL.
- **Roles**: `admin`, `methodology_lead`, `data_curator`, `reviewer`,
  `release_manager`, `legal`, `support`, `viewer` — enforced in both
  `<AuthGuard/>` and the backend.
- **Segregation of duties**: `<ApprovalChain/>` refuses to render an "Approve"
  button when `author.sub === current_user.sub`. Covered by tests.
- **Every mutation** sends `X-Audit-Actor`, `X-Audit-Reason`,
  `X-Audit-Session` headers. Backend writes to the SEC-005 audit log.
- **Tenant scoping**: `/overrides`, `/entitlements`, `/impact` all path-scope
  by `tenant_id`. Cross-tenant leaks prevented at API + UI.

## Source layout

```
src/
  components/       NavSidebar, RoleBadge, IngestionJobRow, ParserLogViewer,
                    MappingEditor, ValidationFailureRow, DiffTable,
                    ApprovalChain, OverrideEditor, ImpactSimulator,
                    EditionPromoter, AuthGuard
  components/ui/    shadcn-style primitives (Button, Card, Badge, Input)
  lib/              api.ts (TanStack Query wrappers for /api/v1/ops/*),
                    auth.ts (SSO + RBAC + SoD), query client, utils
  types/            ops.ts — zod schemas + TS types for every ops payload
  routes/           file-based TanStack Router tree (10 screens)
  __tests__/        vitest suites + fixtures/
  main.tsx          entry
  index.css         Tailwind + design tokens
```

## Non-negotiables (spec §3)

1. Every route wrapped in `<AuthGuard/>`. No public pages.
2. Every mutation writes to the SEC-005 audit log via audit headers.
3. Customer overrides never leak cross-tenant (API + UI enforcement).
4. SoD: author cannot approve their own submission — test-covered.
5. `meta robots = noindex,nofollow` — NOT SEO content.

## Deploy

Static SPA — `npm run build` produces `dist/`. Target is
`ops.greenlang.io` behind the corporate SSO IdP. Cloudflare Access
enforces SSO at the edge; JWT verification happens at the API gateway (Kong /
INFRA-006).

See `docs/frontend/FACTORS-UI-SPEC.md` §3 for the full spec.
