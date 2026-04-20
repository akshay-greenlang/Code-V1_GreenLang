# Factor Explorer UI — Spec

> **Phase 5.1.** React + MUI pages that expose the Factors catalog to developers, auditors, and the public.
> **Files.** `frontend/src/pages/FactorsExplorer.tsx`, `frontend/src/pages/FactorsCatalogStatus.tsx`.

---

## 1. Page catalogue

| Route | Page | Auth | Purpose |
|---|---|---|---|
| `/factors/explorer` | `FactorsExplorer.tsx` | JWT or API key | Interactive search + filter + detail modal |
| `/factors/status` | `FactorsCatalogStatus.tsx` | **Public** | Three-label coverage dashboard |

Both are registered in `frontend/src/App.tsx` and exposed in the top nav via `frontend/src/components/ShellLayout.tsx`.

## 2. Data sources

| UI element | Backend endpoint | Notes |
|---|---|---|
| Filter dropdowns | `GET /api/v1/factors/search/facets` | Returns facet counts for geography, scope, fuel_type |
| Results table | `GET /api/v1/factors/search?q=...&geography=...&limit=50` | Advanced sort/pagination uses `POST /api/v1/factors/search/v2` when added |
| Detail modal | `GET /api/v1/factors/{factor_id}` | Returns full `EmissionFactorResponse` |
| Status dashboard | `GET /api/v1/factors/status/summary` | **Public**, returns counts by label + per-source table |

## 3. Explorer page (`FactorsExplorer.tsx`)

### Layout

```
┌─────────────────────────────────────────────────────┐
│ Title + tagline                                     │
├─────────────────────────────────────────────────────┤
│ [ Search bar ][ Geography ▾ ][ Scope ▾ ][ Fuel ▾ ] [Search] │
├─────────────────────────────────────────────────────┤
│ Factor ID │ Fuel │ Geo │ Scope │ CO₂e │ Source │ DQS │ Label │
│ …rows… (click → detail modal)                              │
└─────────────────────────────────────────────────────┘
     Detail modal (on row click):
     - Source (provenance chain)
     - Data quality (DQS scores)
     - License + redistribution flags
     - Compliance frameworks
     - Raw payload (collapsible JSON)
```

### Tier-badge colours

| Label | Chip color | Meaning |
|---|---|---|
| Certified | `success` (green) | Signed off, regulator-safe |
| Preview | `warning` (amber) | In review, use with disclosure |
| Connector-only | `default` (grey) | Requires licensed connector |
| Deprecated | `error` (red) | Superseded, kept for reproducibility |

### Client-side refinement

The explorer uses `GET /search` (simpler payload, cacheable) and does
light client-side filtering for fuel_type + scope. For heavy workloads
(large result sets, relevance sorting, offset pagination), switch to the
`POST /search/v2` endpoint — the same page component can do it by
flipping the fetch call.

## 4. Status page (`FactorsCatalogStatus.tsx`)

### Design intent

The three-label dashboard is GreenLang's public trust signal. Visitors
must see, **without signing in**:

1. Total factors catalogued
2. Split by Certified / Preview / Connector-only / Deprecated
3. Per-source breakdown with same split
4. Generated-at timestamp so freshness is visible

### Layout

```
┌─────────────────────────────────────────────────────┐
│ "GreenLang Factors — Catalog Status"                │
│ edition + generated_at                               │
├─────────────────────────────────────────────────────┤
│ Totals:  [Certified 245] [Preview 78] [Connector 12]│
│          [Deprecated 5]  [Total 340]                │
│  ┌─────────────────────────────────────────────┐    │
│  │██████████▓▓▓▓░░░▒░░░░░░░░░░░░░░░░░░░░░░░░░│    │ ← proportion bar
│  └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│ By source:                                          │
│  source │ cert │ preview │ conn │ dep │ total       │
│  epa_hub│ 100  │   20    │  0   │  2  │  122        │
│  desnz  │  80  │   15    │  0   │  1  │   96        │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

### Accessibility

- Proportion bar is `role="img"` with an aria-label summarising the distribution.
- Table is marked as `aria-label="Factor counts by source"`.
- All colours satisfy WCAG AA against the table background.

## 5. State + loading

Every page uses a small state-machine pattern:

```
initial → loading → (success | error)
         ↑                         │
         └────── retry (future) ←──┘
```

No external state manager. `useState` + `useEffect` are sufficient for
these pages. Cancellation flags (`cancelled`) guard against unmounted
updates.

## 6. Future extensions

- **Search v2 adoption.** Switch the results table to paginated `POST /search/v2` with explicit sort controls (dqs_score, source_year, co2e_total).
- **Coverage filters on the status page.** Let a visitor filter by source or jurisdiction and re-render the proportion bar.
- **Real donut chart.** The current proportion bar is dependency-free; a future iteration can drop in an MUI X chart for richer interaction.
- **Trend view.** Month-over-month deltas in Certified count per source.
- **Factor-detail deep link.** `/factors/explorer/<factor_id>` opens the modal directly for embedded documentation.

## 7. Testing

- Vitest unit tests (to be added): rendering + interaction tests using `@testing-library/react`.
- Playwright E2E (folder already exists under `frontend/e2e/`): happy-path search + detail flow + public-status smoke.
- Accessibility audit: axe-core CI job.

---

*Last updated: 2026-04-20. Source: FY27_vs_Reality_Analysis.md §5.1 + §5.3.*
