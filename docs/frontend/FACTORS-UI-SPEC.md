# GreenLang Factors v1 — Frontend Specification

Version: 1.0 (spec-freeze candidate)
Author: GL-FrontendDeveloper
Date: 2026-04-22
Status: Ready for design review
Scope: Two shipping surfaces that front the Factors v1 catalog + resolution engine.

---

## 1. Executive summary

Factors v1 is the emissions-factor catalog, resolution cascade, and ingestion/QA platform that sits under every GreenLang calculation. It is already backed by:

- `greenlang/factors/api_endpoints.py` — search v2, explain, alternates, diff, bulk export, impact simulator, audit bundle, ETag + Cache-Control helpers.
- `greenlang/factors/resolution/result.py` — `ResolvedFactor` payload shape: `chosen_factor_id`, `fallback_rank` (1..7), `step_label`, `why_chosen`, `alternates[]`, `gas_breakdown`, `uncertainty`, `unit_conversion_*`, `deprecation_status`, `quality_score`.
- Quality, ingestion, and watch modules under `greenlang/factors/quality/*`, `greenlang/factors/ingestion/*`, `greenlang/factors/watch/*`, `greenlang/factors/mapping/*`.

This spec defines two distinct frontends on top of that backend. They are deliberately built on different stacks because they have different performance, SEO, auth, and deploy requirements:

| | Surface A — Public Factor Explorer | Surface B — Internal Operator Console |
|-|-|-|
| Host | `factors.greenlang.io` | `ops.greenlang.io` |
| Audience | Developers, consultants, platforms (top-of-funnel) | GreenLang methodology + data ops team |
| Auth | Unauth-browseable (anonymous JWT, rate-limited) | SSO/SAML, RBAC from SEC-001/SEC-002 |
| Goal | SEO-indexed discovery + trust + checkout | Methodology day-job tooling |
| Stack | Next.js 14 App Router + RSC + Tailwind + shadcn/ui | React 18 SPA (Vite) + Tailwind + shadcn/ui + TanStack Router + TanStack Query |
| Deploy | Vercel or Cloudflare Pages | Internal cluster behind VPN + allow-list |
| Perf target | LCP < 2s, FCP < 1s, bundle < 200kB initial | p95 interaction < 200ms on broadband |

Non-negotiables enforced by both surfaces:

1. **Every factor shown anywhere** renders source + source version + validity window + license class + FQS next to the number. Bare numbers are a bug.
2. **Explain trace is always one click** from a factor detail view. The 7-step cascade (`fallback_rank`, `step_label`, `why_chosen`) is mandatory UI.
3. **Customer overrides never leak cross-tenant**. Enforced at API (tenant scoping) and at UI (no global search surface shows tenant-scoped rows).
4. **Every mutation in Surface B writes to the SEC-005 audit log** via `Audit-Actor` + `Audit-Reason` request headers.

---

## 2. Surface A — Public Factor Explorer (`factors.greenlang.io`)

### 2.1 Route map

```
/                                       Landing — 3-label dashboard + quickstart + sources
/search                                 Faceted search, URL-as-state
/factors/[factor_id]                    Factor detail + explain + alternates + signed receipt
/sources                                Source catalog
/sources/[source_id]                    Source detail + changelog
/method-packs                           7 method-pack browser
/method-packs/[profile]                 Method-pack deep dive
/editions                               Edition history + current pin + release notes
/docs                                   301 -> developer.greenlang.io
/sitemap.xml                            Static + dynamic (factors + sources)
/robots.txt                             Allow /, /search, /factors/*, /sources/*, /method-packs/*
```

Rendering strategy (Next.js 14 App Router):

| Route | Strategy | Reason |
|-|-|-|
| `/` | Server Component, ISR 10 min | Counts change per release |
| `/search` | Client Component, URL state, Query via RSC handoff | Interactive |
| `/factors/[factor_id]` | Server Component, ISR 1 hour (certified) / 10 min (preview), dynamic head | SEO per-factor |
| `/sources`, `/sources/[source_id]` | Server Component, ISR 1 hour | Low churn |
| `/method-packs`, `/method-packs/[profile]` | Static Generation | Changes only on release |
| `/editions` | Server Component, ISR 5 min | Must be fresh on release day |

### 2.2 Component tree

```
app/
  layout.tsx                <RootShell/>
    <TopNav/>
    <Footer/>
  page.tsx                  <LandingPage/>
    <HeroHeadline/>
    <ThreeLabelDashboard/>
    <QuickstartSnippet/>
    <SourceTileGrid/>
    <EditionPin/>
  search/
    page.tsx                <SearchPage/>
      <SearchBar/>
      <FacetSidebar/>
        <FacetGroup name="family"/>
        <FacetGroup name="jurisdiction"/>
        <FacetGroup name="method_profile"/>
        <FacetGroup name="source"/>
        <FacetGroup name="factor_status"/>
        <FacetGroup name="license_class"/>
        <FacetRange name="dqs_min"/>
        <FacetDateRange name="valid_on_date"/>
      <ResultsHeader/>                 (sort + result count)
      <ResultsList/>
        <FactorCard/> (repeated)
      <Pagination/>
  factors/
    [factor_id]/
      page.tsx              <FactorDetailPage/>
        <FactorHeader/>
          <DeprecationBanner/>         (if status=deprecated)
          <FactorTitle/>
          <ProvenanceBadge/>
          <LicenseBadge/>
          <QualityMeter/>              (FQS 0-100)
          <EditionPin/>
        <FactorNumberCard/>             (co2e_per_unit + unit, gas breakdown)
        <ExplainTrace/>
          <ExplainStepTimeline/>       (7 steps, highlight chosen)
          <WhyChosenCard/>
          <AssumptionsList/>
          <UnitConversionTrace/>       (if target_unit supplied)
        <AlternatesList/>
        <UncertaintyPanel/>
        <SignedReceipt/>                (copy hash, verify button)
        <JsonPeek/>                     (raw record collapsible)
        <CodeSnippets/>                 (Python SDK, REST, curl)
  sources/
    page.tsx                <SourcesIndexPage/>
      <SourceTile/> (repeated)
    [source_id]/
      page.tsx              <SourceDetailPage/>
        <SourceHeader/>
        <SourceChangelog/>
        <SourceFactorsList/>            (filtered FactorCard list)
  method-packs/
    page.tsx                <MethodPacksIndex/>
      <MethodPackCard/> (7)
    [profile]/
      page.tsx              <MethodPackDetail/>
        <SelectionRules/>
        <BoundaryRules/>
        <GwpBasis/>
        <RegionHierarchy/>
        <FallbackLogic/>
        <ReportingLabels/>
  editions/
    page.tsx                <EditionsPage/>
      <EditionTimeline/>
      <ReleaseNotesList/>
```

### 2.3 Component contracts (prop types)

All prop types are TypeScript interfaces the frontend team will implement. Shapes match `greenlang/factors/api_endpoints.py` + `greenlang/factors/resolution/result.py`.

```ts
// ---------- Shared primitive types ----------
type FactorStatus = "certified" | "preview" | "connector_only" | "deprecated";
type LicenseClass = "public" | "open_cc" | "commercial" | "proprietary" | "connector_only";
type FactorId = string;
type EditionId = string;
type SourceId = string;
type MethodProfile =
  | "corporate_scope1"
  | "corporate_scope2_location_based"
  | "corporate_scope2_market_based"
  | "corporate_scope3"
  | "electricity"
  | "freight"
  | "eu_policy"
  | "land_removals"
  | "product_carbon"
  | "finance_proxy";

interface ProvenanceSummary {
  source_org: string;
  source_publication: string;
  source_year: number;
  source_version: string;
  source_url?: string;
  citation?: string;
  methodology: string;      // enum value (ipcc_ar6, defra_2024, …)
}

interface LicenseSummary {
  class: LicenseClass;
  redistribution_allowed: boolean;
  commercial_use_allowed: boolean;
  attribution_required: boolean;
}

interface QualityScore {
  overall: number;          // 0-100 composite FQS
  rating: "A" | "B" | "C" | "D" | "E";
  uncertainty_95ci?: number;
}

// ---------- FactorCard ----------
// Rendered in search results, source-detail list, and any "related factors" row.
interface FactorCardProps {
  factor_id: FactorId;
  fuel_type: string;
  geography: string;
  scope: "1" | "2" | "3";
  boundary: string;
  co2e_per_unit: number;
  unit: string;
  source: string;           // provenance.source_org
  source_year: number;
  dqs_score: number;
  factor_status: FactorStatus;
  source_id?: SourceId;
  highlight?: string;       // search snippet (HTML-safe)
  onClick?: (factor_id: FactorId) => void;
}

// ---------- FactorDetailPage props ----------
// Shape matches build_factor_explain() output merged with repo.get_factor() dict.
interface FactorDetailPageProps {
  factor: {
    factor_id: FactorId;
    factor_version: string;
    fuel_type: string;
    geography: string;
    scope: "1" | "2" | "3";
    boundary: string;
    unit: string;
    valid_from: string;     // ISO date
    valid_to?: string;      // ISO date
    factor_status: FactorStatus;
    gwp_100yr: {
      co2_total_kg: number;
      ch4_kg: number;
      n2o_kg: number;
      hfcs_kg: number;
      pfcs_kg: number;
      sf6_kg: number;
      nf3_kg: number;
      biogenic_co2_kg: number;
      co2e_total: number;
      gwp_basis: string;
    };
    provenance: ProvenanceSummary;
    license_info: LicenseSummary;
    dqs: QualityScore;
    content_hash: string;
    sector_tags?: string[];
    activity_tags?: string[];
  };
  explain: ResolvedFactorDump;         // ResolvedFactor.model_dump() shape
  edition_id: EditionId;
  signed_receipt: {
    payload_sha256: string;
    content_hash: string;
    algorithm: "SHA-256";
  };
  deprecation_replacement?: FactorId;
}

// ---------- ResolvedFactor dump (mirrors result.py exactly) ----------
interface ResolvedFactorDump {
  chosen_factor_id: FactorId;
  chosen_factor_name?: string;
  source_id?: SourceId;
  source_version?: string;
  factor_version?: string;
  vintage?: number;
  method_profile: MethodProfile;
  formula_type?: string;
  redistribution_class?: string;
  fallback_rank: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  step_label:
    | "customer_override"
    | "supplier_specific"
    | "facility_specific"
    | "region_specific"
    | "country_or_sector_average"
    | "global_average"
    | "default_assumption";
  why_chosen: string;
  alternates: AlternateCandidateDump[];
  assumptions: string[];
  deprecation_status?: "active" | "deprecated" | "superseded";
  deprecation_replacement?: FactorId;
  quality_score?: number;
  uncertainty: {
    distribution: string;
    ci_95_percent?: number;
    low?: number;
    high?: number;
    note?: string;
  };
  verification_status?: string;
  gas_breakdown: {
    co2_kg: number;
    ch4_kg: number;
    n2o_kg: number;
    hfcs_kg: number;
    pfcs_kg: number;
    sf6_kg: number;
    nf3_kg: number;
    biogenic_co2_kg: number;
    co2e_total_kg: number;
    gwp_basis: string;
  };
  factor_unit_denominator?: string;
  primary_data_flag?: string;
  target_unit?: string;
  converted_co2e_per_unit?: number;
  unit_conversion_factor?: number;
  unit_conversion_path: string[];
  unit_conversion_note?: string;
  resolved_at: string;                 // ISO datetime
  method_pack_version?: string;
  engine_version: string;
}

interface AlternateCandidateDump {
  factor_id: FactorId;
  tie_break_score: number;
  why_not_chosen: string;
  source_id?: SourceId;
  vintage?: number;
  redistribution_class?: string;
}

// ---------- ExplainTrace ----------
interface ExplainTraceProps {
  steps: Array<{
    rank: 1 | 2 | 3 | 4 | 5 | 6 | 7;
    label: ResolvedFactorDump["step_label"];
    status: "chosen" | "considered" | "skipped";
    note?: string;
  }>;
  chosen_rank: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  why_chosen: string;
  assumptions: string[];
  unit_conversion?: ResolvedFactorDump["unit_conversion_path"] extends infer _
    ? {
        target_unit: string;
        factor: number;
        converted_co2e_per_unit: number;
        path: string[];
        note?: string;
      }
    : never;
}

// ---------- AlternatesList ----------
interface AlternatesListProps {
  chosen_factor_id: FactorId;
  alternates: AlternateCandidateDump[];
  max_shown?: number;                  // defaults to 5, matches EXPLAIN_ALTERNATES_DEFAULT
  onExpand?: () => void;               // show up to 20
}

// ---------- QualityMeter ----------
interface QualityMeterProps {
  score: number;                       // 0-100
  rating: "A" | "B" | "C" | "D" | "E";
  uncertainty_95ci?: number;
  compact?: boolean;
}

// ---------- ProvenanceBadge + LicenseBadge ----------
interface ProvenanceBadgeProps {
  source_org: string;
  source_publication: string;
  source_year: number;
  source_version: string;
  source_url?: string;
  valid_from: string;
  valid_to?: string;
}

interface LicenseBadgeProps {
  class: LicenseClass;
  redistribution_allowed: boolean;
  commercial_use_allowed: boolean;
  attribution_required: boolean;
}

// ---------- EditionPin ----------
interface EditionPinProps {
  current_edition_id: EditionId;
  manifest_fingerprint: string;
  released_at: string;
  clickable?: boolean;                 // link to /editions
}

// ---------- SignedReceipt ----------
interface SignedReceiptProps {
  content_hash: string;                // SHA-256
  payload_sha256: string;
  edition_id: EditionId;
  factor_id: FactorId;
  onVerify: () => Promise<{ ok: boolean; server_hash: string }>;
}

// ---------- SearchBar ----------
interface SearchBarProps {
  value: string;
  onChange: (q: string) => void;
  onSubmit: (q: string) => void;
  suggestions?: string[];              // from /v1/factors/search?suggest=1
  placeholder?: string;
}

// ---------- FacetSidebar ----------
interface FacetSidebarProps {
  value: SearchFilters;
  onChange: (next: SearchFilters) => void;
  availableFacets: {
    families: Array<{ value: string; count: number }>;
    jurisdictions: Array<{ value: string; count: number }>;
    method_profiles: Array<{ value: MethodProfile; count: number }>;
    sources: Array<{ value: SourceId; label: string; count: number }>;
    factor_statuses: Array<{ value: FactorStatus; count: number }>;
    license_classes: Array<{ value: LicenseClass; count: number }>;
  };
}

interface SearchFilters {
  query: string;
  family?: string;
  jurisdiction?: string;       // maps to geography
  method_profile?: MethodProfile;
  source_id?: SourceId;
  factor_status?: FactorStatus;
  license_class?: LicenseClass;
  dqs_min?: number;            // 0-100
  valid_on_date?: string;      // ISO date
  sector_tags?: string[];
  activity_tags?: string[];
  sort_by: "relevance" | "dqs_score" | "co2e_total" | "source_year" | "factor_id";
  sort_order: "asc" | "desc";
  offset: number;
  limit: number;
}

// ---------- ThreeLabelDashboard ----------
interface ThreeLabelDashboardProps {
  certified_count: number;
  preview_count: number;
  connector_only_count: number;
  total: number;
  edition_id: EditionId;
}

// ---------- QuickstartSnippet ----------
interface QuickstartSnippetProps {
  languages: Array<"python" | "typescript" | "curl">;
  defaultLanguage?: "python" | "typescript" | "curl";
  apiBaseUrl: string;          // injected at build time
}

// ---------- SourceTile / SourceDetailPage ----------
interface SourceTileProps {
  source_id: SourceId;
  name: string;
  publisher: string;
  current_version: string;
  license_class: LicenseClass;
  factor_count: number;
  cadence: "annual" | "biannual" | "quarterly" | "ad_hoc";
  last_updated: string;
}

interface SourceDetailProps {
  source: SourceTileProps & {
    description: string;
    validity_start: string;
    validity_end?: string;
    jurisdiction_coverage: string[];
    changelog: Array<{ version: string; date: string; summary: string; diff_url?: string }>;
  };
}

// ---------- MethodPackCard / MethodPackDetail ----------
interface MethodPackCardProps {
  profile: MethodProfile;
  name: string;
  purpose: string;
  scope_coverage: Array<"1" | "2" | "3">;
  gwp_basis: string;
  region_hierarchy_depth: number;
}

interface MethodPackDetailProps {
  profile: MethodProfile;
  selection_rules: Array<{ rank: number; rule: string; example?: string }>;
  boundary_rules: string[];
  gwp_basis: string;
  region_hierarchy: string[];     // ordered list
  fallback_logic: Array<{
    rank: 1 | 2 | 3 | 4 | 5 | 6 | 7;
    step_label: ResolvedFactorDump["step_label"];
    description: string;
  }>;
  reporting_labels: Array<{ standard: string; label: string }>;
}

// ---------- DeprecationBanner ----------
interface DeprecationBannerProps {
  replacement_factor_id?: FactorId;
  deprecation_status: "deprecated" | "superseded";
  note?: string;
}
```

### 2.4 Page-by-page wireframes

#### 2.4.1 `/` — Landing

```
+---------------------------------------------------------------+
|  [logo] GreenLang Factors              Search | Docs | Sign in|
+---------------------------------------------------------------+
|                                                               |
|   Emission factors you can ship.                              |
|   Search 48,000+ factors across 7 method packs, with          |
|   full provenance, signed receipts, and an audit-grade        |
|   resolution cascade.                                         |
|                                                               |
|   +-------------------------------------------+               |
|   | [ certified 32,441 ] [ preview 4,102 ]    |               |
|   | [ connector-only 11,783 ]   [edition: Q2]  |               |
|   +-------------------------------------------+               |
|                                                               |
|   5-minute quickstart                                         |
|   +------------------------------------------------+          |
|   | [python ▼]                              [copy] |          |
|   | from greenlang import Factors                  |          |
|   | f = Factors.resolve(                           |          |
|   |   activity="natural_gas",                      |          |
|   |   jurisdiction="GB",                           |          |
|   |   method="corporate_scope1")                   |          |
|   +------------------------------------------------+          |
|                                                               |
|   Sources                                                     |
|   [DEFRA 2024] [EPA eGRID] [IPCC AR6] [IEA] [Ecoinvent 3.10] |
|                                                               |
+---------------------------------------------------------------+
```

API wiring:

| UI element | Endpoint | Cache |
|-|-|-|
| `<ThreeLabelDashboard/>` | `GET /v1/catalog/summary?edition=current` | `public, max-age=300` |
| `<SourceTileGrid/>` | `GET /v1/sources?limit=8&sort=popularity` | `public, max-age=3600` |
| `<EditionPin/>` | `GET /v1/editions/current` | `public, max-age=300` |
| `<QuickstartSnippet/>` | static (build-time) | — |

#### 2.4.2 `/search` — Faceted search

```
+---------------------------------------------------------------+
|  [search: natural gas combustion             ] [search]       |
+-----------+---------------------------------------------------+
|           |  2,341 results   sort: [relevance ▼] [ desc ▼]    |
| family    |  -------------------------------------------------|
| [ ] fuel  |  natural_gas — GB — scope 1                       |
| [ ] elec  |  2.0296 kgCO2e / m3     FQS: A (92)               |
| [ ] tran  |  DEFRA 2024 v1.2 • CC-BY 4.0 • [certified]        |
|           |  -------------------------------------------------|
| juris.    |  natural_gas — US — scope 1                       |
| [ ] GB    |  53.06 kgCO2e / mmbtu   FQS: B (84)               |
| [ ] US    |  EPA 2024 v1.0 • US-gov public • [certified]      |
| [ ] EU    |  -------------------------------------------------|
|           |  ...                                              |
| method    |                                                   |
| [ ] corp1 |  [ < prev ]   page 1 of 118   [ next > ]          |
| [ ] elec  |                                                   |
|           |                                                   |
| source    |                                                   |
| [ ] DEFRA |                                                   |
| [ ] EPA   |                                                   |
|           |                                                   |
| FQS ≥     |                                                   |
| [ 70 ]    |                                                   |
|           |                                                   |
| valid on  |                                                   |
| [ 2026-  ]|                                                   |
+-----------+---------------------------------------------------+
```

State model: URL is the state. Example URL:

```
/search?q=natural+gas&jurisdiction=GB&family=fuel&method_profile=corporate_scope1&dqs_min=70&valid_on_date=2026-01-01&sort_by=dqs_score&sort_order=desc&offset=0&limit=20
```

API wiring:

| UI element | Endpoint | Cache |
|-|-|-|
| Results list | `POST /v1/factors/search/v2` body = SearchV2Request | `public, max-age=300` |
| Facet counts | same response (computed server-side) | same |

Request body shape (matches `SearchV2Request`):

```json
{
  "query": "natural gas",
  "geography": "GB",
  "method_profile": "corporate_scope1",
  "dqs_min": 70,
  "valid_on_date": "2026-01-01",
  "sort_by": "dqs_score",
  "sort_order": "desc",
  "offset": 0,
  "limit": 20
}
```

#### 2.4.3 `/factors/[factor_id]` — Factor detail

```
+---------------------------------------------------------------+
|  [DEPRECATION BANNER: superseded by DEFRA-NG-GB-2025-001]     |  (only if deprecated)
+---------------------------------------------------------------+
|  natural_gas — GB — scope 1                    [edition Q2-25]|
|  DEFRA 2024 v1.2 • CC-BY 4.0 • [certified] • FQS A (92)       |
|  valid 2024-01-01 → 2024-12-31                                |
+---------------------------------------------------------------+
|  +--------------------------+ +----------------------------+  |
|  | 2.0296 kgCO2e / m3       | | Gases (IPCC AR6 100yr)     |  |
|  | (±1.5% 95% CI)           | |  CO2    1.9876 kg          |  |
|  |                          | |  CH4    0.0082 kg (x29.8)  |  |
|  |                          | |  N2O    0.0033 kg (x273)   |  |
|  +--------------------------+ +----------------------------+  |
|                                                               |
|  Explain trace                                                |
|  ┌─ Step 1 customer_override       skipped (no tenant ctx) ─┐ |
|  │  Step 2 supplier_specific       skipped                  │ |
|  │  Step 3 facility_specific       skipped                  │ |
|  │  Step 4 region_specific         skipped                  │ |
|  │  Step 5 country_or_sector_avg ▶ CHOSEN (rank 5)          │ |
|  │  Step 6 global_average          considered               │ |
|  │  Step 7 default_assumption      considered               │ |
|  └─                                                          ┘ |
|  Why chosen: "DEFRA 2024 natural_gas GB matched jurisdiction  |
|               and method_profile=corporate_scope1; preferred  |
|               over IEA_GLOBAL_2023 by tie-break rank."        |
|                                                               |
|  Alternates (5 of 12)            [ show all 12 ]              |
|  • IEA_GLOBAL_2023 — tie 2.10 — "global, not GB-specific"     |
|  • ECOINVENT_NG_GB — tie 3.00 — "attributional not operational"|
|  • ...                                                        |
|                                                               |
|  Uncertainty  normal • 95%CI ±1.5%  low 1.9991 high 2.0601    |
|                                                               |
|  Signed receipt                                               |
|  content_hash: 8f3d…c41  [ copy ]  [ verify on /v1/verify ]  |
|                                                               |
|  Use this factor                                              |
|  [ Python SDK | TypeScript | curl ]                           |
|  [code]                                                       |
+---------------------------------------------------------------+
```

API wiring:

| UI element | Endpoint | Cache |
|-|-|-|
| Page payload (detail + explain merged) | `GET /v1/factors/{factor_id}?explain=1&edition=current` | `public, max-age=3600` for certified, `max-age=600` for preview |
| Alternates "show all" | `GET /v1/factors/{factor_id}/alternates?limit=20` | `private, max-age=300` |
| Verify receipt | `POST /v1/factors/{factor_id}/verify` body `{ content_hash }` | no-cache |
| Code snippets | client-rendered | — |

ETag: `compute_etag(factor)` — clients pass `If-None-Match` on SWR refetch.

#### 2.4.4 `/sources` and `/sources/[source_id]`

```
/sources
+---------------------------------------------------------------+
|  Sources (18)                                                 |
|  +------------+ +------------+ +------------+ +------------+  |
|  | DEFRA 2024 | | EPA eGRID  | | IPCC AR6   | | IEA WEB    |  |
|  | UK • CC-BY | | US • public| | global     | | global     |  |
|  | 4,421 fac. | | 3,102 fac. | | 812 fac.   | | 2,204 fac. |  |
|  +------------+ +------------+ +------------+ +------------+  |
+---------------------------------------------------------------+

/sources/defra-2024
+---------------------------------------------------------------+
|  DEFRA 2024 Emission Factors                                  |
|  Publisher: UK Department for Environment • CC-BY 4.0         |
|  Current version: 1.2   cadence: annual                       |
|  Validity: 2024-01-01 → 2024-12-31                            |
|                                                               |
|  Changelog                                                    |
|  1.2 — 2024-06-01 — Electricity factors updated for Q1 mix    |
|  1.1 — 2024-03-15 — Refrigerant HFC-32 GWP corrected          |
|  1.0 — 2024-01-15 — Initial 2024 release                      |
|                                                               |
|  Factors in this source (4,421)                  [ browse ]   |
+---------------------------------------------------------------+
```

API wiring:

| UI element | Endpoint | Cache |
|-|-|-|
| Sources index | `GET /v1/sources` | `public, max-age=3600` |
| Source detail | `GET /v1/sources/{source_id}` | `public, max-age=3600` |
| "Browse factors" link | `/search?source_id={id}` | — |

#### 2.4.5 `/method-packs` and `/method-packs/[profile]`

```
/method-packs
7 tiles, one per method profile:
corporate | electricity | freight | eu_policy | land_removals | product_carbon | finance_proxy

/method-packs/electricity
+---------------------------------------------------------------+
|  Electricity Method Pack                                      |
|                                                               |
|  Selection rules                                              |
|  1. Prefer supplier-specific (residual mix) if market-based   |
|  2. Else location-based grid factor                           |
|  3. Else country mix                                          |
|  4. Else regional mix                                         |
|  5. Else global average (IEA)                                 |
|                                                               |
|  Boundary rules                                               |
|  • Cradle-to-grid, excludes T&D losses unless flagged          |
|  • T&D losses reported separately under scope3.cat3            |
|                                                               |
|  GWP basis: IPCC AR6 100yr                                    |
|                                                               |
|  Region hierarchy                                             |
|  facility → subnational → country → region → global           |
|                                                               |
|  Fallback logic (maps to 7-step cascade)                      |
|  rank 1 customer_override                                     |
|  rank 2 supplier_specific                                     |
|  rank 3 facility_specific                                     |
|  rank 4 region_specific                                       |
|  rank 5 country_or_sector_average                             |
|  rank 6 global_average                                        |
|  rank 7 default_assumption                                    |
|                                                               |
|  Reporting labels                                             |
|  GHG Protocol: "Scope 2 electricity (location-based)"         |
|  CDP: "C6.3"                                                  |
+---------------------------------------------------------------+
```

API wiring:

| UI element | Endpoint | Cache |
|-|-|-|
| All 7 method packs | `GET /v1/method-packs` | `public, max-age=3600` |
| Detail | `GET /v1/method-packs/{profile}` | `public, max-age=3600` |

#### 2.4.6 `/editions`

```
+---------------------------------------------------------------+
|  Editions                                                     |
|                                                               |
|  ● 2025.Q2   released 2025-06-15   CURRENT                    |
|     manifest: 7c9e…f421                                       |
|     release notes: [ read ]                                   |
|                                                               |
|  ○ 2025.Q1   released 2025-03-15                              |
|     manifest: 3a1b…9902                                       |
|     release notes: [ read ]                                   |
|                                                               |
|  ○ 2024.Q4   released 2024-12-15                              |
+---------------------------------------------------------------+
```

API wiring: `GET /v1/editions?limit=12`, `GET /v1/editions/current`, `GET /v1/editions/{edition_id}/release-notes`.

### 2.5 API wiring map (Surface A)

All endpoints are under `https://api.greenlang.io/v1/`. Anonymous JWT is minted on first visit (stored in HTTP-only cookie, 7-day TTL) and carries tier = `public`. Rate limits: 100 rpm/IP.

| Call site | Method | Path | Request | Response |
|-|-|-|-|-|
| `<ThreeLabelDashboard/>` | GET | `/catalog/summary` | `?edition=current` | `{ certified, preview, connector_only, total, edition_id }` |
| `<SourceTileGrid/>` | GET | `/sources` | `?limit=N&sort=popularity` | `SourceTile[]` |
| `<EditionPin/>` | GET | `/editions/current` | — | `{ edition_id, manifest_fingerprint, released_at }` |
| `<SearchPage/>` | POST | `/factors/search/v2` | `SearchV2Request` JSON | `SearchV2Result` JSON + `X-Total-Count`, `ETag` |
| `<FactorDetailPage/>` | GET | `/factors/{factor_id}` | `?explain=1&edition=current&method_profile=...&alternates_limit=5` | Merged factor + `ResolvedFactorDump` + `signed_receipt` |
| `<AlternatesList/>` "show all" | GET | `/factors/{factor_id}/alternates` | `?limit=20&method_profile=...` | `{ chosen_factor_id, alternates[] }` |
| `<SignedReceipt/>` verify | POST | `/factors/{factor_id}/verify` | `{ content_hash }` | `{ ok, server_hash }` |
| `<SourceDetailPage/>` | GET | `/sources/{source_id}` | — | `SourceDetailProps["source"]` |
| `<MethodPackDetail/>` | GET | `/method-packs/{profile}` | — | `MethodPackDetailProps` |
| `<EditionsPage/>` | GET | `/editions` | `?limit=12` | edition summaries |

All GETs honour `ETag` (per `compute_etag`, `compute_list_etag`, `compute_search_etag`, `compute_explain_etag`) + `Cache-Control` (per `cache_control_for_status`, `cache_control_for_list`, `cache_control_for_explain`). The Next.js fetch layer sets `If-None-Match` on stale-while-revalidate refetches.

### 2.6 SEO plan

- **Per-factor pages** are the primary SEO surface. Target: each certified factor ranks on its canonical long-tail query (e.g. "natural gas combustion emission factor UK").
- `generateMetadata()` for `/factors/[factor_id]`:
  - `<title>`: `{fuel_type} — {geography} — {scope} emission factor | GreenLang Factors`
  - `<meta name="description">`: first 160 chars of "Why chosen" + `{co2e_per_unit} {unit}` + source + year.
  - Open Graph: generated card with FactorCard content + FQS badge.
  - JSON-LD `Dataset` schema per factor: `name`, `description`, `creator` (source publisher), `license`, `distribution`, `temporalCoverage`, `variableMeasured`.
- **Canonical URL**: `https://factors.greenlang.io/factors/{factor_id}` (no edition query param in canonical — always points to current). Historical editions carry `rel="canonical"` to current factor_id.
- **Sitemap**: `/sitemap.xml` generated nightly; includes `/factors/{id}` for every `certified` factor. Preview/connector_only excluded via `noindex`.
- **`robots.txt`**:
  ```
  User-agent: *
  Allow: /
  Allow: /factors/
  Allow: /sources/
  Allow: /method-packs/
  Allow: /editions
  Disallow: /search?       (pagination-heavy; rely on sitemap)
  Sitemap: https://factors.greenlang.io/sitemap.xml
  ```
- `preview`, `connector_only`, and `deprecated` factors set `<meta name="robots" content="noindex,follow">`.
- Internal linking: every `FactorCard` links to `/factors/[id]`; every factor detail cross-links to source + method-pack + alternates; every source detail links back to filtered search.

---

## 3. Surface B — Internal Operator Console (`ops.greenlang.io`)

### 3.1 Route map

```
/                                               Dashboard (operator home)
/ingestion                                      Ingestion Console (list of jobs)
/ingestion/[job_id]                             Ingestion job detail + parser log
/mapping                                        Mapping Workbench
/mapping/[mapping_set_id]                       Mapping set editor
/qa                                             QA Dashboard
/qa/[failure_id]                                QA failure remediation
/diff                                            Diff Viewer (pick two editions)
/diff/[left]/[right]                            Diff between two editions
/diff/[left]/[right]/[factor_id]                Factor-level diff
/approvals                                      Approval queue
/approvals/[review_id]                          Review detail + chain
/overrides                                      Customer override manager
/overrides/[tenant_id]                          Overrides for one tenant
/overrides/[tenant_id]/[factor_id]              Override editor
/impact                                         Impact simulator
/impact/[simulation_id]                         Impact report
/source-watch                                   Source Watch pending changes
/source-watch/[detection_id]                    Detection detail + classify
/entitlements                                   Entitlements Admin
/entitlements/[tenant_id]                       Tenant tier + pack assignments
/editions/manage                                Edition management
/editions/manage/[edition_id]                   Edition slice promoter + rollback
/audit                                          Audit log viewer (SEC-005)
/settings                                       Operator prefs (not tenant)
```

### 3.2 Component tree

```
src/
  App.tsx                       <AuthGuard/>
    <Router/>
  layouts/
    OperatorShell.tsx           <Sidebar/> + <TopBar/> + <Outlet/>
  guards/
    RoleGuard.tsx               role in ["methodology", "admin", "reviewer", "ops"]
  components/
    AuthGuard/
    Sidebar/
    TopBar/                     (tenant switcher, audit-actor indicator, env badge)
    IngestionJob/
    ParserLogViewer/            (virtualized, line-numbered, copy-line, jump-to-error)
    MappingEditor/              (left: activity text, right: factor picker)
    SuggestionAgent/            (inline LLM suggestion surface)
    ValidationFailureRow/
    DiffTable/                  (F034 diff rendering)
    ApprovalChain/              (SoD: author != approver)
    OverrideEditor/
    ImpactSimulator/            (form + report)
    EditionPromoter/            (slice-by-slice release orchestrator)
    ChangeDetectorCard/         (source_watch.py signal)
    DocDiffView/                (doc_diff.py output)
    RollbackDialog/
    AuditTrailPanel/            (shown inline in every mutation screen)
    ReasonCodeInput/            (mandatory for every mutation)
  pages/
    Dashboard/
    Ingestion/
      IngestionList.tsx
      IngestionDetail.tsx
    Mapping/
    QA/
    Diff/
    Approvals/
    Overrides/
    Impact/
    SourceWatch/
    Entitlements/
    Editions/
    Audit/
```

### 3.3 SSO + RBAC model

Surface B inherits GreenLang platform auth:

- **SSO**: SAML 2.0 via SEC-001 IdP integration (Okta/Azure AD). Session cookie `gl_ops_session` (HTTP-only, Secure, SameSite=Lax), 8-hour TTL, sliding.
- **JWT**: short-lived access token (15 min) + refresh token (8 h). JWT claims include `sub`, `tenant_id`, `roles[]`, `packs[]` — matches SEC-002 RBAC.
- **Roles** (enforced in `<RoleGuard/>` and at API):
  | Role | Permissions |
  |-|-|
  | `ops.viewer` | Read all screens. No mutations. |
  | `ops.ingestion` | Run fetchers, view parser logs, promote to review queue. |
  | `ops.mapping` | Edit mapping sets. |
  | `ops.qa` | Triage/approve/reject QA failures. |
  | `ops.reviewer` | Approve reviews. Cannot approve their own authored item (SoD). |
  | `ops.methodology` | Manage method packs, edition slicing. |
  | `ops.admin` | Entitlements, overrides, rollbacks. |
  | `ops.auditor` | Read audit log. No mutations. |

- **Segregation of duties**: enforced at `<ApprovalChain/>` level + API. Author's `sub` claim must not equal approver's `sub`. UI shows "You authored this — another reviewer must approve" banner if violated.
- **Mutation envelope**: every mutation sends `X-Audit-Actor: <jwt.sub>` + `X-Audit-Reason: <ReasonCodeInput value>` + `X-Audit-Session: <gl_ops_session id>`. Backend writes to SEC-005 audit log.
- **Tenant scoping**: `/overrides`, `/impact`, `/entitlements` all scope by `tenant_id` path param. Sidebar has a tenant switcher; switching fires a RBAC re-check and clears TanStack Query cache for tenant-scoped keys. Cross-tenant leak prevention is enforced at the API (403 if JWT.tenant_id doesn't match, OR admin role+explicit tenant-switch token).

### 3.4 Page-by-page wireframes (Surface B)

#### 3.4.1 `/ingestion` — Ingestion Console

```
+---------------------------------------------------------------+
| Ingestion Jobs                      [ + Run new fetcher ]     |
+---------------------------------------------------------------+
| job_id    source        started      status      rows  parser |
|-----------+--------------+------------+-----------+------+-----|
| j-9823    DEFRA 2025    2026-04-20   completed   4,421  [log] |
| j-9824    EPA eGRID Q1  2026-04-21   running     1,200  [log] |
| j-9821    IEA WEB 2025  2026-04-19   failed      0      [log] |
+---------------------------------------------------------------+
```

Wires to `greenlang/factors/ingestion/fetchers.py` + `parser_harness.py` + `ga/readiness.py`.

`/ingestion/[job_id]`:

```
+---------------------------------------------------------------+
| Job j-9823 • DEFRA 2025 • completed 4,421 rows                |
| started 2026-04-20 14:02 UTC • duration 4m 12s                |
+---------------------------------------------------------------+
| Parser log                                                    |
| +-----------------------------------------------------------+ |
| | L1   INFO  fetching https://.../2025-edition.zip           | |
| | L2   INFO  unpacked 14 files, 18MB                         | |
| | L3   WARN  row 823: missing boundary, defaulted             | |
| | L4   ERROR row 1204: unparseable unit "m^3/yr/m2"           | |
| | ...                                                        | |
| | [ copy ] [ jump to first error ]                           | |
| +-----------------------------------------------------------+ |
|                                                               |
| GA readiness                                                  |
|   parse success: 99.2%                                        |
|   schema validity: 100%                                       |
|   DQS distribution: A 68% B 22% C 8% D 2%                     |
|                                                               |
| [ Promote 4,402 records to review queue ]   [ Reject job ]    |
+---------------------------------------------------------------+
```

Every action (Promote/Reject) opens `<ReasonCodeInput/>` modal and writes audit log.

#### 3.4.2 `/mapping` — Mapping Workbench

```
+---------------------------------------------------------------+
| Mapping set: scope3-purchased-goods-2026                      |
+---------+-----------------------------------------------------+
| Raw     | Suggested factor                                    |
|---------+-----------------------------------------------------|
| "steel  | ● DEFRA-MAT-STEEL-GB-2024 (0.92 kgCO2e/kg, FQS 88) |
|  coil   |   confidence 0.94 (sim 0.97, vintage match)         |
|  UK"    | ○ ECOINVENT-STEEL-GLO-2023 (confidence 0.71)        |
|         | [ accept ] [ reject ] [ manual pick ]               |
|---------+-----------------------------------------------------|
| "alum   | ● DEFRA-MAT-ALU-GB-2024 confidence 0.91             |
|  foil"  | [ accept ] [ reject ]                               |
+---------+-----------------------------------------------------+
| 1,204 unmapped / 4,102 total    [ save draft ] [ submit ]     |
+---------------------------------------------------------------+
```

Wires to `greenlang/factors/mapping/` + `suggestion_agent`. Submit opens approval flow.

#### 3.4.3 `/qa` — QA Dashboard

```
+---------------------------------------------------------------+
| QA Failures (327 open)          filter: [all modules ▼]       |
+---------------------------------------------------------------+
| id     module          severity  factor_id           action   |
|--------+----------------+---------+--------------------+--------|
| q-812  validators      high      DEFRA-NG-GB-2025    [view]  |
|        "uncertainty > 10% but no note"                        |
| q-813  dedup_engine    med       duplicate of q-811   [view]  |
| q-814  cross_source    high      DEFRA vs EPA delta 28% [view]|
| q-815  license_scanner critical  proprietary + redistr [view]|
+---------------------------------------------------------------+
| [ bulk approve ] [ bulk reject ]  (requires role ops.qa)      |
+---------------------------------------------------------------+
```

Wires to `greenlang/factors/quality/*`: validators, dedup_engine, cross_source, license_scanner.

#### 3.4.4 `/diff/[left]/[right]` — Diff Viewer

```
+---------------------------------------------------------------+
| Diff: 2025.Q1 → 2025.Q2                     [ impact sim ]    |
+---------------------------------------------------------------+
| changed 412   added 88   removed 3   unchanged 43,210         |
+---------------------------------------------------------------+
| factor_id                    change type     delta            |
|------------------------------+----------------+---------------|
| DEFRA-ELEC-GB-2025-001        changed         co2e +3.2%     |
| DEFRA-NG-GB-2025-001          changed         uncertainty ↓  |
| EPA-EGRID-RFCW-2025           added           —              |
| IEA-ELEC-GBR-2024             removed         (sup by 2025)  |
+---------------------------------------------------------------+
```

Wires to `F034 diff_factor_between_editions`. Click row → `/diff/[l]/[r]/[factor_id]` with `<DiffTable/>`:

```
field              old                  new                    type
co2e_total         2.0142               2.0796                 changed
provenance.version 1.1                  1.2                    changed
sector_tags        [...]                [..., "industrial"]    changed
```

#### 3.4.5 `/approvals` — Approval Workflow

```
+---------------------------------------------------------------+
| Review queue (18)                                             |
+---------------------------------------------------------------+
| review_id  kind           author       waiting_on    age      |
|------------+---------------+-------------+--------------+------|
| r-412      ingestion      alice        bob          2h       |
| r-413      mapping        bob          carol        6h       |
| r-414      edition-slice  carol        alice,dave   1d       |
+---------------------------------------------------------------+
```

`/approvals/[review_id]`:

```
+---------------------------------------------------------------+
| Review r-412 • ingestion promotion • DEFRA 2025               |
|                                                               |
| Chain:  [alice author] → [bob approver (you)] → [release]     |
|                                                               |
| Context                                                       |
|   4,402 records, parse success 99.2%, DQS avg 87              |
|                                                               |
| Reason (required)                                             |
| [ Matches QA acceptance criteria v4...          ]             |
|                                                               |
| [ Approve ]   [ Request changes ]   [ Reject ]                |
| You cannot approve: you did not author — OK.                  |
+---------------------------------------------------------------+
```

Wires to `quality/review_queue.py`, `quality/review_workflow.py`, `quality/release_signoff.py`.

#### 3.4.6 `/overrides/[tenant_id]` — Customer Override Manager

```
+---------------------------------------------------------------+
| Tenant: acme-corp                         [ + New override ]  |
+---------------------------------------------------------------+
| factor_id                 override    reason           since  |
|---------------------------+-----------+-----------------+------|
| DEFRA-ELEC-GB-2025-001   0.211       "primary data"    Q1-25|
| CUSTOM-ACME-STEEL-001    full-rec    "supplier LCA"    Q1-25|
+---------------------------------------------------------------+
| Audit trail (last 10 events)                                  |
|   2026-04-12  admin-alice  created override                   |
|   2026-04-15  admin-bob    updated co2e value                 |
+---------------------------------------------------------------+
```

Wires to `tenant_overlay.py`. Mutation requires `ops.admin` + reason code.

#### 3.4.7 `/impact` — Impact Simulator

```
+---------------------------------------------------------------+
| What if we replace DEFRA-RF-GB-2024 with UK-2025-freight?     |
|                                                               |
| Factor id:       [ DEFRA-RF-GB-2024        ]                  |
| Hypothetical:    ( ) deprecate  (•) new value  ( ) batch      |
|   new co2e_total: [ 0.142 ]                                   |
| Tenant scope:    [ all ▼ ]  (or pick tenants)                 |
|                                                               |
| [ Run simulation ]                                            |
+---------------------------------------------------------------+
| Results                                                       |
|   affected computations: 12,404                               |
|   affected tenants: 34                                        |
|   avg delta: +2.1%   max: +18% (acme-corp Q3 filing)          |
|   suggested rollback plan: revert to v1.1 in edition 2025.Q2  |
|                                                               |
| [ Export report ]   [ Open rollback ]                         |
+---------------------------------------------------------------+
```

Wires to `quality/impact_simulator.py` + `watch/cross_edition_changelog.py`. API path: `POST /v1/ops/impact-sim`.

#### 3.4.8 `/source-watch` — Source Watch

```
+---------------------------------------------------------------+
| Pending source changes (7)                                    |
+---------------------------------------------------------------+
| source        detected_at   signal            action          |
|---------------+--------------+------------------+--------------|
| DEFRA 2025    2026-04-18    doc hash changed  [ classify ]  |
| IEA WEO       2026-04-19    new version v2025 [ classify ]  |
| EPA eGRID     2026-04-20    checksum drift    [ classify ]  |
+---------------------------------------------------------------+
```

Wires to `watch/source_watch.py`, `change_detector.py`, `doc_diff.py`.

#### 3.4.9 `/entitlements/[tenant_id]` — Entitlements Admin

```
+---------------------------------------------------------------+
| Tenant: acme-corp                                             |
|                                                               |
| Tier:        [ Pro ▼ ]                                        |
| Packs:       [x] corporate   [x] electricity                  |
|              [x] freight     [ ] eu_policy                    |
|              [ ] land_removals [ ] product_carbon [ ] finance_proxy |
| Rate limit:  [ 10,000 rpm ]                                   |
| Preview access:  [x] enabled                                  |
| Connector access: [ ] enabled                                 |
|                                                               |
| [ Save ]   (writes audit log)                                 |
+---------------------------------------------------------------+
```

Wires to `entitlements.py` + `tier_enforcement.py`.

#### 3.4.10 `/editions/manage/[edition_id]` — Edition Management

```
+---------------------------------------------------------------+
| Edition 2025.Q2 (draft)                                       |
|                                                               |
| Slice promoter (sequential)                                   |
|   1. Electricity        [ promoted ]  by alice  2026-04-10    |
|   2. Combustion         [ promoted ]  by bob    2026-04-12    |
|   3. Freight            [ in review ] author carol            |
|   4. Material / CBAM    [ pending ]                           |
|   5. Land / Removals    [ pending ]                           |
|   6. Product Carbon     [ pending ]                           |
|   7. Finance Proxy      [ pending ]                           |
|                                                               |
| [ Promote next slice ]   [ Rollback edition ]                 |
+---------------------------------------------------------------+
```

Wires to `watch/rollback_edition.py` + release orchestrator.

### 3.5 API wiring map (Surface B)

All paths under `https://api.greenlang.io/v1/ops/`. Every mutation requires `X-Audit-Actor`, `X-Audit-Reason`, `X-Audit-Session` headers.

| Screen | Method | Path | Handler |
|-|-|-|-|
| Ingestion list | GET | `/ingestion/jobs` | `fetchers.list_jobs()` |
| Ingestion detail | GET | `/ingestion/jobs/{job_id}` | `fetchers.get_job()` |
| Run fetcher | POST | `/ingestion/run` | `fetchers.run()` |
| Parser log | GET | `/ingestion/jobs/{job_id}/log` | streams parser output |
| GA readiness | GET | `/ingestion/jobs/{job_id}/readiness` | `ga/readiness.py` |
| Promote | POST | `/ingestion/jobs/{job_id}/promote` | `ga/readiness.promote()` |
| Mapping list | GET | `/mapping/sets` | `mapping/repository` |
| Mapping set | GET | `/mapping/sets/{id}` | `mapping/repository` |
| Mapping suggest | POST | `/mapping/suggest` | `mapping/suggestion_agent` |
| Mapping save | PUT | `/mapping/sets/{id}` | `mapping/repository` |
| QA failures | GET | `/qa/failures` | `quality/failure_store` |
| QA detail | GET | `/qa/failures/{id}` | `quality/failure_store` |
| QA remediate | POST | `/qa/failures/{id}/remediate` | `quality/*` |
| Diff | GET | `/editions/{l}/diff/{r}` | `F034 diff_factor_between_editions` |
| Approval queue | GET | `/reviews` | `quality/review_queue.py` |
| Approval detail | GET | `/reviews/{id}` | `quality/review_workflow.py` |
| Approve | POST | `/reviews/{id}/approve` | `quality/review_workflow.approve()` |
| Release signoff | POST | `/reviews/{id}/signoff` | `quality/release_signoff.py` |
| Overrides list | GET | `/tenants/{tid}/overrides` | `tenant_overlay.py` |
| Override save | PUT | `/tenants/{tid}/overrides/{fid}` | `tenant_overlay.py` |
| Impact sim single | POST | `/impact-sim` | `build_impact_simulation` |
| Impact sim batch | POST | `/impact-sim/batch` | `build_impact_simulation_batch` |
| Dependent comps | GET | `/impact-sim/dependents/{fid}` | `list_dependent_computations` |
| Source watch | GET | `/source-watch` | `watch/source_watch.py` |
| Source watch classify | POST | `/source-watch/{id}/classify` | `watch/change_detector.py` |
| Entitlements | GET | `/tenants/{tid}/entitlements` | `entitlements.py` |
| Entitlements save | PUT | `/tenants/{tid}/entitlements` | `entitlements.py` + `tier_enforcement.py` |
| Edition create | POST | `/editions` | edition service |
| Edition promote | POST | `/editions/{id}/promote/{slice}` | release orchestrator |
| Edition rollback | POST | `/editions/{id}/rollback` | `watch/rollback_edition.py` |
| Audit log | GET | `/audit` | SEC-005 read API |

### 3.6 Component prop contracts (Surface B)

```ts
interface AuthGuardProps {
  requiredRoles: Array<"ops.viewer" | "ops.ingestion" | "ops.mapping" | "ops.qa"
                   | "ops.reviewer" | "ops.methodology" | "ops.admin" | "ops.auditor">;
  children: ReactNode;
}

interface IngestionJobProps {
  job_id: string;
  source_id: SourceId;
  status: "queued" | "running" | "completed" | "failed" | "promoted" | "rejected";
  started_at: string;
  duration_seconds?: number;
  row_count?: number;
  parser_log_url: string;
}

interface ParserLogViewerProps {
  stream_url: string;         // SSE
  initial_lines?: string[];
  onError?: (line: number) => void;
  virtualized?: boolean;      // default true for > 1000 lines
}

interface MappingEditorProps {
  mapping_set_id: string;
  rows: Array<{
    raw_text: string;
    suggested: Array<{ factor_id: FactorId; confidence: number; reason: string }>;
    accepted?: FactorId;
    state: "unmapped" | "suggested" | "accepted" | "rejected";
  }>;
  onAccept: (index: number, factor_id: FactorId) => void;
  onReject: (index: number) => void;
  onManualPick: (index: number) => void;
  onSubmit: (reason: string) => Promise<void>;
}

interface SuggestionAgentProps {
  raw_text: string;
  onPick: (factor_id: FactorId) => void;
  max_results?: number;
}

interface ValidationFailureRowProps {
  id: string;
  module: "validators" | "dedup_engine" | "cross_source" | "license_scanner";
  severity: "low" | "med" | "high" | "critical";
  factor_id?: FactorId;
  message: string;
  onView: () => void;
}

interface DiffTableProps {
  factor_id: FactorId;
  left_edition: EditionId;
  right_edition: EditionId;
  status: "added" | "removed" | "changed" | "unchanged" | "not_found";
  changes: Array<{
    field: string;
    type: "added" | "removed" | "changed";
    old_value?: unknown;
    new_value?: unknown;
  }>;
  left_content_hash?: string;
  right_content_hash?: string;
}

interface ApprovalChainProps {
  review_id: string;
  kind: "ingestion" | "mapping" | "qa-remediation" | "edition-slice" | "override";
  author: { sub: string; display_name: string };
  steps: Array<{
    approver: { sub: string; display_name: string } | null;
    status: "pending" | "approved" | "rejected" | "changes_requested";
    at?: string;
    comment?: string;
  }>;
  current_user_sub: string;           // used for SoD check
  onApprove: (reason: string) => Promise<void>;
  onReject: (reason: string) => Promise<void>;
  onRequestChanges: (reason: string) => Promise<void>;
}

interface OverrideEditorProps {
  tenant_id: string;
  factor_id: FactorId;
  current?: {
    co2e_total?: number;
    reason: string;
    effective_from: string;
    effective_to?: string;
  };
  onSave: (next: { co2e_total?: number; reason: string; effective_from: string; effective_to?: string }) => Promise<void>;
  onDelete: () => Promise<void>;
}

interface ImpactSimulatorProps {
  factor_id: FactorId;
  mode: "listing_only" | "value_override" | "deprecation";
  hypothetical_value?: number | { co2e_total: number };
  tenant_scope?: string[];
  onRun: () => Promise<ImpactReportDump>;
}

interface ImpactReportDump {
  edition_id: EditionId;
  factor_id: FactorId;
  simulation_mode: "listing_only" | "value_override" | "deprecation";
  tenant_scope: string[] | null;
  summary: { affected_computations: number };
  tenants: string[];
  computations: Array<{
    computation_id: string;
    tenant_id: string;
    old_value: number;
    new_value: number;
    delta_pct: number;
  }>;
  suggested_rollback_plan?: string;
}

interface EditionPromoterProps {
  edition_id: EditionId;
  slices: Array<{
    order: 1 | 2 | 3 | 4 | 5 | 6 | 7;
    name: "Electricity" | "Combustion" | "Freight" | "Material/CBAM" | "Land/Removals" | "Product Carbon" | "Finance Proxy";
    status: "pending" | "in_review" | "promoted" | "rolled_back";
    by?: string;
    at?: string;
  }>;
  onPromoteNext: (reason: string) => Promise<void>;
  onRollback: (reason: string) => Promise<void>;
}

interface ReasonCodeInputProps {
  required: true;
  minLength: 10;
  maxLength: 500;
  value: string;
  onChange: (v: string) => void;
  suggestions?: string[];     // common reason codes
}

interface AuditTrailPanelProps {
  entity_type: "factor" | "mapping" | "override" | "edition" | "entitlement" | "review";
  entity_id: string;
  limit?: number;
}
```

---

## 4. Design system

### 4.1 Tokens (Tailwind config — identical between Surface A and B)

```
Color (semantic, HSL):
  --gl-bg              0 0% 100%          (dark: 220 14% 10%)
  --gl-surface         220 14% 98%        (dark: 220 14% 13%)
  --gl-border          220 13% 91%        (dark: 220 13% 22%)
  --gl-fg              220 14% 12%        (dark: 220 14% 92%)
  --gl-muted           220 9% 46%
  --gl-primary         158 84% 32%        emerald 700
  --gl-primary-fg      0 0% 100%
  --gl-accent          200 90% 42%        ocean
  --gl-warn            38 92% 50%
  --gl-danger          0 84% 55%
  --gl-success         142 72% 38%

Factor-status semantic:
  certified      → success
  preview        → accent
  connector_only → muted + lock icon
  deprecated     → danger (banner)

FQS rating:
  A (90-100) success
  B (75-89)  accent
  C (60-74)  warn
  D (40-59)  danger-soft
  E (<40)    danger

Typography:
  font-sans   : "Inter var", system-ui
  font-mono   : "JetBrains Mono", ui-monospace
  scale       : tw default + `text-2xs` (0.6875rem)

Spacing: tw default (4px base)
Radius:  rounded-md default, rounded-lg for cards, rounded-full for badges
Shadows: shadow-sm / shadow-md / shadow-lg — no heavy drop shadows
```

### 4.2 Component library

- Base: shadcn/ui (Button, Input, Select, Dialog, DropdownMenu, Tabs, Badge, Card, Table, Tooltip, Popover, Toast, Skeleton).
- Custom (shared between A and B): `<FactorCard/>`, `<ExplainTrace/>`, `<DiffTable/>`, `<QualityMeter/>`, `<ProvenanceBadge/>`, `<LicenseBadge/>`, `<EditionPin/>`, `<SignedReceipt/>`, `<SourceTile/>`, `<MethodPackCard/>`.
- Icons: Lucide. Charts: Recharts (small) + Plotly only where interactive drill-downs are required (Surface B impact sim).

### 4.3 Dark mode

Both surfaces ship dark mode at GA. Default = system. Tailwind `dark:` variants. No color is hardcoded outside tokens.

### 4.4 Empty states

Every list has (a) loading skeleton (`<Skeleton/>` from shadcn), (b) empty state with illustration + call to action, (c) error state with retry + incident ID.

---

## 5. State management

### 5.1 Surface A — Next.js

- **Server state** (factor detail, sources, method-packs, editions): React Server Components with `fetch()` and Next.js ISR. No client-side cache layer.
- **Client state** (search filters, sort, pagination, suggestions): URL is the state. Helper hook `useSearchState()` syncs URL `<->` React state via `useSearchParams()` + `router.replace(url, { scroll: false })`.
- **Light TanStack Query** on the client for `<SearchPage/>` only: `queryKey = ["search", SearchFilters]`, `staleTime = 5m`, `keepPreviousData = true` (for pagination).
- **Mutations**: only `/verify` (signed receipt). Optimistic: no — verify is fast and deterministic.

### 5.2 Surface B — SPA

- **TanStack Query** everywhere, with explicit boundaries:
  - `queryKeys.ingestion.jobs` → `["ingestion", "jobs"]`
  - `queryKeys.ingestion.job(id)` → `["ingestion", "job", id]`
  - `queryKeys.tenant(tid).overrides` → `["tenant", tid, "overrides"]`
  - `queryKeys.reviews.queue` → `["reviews", "queue"]`
  - `queryKeys.diff(l,r)` → `["diff", l, r]`
  - Tenant-scoped keys always include `tid` so tenant switch `queryClient.invalidateQueries({ queryKey: ["tenant", tid] })` is a single call.
- **Optimistic updates on approval**:
  - `POST /reviews/{id}/approve` → mutation optimistically sets `status="approved"` on the review cache entry + removes it from the queue list.
  - On 409 (SoD violation, stale version) → rollback + toast.
- **Cache invalidation** on mutations:
  - Promote ingestion → invalidate `["ingestion"]` + `["reviews"]`.
  - Save override → invalidate `["tenant", tid, "overrides"]` + `["impact", fid]`.
  - Promote edition slice → invalidate `["editions"]` + `["diff"]`.
- **URL as state for Surface B**: QA filter, diff pair, tenant switcher — all in URL so screens are linkable between operators.
- **SSE for streaming**: parser log uses EventSource; no polling.

---

## 6. Accessibility — WCAG 2.1 AA

Both surfaces. Automated: `eslint-plugin-jsx-a11y`, axe-core in Playwright CI.

| Concern | Implementation |
|-|-|
| Color contrast | All tokens pass 4.5:1 on body, 3:1 on ≥18pt. FQS and factor-status badges include a label icon, not color alone. |
| Keyboard nav | Every interactive element `tabIndex=0`. Dialogs focus-trap (shadcn default). Skip-to-main-content link. |
| Focus visible | `focus-visible:ring-2 ring-offset-2` always on. No `outline:none` without replacement. |
| Screen reader — `<ExplainTrace/>` | Ordered list of 7 `<li>` with `aria-current="true"` on chosen step. "Why chosen" is an `<p>` inside the chosen `<li>` and is read immediately after the step label. Alternates: `<ol aria-label="Alternate factors considered">`. |
| Screen reader — `<DiffTable/>` | Rendered as native `<table>` with `<caption>`, `<thead>`, row `scope="row"`. Change type in a `<td>` with `<span class="sr-only">field added</span>` prefix. |
| Forms | Every `<Input/>` has a `<label/>`. Errors linked via `aria-describedby` + `aria-invalid`. `<ReasonCodeInput/>` announces min length with live region on blur. |
| Modal dialogs | `role="dialog"` + `aria-modal="true"` + initial focus on first field + Esc to close. |
| Live regions | Toasts = `role="status"`. Impact sim "running" = `aria-live="polite"`. |
| Reduced motion | All transitions wrapped in `@media (prefers-reduced-motion: no-preference)`. |
| Zoom | Layout works at 200% zoom. No horizontal scroll under 1024px at 200%. |

---

## 7. Performance budgets

### 7.1 Surface A

| Metric | Budget | How enforced |
|-|-|-|
| LCP (p75) | < 2.0s | RSC + ISR; landing LCP = `<HeroHeadline/>` text only. |
| FCP (p75) | < 1.0s | Minimal critical CSS via Tailwind JIT + `@next/font`. |
| TTI (p75) | < 2.5s | < 200kB initial JS; route chunks lazy. |
| Initial JS (gz) | < 200 kB | `next-bundle-analyzer` in CI, block merge on regression > 10 kB. |
| CLS | < 0.05 | Reserved space for dashboards, badge widths fixed. |
| Factor detail navigation | < 400ms client transition | Prefetch on `FactorCard` hover. |
| Sitemap | < 5s to generate 60k URLs | Streamed; regenerated nightly. |

### 7.2 Surface B

| Metric | Budget |
|-|-|
| p95 interaction | < 200ms on broadband |
| p95 search-in-console (uses pgvector) | < 300ms (enforced at API; UI shows skeleton after 100ms) |
| Parser log stream | first line visible < 500ms |
| Diff page (500 changes) | render < 1s |

### 7.3 Backend-facing budgets

| Endpoint | p95 |
|-|-|
| `POST /v1/factors/search/v2` (pgvector + post-filter) | 300 ms |
| `GET /v1/factors/{id}?explain=1` | 150 ms (hot), 500 ms (cold) |
| `POST /v1/ops/impact-sim` | 2 s (async fallback to job if > 5s) |

---

## 8. Analytics

PostHog (self-hosted). Anonymous user_id for Surface A (cookie-based), SSO sub for Surface B.

Events (Surface A):

| Event | Properties |
|-|-|
| `search_submitted` | query, facet_count, sort_by, result_count |
| `factor_resolve_viewed` | factor_id, source_id, factor_status, fqs, edition_id, method_profile |
| `explain_viewed` | factor_id, fallback_rank, chosen_step_label |
| `alternate_clicked` | from_factor_id, alternate_factor_id, rank |
| `snippet_copied` | factor_id, language |
| `receipt_verify_clicked` | factor_id, verify_ok |
| `source_detail_viewed` | source_id |
| `method_pack_viewed` | profile |
| `checkout_clicked` | tier, pack, factor_id |
| `signup_clicked` | source_surface |

Events (Surface B): every mutation emits `ops_action` with `{ action, entity, entity_id, role, tenant_id, reason_length }` in addition to the SEC-005 audit log.

Dashboards (mandatory at GA):
- Funnel: landing → search → factor detail → snippet copied → checkout clicked.
- Engagement: median explain views per session, median alternates opened.
- Source coverage: top 20 factors by view count grouped by `source_id`.

---

## 9. Rollout plan

### Week 5 (MVP alpha — public)
- Surface A: `/search` (POST /factors/search/v2), `/factors/[factor_id]` (with `explain=1`), `<ExplainTrace/>`, `<FactorCard/>`, `<QualityMeter/>`, `<ProvenanceBadge/>`, `<LicenseBadge/>`.
- Anonymous JWT + rate limit.
- Dark mode off (ship in Week 7).

### Week 6 (public expansion)
- `/` landing + `<ThreeLabelDashboard/>`.
- `/sources` + `/sources/[source_id]`.
- `/method-packs` + `/method-packs/[profile]`.
- SEO: sitemap, JSON-LD, OG images.
- PostHog wired.

### Week 7 (public polish)
- `/editions`.
- Dark mode.
- `<SignedReceipt/>` verify.
- `<DeprecationBanner/>`.
- a11y audit pass (axe + manual screen-reader).

### Week 9 (internal alpha)
- Surface B: `<AuthGuard/>` + SSO.
- `/ingestion` + `/ingestion/[job_id]` + `<ParserLogViewer/>` + `<ReasonCodeInput/>` + `<AuditTrailPanel/>`.
- `/qa` + `<ValidationFailureRow/>`.
- Tenant switcher (but overrides locked until Week 10).

### Week 10 (internal beta)
- `/approvals` + `<ApprovalChain/>` (with SoD enforcement).
- `/overrides` + `<OverrideEditor/>`.
- `/impact` + `<ImpactSimulator/>`.
- `/diff` + `<DiffTable/>`.

### Week 11 (internal GA)
- `/source-watch`.
- `/entitlements`.
- `/editions/manage` + `<EditionPromoter/>` + rollback.
- `/audit` read-only viewer.
- Load + security testing. Promote to `ops.greenlang.io`.

### Week 12 (post-launch)
- Mapping Workbench (`/mapping` + `<SuggestionAgent/>`) — deferred to Week 12 because suggestion_agent latency budget is still being tuned.
- Analytics dashboards live.

---

## 10. Open questions for design review

1. **Factor URL identity under editions**: `/factors/{factor_id}` currently resolves to `edition=current`. If a user deep-links a factor that was removed in the current edition, we 404 or redirect? Proposal: show "this factor is retired — last known version in edition X" with link. Need design for the retired state.
2. **Explain step 1 (customer_override) on public**: public surface has no tenant context, so step 1 is always "skipped". Do we render it at all, or collapse the 7 steps to 6 in public view? Recommendation: always render 7 (pedagogical), but grey out step 1 with "available only in signed-in Console" tooltip.
3. **Unit conversion UX**: `target_unit` is a query param. Do we expose a converter widget on the detail page, or only honour `?unit=`? Recommend: widget with popular units for the factor's family (e.g. kWh, MWh, kBTU, mmBTU for energy).
4. **Signed receipt verification surface**: today the button POSTs to `/verify`. Do we also expose a public standalone "paste a receipt" page at `/verify`? Useful for auditors who land from emails. Proposal: yes, Week 7.
5. **Preview-tier gating on Surface A**: `preview` factors show in search (with a badge) but `connector_only` factors don't list at all. Confirm: connector-only factors are completely invisible on public, or are they listed "shell-only" with "contact sales" CTA? The non-negotiable wording argues invisible. Need business sign-off.
6. **Operator Console tenant switcher**: should we allow multi-tenant merged views (e.g. "show me overrides across all my tenants") or force single-tenant context? Single-tenant is safer for cross-tenant leak prevention but hurts ops productivity. Proposal: single-tenant by default; admin-only "merged view" behind feature flag.
7. **Impact simulator on large ledgers**: ledgers > 10M rows will not fit a sub-5s sync response. Do we ship sync for MVP and switch to async jobs in Week 11, or async-first? Recommend: sync for < 100k ledger rows; async-first otherwise with job-status polling.
8. **Edition rollback blast radius**: a full edition rollback cascades to every tenant. Do we require two-approver signoff specifically for rollback, separate from normal approval chain? Proposal: yes — new role `ops.rollback_approver` gating rollback button.
9. **Dark mode default on Surface B**: ops team works long hours; default to dark? Default to system preference is safer for first launch; revisit after user feedback.
10. **Mapping Workbench LLM suggestions**: `SuggestionAgent` uses the suggestion_agent module. Do we surface model confidence numerically (0.0-1.0) or categorically (low/med/high)? Recommend: numeric + bucketed color. Product decision needed on whether to auto-accept high-confidence (≥ 0.95) with audit trail, or always require human-in-loop. Proposal: always human-in-loop for v1; revisit after 1,000 accepted mappings of telemetry.

---

End of spec. Ready for design review and frontend team estimation.
