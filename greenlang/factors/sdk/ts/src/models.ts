/**
 * TypeScript model interfaces for the GreenLang Factors SDK.
 *
 * These mirror the Pydantic v2 models in
 * `greenlang/factors/sdk/python/models.py`. Fields are declared
 * `Optional` (via `?`) wherever the Python model defaults them to
 * `None` so forward-compatibility with server drift is preserved.
 */

// ---------------------------------------------------------------------------
// Primitive / shared shapes
// ---------------------------------------------------------------------------

export interface Jurisdiction {
  code: string;
  name?: string | null;
  level?: string | null;
  [extra: string]: unknown;
}

export interface ActivitySchema {
  activity_id: string;
  label?: string | null;
  unit?: string | null;
  category?: string | null;
  description?: string | null;
  [extra: string]: unknown;
}

export interface QualityScore {
  overall_score: number;
  rating?: string | null;
  temporal?: number | null;
  geographical?: number | null;
  technological?: number | null;
  representativeness?: number | null;
  methodological?: number | null;
  [extra: string]: unknown;
}

export interface Uncertainty {
  ci_95?: number | null;
  distribution?: string | null;
  std_dev?: number | null;
  sample_size?: number | null;
  /**
   * v1.3: ABSOLUTE uncertainty in the factor's native unit
   * (e.g. kg CO2e per activity unit). See `uncertainty_percent`
   * for the relative form.
   */
  uncertainty?: number | null;
  /**
   * v1.3: RELATIVE uncertainty as a percentage 0-100
   * (`5.0` = 5 %). Complements the absolute `uncertainty` field.
   */
  uncertainty_percent?: number | null;
  [extra: string]: unknown;
}

export interface GasBreakdown {
  CO2?: number | null;
  CH4?: number | null;
  N2O?: number | null;
  HFCs?: number | null;
  PFCs?: number | null;
  SF6?: number | null;
  NF3?: number | null;
  biogenic_CO2?: number | null;
  ch4_gwp?: number | null;
  n2o_gwp?: number | null;
  [extra: string]: unknown;
}

// ---------------------------------------------------------------------------
// Core entities
// ---------------------------------------------------------------------------

export interface Source {
  source_id: string;
  organization?: string | null;
  publication?: string | null;
  year?: number | null;
  url?: string | null;
  methodology?: string | null;
  license?: string | null;
  version?: string | null;
  [extra: string]: unknown;
}

export interface MethodPack {
  method_pack_id: string;
  name?: string | null;
  version?: string | null;
  scope?: string | null;
  description?: string | null;
  jurisdictions?: string[];
  [extra: string]: unknown;
}

export interface Edition {
  edition_id: string;
  status?: string | null;
  label?: string | null;
  manifest_hash?: string | null;
  published_at?: string | null;
  [extra: string]: unknown;
}

export interface Factor {
  factor_id: string;
  fuel_type?: string | null;
  unit?: string | null;
  geography?: string | null;
  geography_level?: string | null;
  scope?: string | null;
  boundary?: string | null;

  co2_per_unit?: number | null;
  ch4_per_unit?: number | null;
  n2o_per_unit?: number | null;
  co2e_per_unit?: number | null;

  data_quality?: QualityScore | null;
  source?: Source | null;
  uncertainty_95ci?: number | null;

  valid_from?: string | null;
  valid_to?: string | null;

  factor_status?: string | null;
  license?: string | null;
  license_class?: string | null;
  compliance_frameworks?: string[];
  tags?: string[];
  activity_tags?: string[];
  sector_tags?: string[];
  notes?: string | null;

  edition_id?: string | null;
  source_id?: string | null;
  source_release?: string | null;
  release_version?: string | null;
  replacement_factor_id?: string | null;
  content_hash?: string | null;

  [extra: string]: unknown;
}

export interface FactorMatch {
  factor_id: string;
  score: number;
  explanation?: Record<string, unknown>;
  [extra: string]: unknown;
}

// ---------------------------------------------------------------------------
// Search / list responses
// ---------------------------------------------------------------------------

export interface SearchResponse {
  factors: Factor[];
  count?: number | null;
  total_count?: number | null;
  page?: number | null;
  page_size?: number | null;
  total_pages?: number | null;
  offset?: number | null;
  limit?: number | null;
  query?: string | null;
  edition_id?: string | null;
  search_time_ms?: number | null;
  sort_by?: string | null;
  sort_order?: string | null;
  next_cursor?: string | null;
  [extra: string]: unknown;
}

export interface CoverageReport {
  total_factors?: number | null;
  by_geography?: Record<string, number>;
  by_scope?: Record<string, number>;
  by_fuel_type?: Record<string, number>;
  by_source?: Record<string, number>;
  edition_id?: string | null;
  [extra: string]: unknown;
}

/**
 * v1.3: single-pack entry returned by
 * `/v1/method-packs/coverage` under `packs[]` / `overall`.
 */
export interface MethodPackCoverage {
  /** Canonical pack slug (or `null` on the `overall` roll-up). */
  slug?: string | null;
  version?: string | null;
  total_activities?: number | null;
  covered?: number | null;
  /** covered / total_activities as a 0-1 float. */
  fraction?: number | null;
  by_family?: Record<string, unknown>;
  by_jurisdiction?: Record<string, unknown>;
  [extra: string]: unknown;
}

/**
 * v1.3 canonical top-level shape for
 * `GET /v1/method-packs/coverage`.
 *
 * A single canonical shape is returned whether or not a `?pack=<slug>`
 * filter was applied. The Wave 4-G legacy payload
 * `{ packs: [{ pack_id, version, resolved_case_count_7d, ... }], total }`
 * is inflated transparently by {@link inflateMethodPackCoverage}.
 */
export interface MethodPackCoverageReport {
  packs: MethodPackCoverage[];
  overall?: MethodPackCoverage | null;
  /** Back-compat: legacy Wave 4-G payload preserved verbatim. */
  legacy_packs?: Array<Record<string, unknown>>;
  [extra: string]: unknown;
}

function legacyPackToCanonical(
  raw: Record<string, unknown>,
): MethodPackCoverage {
  const slug =
    (raw.slug as string | undefined) ?? (raw.pack_id as string | undefined) ?? null;
  const resolved = Number(raw.resolved_case_count_7d ?? 0) || 0;
  const unresolved = Number(raw.unresolved_case_count_7d ?? 0) || 0;
  let total_activities =
    (raw.total_activities as number | null | undefined) ?? null;
  let covered = (raw.covered as number | null | undefined) ?? null;
  let fraction = (raw.fraction as number | null | undefined) ?? null;
  if (total_activities == null) total_activities = resolved + unresolved;
  if (covered == null) covered = resolved;
  if (fraction == null && total_activities) {
    fraction = Number(covered) / Number(total_activities);
  }
  return {
    slug,
    version: (raw.version as string | null | undefined) ?? null,
    total_activities,
    covered,
    fraction,
    by_family: (raw.by_family as Record<string, unknown>) ?? {},
    by_jurisdiction: (raw.by_jurisdiction as Record<string, unknown>) ?? {},
  };
}

/**
 * Inflate either the v1.3 canonical shape or the legacy Wave 4-G shape
 * into {@link MethodPackCoverageReport}.
 */
export function inflateMethodPackCoverage(
  payload: unknown,
): MethodPackCoverageReport {
  if (!payload || typeof payload !== 'object') {
    return { packs: [], overall: null };
  }
  const obj = payload as Record<string, unknown>;
  // v1.3 canonical — already has `overall` or `packs[0].slug`.
  const packs = obj.packs;
  if (
    'overall' in obj ||
    (Array.isArray(packs) &&
      packs.length > 0 &&
      typeof packs[0] === 'object' &&
      packs[0] !== null &&
      'slug' in (packs[0] as Record<string, unknown>))
  ) {
    return obj as unknown as MethodPackCoverageReport;
  }
  // Legacy single-pack object (no `packs` wrapper).
  if ('pack_id' in obj && !('packs' in obj)) {
    const entry = legacyPackToCanonical(obj);
    return { packs: [entry], overall: entry, legacy_packs: [obj] };
  }
  // Legacy list-of-packs (Wave 4-G).
  const legacy = Array.isArray(packs) ? (packs as Array<Record<string, unknown>>) : [];
  const canonical = legacy.map(legacyPackToCanonical);
  let overall: MethodPackCoverage | null = null;
  if (canonical.length === 1) {
    overall = canonical[0];
  } else if (canonical.length > 0) {
    const total_activities = canonical.reduce(
      (acc, c) => acc + (c.total_activities ?? 0),
      0,
    );
    const covered = canonical.reduce((acc, c) => acc + (c.covered ?? 0), 0);
    overall = {
      slug: null,
      version: null,
      total_activities,
      covered,
      fraction: total_activities ? covered / total_activities : null,
      by_family: {},
      by_jurisdiction: {},
    };
  }
  return {
    packs: canonical,
    overall,
    legacy_packs: legacy,
  };
}

// ---------------------------------------------------------------------------
// Resolution (7-step cascade)
// ---------------------------------------------------------------------------

export interface ResolutionRequest {
  activity: string;
  method_profile: string;
  jurisdiction?: string | null;
  reporting_date?: string | null;
  supplier_id?: string | null;
  facility_id?: string | null;
  utility_or_grid_region?: string | null;
  preferred_sources?: string[];
  extras?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Wave 2 envelope types
// ---------------------------------------------------------------------------

/** Typed ``chosen_factor`` block the Wave 2 resolver returns. */
export interface ChosenFactor {
  factor_id: string;
  factor_version?: string | null;
  /** Method-pack release, distinct from ``factor_version`` (record rev). */
  release_version?: string | null;
  method_profile?: string | null;
  method_pack_id?: string | null;
  method_pack_version?: string | null;
  /** Alias of ``method_pack_id`` used by some Wave 2 endpoints. */
  pack_id?: string | null;
  co2e_per_unit?: number | null;
  unit?: string | null;
  geography?: string | null;
  scope?: string | null;
  [extra: string]: unknown;
}

/** Nested ``source`` block on the Wave 2 envelope. */
export interface SourceDescriptor {
  source_id: string;
  organization?: string | null;
  publication?: string | null;
  year?: number | null;
  url?: string | null;
  methodology?: string | null;
  license?: string | null;
  license_class?: string | null;
  version?: string | null;
  release_version?: string | null;
  provenance?: Record<string, unknown>;
  [extra: string]: unknown;
}

/** Composite quality envelope — Wave 2 surfaces ``composite_fqs_0_100``. */
export interface QualityEnvelope {
  composite_fqs_0_100?: number | null;
  overall_score?: number | null;
  rating?: string | null;
  temporal?: number | null;
  geographical?: number | null;
  technological?: number | null;
  representativeness?: number | null;
  methodological?: number | null;
  [extra: string]: unknown;
}

/**
 * Richer Wave 2 uncertainty envelope (superset of {@link Uncertainty}).
 *
 * v1.3: `uncertainty` is ABSOLUTE (native unit of the factor, e.g.
 * kg CO2e / activity unit); `uncertainty_percent` is RELATIVE (0-100,
 * `5.0` = 5 %). Both are optional and the resolver emits both when it
 * can compute them.
 */
export interface UncertaintyEnvelope {
  ci_95?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  distribution?: string | null;
  std_dev?: number | null;
  sample_size?: number | null;
  pedigree_matrix?: Record<string, unknown>;
  /** v1.3: absolute magnitude in the factor's native unit. */
  uncertainty?: number | null;
  /** v1.3: relative percentage 0-100 (`5.0` = 5 %). */
  uncertainty_percent?: number | null;
  [extra: string]: unknown;
}

/** Licensing envelope — full upstream chain surfaced by Wave 2. */
export interface LicensingEnvelope {
  license?: string | null;
  /** ``certified | preview | connector_only | redistributable`` */
  license_class?: string | null;
  redistribution_class?: string | null;
  upstream_licenses?: string[];
  attribution?: string | null;
  restrictions?: string[];
  [extra: string]: unknown;
}

/**
 * Structured deprecation status — canonical v1.3 shape.
 *
 * The wire may emit a bare string (pre-Wave-2), the Wave 2 dict with
 * legacy keys (`replacement_factor_id` / `effective_from`), or the new
 * canonical v1.3 dict (`successor_id` / `deprecated_at`). Use
 * {@link inflateDeprecationStatus} to normalize any wire shape into
 * this interface before reading it; the SDK `ResolvedFactor` builder
 * calls that helper automatically.
 */
export interface DeprecationStatus {
  /** active | deprecated | sunset (legacy: current | scheduled | retired) */
  status?: string | null;
  /** v1.3 canonical: factor id that supersedes this one. */
  successor_id?: string | null;
  /** v1.3 canonical: human-readable reason. */
  reason?: string | null;
  /** v1.3 canonical: ISO-8601 timestamp the deprecation took effect. */
  deprecated_at?: string | null;
  /** Legacy alias for `deprecated_at`. */
  effective_from?: string | null;
  /** Legacy; no canonical equivalent. */
  effective_to?: string | null;
  /** Legacy alias for `successor_id`. */
  replacement_factor_id?: string | null;
  /** Legacy; optional notice URL. */
  notice_url?: string | null;
  [extra: string]: unknown;
}

/**
 * Inflate any wire value into a canonical v1.3 {@link DeprecationStatus}.
 *
 * - `null` / `undefined` -> `null`.
 * - Bare string (e.g. `"active"`, `"deprecated"`) ->
 *   `{ status, successor_id: null, reason: null, deprecated_at: null }`.
 * - Legacy dict with `replacement_factor_id` / `effective_from` -> canonical
 *   fields backfilled from the legacy aliases (only when canonical is
 *   missing).
 * - Canonical dict -> pass through.
 */
export function inflateDeprecationStatus(
  raw: unknown,
): DeprecationStatus | null {
  if (raw === null || raw === undefined) return null;
  if (typeof raw === 'string') {
    return {
      status: raw,
      successor_id: null,
      reason: null,
      deprecated_at: null,
    };
  }
  if (typeof raw === 'object') {
    const src = raw as Record<string, unknown>;
    const out: DeprecationStatus = { ...(src as DeprecationStatus) };
    if (out.successor_id == null && typeof src.replacement_factor_id === 'string') {
      out.successor_id = src.replacement_factor_id;
    }
    if (out.deprecated_at == null && typeof src.effective_from === 'string') {
      out.deprecated_at = src.effective_from;
    }
    if (out.successor_id === undefined) out.successor_id = null;
    if (out.reason === undefined) out.reason = null;
    if (out.deprecated_at === undefined) out.deprecated_at = null;
    return out;
  }
  return { status: String(raw), successor_id: null, reason: null, deprecated_at: null };
}

export interface ResolvedFactor {
  chosen_factor_id?: string | null;
  /** Wave 2: typed ``chosen_factor`` envelope. */
  chosen_factor?: ChosenFactor | null;
  factor_id?: string | null;
  factor_version?: string | null;
  /** Method-pack release — distinct from ``factor_version``. */
  release_version?: string | null;
  method_profile?: string | null;
  method_pack_version?: string | null;

  fallback_rank?: number | null;
  step_label?: string | null;
  why_chosen?: string | null;

  /** Wave 2: nested source descriptor. */
  source?: SourceDescriptor | null;
  quality_score?: QualityScore | null;
  /** Wave 2: composite FQS envelope (``composite_fqs_0_100``). */
  quality?: QualityEnvelope | null;
  /** Wave 2: richer uncertainty envelope. */
  uncertainty?: UncertaintyEnvelope | null;
  gas_breakdown?: GasBreakdown | null;
  co2e_basis?: string | null;

  assumptions?: string[];
  alternates?: Array<Record<string, unknown>>;

  /** Wave 2: structured licensing envelope. */
  licensing?: LicensingEnvelope | null;
  /**
   * May be a plain string (pre-Wave-2) or a structured
   * {@link DeprecationStatus} object (Wave 2+).
   */
  deprecation_status?: string | DeprecationStatus | null;
  deprecation_replacement?: string | null;

  /** Wave 2.5: human-readable narrative explaining the resolution. */
  audit_text?: string | null;
  /** Wave 2.5: true when ``audit_text`` is a draft from an unapproved template. */
  audit_text_draft?: boolean | null;

  /** Wave 2a: canonical top-level signed receipt. */
  signed_receipt?: SignedReceiptEnvelope | null;

  explain?: Record<string, unknown>;
  edition_id?: string | null;

  [extra: string]: unknown;
}

/**
 * Wave 2a signed receipt envelope as it lives on response objects.
 *
 * Separate from the ``SignedReceipt`` interface in ``verify.ts`` so
 * consumers of ``models.ts`` do not need to import the verifier. The
 * shape is identical; re-export here keeps the model surface self-
 * contained.
 */
export interface SignedReceiptEnvelope {
  receipt_id?: string | null;
  signature: string;
  verification_key_hint?: string | null;
  /** Wave 2a canonical — was ``algorithm`` pre-2a. */
  alg: string;
  /** Wave 2a canonical — was ``signed_over`` pre-2a. */
  payload_hash?: string | null;
  signed_at?: string | null;
  key_id?: string | null;
  edition_id?: string | null;
  [extra: string]: unknown;
}

// ---------------------------------------------------------------------------
// Diff / Audit
// ---------------------------------------------------------------------------

export interface FactorDiff {
  factor_id: string;
  left_edition: string;
  right_edition: string;
  status: 'unchanged' | 'changed' | 'added' | 'removed' | 'not_found' | string;
  left_exists?: boolean | null;
  right_exists?: boolean | null;
  changes?: Array<Record<string, unknown>>;
  left_content_hash?: string | null;
  right_content_hash?: string | null;
  [extra: string]: unknown;
}

export interface AuditBundle {
  factor_id: string;
  edition_id: string;
  content_hash?: string | null;
  payload_sha256?: string | null;
  normalized_record?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  license_info?: Record<string, unknown>;
  quality?: Record<string, unknown>;
  verification_chain?: Record<string, unknown>;
  raw_artifact_uri?: string | null;
  parser_log?: string | null;
  qa_errors?: string[];
  reviewer_decision?: string | null;
  [extra: string]: unknown;
}

// ---------------------------------------------------------------------------
// Tenant overrides (Consulting/Platform tier)
// ---------------------------------------------------------------------------

export interface Override {
  factor_id: string;
  tenant_id?: string | null;
  co2e_per_unit?: number | null;
  justification?: string | null;
  effective_from?: string | null;
  effective_to?: string | null;
  metadata?: Record<string, unknown>;
  [extra: string]: unknown;
}

// ---------------------------------------------------------------------------
// Batch jobs
// ---------------------------------------------------------------------------

export type BatchJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | string;

export interface BatchJobHandle {
  job_id: string;
  status: BatchJobStatus;
  total_items?: number | null;
  processed_items?: number | null;
  progress_percent?: number | null;
  results_url?: string | null;
  created_at?: string | null;
  completed_at?: string | null;
  error_message?: string | null;
  results?: Array<Record<string, unknown>> | null;
  [extra: string]: unknown;
}

/** Helper — true when the batch job has reached a terminal state. */
export function isTerminalBatchStatus(status: BatchJobStatus | undefined | null): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}
