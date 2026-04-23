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

/** Richer Wave 2 uncertainty envelope (superset of {@link Uncertainty}). */
export interface UncertaintyEnvelope {
  ci_95?: number | null;
  ci_lower?: number | null;
  ci_upper?: number | null;
  distribution?: string | null;
  std_dev?: number | null;
  sample_size?: number | null;
  pedigree_matrix?: Record<string, unknown>;
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

/** Structured Wave 2 deprecation status (pre-Wave-2 was a bare string). */
export interface DeprecationStatus {
  status?: string | null;
  effective_from?: string | null;
  effective_to?: string | null;
  replacement_factor_id?: string | null;
  reason?: string | null;
  notice_url?: string | null;
  [extra: string]: unknown;
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
