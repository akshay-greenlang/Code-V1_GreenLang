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

export interface ResolvedFactor {
  chosen_factor_id?: string | null;
  factor_id?: string | null;
  factor_version?: string | null;
  method_profile?: string | null;
  method_pack_version?: string | null;

  fallback_rank?: number | null;
  step_label?: string | null;
  why_chosen?: string | null;

  quality_score?: QualityScore | null;
  uncertainty?: Uncertainty | null;
  gas_breakdown?: GasBreakdown | null;
  co2e_basis?: string | null;

  assumptions?: string[];
  alternates?: Array<Record<string, unknown>>;

  deprecation_status?: string | null;
  deprecation_replacement?: string | null;

  explain?: Record<string, unknown>;
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
