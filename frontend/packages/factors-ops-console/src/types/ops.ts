/**
 * Zod schemas + TS types for every Operator Console payload.
 *
 * Mirrors the Python ops API under /api/v1/ops/*. Every request and response
 * passes through a zod parse at the fetch boundary so drift between backend
 * and frontend fails loudly.
 *
 * Keep this file in sync with:
 *   - greenlang/factors/ingestion/fetchers.py
 *   - greenlang/factors/mapping/*
 *   - greenlang/factors/quality/*
 *   - greenlang/factors/watch/*
 *   - greenlang/factors/tenant_overlay.py
 *   - greenlang/factors/entitlements.py
 */
import { z } from "zod";

// ---------- Primitives ----------

export const RoleSchema = z.enum([
  "admin",
  "methodology_lead",
  "data_curator",
  "reviewer",
  "release_manager",
  "legal",
  "support",
  "viewer",
]);
export type Role = z.infer<typeof RoleSchema>;

export const SeveritySchema = z.enum(["low", "med", "high", "critical"]);
export type Severity = z.infer<typeof SeveritySchema>;

export const JobStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
  "promoted",
  "rejected",
]);
export type JobStatus = z.infer<typeof JobStatusSchema>;

export const LogLevelSchema = z.enum(["DEBUG", "INFO", "WARN", "ERROR"]);
export type LogLevel = z.infer<typeof LogLevelSchema>;

// ---------- Ingestion ----------

export const IngestionJobSchema = z.object({
  job_id: z.string(),
  source_id: z.string(),
  source_label: z.string(),
  status: JobStatusSchema,
  started_at: z.string(),
  finished_at: z.string().nullable().optional(),
  duration_seconds: z.number().nullable().optional(),
  row_count: z.number().int().nonnegative().default(0),
  parser_log_url: z.string(),
  triggered_by: z.string(),
});
export type IngestionJob = z.infer<typeof IngestionJobSchema>;

export const ParserLogEntrySchema = z.object({
  line: z.number().int().nonnegative(),
  level: LogLevelSchema,
  message: z.string(),
  at: z.string(),
});
export type ParserLogEntry = z.infer<typeof ParserLogEntrySchema>;

// ---------- Mapping ----------

export const MappingSuggestionSchema = z.object({
  factor_id: z.string(),
  label: z.string(),
  confidence: z.number().min(0).max(1),
  reason: z.string(),
});
export type MappingSuggestion = z.infer<typeof MappingSuggestionSchema>;

export const MappingRowSchema = z.object({
  index: z.number().int().nonnegative(),
  raw_text: z.string(),
  suggested: z.array(MappingSuggestionSchema),
  accepted: z.string().nullable().optional(),
  state: z.enum(["unmapped", "suggested", "accepted", "rejected"]),
});
export type MappingRow = z.infer<typeof MappingRowSchema>;

export const MappingSetSchema = z.object({
  mapping_set_id: z.string(),
  name: z.string(),
  rows: z.array(MappingRowSchema),
  unmapped_count: z.number().int().nonnegative(),
  total_count: z.number().int().nonnegative(),
});
export type MappingSet = z.infer<typeof MappingSetSchema>;

// ---------- QA ----------

export const ValidationFailureSchema = z.object({
  id: z.string(),
  module: z.enum(["validators", "dedup_engine", "cross_source", "license_scanner"]),
  severity: SeveritySchema,
  factor_id: z.string().nullable().optional(),
  message: z.string(),
  detected_at: z.string(),
  remediation_hint: z.string().nullable().optional(),
});
export type ValidationFailure = z.infer<typeof ValidationFailureSchema>;

// ---------- Diff ----------

export const DiffChangeSchema = z.object({
  field: z.string(),
  type: z.enum(["added", "removed", "changed"]),
  old_value: z.unknown().optional(),
  new_value: z.unknown().optional(),
});
export type DiffChange = z.infer<typeof DiffChangeSchema>;

export const FactorDiffSchema = z.object({
  factor_id: z.string(),
  left_edition: z.string(),
  right_edition: z.string(),
  status: z.enum(["added", "removed", "changed", "unchanged", "not_found"]),
  changes: z.array(DiffChangeSchema).default([]),
  left_content_hash: z.string().nullable().optional(),
  right_content_hash: z.string().nullable().optional(),
});
export type FactorDiff = z.infer<typeof FactorDiffSchema>;

// ---------- Approvals ----------

export const ApprovalStepSchema = z.object({
  approver: z
    .object({ sub: z.string(), display_name: z.string() })
    .nullable(),
  status: z.enum(["pending", "approved", "rejected", "changes_requested"]),
  at: z.string().nullable().optional(),
  comment: z.string().nullable().optional(),
});
export type ApprovalStep = z.infer<typeof ApprovalStepSchema>;

export const ReviewItemSchema = z.object({
  review_id: z.string(),
  kind: z.enum(["ingestion", "mapping", "qa-remediation", "edition-slice", "override"]),
  author: z.object({ sub: z.string(), display_name: z.string() }),
  steps: z.array(ApprovalStepSchema),
  age_hours: z.number().nonnegative(),
  context: z.record(z.unknown()).optional(),
});
export type ReviewItem = z.infer<typeof ReviewItemSchema>;

// ---------- Overrides ----------

export const TenantOverlayEntrySchema = z.object({
  tenant_id: z.string(),
  factor_id: z.string(),
  override_kind: z.enum(["value", "replacement", "deprecation"]),
  co2e_total: z.number().nullable().optional(),
  replacement_factor_id: z.string().nullable().optional(),
  reason: z.string().min(10),
  effective_from: z.string(),
  effective_to: z.string().nullable().optional(),
  created_by: z.string(),
  created_at: z.string(),
});
export type TenantOverlayEntry = z.infer<typeof TenantOverlayEntrySchema>;

// ---------- Impact simulation ----------

export const ImpactSimulationRequestSchema = z.object({
  factor_id: z.string(),
  mode: z.enum(["listing_only", "value_override", "deprecation"]),
  hypothetical_value: z.number().nullable().optional(),
  tenant_scope: z.array(z.string()).nullable().optional(),
});
export type ImpactSimulationRequest = z.infer<typeof ImpactSimulationRequestSchema>;

export const ImpactSimulationResultSchema = z.object({
  simulation_id: z.string(),
  edition_id: z.string(),
  factor_id: z.string(),
  simulation_mode: z.enum(["listing_only", "value_override", "deprecation"]),
  tenant_scope: z.array(z.string()).nullable(),
  summary: z.object({
    affected_computations: z.number().int().nonnegative(),
    affected_tenants: z.number().int().nonnegative(),
    avg_delta_pct: z.number(),
    max_delta_pct: z.number(),
  }),
  tenants: z.array(z.string()),
  suggested_rollback_plan: z.string().nullable().optional(),
});
export type ImpactSimulationResult = z.infer<typeof ImpactSimulationResultSchema>;

// ---------- Watch ----------

export const WatchEventSchema = z.object({
  detection_id: z.string(),
  source_id: z.string(),
  source_label: z.string(),
  detected_at: z.string(),
  signal: z.enum(["doc_hash_changed", "new_version", "checksum_drift", "license_change"]),
  doc_diff_url: z.string().nullable().optional(),
  classified: z.boolean().default(false),
  classification: z.enum(["major", "minor", "patch", "noop"]).nullable().optional(),
});
export type WatchEvent = z.infer<typeof WatchEventSchema>;

// ---------- Entitlements ----------

export const TierSchema = z.enum(["free", "starter", "pro", "enterprise"]);
export type Tier = z.infer<typeof TierSchema>;

export const PackAssignmentSchema = z.enum([
  "corporate",
  "electricity",
  "freight",
  "eu_policy",
  "land_removals",
  "product_carbon",
  "finance_proxy",
]);
export type PackAssignment = z.infer<typeof PackAssignmentSchema>;

export const EntitlementSchema = z.object({
  tenant_id: z.string(),
  tier: TierSchema,
  packs: z.array(PackAssignmentSchema),
  rate_limit_rpm: z.number().int().positive(),
  preview_access: z.boolean(),
  connector_access: z.boolean(),
});
export type Entitlement = z.infer<typeof EntitlementSchema>;

// ---------- Editions ----------

export const EditionSliceSchema = z.object({
  order: z.number().int().min(1).max(7),
  name: z.enum([
    "Electricity",
    "Combustion",
    "Freight",
    "Material/CBAM",
    "Land/Removals",
    "Product Carbon",
    "Finance Proxy",
  ]),
  status: z.enum(["pending", "in_review", "promoted", "rolled_back"]),
  by: z.string().nullable().optional(),
  at: z.string().nullable().optional(),
});
export type EditionSlice = z.infer<typeof EditionSliceSchema>;

export const EditionPromotionEventSchema = z.object({
  edition_id: z.string(),
  slice_name: z.string(),
  event: z.enum(["promote", "rollback"]),
  actor: z.string(),
  reason: z.string(),
  at: z.string(),
});
export type EditionPromotionEvent = z.infer<typeof EditionPromotionEventSchema>;

export const EditionDetailSchema = z.object({
  edition_id: z.string(),
  label: z.string(),
  status: z.enum(["draft", "published", "deprecated"]),
  slices: z.array(EditionSliceSchema),
  recent_events: z.array(EditionPromotionEventSchema).default([]),
});
export type EditionDetail = z.infer<typeof EditionDetailSchema>;

// ---------- Audit ----------

export const AuditEnvelopeSchema = z.object({
  actor: z.string(),
  reason: z.string().min(10).max(500),
  session: z.string(),
});
export type AuditEnvelope = z.infer<typeof AuditEnvelopeSchema>;
