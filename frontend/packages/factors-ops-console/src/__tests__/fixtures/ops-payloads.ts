/**
 * Realistic payload fixtures mirroring the shape the Python ops API emits.
 * These run through their zod schemas in api.test.ts to catch drift.
 */
export const INGESTION_JOB_PAYLOAD = {
  job_id: "j-9823",
  source_id: "defra_2025",
  source_label: "DEFRA 2025",
  status: "completed",
  started_at: "2026-04-20T14:02:00Z",
  finished_at: "2026-04-20T14:06:12Z",
  duration_seconds: 252,
  row_count: 4421,
  parser_log_url: "/api/v1/ops/ingestion/jobs/j-9823/log",
  triggered_by: "alice@greenlang.io",
};

export const PARSER_LOG_PAYLOAD = [
  { line: 1, level: "INFO", message: "fetching https://…/2025-edition.zip", at: "2026-04-20T14:02:00Z" },
  { line: 2, level: "INFO", message: "unpacked 14 files, 18MB", at: "2026-04-20T14:02:04Z" },
  { line: 3, level: "WARN", message: "row 823: missing boundary, defaulted", at: "2026-04-20T14:02:30Z" },
  { line: 4, level: "ERROR", message: "row 1204: unparseable unit \"m^3/yr/m2\"", at: "2026-04-20T14:02:35Z" },
];

export const MAPPING_SET_PAYLOAD = {
  mapping_set_id: "scope3-purchased-goods-2026",
  name: "Scope 3 purchased goods 2026",
  total_count: 4102,
  unmapped_count: 1204,
  rows: [
    {
      index: 0,
      raw_text: "steel coil UK",
      suggested: [
        {
          factor_id: "DEFRA-MAT-STEEL-GB-2024",
          label: "Steel, GB, 2024",
          confidence: 0.94,
          reason: "sim 0.97, vintage match",
        },
      ],
      accepted: null,
      state: "suggested",
    },
  ],
};

export const VALIDATION_FAILURE_PAYLOAD = {
  id: "q-812",
  module: "validators",
  severity: "high",
  factor_id: "DEFRA-NG-GB-2025",
  message: "uncertainty > 10% but no note",
  detected_at: "2026-04-20T16:00:00Z",
  remediation_hint: "Add uncertainty justification per QA criteria v4.",
};

export const FACTOR_DIFF_PAYLOAD = {
  factor_id: "DEFRA-ELEC-GB-2025-001",
  left_edition: "2025.Q1",
  right_edition: "2025.Q2",
  status: "changed",
  changes: [
    { field: "co2e_total", type: "changed", old_value: 2.0142, new_value: 2.0796 },
    { field: "provenance.version", type: "changed", old_value: "1.1", new_value: "1.2" },
  ],
  left_content_hash: "abc123def456",
  right_content_hash: "def456abc789",
};

export const TENANT_OVERLAY_PAYLOAD = {
  tenant_id: "acme-corp",
  factor_id: "DEFRA-ELEC-GB-2025-001",
  override_kind: "value",
  co2e_total: 0.211,
  replacement_factor_id: null,
  reason: "Primary supplier data from LCA report",
  effective_from: "2025-01-01T00:00:00Z",
  effective_to: null,
  created_by: "admin@greenlang.io",
  created_at: "2026-04-12T10:00:00Z",
};

export const IMPACT_SIM_RESULT_PAYLOAD = {
  simulation_id: "sim-001",
  edition_id: "2025.Q2",
  factor_id: "DEFRA-RF-GB-2024",
  simulation_mode: "value_override",
  tenant_scope: null,
  summary: {
    affected_computations: 12404,
    affected_tenants: 34,
    avg_delta_pct: 2.1,
    max_delta_pct: 18,
  },
  tenants: ["acme-corp", "globex"],
  suggested_rollback_plan: "Revert to v1.1 in edition 2025.Q2",
};

export const WATCH_EVENT_PAYLOAD = {
  detection_id: "w-001",
  source_id: "defra_2025",
  source_label: "DEFRA 2025",
  detected_at: "2026-04-18T09:00:00Z",
  signal: "doc_hash_changed",
  doc_diff_url: "/api/v1/ops/source-watch/w-001/diff",
  classified: false,
  classification: null,
};

export const ENTITLEMENT_PAYLOAD = {
  tenant_id: "acme-corp",
  tier: "pro",
  packs: ["corporate", "electricity", "freight"],
  rate_limit_rpm: 10000,
  preview_access: true,
  connector_access: false,
};

export const EDITION_DETAIL_PAYLOAD = {
  edition_id: "2025.Q2",
  label: "2025.Q2",
  status: "draft",
  slices: [
    { order: 1, name: "Electricity", status: "promoted", by: "alice", at: "2026-04-10T12:00:00Z" },
    { order: 2, name: "Combustion", status: "promoted", by: "bob", at: "2026-04-12T12:00:00Z" },
    { order: 3, name: "Freight", status: "in_review", by: null, at: null },
    { order: 4, name: "Material/CBAM", status: "pending", by: null, at: null },
    { order: 5, name: "Land/Removals", status: "pending", by: null, at: null },
    { order: 6, name: "Product Carbon", status: "pending", by: null, at: null },
    { order: 7, name: "Finance Proxy", status: "pending", by: null, at: null },
  ],
  recent_events: [],
};

export const EDITION_PROMOTION_EVENT_PAYLOAD = {
  edition_id: "2025.Q2",
  slice_name: "Electricity",
  event: "promote",
  actor: "alice@greenlang.io",
  reason: "Release gate passed for electricity slice",
  at: "2026-04-10T12:00:00Z",
};
