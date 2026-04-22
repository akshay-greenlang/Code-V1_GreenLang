/**
 * TypeScript + zod mirror of the Python ResolvedFactor payload.
 *
 * Source of truth: `greenlang/factors/resolution/result.py`
 *                  `greenlang/factors/api_endpoints.py`
 *
 * Rule: every API response is parsed through one of these zod schemas before
 * it reaches a component, so runtime drift between Python and TS cannot cause
 * a silent render bug.
 */
import { z } from "zod";

// ---------- Enumerations ----------

export const FactorStatusSchema = z.enum([
  "certified",
  "preview",
  "connector_only",
  "deprecated",
]);
export type FactorStatus = z.infer<typeof FactorStatusSchema>;

export const LicenseClassSchema = z.enum([
  "public",
  "open",
  "open_cc",
  "commercial",
  "restricted",
  "licensed",
  "proprietary",
  "customer_private",
  "connector_only",
]);
export type LicenseClass = z.infer<typeof LicenseClassSchema>;

export const MethodProfileSchema = z.enum([
  "corporate_scope1",
  "corporate_scope2_location_based",
  "corporate_scope2_market_based",
  "corporate_scope3",
  "electricity",
  "freight",
  "eu_policy",
  "land_removals",
  "product_carbon",
  "finance_proxy",
]);
export type MethodProfile = z.infer<typeof MethodProfileSchema>;

export const StepLabelSchema = z.enum([
  "customer_override",
  "supplier_specific",
  "facility_specific",
  "region_specific",
  "country_or_sector_average",
  "global_average",
  "default_assumption",
]);
export type StepLabel = z.infer<typeof StepLabelSchema>;

export const FallbackRankSchema = z.union([
  z.literal(1),
  z.literal(2),
  z.literal(3),
  z.literal(4),
  z.literal(5),
  z.literal(6),
  z.literal(7),
]);
export type FallbackRank = z.infer<typeof FallbackRankSchema>;

export const DeprecationStatusSchema = z
  .enum(["active", "deprecated", "superseded"])
  .optional();
export type DeprecationStatus = z.infer<typeof DeprecationStatusSchema>;

// ---------- Composite sub-objects ----------

export const GasBreakdownSchema = z.object({
  co2_kg: z.number().default(0),
  ch4_kg: z.number().default(0),
  n2o_kg: z.number().default(0),
  hfcs_kg: z.number().default(0),
  pfcs_kg: z.number().default(0),
  sf6_kg: z.number().default(0),
  nf3_kg: z.number().default(0),
  biogenic_co2_kg: z.number().default(0),
  co2e_total_kg: z.number().default(0),
  gwp_basis: z.string().default("IPCC_AR6_100"),
});
export type GasBreakdown = z.infer<typeof GasBreakdownSchema>;

export const UncertaintyBandSchema = z.object({
  distribution: z.string().default("unknown"),
  ci_95_percent: z.number().nullable().optional(),
  low: z.number().nullable().optional(),
  high: z.number().nullable().optional(),
  note: z.string().nullable().optional(),
});
export type UncertaintyBand = z.infer<typeof UncertaintyBandSchema>;

export const AlternateCandidateSchema = z.object({
  factor_id: z.string(),
  tie_break_score: z.number(),
  why_not_chosen: z.string(),
  source_id: z.string().nullable().optional(),
  vintage: z.number().int().nullable().optional(),
  redistribution_class: z.string().nullable().optional(),
});
export type AlternateCandidate = z.infer<typeof AlternateCandidateSchema>;

// ---------- Composite FQS (5 components from quality/dqs scoring) ----------

export const FqsRatingSchema = z.enum(["A", "B", "C", "D", "E"]);
export type FqsRating = z.infer<typeof FqsRatingSchema>;

export const CompositeFqsSchema = z.object({
  overall: z.number().min(0).max(100),
  rating: FqsRatingSchema,
  temporal_representativeness: z.number().min(0).max(100),
  geographic_representativeness: z.number().min(0).max(100),
  technology_representativeness: z.number().min(0).max(100),
  verification: z.number().min(0).max(100),
  completeness: z.number().min(0).max(100),
  uncertainty_95ci: z.number().nullable().optional(),
});
export type CompositeFqs = z.infer<typeof CompositeFqsSchema>;

// ---------- Signed receipt ----------

export const SignedReceiptSchema = z.object({
  payload_sha256: z.string(),
  content_hash: z.string(),
  algorithm: z.string().default("SHA-256"),
  key_id: z.string().nullable().optional(),
  signed_at: z.string().nullable().optional(),
  signature: z.string().nullable().optional(),
});
export type SignedReceipt = z.infer<typeof SignedReceiptSchema>;

// ---------- Source & method-pack & edition ----------

export const SourceSchema = z.object({
  source_id: z.string(),
  name: z.string(),
  publisher: z.string(),
  current_version: z.string(),
  license_class: LicenseClassSchema,
  factor_count: z.number().int().default(0),
  cadence: z
    .enum(["annual", "biannual", "quarterly", "ad_hoc"])
    .default("ad_hoc"),
  last_updated: z.string(),
  jurisdiction: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
});
export type Source = z.infer<typeof SourceSchema>;

export const SourceDetailSchema = SourceSchema.extend({
  description: z.string().default(""),
  validity_start: z.string(),
  validity_end: z.string().nullable().optional(),
  jurisdiction_coverage: z.array(z.string()).default([]),
  changelog: z
    .array(
      z.object({
        version: z.string(),
        date: z.string(),
        summary: z.string(),
        diff_url: z.string().nullable().optional(),
      })
    )
    .default([]),
});
export type SourceDetail = z.infer<typeof SourceDetailSchema>;

export const MethodPackSchema = z.object({
  profile: MethodProfileSchema,
  name: z.string(),
  purpose: z.string(),
  scope_coverage: z.array(z.enum(["1", "2", "3"])).default([]),
  gwp_basis: z.string(),
  region_hierarchy_depth: z.number().int().default(5),
});
export type MethodPack = z.infer<typeof MethodPackSchema>;

export const MethodPackDetailSchema = MethodPackSchema.extend({
  selection_rules: z
    .array(
      z.object({
        rank: z.number().int(),
        rule: z.string(),
        example: z.string().nullable().optional(),
      })
    )
    .default([]),
  boundary_rules: z.array(z.string()).default([]),
  region_hierarchy: z.array(z.string()).default([]),
  fallback_logic: z
    .array(
      z.object({
        rank: FallbackRankSchema,
        step_label: StepLabelSchema,
        description: z.string(),
      })
    )
    .default([]),
  reporting_labels: z
    .array(z.object({ standard: z.string(), label: z.string() }))
    .default([]),
});
export type MethodPackDetail = z.infer<typeof MethodPackDetailSchema>;

export const EditionSchema = z.object({
  edition_id: z.string(),
  manifest_fingerprint: z.string(),
  released_at: z.string(),
  is_current: z.boolean().default(false),
  release_notes_url: z.string().nullable().optional(),
  summary: z.string().nullable().optional(),
});
export type Edition = z.infer<typeof EditionSchema>;

// ---------- Factor catalog record ----------

export const ProvenanceSummarySchema = z.object({
  source_org: z.string(),
  source_publication: z.string().default(""),
  source_year: z.number().int(),
  source_version: z.string().default(""),
  source_url: z.string().nullable().optional(),
  citation: z.string().nullable().optional(),
  methodology: z.string().default(""),
});
export type ProvenanceSummary = z.infer<typeof ProvenanceSummarySchema>;

export const LicenseSummarySchema = z.object({
  class: LicenseClassSchema,
  redistribution_allowed: z.boolean().default(false),
  commercial_use_allowed: z.boolean().default(false),
  attribution_required: z.boolean().default(true),
});
export type LicenseSummary = z.infer<typeof LicenseSummarySchema>;

export const FactorRecordSchema = z.object({
  factor_id: z.string(),
  factor_version: z.string().default("1.0"),
  factor_family: z.string().default("fuel"),
  fuel_type: z.string().default(""),
  geography: z.string().default(""),
  jurisdiction: z.string().default(""),
  scope: z.enum(["1", "2", "3"]).default("1"),
  boundary: z.string().default(""),
  unit: z.string().default(""),
  valid_from: z.string(),
  valid_to: z.string().nullable().optional(),
  factor_status: FactorStatusSchema.default("certified"),
  co2e_per_unit: z.number().default(0),
  source_id: z.string().nullable().optional(),
  gwp_100yr: z
    .object({
      co2_total_kg: z.number().default(0),
      ch4_kg: z.number().default(0),
      n2o_kg: z.number().default(0),
      hfcs_kg: z.number().default(0),
      pfcs_kg: z.number().default(0),
      sf6_kg: z.number().default(0),
      nf3_kg: z.number().default(0),
      biogenic_co2_kg: z.number().default(0),
      co2e_total: z.number().default(0),
      gwp_basis: z.string().default("IPCC_AR6_100"),
    })
    .default({
      co2_total_kg: 0,
      ch4_kg: 0,
      n2o_kg: 0,
      hfcs_kg: 0,
      pfcs_kg: 0,
      sf6_kg: 0,
      nf3_kg: 0,
      biogenic_co2_kg: 0,
      co2e_total: 0,
      gwp_basis: "IPCC_AR6_100",
    }),
  provenance: ProvenanceSummarySchema,
  license_info: LicenseSummarySchema,
  quality: CompositeFqsSchema,
  content_hash: z.string(),
  sector_tags: z.array(z.string()).default([]),
  activity_tags: z.array(z.string()).default([]),
});
export type FactorRecord = z.infer<typeof FactorRecordSchema>;

// ---------- ResolvedFactor payload (matches ResolvedFactor.model_dump()) ----------

export const ResolvedFactorSchema = z.object({
  chosen_factor_id: z.string(),
  chosen_factor_name: z.string().nullable().optional(),
  source_id: z.string().nullable().optional(),
  source_version: z.string().nullable().optional(),
  factor_version: z.string().nullable().optional(),
  vintage: z.number().int().nullable().optional(),
  method_profile: z.string(),
  formula_type: z.string().nullable().optional(),
  redistribution_class: z.string().nullable().optional(),

  fallback_rank: FallbackRankSchema,
  step_label: StepLabelSchema,
  why_chosen: z.string(),
  alternates: z.array(AlternateCandidateSchema).default([]),
  assumptions: z.array(z.string()).default([]),
  deprecation_status: z
    .enum(["active", "deprecated", "superseded"])
    .nullable()
    .optional(),
  deprecation_replacement: z.string().nullable().optional(),

  quality_score: z.number().nullable().optional(),
  uncertainty: UncertaintyBandSchema.default({
    distribution: "unknown",
  }),
  verification_status: z.string().nullable().optional(),

  gas_breakdown: GasBreakdownSchema.default({
    co2_kg: 0,
    ch4_kg: 0,
    n2o_kg: 0,
    hfcs_kg: 0,
    pfcs_kg: 0,
    sf6_kg: 0,
    nf3_kg: 0,
    biogenic_co2_kg: 0,
    co2e_total_kg: 0,
    gwp_basis: "IPCC_AR6_100",
  }),
  factor_unit_denominator: z.string().nullable().optional(),
  primary_data_flag: z.string().nullable().optional(),

  target_unit: z.string().nullable().optional(),
  converted_co2e_per_unit: z.number().nullable().optional(),
  unit_conversion_factor: z.number().nullable().optional(),
  unit_conversion_path: z.array(z.string()).default([]),
  unit_conversion_note: z.string().nullable().optional(),

  resolved_at: z.string(),
  method_pack_version: z.string().nullable().optional(),
  engine_version: z.string().default("resolution-1.0.0"),
});
export type ResolvedFactor = z.infer<typeof ResolvedFactorSchema>;

// ---------- Explain-endpoint compact form (ResolvedFactor.explain()) ----------

export const ResolvedFactorExplainSchema = z.object({
  chosen: z.object({
    factor_id: z.string(),
    factor_name: z.string().nullable().optional(),
    source: z.string().nullable().optional(),
    source_version: z.string().nullable().optional(),
    factor_version: z.string().nullable().optional(),
    vintage: z.number().int().nullable().optional(),
    method_profile: z.string(),
    redistribution_class: z.string().nullable().optional(),
  }),
  derivation: z.object({
    fallback_rank: FallbackRankSchema,
    step_label: StepLabelSchema,
    why_chosen: z.string(),
    assumptions: z.array(z.string()).default([]),
    deprecation_status: z.string().nullable().optional(),
    deprecation_replacement: z.string().nullable().optional(),
  }),
  quality: z.object({
    score: z.number().nullable().optional(),
    verification_status: z.string().nullable().optional(),
    primary_data_flag: z.string().nullable().optional(),
  }),
  uncertainty: UncertaintyBandSchema,
  emissions: GasBreakdownSchema,
  unit_conversion: z
    .object({
      target_unit: z.string(),
      factor: z.number().nullable().optional(),
      converted_co2e_per_unit: z.number().nullable().optional(),
      path: z.array(z.string()).default([]),
      note: z.string().nullable().optional(),
    })
    .nullable(),
  alternates: z.array(AlternateCandidateSchema).default([]),
  meta: z.object({
    resolved_at: z.string(),
    method_pack_version: z.string().nullable().optional(),
    engine_version: z.string(),
  }),
  _signed_receipt: SignedReceiptSchema.nullable().optional(),
});
export type ResolvedFactorExplain = z.infer<typeof ResolvedFactorExplainSchema>;

// ---------- Merged factor-detail payload ----------

export const FactorDetailPayloadSchema = z.object({
  factor: FactorRecordSchema,
  explain: ResolvedFactorExplainSchema,
  edition_id: z.string(),
  signed_receipt: SignedReceiptSchema,
  deprecation_replacement: z.string().nullable().optional(),
});
export type FactorDetailPayload = z.infer<typeof FactorDetailPayloadSchema>;

// ---------- Search ----------

export const SearchFiltersSchema = z.object({
  query: z.string().default(""),
  family: z.string().nullable().optional(),
  jurisdiction: z.string().nullable().optional(),
  method_profile: MethodProfileSchema.nullable().optional(),
  source_id: z.string().nullable().optional(),
  factor_status: FactorStatusSchema.nullable().optional(),
  license_class: LicenseClassSchema.nullable().optional(),
  dqs_min: z.number().min(0).max(100).nullable().optional(),
  valid_on_date: z.string().nullable().optional(),
  sort_by: z
    .enum(["relevance", "dqs_score", "co2e_total", "source_year", "factor_id"])
    .default("relevance"),
  sort_order: z.enum(["asc", "desc"]).default("desc"),
  offset: z.number().int().min(0).default(0),
  limit: z.number().int().min(1).max(200).default(20),
});
export type SearchFilters = z.infer<typeof SearchFiltersSchema>;

export const SearchResultSchema = z.object({
  items: z.array(FactorRecordSchema),
  total: z.number().int(),
  offset: z.number().int(),
  limit: z.number().int(),
  facets: z
    .object({
      families: z.array(z.object({ value: z.string(), count: z.number() })),
      jurisdictions: z.array(
        z.object({ value: z.string(), count: z.number() })
      ),
      method_profiles: z.array(
        z.object({ value: MethodProfileSchema, count: z.number() })
      ),
      sources: z.array(
        z.object({
          value: z.string(),
          label: z.string(),
          count: z.number(),
        })
      ),
      factor_statuses: z.array(
        z.object({ value: FactorStatusSchema, count: z.number() })
      ),
      license_classes: z.array(
        z.object({ value: LicenseClassSchema, count: z.number() })
      ),
    })
    .nullable()
    .optional(),
  _signed_receipt: SignedReceiptSchema.nullable().optional(),
});
export type SearchResult = z.infer<typeof SearchResultSchema>;

// ---------- Catalog summary (for ThreeLabelDashboard) ----------

export const CatalogSummarySchema = z.object({
  certified_count: z.number().int(),
  preview_count: z.number().int(),
  connector_only_count: z.number().int(),
  total: z.number().int(),
  edition_id: z.string(),
  generated_at: z.string(),
});
export type CatalogSummary = z.infer<typeof CatalogSummarySchema>;
