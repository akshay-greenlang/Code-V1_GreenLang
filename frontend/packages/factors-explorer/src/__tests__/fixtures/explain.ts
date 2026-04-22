/**
 * Test helper: convert a ResolvedFactor dump (matches result.py) into the
 * compact "explain()" envelope the API ships at /factors/{id}?explain=1 and
 * /factors/{id}/explain.
 */
import type {
  ResolvedFactor,
  ResolvedFactorExplain,
} from "@/types/factors";
import fixture from "./resolved_factor.json";

export const RESOLVED_FACTOR_FIXTURE = fixture as unknown as ResolvedFactor;

export function toExplainEnvelope(
  rf: ResolvedFactor
): ResolvedFactorExplain {
  return {
    chosen: {
      factor_id: rf.chosen_factor_id,
      factor_name: rf.chosen_factor_name ?? null,
      source: rf.source_id ?? null,
      source_version: rf.source_version ?? null,
      factor_version: rf.factor_version ?? null,
      vintage: rf.vintage ?? null,
      method_profile: rf.method_profile,
      redistribution_class: rf.redistribution_class ?? null,
    },
    derivation: {
      fallback_rank: rf.fallback_rank,
      step_label: rf.step_label,
      why_chosen: rf.why_chosen,
      assumptions: rf.assumptions,
      deprecation_status: rf.deprecation_status ?? null,
      deprecation_replacement: rf.deprecation_replacement ?? null,
    },
    quality: {
      score: rf.quality_score ?? null,
      verification_status: rf.verification_status ?? null,
      primary_data_flag: rf.primary_data_flag ?? null,
    },
    uncertainty: rf.uncertainty,
    emissions: rf.gas_breakdown,
    unit_conversion:
      rf.target_unit === null || rf.target_unit === undefined
        ? null
        : {
            target_unit: rf.target_unit,
            factor: rf.unit_conversion_factor ?? null,
            converted_co2e_per_unit: rf.converted_co2e_per_unit ?? null,
            path: rf.unit_conversion_path,
            note: rf.unit_conversion_note ?? null,
          },
    alternates: rf.alternates,
    meta: {
      resolved_at: rf.resolved_at,
      method_pack_version: rf.method_pack_version ?? null,
      engine_version: rf.engine_version,
    },
    _signed_receipt: {
      algorithm: "SHA-256",
      payload_sha256:
        "8f3d1c9e4b2a7d5f6c8a3e1b9d0f2a4c6e8b1d3f5a7c9e2b4d6f8a0c2e4b6d8f",
      content_hash:
        "a1b2c3d4e5f67890abcdef1234567890abcdef1234567890abcdef1234567890",
      key_id: "gl-factors-2026-q2",
      signed_at: "2026-04-22T10:15:31.000000+00:00",
    },
  };
}
