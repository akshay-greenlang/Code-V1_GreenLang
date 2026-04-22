import { describe, it, expect } from "vitest";
import {
  AlternateCandidateSchema,
  FallbackRankSchema,
  GasBreakdownSchema,
  ResolvedFactorExplainSchema,
  ResolvedFactorSchema,
  SignedReceiptSchema,
  StepLabelSchema,
  UncertaintyBandSchema,
} from "@/types/factors";
import { extractReceipt } from "@/lib/api";
import { RESOLVED_FACTOR_FIXTURE, toExplainEnvelope } from "./fixtures/explain";

describe("zod ResolvedFactor schema", () => {
  it("accepts a realistic ResolvedFactor payload end-to-end", () => {
    const parsed = ResolvedFactorSchema.parse(RESOLVED_FACTOR_FIXTURE);
    expect(parsed.chosen_factor_id).toBe("DEFRA-NG-GB-2024-001");
    expect(parsed.method_profile).toBe("corporate_scope1");
    expect(parsed.fallback_rank).toBe(5);
    expect(parsed.step_label).toBe("country_or_sector_average");
    expect(parsed.alternates).toHaveLength(2);
  });

  it("enforces fallback_rank is one of 1..7", () => {
    expect(FallbackRankSchema.safeParse(5).success).toBe(true);
    expect(FallbackRankSchema.safeParse(8).success).toBe(false);
    expect(FallbackRankSchema.safeParse(0).success).toBe(false);
  });

  it("enforces step_label is one of the 7 known labels", () => {
    expect(StepLabelSchema.safeParse("customer_override").success).toBe(true);
    expect(StepLabelSchema.safeParse("supplier_specific").success).toBe(true);
    expect(StepLabelSchema.safeParse("default_assumption").success).toBe(true);
    expect(StepLabelSchema.safeParse("totally_made_up").success).toBe(false);
  });

  it("validates gas_breakdown and sums the co2e total", () => {
    const gb = GasBreakdownSchema.parse(RESOLVED_FACTOR_FIXTURE.gas_breakdown);
    expect(gb.co2_kg).toBeCloseTo(1.9876);
    expect(gb.ch4_kg).toBeCloseTo(0.0082);
    expect(gb.co2e_total_kg).toBeCloseTo(2.0296);
    expect(gb.gwp_basis).toBe("IPCC_AR6_100");
  });

  it("validates the uncertainty band", () => {
    const u = UncertaintyBandSchema.parse(RESOLVED_FACTOR_FIXTURE.uncertainty);
    expect(u.distribution).toBe("normal");
    expect(u.ci_95_percent).toBeCloseTo(0.015);
  });

  it("validates an alternate candidate", () => {
    const a = AlternateCandidateSchema.parse(
      RESOLVED_FACTOR_FIXTURE.alternates[0]
    );
    expect(a.factor_id).toBe("IEA-GLOBAL-NG-2023-001");
    expect(a.tie_break_score).toBe(2.1);
  });
});

describe("ResolvedFactorExplain schema", () => {
  it("accepts the compact explain() envelope built from our fixture", () => {
    const envelope = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    const parsed = ResolvedFactorExplainSchema.parse(envelope);
    expect(parsed.chosen.factor_id).toBe("DEFRA-NG-GB-2024-001");
    expect(parsed.derivation.fallback_rank).toBe(5);
    expect(parsed.derivation.why_chosen).toMatch(/DEFRA 2024/);
    expect(parsed.meta.engine_version).toBe("resolution-1.0.0");
    expect(parsed.unit_conversion).toBeNull();
  });

  it("accepts a present unit_conversion trace", () => {
    const envelope = toExplainEnvelope({
      ...RESOLVED_FACTOR_FIXTURE,
      target_unit: "therm",
      unit_conversion_factor: 29.3071,
      converted_co2e_per_unit: 59.479,
      unit_conversion_path: ["m3", "MJ", "therm"],
    });
    const parsed = ResolvedFactorExplainSchema.parse(envelope);
    expect(parsed.unit_conversion?.target_unit).toBe("therm");
    expect(parsed.unit_conversion?.path).toEqual(["m3", "MJ", "therm"]);
  });
});

describe("extractReceipt helper", () => {
  it("pulls _signed_receipt from an envelope", () => {
    const envelope = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    const receipt = extractReceipt(envelope);
    expect(receipt).not.toBeNull();
    expect(receipt?.algorithm).toBe("SHA-256");
    expect(receipt?.content_hash).toHaveLength(64);
    // round-trips through its own schema
    expect(SignedReceiptSchema.safeParse(receipt).success).toBe(true);
  });

  it("returns null when no receipt present", () => {
    expect(extractReceipt({ foo: "bar" })).toBeNull();
    expect(extractReceipt(null)).toBeNull();
    expect(extractReceipt(undefined)).toBeNull();
  });
});
