import { describe, expect, it } from "vitest";
import {
  EditionDetailSchema,
  EditionPromotionEventSchema,
  EntitlementSchema,
  FactorDiffSchema,
  ImpactSimulationResultSchema,
  IngestionJobSchema,
  MappingSetSchema,
  ParserLogEntrySchema,
  TenantOverlayEntrySchema,
  ValidationFailureSchema,
  WatchEventSchema,
  AuditEnvelopeSchema,
  RoleSchema,
} from "@/types/ops";
import {
  EDITION_DETAIL_PAYLOAD,
  EDITION_PROMOTION_EVENT_PAYLOAD,
  ENTITLEMENT_PAYLOAD,
  FACTOR_DIFF_PAYLOAD,
  IMPACT_SIM_RESULT_PAYLOAD,
  INGESTION_JOB_PAYLOAD,
  MAPPING_SET_PAYLOAD,
  PARSER_LOG_PAYLOAD,
  TENANT_OVERLAY_PAYLOAD,
  VALIDATION_FAILURE_PAYLOAD,
  WATCH_EVENT_PAYLOAD,
} from "./fixtures/ops-payloads";

describe("zod round-trip: ops API payloads", () => {
  it("parses an IngestionJob", () => {
    const parsed = IngestionJobSchema.parse(INGESTION_JOB_PAYLOAD);
    expect(parsed.job_id).toBe("j-9823");
    expect(parsed.status).toBe("completed");
    expect(parsed.row_count).toBe(4421);
  });

  it("parses an array of ParserLogEntry", () => {
    const parsed = PARSER_LOG_PAYLOAD.map((e) => ParserLogEntrySchema.parse(e));
    expect(parsed).toHaveLength(4);
    expect(parsed[3]?.level).toBe("ERROR");
  });

  it("parses a MappingSet with one suggested row", () => {
    const parsed = MappingSetSchema.parse(MAPPING_SET_PAYLOAD);
    expect(parsed.rows[0]?.state).toBe("suggested");
    expect(parsed.rows[0]?.suggested[0]?.confidence).toBeGreaterThan(0.9);
  });

  it("parses a ValidationFailure", () => {
    const parsed = ValidationFailureSchema.parse(VALIDATION_FAILURE_PAYLOAD);
    expect(parsed.severity).toBe("high");
    expect(parsed.module).toBe("validators");
  });

  it("parses a FactorDiff with two field-level changes", () => {
    const parsed = FactorDiffSchema.parse(FACTOR_DIFF_PAYLOAD);
    expect(parsed.changes).toHaveLength(2);
    expect(parsed.status).toBe("changed");
  });

  it("parses a TenantOverlayEntry", () => {
    const parsed = TenantOverlayEntrySchema.parse(TENANT_OVERLAY_PAYLOAD);
    expect(parsed.override_kind).toBe("value");
    expect(parsed.co2e_total).toBeCloseTo(0.211);
  });

  it("parses an ImpactSimulationResult", () => {
    const parsed = ImpactSimulationResultSchema.parse(IMPACT_SIM_RESULT_PAYLOAD);
    expect(parsed.summary.affected_computations).toBe(12404);
    expect(parsed.summary.affected_tenants).toBe(34);
  });

  it("parses a WatchEvent", () => {
    const parsed = WatchEventSchema.parse(WATCH_EVENT_PAYLOAD);
    expect(parsed.signal).toBe("doc_hash_changed");
    expect(parsed.classified).toBe(false);
  });

  it("parses an Entitlement", () => {
    const parsed = EntitlementSchema.parse(ENTITLEMENT_PAYLOAD);
    expect(parsed.tier).toBe("pro");
    expect(parsed.packs).toContain("electricity");
  });

  it("parses an EditionDetail with 7 slices", () => {
    const parsed = EditionDetailSchema.parse(EDITION_DETAIL_PAYLOAD);
    expect(parsed.slices).toHaveLength(7);
    expect(parsed.slices.filter((s) => s.status === "promoted")).toHaveLength(2);
  });

  it("parses an EditionPromotionEvent", () => {
    const parsed = EditionPromotionEventSchema.parse(EDITION_PROMOTION_EVENT_PAYLOAD);
    expect(parsed.event).toBe("promote");
    expect(parsed.slice_name).toBe("Electricity");
  });
});

describe("zod: enums and guards", () => {
  it("rejects an unknown role", () => {
    expect(RoleSchema.safeParse("admin").success).toBe(true);
    expect(RoleSchema.safeParse("god_mode").success).toBe(false);
  });

  it("AuditEnvelope rejects short reasons (< 10 chars)", () => {
    expect(
      AuditEnvelopeSchema.safeParse({ actor: "a", reason: "ok", session: "s" }).success
    ).toBe(false);
    expect(
      AuditEnvelopeSchema.safeParse({
        actor: "a",
        reason: "this is long enough to pass",
        session: "s",
      }).success
    ).toBe(true);
  });

  it("AuditEnvelope rejects reasons > 500 chars", () => {
    expect(
      AuditEnvelopeSchema.safeParse({
        actor: "a",
        reason: "x".repeat(501),
        session: "s",
      }).success
    ).toBe(false);
  });
});
