/**
 * factorsClient unit tests — exercise the typed API wrapper without a
 * DOM. Covers:
 *   - successful requests attach the bearer token + edition pin
 *   - 401/402/403/429/404 errors are normalized to FactorsApiError
 *   - the licensing_gap upgrade path includes upgradeUrl
 *   - composeImpactRequest validates the three input modes
 *   - proposeQueueItemFromSimulation builds a well-formed evidence body
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  FactorsApiError,
  approve,
  composeImpactRequest,
  getCoverage,
  proposeQueueItem,
  proposeQueueItemFromSimulation,
  simulateImpact,
  type ImpactReport,
} from "./factorsClient";
import { setPinnedEdition } from "./editionStore";

// Minimal localStorage shim — Vitest in default mode runs in Node without
// a DOM, so window.localStorage doesn't exist. We attach one so the
// client's auth + edition reads don't blow up.
function attachLocalStorage() {
  const store = new Map<string, string>();
  const ls = {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => {
      store.set(k, v);
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    clear: () => store.clear(),
    key: (i: number) => Array.from(store.keys())[i] ?? null,
    get length() {
      return store.size;
    },
  };
  // @ts-expect-error -- attach to global for the client to find.
  globalThis.window = { localStorage: ls };
  return ls;
}

function jsonResponse(body: unknown, init: { status?: number; headers?: Record<string, string> } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
}

describe("factorsClient", () => {
  let ls: ReturnType<typeof attachLocalStorage>;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    ls = attachLocalStorage();
    fetchMock = vi.fn();
    // @ts-expect-error -- override the global fetch
    globalThis.fetch = fetchMock;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // @ts-expect-error
    delete globalThis.window;
    // @ts-expect-error
    delete globalThis.fetch;
  });

  it("attaches Authorization and X-GL-Edition headers", async () => {
    ls.setItem("gl.auth.token", "abc.def.ghi");
    setPinnedEdition("v1.2.3");
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        edition_id: "v1.2.3",
        totals: { certified: 10, preview: 2, connector_only: 1, all: 13 },
        by_family: [],
        generated_at: "2026-04-22T00:00:00Z",
      }),
    );

    const r = await getCoverage();
    expect(r.edition_id).toBe("v1.2.3");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/coverage");
    const headers = init.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer abc.def.ghi");
    expect(headers.get("X-GL-Edition")).toBe("v1.2.3");

    setPinnedEdition(null);
  });

  it("normalizes 401 to FactorsApiError with code 'unauthorized'", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ message: "Token expired" }, { status: 401 }),
    );
    await expect(getCoverage()).rejects.toMatchObject({
      name: "FactorsApiError",
      status: 401,
      code: "unauthorized",
      userMessage: "Token expired",
    });
  });

  it("normalizes 402 / licensing_gap with friendly upgrade URL", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        {
          message: "Upgrade required for connector-only factors",
          code: "licensing_gap",
          upgrade_url: "/pricing?cta=cbam-connector",
        },
        { status: 402 },
      ),
    );
    try {
      await getCoverage();
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(FactorsApiError);
      const err = e as FactorsApiError;
      expect(err.code).toBe("licensing_gap");
      expect(err.upgradeUrl).toBe("/pricing?cta=cbam-connector");
    }
  });

  it("normalizes 403 with code=licensing_gap as licensing_gap, plain 403 as forbidden", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ code: "licensing_gap", message: "ng" }, { status: 403 }),
    );
    await expect(getCoverage()).rejects.toMatchObject({ code: "licensing_gap" });

    fetchMock.mockResolvedValueOnce(
      jsonResponse({ message: "no scope" }, { status: 403 }),
    );
    await expect(getCoverage()).rejects.toMatchObject({ code: "forbidden" });
  });

  it("normalizes 429 to rate_limited with friendly message", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, { status: 429 }));
    await expect(getCoverage()).rejects.toMatchObject({ code: "rate_limited" });
  });

  it("normalizes network errors to network_error code", async () => {
    fetchMock.mockRejectedValueOnce(new Error("ECONNREFUSED"));
    try {
      await getCoverage();
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(FactorsApiError);
      expect((e as FactorsApiError).code).toBe("network_error");
    }
  });

  it("approve POSTs to the right endpoint with a JSON body", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true, new_status: "certified" }));
    const r = await approve("rev_123", { note: "looks good" });
    expect(r).toEqual({ ok: true, new_status: "certified" });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/admin/queue/rev_123/approve");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ note: "looks good" });
  });
});

describe("composeImpactRequest", () => {
  it("requires from_factor_id in single mode", () => {
    expect(() => composeImpactRequest({ toFactorId: "x" })).toThrow(/from_factor_id/);
  });

  it("requires either to_factor_id or proposed payload", () => {
    expect(() => composeImpactRequest({ fromFactorId: "x" })).toThrow(/to_factor_id/);
  });

  it("rejects supplying both to_factor_id and payload", () => {
    expect(() =>
      composeImpactRequest({
        fromFactorId: "x",
        toFactorId: "y",
        toFactorPayloadJson: '{"k":1}',
      }),
    ).toThrow(/not both/);
  });

  it("parses a valid proposed payload", () => {
    const r = composeImpactRequest({
      fromFactorId: "x",
      toFactorPayloadJson: '{"co2e_per_unit":0.42}',
    });
    expect(r.from_factor_id).toBe("x");
    expect(r.to_factor_payload).toEqual({ co2e_per_unit: 0.42 });
    expect(r.to_factor_id).toBeUndefined();
  });

  it("produces a single-factor pair request", () => {
    const r = composeImpactRequest({ fromFactorId: "a", toFactorId: "b" });
    expect(r).toEqual({ from_factor_id: "a", to_factor_id: "b" });
  });

  it("supports bulk-replace mode with mixed delimiters", () => {
    const r = composeImpactRequest({ bulkIds: "a,b\n c\nd" });
    expect(r.replaced_factor_ids).toEqual(["a", "b", "c", "d"]);
  });

  it("rejects malformed JSON in payload", () => {
    expect(() =>
      composeImpactRequest({ fromFactorId: "x", toFactorPayloadJson: "{not-json}" }),
    ).toThrow(/JSON/);
  });

  it("rejects array payloads", () => {
    expect(() =>
      composeImpactRequest({ fromFactorId: "x", toFactorPayloadJson: "[1,2,3]" }),
    ).toThrow(/object/);
  });
});

describe("proposeQueueItemFromSimulation", () => {
  it("attaches simulation summary as evidence", () => {
    const report: ImpactReport = {
      simulated_at: "2026-04-22T00:00:00Z",
      affected_factor_ids: ["EF:UK:road_freight_40t:2025:v1"],
      tenants: ["t1", "t2", "t3"],
      computation_count: 12,
      inventory_count: 4,
      customer_count: 3,
      summary: { mean_pct_delta: 1.2, max_pct_delta: 5.6, min_pct_delta: -0.8 },
      computations: [],
    };
    const payload = proposeQueueItemFromSimulation({
      fromFactorId: "EF:UK:road_freight_40t:2025:v1",
      toFactorId: "EF:UK:road_freight_40t:2026:v1",
      rationale: "DEFRA 2026 update",
      report,
    });
    expect(payload.factor_id).toBe("EF:UK:road_freight_40t:2026:v1");
    expect(payload.proposed_status).toBe("preview");
    expect(payload.rationale).toBe("DEFRA 2026 update");
    const ev = payload.evidence as Record<string, unknown>;
    const sim = ev.simulation as Record<string, unknown>;
    expect(sim.computation_count).toBe(12);
    expect(sim.tenant_count).toBe(3);
    expect(sim.max_pct_delta).toBe(5.6);
    expect(ev.from_factor_id).toBe("EF:UK:road_freight_40t:2025:v1");
  });

  it("falls back to from_factor_id when no to_factor_id is supplied", () => {
    const report: ImpactReport = {
      simulated_at: "2026-04-22T00:00:00Z",
      affected_factor_ids: [],
      tenants: [],
      computation_count: 0,
      summary: {},
      computations: [],
    };
    const payload = proposeQueueItemFromSimulation({
      fromFactorId: "from",
      rationale: "r",
      report,
    });
    expect(payload.factor_id).toBe("from");
  });
});

describe("simulator end-to-end (propose -> simulate -> approve)", () => {
  // Reproduces the launch-gate test journey at the client layer:
  //   1. operator composes a request
  //   2. simulateImpact returns a report
  //   3. operator proposes a queue item with simulation as evidence
  //   4. methodology lead approves the queue item
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    attachLocalStorage();
    fetchMock = vi.fn();
    // @ts-expect-error
    globalThis.fetch = fetchMock;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    // @ts-expect-error
    delete globalThis.window;
    // @ts-expect-error
    delete globalThis.fetch;
  });

  it("threads through simulate -> proposeQueueItem -> approve", async () => {
    // 1) compose
    const req = composeImpactRequest({ fromFactorId: "from-id", toFactorId: "to-id" });
    expect(req).toEqual({ from_factor_id: "from-id", to_factor_id: "to-id" });

    // 2) simulate
    const fakeReport: ImpactReport = {
      simulated_at: "2026-04-22T00:00:00Z",
      affected_factor_ids: ["from-id"],
      tenants: ["t1", "t2"],
      computation_count: 7,
      inventory_count: 3,
      customer_count: 2,
      summary: { mean_pct_delta: 0.6, max_pct_delta: 2.1, min_pct_delta: -1.0 },
      computations: [],
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(fakeReport));
    const report = await simulateImpact(req);
    const [simUrl, simInit] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(simUrl).toBe("/api/v1/admin/impact-simulate");
    expect(simInit.method).toBe("POST");
    expect(report.computation_count).toBe(7);

    // 3) propose with simulation as evidence
    const proposalPayload = proposeQueueItemFromSimulation({
      fromFactorId: "from-id",
      toFactorId: "to-id",
      rationale: "vendor-driven minor revision",
      report,
    });
    fetchMock.mockResolvedValueOnce(jsonResponse({ review_id: "rev_42" }));
    const { review_id } = await proposeQueueItem(proposalPayload);
    expect(review_id).toBe("rev_42");
    const [propUrl, propInit] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(propUrl).toBe("/api/v1/admin/queue");
    const sentBody = JSON.parse(propInit.body as string);
    expect(sentBody.factor_id).toBe("to-id");
    expect(sentBody.proposed_status).toBe("preview");
    expect(sentBody.evidence.simulation.computation_count).toBe(7);

    // 4) methodology lead approves
    fetchMock.mockResolvedValueOnce(jsonResponse({ ok: true, new_status: "preview" }));
    const approval = await approve("rev_42", { note: "ok" });
    expect(approval.new_status).toBe("preview");
    const [appUrl] = fetchMock.mock.calls[2] as [string, RequestInit];
    expect(appUrl).toBe("/api/v1/admin/queue/rev_42/approve");
  });
});
