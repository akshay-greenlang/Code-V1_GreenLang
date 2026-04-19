/**
 * k6 load test script for GreenLang Factors API.
 *
 * Target: 1000 req/s sustained, p95 < 50ms, error rate < 1%.
 *
 * Stages:
 *   - Ramp up:  0 -> 100 VUs over 1 minute
 *   - Sustain:  100 VUs for 5 minutes
 *   - Ramp down: 100 -> 0 VUs over 1 minute
 *
 * Usage:
 *   k6 run tests/factors/load/k6_script.js
 *   k6 run --out json=results.json tests/factors/load/k6_script.js
 *
 * Environment variables:
 *   GL_FACTORS_BASE_URL: Target API base URL (default: http://localhost:8000)
 *   GL_LOAD_TEST_API_KEY: Test API key
 */

import http from "k6/http";
import { check, group, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

// ── Configuration ─────────────────────────────────────────────────

const BASE_URL = __ENV.GL_FACTORS_BASE_URL || "http://localhost:8000";
const API_KEY =
  __ENV.GL_LOAD_TEST_API_KEY ||
  "gl_test_load_key_xxxxxxxxxxxxxxxxxxxxxxxx";

// ── Custom metrics ────────────────────────────────────────────────

const searchLatency = new Trend("factors_search_latency", true);
const listLatency = new Trend("factors_list_latency", true);
const getLatency = new Trend("factors_get_latency", true);
const matchLatency = new Trend("factors_match_latency", true);
const editionLatency = new Trend("factors_edition_latency", true);
const exportLatency = new Trend("factors_export_latency", true);
const errorRate = new Rate("factors_error_rate");

// ── k6 options ────────────────────────────────────────────────────

export const options = {
  stages: [
    { duration: "1m", target: 100 }, // Ramp up
    { duration: "5m", target: 100 }, // Sustain
    { duration: "1m", target: 0 }, // Ramp down
  ],
  thresholds: {
    // Global thresholds
    http_req_duration: ["p(95)<50", "p(99)<100"],
    factors_error_rate: ["rate<0.01"],
    // Per-scenario thresholds
    factors_search_latency: ["p(95)<50"],
    factors_list_latency: ["p(95)<40"],
    factors_get_latency: ["p(95)<25"],
    factors_match_latency: ["p(95)<50"],
    factors_edition_latency: ["p(95)<20"],
    factors_export_latency: ["p(95)<500"],
  },
};

// ── Shared data ───────────────────────────────────────────────────

const SEARCH_QUERIES = [
  "diesel",
  "natural gas",
  "electricity",
  "gasoline",
  "coal",
  "fuel oil",
  "propane",
  "biomass",
  "solar",
  "wind",
  "jet fuel",
  "LPG",
  "CNG",
  "hydrogen",
  "ethanol",
  "biodiesel",
  "refrigerant R-410A",
  "HFC-134a",
  "methane fugitive",
  "waste landfill",
];

const GEOGRAPHIES = [
  "US",
  "GB",
  "DE",
  "FR",
  "JP",
  "AU",
  "CA",
  "IN",
  "BR",
  "CN",
];

const ACTIVITY_DESCRIPTIONS = [
  "Burned 1000 gallons of diesel in stationary generators",
  "Purchased 500 MWh of grid electricity in California",
  "Fleet consumed 2000 liters of petrol",
  "Used 300 therms of natural gas for heating",
  "Refrigerant leak of 5 kg R-410A from HVAC",
  "Employee commuting 10000 km by car",
  "Air freight 2000 kg-km international",
  "Waste sent to landfill 50 tonnes",
  "Business travel 5000 passenger-km domestic flights",
  "Purchased 100 tonnes of steel",
];

// Collected factor IDs for GET requests
let knownFactorIds = [];
let knownEditionIds = [];

// ── Helpers ───────────────────────────────────────────────────────

function authHeaders() {
  return {
    Authorization: `Bearer ${API_KEY}`,
    Accept: "application/json",
    "Content-Type": "application/json",
  };
}

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ── Setup ─────────────────────────────────────────────────────────

export function setup() {
  // Seed factor IDs and edition IDs
  const searchResp = http.get(
    `${BASE_URL}/v2/factors/search?q=diesel&limit=50`,
    { headers: authHeaders() }
  );
  if (searchResp.status === 200) {
    try {
      const data = JSON.parse(searchResp.body);
      knownFactorIds = (data.factors || [])
        .map((f) => f.factor_id)
        .filter(Boolean);
    } catch (_) {
      // Ignore parse errors in setup
    }
  }

  const editionResp = http.get(`${BASE_URL}/v2/editions`, {
    headers: authHeaders(),
  });
  if (editionResp.status === 200) {
    try {
      const data = JSON.parse(editionResp.body);
      knownEditionIds = (data.editions || [])
        .map((e) => e.edition_id)
        .filter(Boolean);
    } catch (_) {
      // Ignore parse errors in setup
    }
  }

  return { factorIds: knownFactorIds, editionIds: knownEditionIds };
}

// ── Main scenario (weighted by iteration distribution) ────────────

export default function (data) {
  const roll = Math.random() * 100;

  if (roll < 40) {
    searchScenario();
  } else if (roll < 60) {
    listScenario();
  } else if (roll < 80) {
    getScenario(data);
  } else if (roll < 90) {
    matchScenario();
  } else if (roll < 95) {
    editionScenario(data);
  } else {
    exportScenario();
  }

  sleep(0.1 + Math.random() * 0.4);
}

// ── Scenario implementations ──────────────────────────────────────

function searchScenario() {
  group("search", function () {
    const query = randomChoice(SEARCH_QUERIES);
    const geo = randomChoice(GEOGRAPHIES);

    // Simple search
    const resp = http.get(
      `${BASE_URL}/v2/factors/search?q=${encodeURIComponent(query)}&limit=10`,
      { headers: authHeaders(), tags: { endpoint: "search" } }
    );
    searchLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200);

    check(resp, {
      "search status 200": (r) => r.status === 200,
      "search has factors": (r) => {
        try {
          return JSON.parse(r.body).factors.length > 0;
        } catch (_) {
          return false;
        }
      },
    });

    // Filtered search (50% of time)
    if (Math.random() < 0.5) {
      const filteredResp = http.get(
        `${BASE_URL}/v2/factors/search?q=${encodeURIComponent(query)}&geography=${geo}&limit=20`,
        { headers: authHeaders(), tags: { endpoint: "search_filtered" } }
      );
      searchLatency.add(filteredResp.timings.duration);
      errorRate.add(filteredResp.status !== 200);
    }
  });
}

function listScenario() {
  group("list", function () {
    const offsets = [0, 25, 50, 100];
    const offset = randomChoice(offsets);

    const resp = http.get(
      `${BASE_URL}/v2/factors?limit=25&offset=${offset}`,
      { headers: authHeaders(), tags: { endpoint: "list" } }
    );
    listLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200);

    check(resp, {
      "list status 200": (r) => r.status === 200,
    });

    // Filtered list (50% of time)
    if (Math.random() < 0.5) {
      const geo = randomChoice(GEOGRAPHIES);
      const filteredResp = http.get(
        `${BASE_URL}/v2/factors?geography=${geo}&limit=25`,
        { headers: authHeaders(), tags: { endpoint: "list_filtered" } }
      );
      listLatency.add(filteredResp.timings.duration);
      errorRate.add(filteredResp.status !== 200);
    }
  });
}

function getScenario(data) {
  group("get", function () {
    const factorIds = data.factorIds || [];
    if (factorIds.length === 0) {
      return;
    }
    const factorId = randomChoice(factorIds);

    const resp = http.get(`${BASE_URL}/v2/factors/${factorId}`, {
      headers: authHeaders(),
      tags: { endpoint: "get" },
    });
    getLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200 && resp.status !== 404);

    check(resp, {
      "get status 200 or 404": (r) =>
        r.status === 200 || r.status === 404,
    });

    // ETag conditional request (30% of time)
    if (Math.random() < 0.3 && resp.status === 200) {
      const etag = resp.headers["ETag"];
      if (etag) {
        const conditionalHeaders = authHeaders();
        conditionalHeaders["If-None-Match"] = etag;
        const condResp = http.get(
          `${BASE_URL}/v2/factors/${factorId}`,
          { headers: conditionalHeaders, tags: { endpoint: "get_conditional" } }
        );
        getLatency.add(condResp.timings.duration);
        errorRate.add(
          condResp.status !== 200 && condResp.status !== 304
        );
      }
    }
  });
}

function matchScenario() {
  group("match", function () {
    const description = randomChoice(ACTIVITY_DESCRIPTIONS);
    const payload = JSON.stringify({
      activity_description: description,
      max_candidates: 5,
    });

    const resp = http.post(`${BASE_URL}/v2/factors/match`, payload, {
      headers: authHeaders(),
      tags: { endpoint: "match" },
    });
    matchLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200);

    check(resp, {
      "match status 200": (r) => r.status === 200,
    });
  });
}

function editionScenario(data) {
  group("edition", function () {
    // List editions
    const resp = http.get(`${BASE_URL}/v2/editions`, {
      headers: authHeaders(),
      tags: { endpoint: "edition_list" },
    });
    editionLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200);

    check(resp, {
      "edition list status 200": (r) => r.status === 200,
    });

    // Changelog for random edition (if any known)
    const editionIds = data.editionIds || [];
    if (editionIds.length > 0) {
      const editionId = randomChoice(editionIds);
      const changelogResp = http.get(
        `${BASE_URL}/v2/editions/${editionId}/changelog`,
        { headers: authHeaders(), tags: { endpoint: "edition_changelog" } }
      );
      editionLatency.add(changelogResp.timings.duration);
      errorRate.add(
        changelogResp.status !== 200 && changelogResp.status !== 404
      );
    }
  });
}

function exportScenario() {
  group("export", function () {
    const resp = http.get(
      `${BASE_URL}/v2/factors/export?tier=enterprise&format=json`,
      { headers: authHeaders(), tags: { endpoint: "export" } }
    );
    exportLatency.add(resp.timings.duration);
    errorRate.add(resp.status !== 200);

    check(resp, {
      "export status 200": (r) => r.status === 200,
    });
  });
}
