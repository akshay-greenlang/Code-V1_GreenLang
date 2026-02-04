/**
 * GreenLang Climate OS - pgvector Load Test
 * PRD: INFRA-005 Vector Database Infrastructure with pgvector
 *
 * k6 load test configuration for vector search and batch insert operations.
 *
 * Usage:
 *   k6 run --env BASE_URL=http://localhost:8000 pgvector_search_test.js
 *
 * SLOs (from PRD):
 *   - Search P50 < 20ms
 *   - Search P99 < 100ms
 *   - Insert throughput > 10,000/sec
 *   - Availability > 99.9%
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// Custom metrics
const searchLatency = new Trend('pgvector_search_latency_ms', true);
const hybridSearchLatency = new Trend('pgvector_hybrid_search_latency_ms', true);
const insertLatency = new Trend('pgvector_insert_latency_ms', true);
const searchErrors = new Rate('pgvector_search_error_rate');
const insertErrors = new Rate('pgvector_insert_error_rate');
const searchCount = new Counter('pgvector_search_total');
const insertCount = new Counter('pgvector_insert_total');

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_PREFIX = '/api/v1/vectors';

// Test scenarios
export const options = {
  scenarios: {
    // Scenario 1: Vector search ramping load
    vector_search: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 100 },    // Ramp to 100 VUs
        { duration: '5m', target: 100 },    // Sustain 100 VUs
        { duration: '2m', target: 200 },    // Ramp to 200 VUs
        { duration: '5m', target: 200 },    // Sustain 200 VUs
        { duration: '2m', target: 0 },      // Ramp down
      ],
      exec: 'searchTest',
      gracefulRampDown: '30s',
    },

    // Scenario 2: Hybrid search
    hybrid_search: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 50 },
        { duration: '2m', target: 0 },
      ],
      exec: 'hybridSearchTest',
      startTime: '1m',
      gracefulRampDown: '30s',
    },

    // Scenario 3: Batch insert throughput
    batch_insert: {
      executor: 'constant-arrival-rate',
      rate: 100,          // 100 requests/s (each with 10 texts = 1000 inserts/s)
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 50,
      maxVUs: 100,
      exec: 'insertTest',
      startTime: '5m',
    },
  },

  // PRD SLO thresholds
  thresholds: {
    'pgvector_search_latency_ms': [
      { threshold: 'p(50)<20', abortOnFail: false },
      { threshold: 'p(99)<100', abortOnFail: true },
    ],
    'pgvector_hybrid_search_latency_ms': [
      { threshold: 'p(99)<150', abortOnFail: false },
    ],
    'pgvector_search_error_rate': [
      { threshold: 'rate<0.01', abortOnFail: true },
    ],
    'pgvector_insert_error_rate': [
      { threshold: 'rate<0.01', abortOnFail: false },
    ],
    'http_req_failed': [
      { threshold: 'rate<0.01', abortOnFail: true },
    ],
  },
};

// Sample queries for search testing
const SAMPLE_QUERIES = [
  'carbon emission reduction strategies',
  'EU taxonomy alignment criteria',
  'scope 3 supply chain emissions',
  'deforestation risk assessment',
  'building energy performance standards',
  'CBAM carbon border adjustment',
  'green bond taxonomy classification',
  'climate risk disclosure requirements',
  'renewable energy certificates',
  'waste management circular economy',
];

const NAMESPACES = ['csrd', 'cbam', 'eudr', 'vcci', 'sb253', 'taxonomy', 'default'];

function getRandomQuery() {
  return SAMPLE_QUERIES[Math.floor(Math.random() * SAMPLE_QUERIES.length)];
}

function getRandomNamespace() {
  return NAMESPACES[Math.floor(Math.random() * NAMESPACES.length)];
}

// Scenario 1: Similarity Search
export function searchTest() {
  group('similarity_search', function() {
    const payload = JSON.stringify({
      query: getRandomQuery(),
      namespace: getRandomNamespace(),
      top_k: 10,
      threshold: 0.5,
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
      timeout: '10s',
    };

    const res = http.post(`${BASE_URL}${API_PREFIX}/search`, payload, params);

    const success = check(res, {
      'search status 200': (r) => r.status === 200,
      'search has matches': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.matches !== undefined;
        } catch {
          return false;
        }
      },
      'search latency < 100ms': (r) => r.timings.duration < 100,
    });

    searchLatency.add(res.timings.duration);
    searchErrors.add(!success);
    searchCount.add(1);

    sleep(0.1);
  });
}

// Scenario 2: Hybrid Search
export function hybridSearchTest() {
  group('hybrid_search', function() {
    const payload = JSON.stringify({
      query: getRandomQuery(),
      namespace: getRandomNamespace(),
      top_k: 10,
      rrf_k: 60,
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
      timeout: '15s',
    };

    const res = http.post(`${BASE_URL}${API_PREFIX}/hybrid-search`, payload, params);

    const success = check(res, {
      'hybrid search status 200': (r) => r.status === 200,
      'hybrid search has matches': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.matches !== undefined;
        } catch {
          return false;
        }
      },
      'hybrid search latency < 150ms': (r) => r.timings.duration < 150,
    });

    hybridSearchLatency.add(res.timings.duration);
    searchErrors.add(!success);

    sleep(0.2);
  });
}

// Scenario 3: Batch Insert
export function insertTest() {
  group('batch_insert', function() {
    const texts = [];
    for (let i = 0; i < 10; i++) {
      texts.push(`Sample text for embedding ${randomString(20)} about climate ${getRandomQuery()}`);
    }

    const payload = JSON.stringify({
      texts: texts,
      source_type: 'document',
      namespace: getRandomNamespace(),
      metadata: { test: true, load_test: true },
    });

    const params = {
      headers: { 'Content-Type': 'application/json' },
      timeout: '30s',
    };

    const res = http.post(`${BASE_URL}${API_PREFIX}/embed-and-store`, payload, params);

    const success = check(res, {
      'insert status 200': (r) => r.status === 200,
      'insert has count': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.inserted_count > 0;
        } catch {
          return false;
        }
      },
    });

    insertLatency.add(res.timings.duration);
    insertErrors.add(!success);
    insertCount.add(10); // 10 texts per request

    sleep(0.01);
  });
}

// Health check (runs once at start)
export function setup() {
  const res = http.get(`${BASE_URL}${API_PREFIX}/health`);
  check(res, {
    'health check passed': (r) => r.status === 200,
    'pgvector available': (r) => {
      try {
        return JSON.parse(r.body).pgvector === true;
      } catch {
        return false;
      }
    },
  });
  return { baseUrl: BASE_URL };
}
