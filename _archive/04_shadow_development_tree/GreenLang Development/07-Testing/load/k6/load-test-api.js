// =============================================================================
// GreenLang API Load Testing Suite
// =============================================================================
// TEST-001: k6 load testing for API endpoints
//
// This test suite covers:
//   - API endpoint performance
//   - Stress testing
//   - Spike testing
//   - Soak testing
//
// Usage:
//   k6 run load-test-api.js
//   k6 run --vus 100 --duration 5m load-test-api.js
//   k6 run --config load-test-config.json load-test-api.js
//
// Author: GreenLang Platform Team
// Version: 1.0.0
// =============================================================================

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

// =============================================================================
// CONFIGURATION
// =============================================================================

// Environment configuration
const BASE_URL = __ENV.BASE_URL || 'https://api.greenlang.io';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Custom metrics
const errorRate = new Rate('errors');
const authFailures = new Counter('auth_failures');
const apiLatency = new Trend('api_latency', true);
const agentExecutionTime = new Trend('agent_execution_time', true);

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - verify system works
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      tags: { test_type: 'smoke' },
      exec: 'smokeTest',
    },

    // Load test - normal traffic patterns
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'load' },
      exec: 'loadTest',
      startTime: '1m', // Start after smoke test
    },

    // Stress test - find breaking point
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 300 },
        { duration: '5m', target: 300 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'stress' },
      exec: 'stressTest',
      startTime: '17m', // Start after load test
    },

    // Spike test - sudden traffic surge
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '30s', target: 500 }, // Sudden spike
        { duration: '2m', target: 500 },
        { duration: '30s', target: 10 },
        { duration: '1m', target: 10 },
      ],
      tags: { test_type: 'spike' },
      exec: 'spikeTest',
      startTime: '40m', // Start after stress test
    },
  },

  // Thresholds - pass/fail criteria
  thresholds: {
    // HTTP request duration thresholds
    http_req_duration: [
      'p(95)<500',  // 95% of requests under 500ms
      'p(99)<1000', // 99% of requests under 1s
    ],

    // Error rate thresholds
    errors: ['rate<0.01'], // Error rate under 1%

    // Custom metric thresholds
    api_latency: ['p(95)<300'],
    agent_execution_time: ['p(95)<5000'],

    // HTTP failures
    http_req_failed: ['rate<0.01'],
  },
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Get authentication headers
function getHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`,
    'X-Request-ID': randomString(16),
  };
}

// Check response and record metrics
function checkResponse(response, name) {
  const success = check(response, {
    [`${name} status is 200`]: (r) => r.status === 200,
    [`${name} response time < 500ms`]: (r) => r.timings.duration < 500,
    [`${name} has valid JSON`]: (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!success);
  apiLatency.add(response.timings.duration);

  if (response.status === 401 || response.status === 403) {
    authFailures.add(1);
  }

  return success;
}

// =============================================================================
// TEST FUNCTIONS
// =============================================================================

// Smoke test - basic functionality verification
export function smokeTest() {
  group('Smoke Test', () => {
    // Health check
    group('Health Check', () => {
      const healthRes = http.get(`${BASE_URL}/health`, {
        headers: getHeaders(),
        tags: { endpoint: 'health' },
      });
      checkResponse(healthRes, 'Health check');
    });

    // Ready check
    group('Ready Check', () => {
      const readyRes = http.get(`${BASE_URL}/ready`, {
        headers: getHeaders(),
        tags: { endpoint: 'ready' },
      });
      checkResponse(readyRes, 'Ready check');
    });

    sleep(1);
  });
}

// Load test - normal operation patterns
export function loadTest() {
  group('Load Test', () => {
    // API v2 endpoints
    group('API Endpoints', () => {
      // List agents
      const agentsRes = http.get(`${BASE_URL}/api/v2/agents`, {
        headers: getHeaders(),
        tags: { endpoint: 'agents-list' },
      });
      checkResponse(agentsRes, 'List agents');

      // Get specific agent
      const agentRes = http.get(`${BASE_URL}/api/v2/agents/GL-CALC-S1-001`, {
        headers: getHeaders(),
        tags: { endpoint: 'agent-get' },
      });
      checkResponse(agentRes, 'Get agent');

      // List pipelines
      const pipelinesRes = http.get(`${BASE_URL}/api/v2/pipelines`, {
        headers: getHeaders(),
        tags: { endpoint: 'pipelines-list' },
      });
      checkResponse(pipelinesRes, 'List pipelines');
    });

    // Calculation endpoints
    group('Calculations', () => {
      // Emissions calculation
      const calcPayload = JSON.stringify({
        organization_id: `org-${randomString(8)}`,
        reporting_period: '2025',
        scope: 1,
        activity_data: {
          fuel_type: 'natural_gas',
          quantity: randomIntBetween(1000, 10000),
          unit: 'therms',
        },
      });

      const calcRes = http.post(`${BASE_URL}/api/v2/calculations/emissions`, calcPayload, {
        headers: getHeaders(),
        tags: { endpoint: 'emissions-calc' },
      });
      checkResponse(calcRes, 'Emissions calculation');

      if (calcRes.status === 200) {
        const calcData = JSON.parse(calcRes.body);
        agentExecutionTime.add(calcData.execution_time_ms || 0);
      }
    });

    // Data endpoints
    group('Data Operations', () => {
      // Get emission factors
      const efRes = http.get(`${BASE_URL}/api/v2/data/emission-factors?category=stationary`, {
        headers: getHeaders(),
        tags: { endpoint: 'emission-factors' },
      });
      checkResponse(efRes, 'Emission factors');

      // Get carbon prices
      const pricesRes = http.get(`${BASE_URL}/api/v2/data/carbon-prices?market=eu_ets`, {
        headers: getHeaders(),
        tags: { endpoint: 'carbon-prices' },
      });
      checkResponse(pricesRes, 'Carbon prices');
    });

    sleep(randomIntBetween(1, 3));
  });
}

// Stress test - push system limits
export function stressTest() {
  group('Stress Test', () => {
    // Heavy computation - batch processing
    group('Batch Processing', () => {
      const batchPayload = JSON.stringify({
        organization_id: `org-${randomString(8)}`,
        reporting_period: '2025',
        items: Array.from({ length: 100 }, (_, i) => ({
          id: `item-${i}`,
          category: 'electricity',
          quantity: randomIntBetween(100, 10000),
          unit: 'kWh',
        })),
      });

      const batchRes = http.post(`${BASE_URL}/api/v2/calculations/batch`, batchPayload, {
        headers: getHeaders(),
        tags: { endpoint: 'batch-calc' },
        timeout: '30s',
      });
      checkResponse(batchRes, 'Batch calculation');
    });

    // Complex query
    group('Complex Queries', () => {
      const queryParams = new URLSearchParams({
        organization_id: `org-${randomString(8)}`,
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        scope: '1,2,3',
        group_by: 'category,month',
        include_details: 'true',
      });

      const queryRes = http.get(`${BASE_URL}/api/v2/reports/emissions?${queryParams}`, {
        headers: getHeaders(),
        tags: { endpoint: 'complex-query' },
        timeout: '10s',
      });
      checkResponse(queryRes, 'Complex query');
    });

    sleep(randomIntBetween(0, 2));
  });
}

// Spike test - sudden traffic surge
export function spikeTest() {
  group('Spike Test', () => {
    // Quick, lightweight requests
    const endpoints = [
      '/health',
      '/ready',
      '/api/v2/agents',
      '/api/v2/data/emission-factors',
    ];

    const endpoint = endpoints[randomIntBetween(0, endpoints.length - 1)];
    const res = http.get(`${BASE_URL}${endpoint}`, {
      headers: getHeaders(),
      tags: { endpoint: endpoint.replace(/\//g, '-') },
    });
    checkResponse(res, `Spike request ${endpoint}`);

    // Minimal sleep during spike
    sleep(0.1);
  });
}

// =============================================================================
// LIFECYCLE HOOKS
// =============================================================================

// Setup - runs once before all VUs start
export function setup() {
  console.log('Starting GreenLang API load tests');
  console.log(`Target URL: ${BASE_URL}`);

  // Verify connectivity
  const healthRes = http.get(`${BASE_URL}/health`);
  if (healthRes.status !== 200) {
    throw new Error(`Health check failed: ${healthRes.status}`);
  }

  return {
    startTime: new Date().toISOString(),
    baseUrl: BASE_URL,
  };
}

// Teardown - runs once after all VUs complete
export function teardown(data) {
  console.log('Load tests completed');
  console.log(`Test started at: ${data.startTime}`);
  console.log(`Test ended at: ${new Date().toISOString()}`);
}

// =============================================================================
// DEFAULT EXPORT (for simple execution)
// =============================================================================

export default function () {
  loadTest();
}
