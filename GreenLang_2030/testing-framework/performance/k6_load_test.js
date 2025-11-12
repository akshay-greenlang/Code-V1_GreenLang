/**
 * k6 Load Testing Script for GreenLang APIs
 *
 * Tests API performance under various load conditions.
 * Validates latency, throughput, and error rates.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency', true);
const successRate = new Rate('success_rate');
const calculationAccuracy = new Rate('calculation_accuracy');

// Test configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test - minimal load
    smoke: {
      executor: 'constant-vus',
      vus: 2,
      duration: '1m',
      startTime: '0s',
      tags: { test_type: 'smoke' },
    },

    // Load test - normal traffic
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      startTime: '2m',
      tags: { test_type: 'load' },
    },

    // Stress test - beyond normal capacity
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },  // Ramp up
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 200 },  // Push to 200 users
        { duration: '5m', target: 200 },  // Stay at 200 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      startTime: '12m',
      tags: { test_type: 'stress' },
    },

    // Spike test - sudden traffic increase
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 2 },   // Warm up
        { duration: '10s', target: 500 }, // Spike to 500 users
        { duration: '30s', target: 500 }, // Stay at spike
        { duration: '10s', target: 2 },   // Back to normal
        { duration: '30s', target: 2 },   // Recover
      ],
      startTime: '30m',
      tags: { test_type: 'spike' },
    },

    // Soak test - sustained load
    soak: {
      executor: 'constant-vus',
      vus: 100,
      duration: '30m',
      startTime: '35m',
      tags: { test_type: 'soak' },
    },
  },

  // Thresholds (Quality Gates)
  thresholds: {
    // HTTP metrics
    http_req_duration: [
      'p(50)<100',  // 50% of requests under 100ms
      'p(95)<500',  // 95% of requests under 500ms
      'p(99)<1000', // 99% of requests under 1000ms
    ],
    http_req_failed: ['rate<0.05'], // Error rate under 5%

    // Custom metrics
    errors: ['rate<0.05'],
    success_rate: ['rate>0.95'],
    calculation_accuracy: ['rate>0.999'], // 99.9% accuracy
    api_latency: ['p(95)<500', 'p(99)<1000'],
  },
};

// Test data generators
function generateEmissionData() {
  const fuelTypes = ['diesel', 'natural_gas', 'coal', 'electricity'];
  const regions = ['US', 'EU', 'ASIA', 'GLOBAL'];

  return {
    fuel_type: randomItem(fuelTypes),
    fuel_quantity: Math.random() * 1000 + 100,
    region: randomItem(regions),
    combustion_type: 'stationary',
    reporting_period: '2025-Q1',
  };
}

function generateCBAMData() {
  const products = ['cement', 'steel', 'aluminum', 'fertilizer'];
  const countries = ['CN', 'IN', 'RU', 'TR'];

  return {
    product_category: randomItem(products),
    weight_tonnes: Math.random() * 100 + 1,
    origin_country: randomItem(countries),
    import_date: new Date().toISOString().split('T')[0],
    declared_emissions: Math.random() * 5 + 0.5,
  };
}

// API test functions
export function testEmissionCalculation() {
  const data = generateEmissionData();

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
    tags: { endpoint: 'emission_calculation' },
  };

  const startTime = new Date();
  const response = http.post(
    `${BASE_URL}/api/v1/emissions/calculate`,
    JSON.stringify(data),
    params
  );
  const latency = new Date() - startTime;

  // Track metrics
  apiLatency.add(latency, { endpoint: 'emission_calculation' });

  // Validations
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response has emissions': (r) => {
      const body = JSON.parse(r.body);
      return body.total_emissions !== undefined;
    },
    'response time < 500ms': (r) => r.timings.duration < 500,
    'calculation is accurate': (r) => {
      const body = JSON.parse(r.body);
      // Validate calculation: emissions = quantity * factor
      const expectedEmissions = data.fuel_quantity * body.emission_factor;
      const accuracy = Math.abs(body.total_emissions - expectedEmissions) < 0.01;
      calculationAccuracy.add(accuracy);
      return accuracy;
    },
    'has provenance hash': (r) => {
      const body = JSON.parse(r.body);
      return body.provenance_hash && body.provenance_hash.length === 64;
    },
  });

  successRate.add(success);
  errorRate.add(!success);

  return response;
}

export function testCBAMValidation() {
  const data = generateCBAMData();

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
    tags: { endpoint: 'cbam_validation' },
  };

  const response = http.post(
    `${BASE_URL}/api/v1/cbam/validate`,
    JSON.stringify(data),
    params
  );

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response has validation result': (r) => {
      const body = JSON.parse(r.body);
      return body.validation_status !== undefined;
    },
    'response time < 200ms': (r) => r.timings.duration < 200,
  });

  successRate.add(success);
  errorRate.add(!success);

  return response;
}

export function testBatchProcessing() {
  // Generate batch of 100 records
  const batchData = {
    records: Array.from({ length: 100 }, () => generateEmissionData()),
  };

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
    tags: { endpoint: 'batch_processing' },
  };

  const response = http.post(
    `${BASE_URL}/api/v1/emissions/batch`,
    JSON.stringify(batchData),
    params
  );

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'processed all records': (r) => {
      const body = JSON.parse(r.body);
      return body.processed_count === 100;
    },
    'batch processing < 5s': (r) => r.timings.duration < 5000,
  });

  successRate.add(success);
  errorRate.add(!success);

  return response;
}

// Main test scenario
export default function () {
  const scenario = __ENV.SCENARIO || 'mixed';

  switch (scenario) {
    case 'emissions':
      testEmissionCalculation();
      break;
    case 'cbam':
      testCBAMValidation();
      break;
    case 'batch':
      testBatchProcessing();
      break;
    case 'mixed':
    default:
      // Mix of different API calls
      const rand = Math.random();
      if (rand < 0.5) {
        testEmissionCalculation();
      } else if (rand < 0.8) {
        testCBAMValidation();
      } else {
        testBatchProcessing();
      }
  }

  sleep(1); // Think time between requests
}

// Setup and teardown
export function setup() {
  console.log('Starting GreenLang load test...');

  // Verify API is accessible
  const response = http.get(`${BASE_URL}/health`);
  check(response, {
    'API is healthy': (r) => r.status === 200,
  });

  return {
    startTime: new Date(),
    testId: Math.random().toString(36).substring(7),
  };
}

export function teardown(data) {
  console.log('Load test completed');
  console.log(`Test ID: ${data.testId}`);
  console.log(`Duration: ${(new Date() - data.startTime) / 1000}s`);
}

// Custom summary
export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data, null, 2),
    'summary.txt': generateTextSummary(data),
    stdout: generateConsoleSummary(data),
  };
}

function generateTextSummary(data) {
  let summary = 'GreenLang Load Test Results\n';
  summary += '===========================\n\n';

  // Extract key metrics
  const metrics = data.metrics;

  if (metrics.http_req_duration) {
    summary += 'Response Times:\n';
    summary += `  Median: ${metrics.http_req_duration.values['p(50)'].toFixed(2)}ms\n`;
    summary += `  95th percentile: ${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
    summary += `  99th percentile: ${metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n\n`;
  }

  if (metrics.http_reqs) {
    summary += `Total Requests: ${metrics.http_reqs.values.count}\n`;
    summary += `Requests/sec: ${metrics.http_reqs.values.rate.toFixed(2)}\n\n`;
  }

  if (metrics.errors) {
    summary += `Error Rate: ${(metrics.errors.values.rate * 100).toFixed(2)}%\n`;
  }

  if (metrics.success_rate) {
    summary += `Success Rate: ${(metrics.success_rate.values.rate * 100).toFixed(2)}%\n`;
  }

  if (metrics.calculation_accuracy) {
    summary += `Calculation Accuracy: ${(metrics.calculation_accuracy.values.rate * 100).toFixed(3)}%\n`;
  }

  // Check thresholds
  summary += '\nThreshold Results:\n';
  for (const [metric, threshold] of Object.entries(data.metrics)) {
    if (threshold.thresholds) {
      const passed = Object.values(threshold.thresholds).every(t => t.ok);
      summary += `  ${metric}: ${passed ? 'PASS' : 'FAIL'}\n`;
    }
  }

  return summary;
}

function generateConsoleSummary(data) {
  // Simple console output
  const passed = Object.values(data.metrics)
    .filter(m => m.thresholds)
    .every(m => Object.values(m.thresholds).every(t => t.ok));

  return passed
    ? '✅ All performance thresholds passed!\n'
    : '❌ Some performance thresholds failed. Check summary.json for details.\n';
}