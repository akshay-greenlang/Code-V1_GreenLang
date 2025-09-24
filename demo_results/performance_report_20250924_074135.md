# GreenLang v0.2.0 Performance Report

Generated: 2025-09-24T07:41:35.871380

## System Information
- CPU Cores: 12
- Total Memory: 15.24 GB
- Python Version: 3.13.5

## Summary
- Status: **PASSED**
- Total Benchmarks: 4
- Successful: 4
- Failed: 0
- Average P95 Latency: 5.15 ms
- Average Throughput: 1099.71 ops/sec
- Max Memory Usage: 46.44 MB

## Benchmark Results

| Benchmark | Status | P95 (ms) | Throughput (ops/sec) | Memory (MB) |
|-----------|--------|----------|---------------------|-------------|
| agent_execution | PASS | 3.42 | 91.74 | 40.25 |
| pipeline_simple | PASS | 2.33 | 47.59 | 40.30 |
| memory_intensive | PASS | 14.15 | 49.42 | 46.44 |
| context_creation | PASS | 0.68 | 4210.09 | 42.79 |

## Load Test Results
- Concurrent Users: 5
- Total Requests: 100
- Success Rate: 100/100 (100.0%)
- Average Response Time: 0.61 ms
- P95 Response Time: 0.56 ms
- P99 Response Time: 18.72 ms
- Requests per Second: 2262.80
