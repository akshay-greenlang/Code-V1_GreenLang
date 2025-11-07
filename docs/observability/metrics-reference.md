# GreenLang Metrics Reference

Complete reference of all metrics exposed by GreenLang.

## Core Metrics

### Pipeline Metrics

#### `gl_pipeline_runs_total`
- **Type**: Counter
- **Description**: Total number of pipeline executions
- **Labels**:
  - `pipeline`: Pipeline name
  - `status`: Execution status (success, failure)
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  rate(gl_pipeline_runs_total[5m])
  ```

#### `gl_pipeline_duration_seconds`
- **Type**: Histogram
- **Description**: Pipeline execution duration in seconds
- **Labels**:
  - `pipeline`: Pipeline name
  - `tenant_id`: Tenant identifier
- **Buckets**: 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0
- **Example Query**:
  ```promql
  histogram_quantile(0.95, rate(gl_pipeline_duration_seconds_bucket[5m]))
  ```

#### `gl_active_executions`
- **Type**: Gauge
- **Description**: Currently running pipeline executions
- **Labels**:
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  sum(gl_active_executions)
  ```

### API Metrics

#### `gl_api_requests_total`
- **Type**: Counter
- **Description**: Total number of API requests
- **Labels**:
  - `method`: HTTP method (GET, POST, etc.)
  - `endpoint`: API endpoint path
  - `status_code`: HTTP status code
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  sum(rate(gl_api_requests_total[5m])) by (endpoint)
  ```

#### `gl_api_latency_seconds`
- **Type**: Histogram
- **Description**: API request latency in seconds
- **Labels**:
  - `method`: HTTP method
  - `endpoint`: API endpoint path
  - `tenant_id`: Tenant identifier
- **Buckets**: 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
- **Example Query**:
  ```promql
  histogram_quantile(0.99, rate(gl_api_latency_seconds_bucket[5m]))
  ```

### Resource Metrics

#### `gl_cpu_usage_percent`
- **Type**: Gauge
- **Description**: CPU usage percentage
- **Labels**:
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  avg(gl_cpu_usage_percent)
  ```

#### `gl_memory_usage_bytes`
- **Type**: Gauge
- **Description**: Memory usage in bytes
- **Labels**:
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  gl_memory_usage_bytes / 1024 / 1024 / 1024
  ```

#### `gl_disk_usage_bytes`
- **Type**: Gauge
- **Description**: Disk usage in bytes
- **Labels**:
  - `tenant_id`: Tenant identifier
  - `path`: Filesystem path
- **Example Query**:
  ```promql
  sum(gl_disk_usage_bytes) by (path)
  ```

#### `gl_resource_usage`
- **Type**: Gauge
- **Description**: General resource usage metrics
- **Labels**:
  - `resource_type`: Type of resource (memory_percent, network_bytes_sent, etc.)
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  gl_resource_usage{resource_type="network_bytes_sent"}
  ```

### Error Metrics

#### `gl_errors_total`
- **Type**: Counter
- **Description**: Total number of errors
- **Labels**:
  - `error_type`: Type of error
  - `component`: Component where error occurred
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  sum(rate(gl_errors_total[5m])) by (component)
  ```

### Cache Metrics

#### `gl_cache_hits_total`
- **Type**: Counter
- **Description**: Total cache hits
- **Labels**:
  - `cache_name`: Name of cache
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  rate(gl_cache_hits_total[5m])
  ```

#### `gl_cache_misses_total`
- **Type**: Counter
- **Description**: Total cache misses
- **Labels**:
  - `cache_name`: Name of cache
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  rate(gl_cache_misses_total[5m])
  ```

### Database Metrics

#### `gl_db_queries_total`
- **Type**: Counter
- **Description**: Total database queries
- **Labels**:
  - `query_type`: Type of query (select, insert, update, delete)
  - `table`: Database table name
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  sum(rate(gl_db_queries_total[5m])) by (query_type)
  ```

#### `gl_db_query_duration_seconds`
- **Type**: Histogram
- **Description**: Database query duration in seconds
- **Labels**:
  - `query_type`: Type of query
  - `table`: Database table name
  - `tenant_id`: Tenant identifier
- **Buckets**: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5
- **Example Query**:
  ```promql
  histogram_quantile(0.95, rate(gl_db_query_duration_seconds_bucket[5m]))
  ```

#### `gl_db_connections`
- **Type**: Gauge
- **Description**: Database connection count
- **Labels**:
  - `state`: Connection state (idle, active)
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  sum(gl_db_connections) by (state)
  ```

### Pack Metrics

#### `gl_pack_operations_total`
- **Type**: Counter
- **Description**: Total pack operations
- **Labels**:
  - `operation`: Operation type (install, update, remove)
  - `status`: Operation status (success, failure)
  - `tenant_id`: Tenant identifier
- **Example Query**:
  ```promql
  rate(gl_pack_operations_total{status="failure"}[5m])
  ```

#### `gl_pack_size_bytes`
- **Type**: Histogram
- **Description**: Pack size distribution in bytes
- **Labels**:
  - `pack_name`: Name of pack
  - `tenant_id`: Tenant identifier
- **Buckets**: 1KB, 10KB, 100KB, 1MB, 10MB, 100MB, 1GB
- **Example Query**:
  ```promql
  histogram_quantile(0.90, rate(gl_pack_size_bytes_bucket[5m]))
  ```

## Useful PromQL Queries

### Error Rate
```promql
# Overall error rate
rate(gl_errors_total[5m]) / rate(gl_pipeline_runs_total[5m])

# Error rate by component
sum(rate(gl_errors_total[5m])) by (component) / sum(rate(gl_pipeline_runs_total[5m])) by (component)
```

### Latency Percentiles
```promql
# P50, P95, P99 latency
histogram_quantile(0.50, rate(gl_api_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(gl_api_latency_seconds_bucket[5m]))
```

### Cache Hit Rate
```promql
sum(rate(gl_cache_hits_total[5m])) / (sum(rate(gl_cache_hits_total[5m])) + sum(rate(gl_cache_misses_total[5m])))
```

### Request Rate
```promql
# Total request rate
sum(rate(gl_api_requests_total[5m]))

# Request rate by endpoint
sum(rate(gl_api_requests_total[5m])) by (endpoint)

# Request rate by status code
sum(rate(gl_api_requests_total[5m])) by (status_code)
```

### Resource Usage
```promql
# Average CPU usage
avg(gl_cpu_usage_percent)

# Memory usage in GB
gl_memory_usage_bytes / 1024 / 1024 / 1024

# Disk usage percentage
(gl_disk_usage_bytes / (gl_disk_usage_bytes + 1073741824)) * 100
```

### Top N Queries
```promql
# Top 10 slowest endpoints
topk(10, histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m])))

# Top 10 error producers
topk(10, sum(rate(gl_errors_total[5m])) by (component))

# Top 10 most active pipelines
topk(10, sum(rate(gl_pipeline_runs_total[5m])) by (pipeline))
```

## Aggregation Examples

### By Tenant
```promql
sum(rate(gl_pipeline_runs_total[5m])) by (tenant_id)
```

### By Time Window
```promql
# Increase over last hour
increase(gl_api_requests_total[1h])

# Average over last day
avg_over_time(gl_cpu_usage_percent[24h])
```

### With Filters
```promql
# Only successful requests
rate(gl_api_requests_total{status_code=~"2.."}[5m])

# Only errors
rate(gl_api_requests_total{status_code=~"[45].."}[5m])
```

## Metric Best Practices

### 1. Use Rate for Counters
```promql
# Good: Use rate() for counters
rate(gl_api_requests_total[5m])

# Bad: Don't use raw counter values
gl_api_requests_total
```

### 2. Use Histogram Quantiles for Latency
```promql
# Good: Use histogram_quantile
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Bad: Don't use average
avg(gl_api_latency_seconds)
```

### 3. Aggregate Before Rate
```promql
# Good: Sum first, then rate
sum(rate(gl_pipeline_runs_total[5m])) by (pipeline)

# Works but less efficient
rate(sum(gl_pipeline_runs_total) by (pipeline)[5m])
```

### 4. Use Appropriate Time Windows
```promql
# For alerts: 5-15 minute windows
rate(gl_errors_total[5m])

# For dashboards: Match graph range
rate(gl_api_requests_total[1h])
```

## System Information

#### `gl_system_info`
- **Type**: Info
- **Description**: System information
- **Labels**:
  - `version`: GreenLang version
  - `python_version`: Python version
  - `platform`: Operating system platform
- **Example Query**:
  ```promql
  gl_system_info
  ```
