# GreenLang Analytics WebSocket API

Real-time metrics streaming via WebSocket for the GreenLang analytics dashboard.

## Overview

The WebSocket API provides real-time streaming of metrics from various GreenLang components including system resources, workflow executions, agent calls, and distributed cluster status.

## Quick Start

### Start the Server

```python
from greenlang.api.websocket import MetricsWebSocketServer, MetricCollector
from greenlang.config import get_config

config = get_config()

# Start metric collector
collector = MetricCollector(
    redis_url=config.redis.url,
    collection_interval=config.metrics.collection_interval
)
await collector.start()

# Start WebSocket server
server = MetricsWebSocketServer(
    redis_url=config.redis.url,
    jwt_secret=config.jwt.secret_key,
    jwt_algorithm=config.jwt.algorithm
)
await server.start()
```

### Connect from Frontend

```typescript
import { getMetricService } from './MetricService';

const service = getMetricService(
  'ws://localhost:8000/ws/metrics',
  userToken  // JWT token
);

// Connect
service.connect();

// Subscribe to metrics
service.subscribe({
  channels: ['system.metrics', 'workflow.metrics'],
  tags: { env: 'production' },
  aggregation_interval: '5s',
  compression: true
});

// Listen for metrics
service.onMetric('system.metrics', (metric) => {
  console.log('Received metric:', metric);
});

// Handle connection status
service.onConnectionChange((connected) => {
  console.log('Connected:', connected);
});
```

## WebSocket Protocol

### Connection

Connect to the WebSocket endpoint with optional JWT token:

```
ws://localhost:8000/ws/metrics?token=<jwt-token>
```

### Message Format

All messages are JSON or MessagePack encoded.

#### Client → Server

**Subscribe to Channels**
```json
{
  "type": "subscribe",
  "data": {
    "channels": ["system.metrics", "workflow.metrics"],
    "tags": { "env": "production" },
    "aggregation_interval": "5s",
    "compression": true
  }
}
```

**Unsubscribe from Channels**
```json
{
  "type": "unsubscribe",
  "channels": ["system.metrics"]
}
```

**Request Historical Metrics**
```json
{
  "type": "get_history",
  "data": {
    "channel": "system.metrics",
    "start_time": "2025-11-08T00:00:00Z",
    "end_time": "2025-11-08T01:00:00Z"
  }
}
```

**Heartbeat Response**
```json
{
  "type": "pong",
  "timestamp": 1699459200.0
}
```

#### Server → Client

**Welcome Message**
```json
{
  "type": "welcome",
  "client_id": "127.0.0.1:12345",
  "timestamp": 1699459200.0
}
```

**Subscription Confirmation**
```json
{
  "type": "subscribed",
  "channels": ["system.metrics", "workflow.metrics"]
}
```

**Metric Data**
```json
{
  "channel": "system.metrics",
  "name": "cpu.percent",
  "type": "gauge",
  "value": 75.5,
  "timestamp": "2025-11-08T12:00:00Z",
  "tags": { "host": "server1", "env": "production" }
}
```

**Heartbeat Ping**
```json
{
  "type": "ping",
  "timestamp": 1699459200.0
}
```

**Error Message**
```json
{
  "type": "error",
  "message": "Invalid subscription: unknown channel"
}
```

## Metric Channels

### system.metrics

System resource metrics collected every second.

**Metrics**:
- `cpu.percent` - CPU usage percentage
- `memory.percent` - Memory usage percentage
- `disk.percent` - Disk usage percentage
- `network.bytes_sent` - Network bytes sent
- `network.bytes_recv` - Network bytes received

**Example**:
```json
{
  "channel": "system.metrics",
  "timestamp": "2025-11-08T12:00:00Z",
  "cpu": {
    "percent": 45.2,
    "count": 8,
    "frequency": 2400.0
  },
  "memory": {
    "total": 16777216000,
    "used": 8388608000,
    "percent": 50.0
  }
}
```

### workflow.metrics

Workflow execution metrics collected every 5 seconds.

**Metrics**:
- `executions.total` - Total executions
- `executions.successful` - Successful executions
- `executions.failed` - Failed executions
- `success_rate` - Success rate percentage
- `duration.avg` - Average duration
- `duration.p95` - 95th percentile duration

**Example**:
```json
{
  "channel": "workflow.metrics",
  "timestamp": "2025-11-08T12:00:00Z",
  "executions": {
    "total": 1000,
    "successful": 950,
    "failed": 50
  },
  "success_rate": 95.0,
  "duration": {
    "avg": 125.5,
    "p95": 250.0
  }
}
```

### agent.metrics

Agent execution metrics collected every 5 seconds.

**Metrics**:
- `calls.total` - Total agent calls
- `calls.successful` - Successful calls
- `calls.failed` - Failed calls
- `latency.avg` - Average latency
- `errors.total` - Total errors

**Example**:
```json
{
  "channel": "agent.metrics",
  "timestamp": "2025-11-08T12:00:00Z",
  "calls": {
    "total": 5000,
    "successful": 4900,
    "failed": 100
  },
  "latency": {
    "avg": 45.5,
    "p95": 120.0
  }
}
```

### distributed.metrics

Distributed cluster metrics collected every 5 seconds.

**Metrics**:
- `nodes.active` - Active nodes
- `tasks.pending` - Pending tasks
- `tasks.running` - Running tasks
- `throughput` - Tasks per second

**Example**:
```json
{
  "channel": "distributed.metrics",
  "timestamp": "2025-11-08T12:00:00Z",
  "nodes": {
    "active": 5,
    "total": 10
  },
  "tasks": {
    "pending": 100,
    "running": 50,
    "completed": 10000
  },
  "throughput": 50.5
}
```

## Features

### Metric Filtering

Filter metrics by tags:

```json
{
  "type": "subscribe",
  "data": {
    "channels": ["system.metrics"],
    "tags": {
      "env": "production",
      "region": "us-east-1"
    }
  }
}
```

### Metric Aggregation

Aggregate metrics over time intervals:

```json
{
  "type": "subscribe",
  "data": {
    "channels": ["system.metrics"],
    "aggregation_interval": "1m"
  }
}
```

Supported intervals: `1s`, `5s`, `1m`, `5m`, `1h`

### Compression

Enable MessagePack compression for large payloads:

```json
{
  "type": "subscribe",
  "data": {
    "channels": ["system.metrics"],
    "compression": true
  }
}
```

### Rate Limiting

Clients are limited to 1000 messages per minute by default. Exceeding the limit will result in dropped messages.

### Authentication

Connections require a valid JWT token:

```
ws://localhost:8000/ws/metrics?token=<jwt-token>
```

Token payload must include:
```json
{
  "sub": "user-id",
  "exp": 1699459200
}
```

## Error Handling

### Client-Side Reconnection

The client automatically reconnects with exponential backoff:

```typescript
// Reconnection is handled automatically
service.connect();

// Monitor connection status
service.onConnectionChange((connected) => {
  if (!connected) {
    console.log('Disconnected, reconnecting...');
  }
});
```

### Error Messages

Errors are sent as JSON messages:

```json
{
  "type": "error",
  "message": "Rate limit exceeded"
}
```

Common error messages:
- `Invalid authentication token`
- `Rate limit exceeded`
- `Invalid subscription: unknown channel`
- `Subscription error: <details>`

## Performance

### Throughput
- Server: 10,000+ messages/sec per instance
- Client: 1,000 messages/min (rate limited)

### Latency
- Metric collection: <50ms
- WebSocket delivery: <10ms
- End-to-end: <100ms

### Scalability
- Concurrent clients: 10,000+ per server
- Horizontal scaling via load balancer
- Redis pub/sub for message distribution

## Best Practices

### Subscribe Efficiently

Only subscribe to channels you need:

```typescript
// Good: Subscribe to specific channels
service.subscribe({
  channels: ['system.metrics'],
  tags: { host: 'server1' }
});

// Bad: Subscribe to all channels
service.subscribe({
  channels: [
    'system.metrics',
    'workflow.metrics',
    'agent.metrics',
    'distributed.metrics'
  ]
});
```

### Handle Disconnections

Always handle connection state changes:

```typescript
service.onConnectionChange((connected) => {
  if (connected) {
    // Resubscribe to channels
    service.subscribe({
      channels: ['system.metrics']
    });
  } else {
    // Show disconnected UI state
    showDisconnectedBanner();
  }
});
```

### Buffer Offline Data

Buffer metrics when offline:

```typescript
const bufferedMetrics = service.getBufferedMetrics();
```

### Use Compression

Enable compression for high-frequency metrics:

```typescript
service.subscribe({
  channels: ['system.metrics'],
  compression: true
});
```

## Configuration

See `greenlang/config/analytics_config.py` for configuration options.

### Environment Variables

```bash
# WebSocket
WS_HOST=0.0.0.0
WS_PORT=8000
WS_HEARTBEAT_INTERVAL=30

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256

# Rate Limiting
RATE_LIMIT_MAX_MESSAGES_PER_MINUTE=1000
```

## Testing

Run WebSocket tests:

```bash
pytest tests/phase4/test_metrics_websocket.py -v
```

## Monitoring

Monitor WebSocket server:

```python
stats = server.get_stats()
print(f"Connected clients: {stats['connected_clients']}")
print(f"Total subscriptions: {stats['total_subscriptions']}")
```

## Troubleshooting

### Connection Refused

Check that the server is running:

```bash
netstat -an | grep 8000
```

### Authentication Failed

Verify JWT token is valid:

```python
from jose import jwt

payload = jwt.decode(token, secret_key, algorithms=["HS256"])
print(payload)
```

### No Metrics Received

Check metric collector is running:

```python
stats = collector.get_stats()
print(f"Collector running: {stats['running']}")
```

## Support

For issues or questions, please refer to:
- Main documentation: `PHASE4B_ANALYTICS_DASHBOARD_SUMMARY.md`
- Configuration: `greenlang/config/analytics_config.py`
- Tests: `tests/phase4/test_metrics_websocket.py`
