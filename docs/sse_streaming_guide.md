# SSE Streaming Guide for Process Heat Agents

## Overview

Server-Sent Events (SSE) streaming provides real-time, one-way communication from GreenLang Process Heat agents to web clients. This guide covers implementation and usage patterns.

## Features

- **Agent Status Updates**: IDLE, RUNNING, PAUSED, COMPLETED, ERROR, SHUTDOWN
- **Job Progress Tracking**: Real-time job progress (0-100%) with step details
- **Alarm Notifications**: Multi-level alarms (INFO, WARNING, CRITICAL, ALERT)
- **Live Metrics Streaming**: Temperature, pressure, efficiency, and custom metrics
- **Automatic Heartbeat**: 30-second heartbeat to maintain connection
- **Last-Event-ID Support**: Automatic replay of missed events on reconnection
- **Client Filtering**: Subscribe to specific event types
- **Connection Limits**: Per-stream client capacity management

## Architecture

```
┌─────────────────┐
│  Process Heat   │
│     Agent       │
└────────┬────────┘
         │
         ├─→ send_agent_status()
         ├─→ send_job_progress()
         ├─→ send_alarm_update()
         └─→ send_metrics()
         │
         v
┌─────────────────────────────┐
│   SSEStreamManager          │
│  - Stream routing           │
│  - Event buffering          │
│  - Client management        │
│  - Heartbeat control        │
└──────────┬──────────────────┘
           │
           ├─ /api/v1/stream/events (Global)
           ├─ /api/v1/stream/agents/{id} (Agent-specific)
           ├─ /api/v1/stream/jobs/{id} (Job progress)
           └─ /api/v1/stream/status (Manager status)
           │
           v
    ┌──────────────┐
    │  Web Client  │
    │  EventSource │
    └──────────────┘
```

## Integration with FastAPI

### 1. Initialize SSE Manager

```python
from fastapi import FastAPI
from greenlang.infrastructure.api.sse_streaming import SSEStreamManager

app = FastAPI()
sse_manager = SSEStreamManager()

@app.on_event("startup")
async def startup():
    await sse_manager.start()

@app.on_event("shutdown")
async def shutdown():
    await sse_manager.stop()

# Register SSE routes
app.include_router(sse_manager.router)
```

### 2. Send Agent Status Updates

```python
from greenlang.infrastructure.api.sse_streaming import AgentStatusEnum

# From agent processing
async def run_agent(agent_id: str):
    # Agent starting
    await sse_manager.send_agent_status(
        agent_id=agent_id,
        status=AgentStatusEnum.RUNNING,
        current_step="Initializing",
        progress_percent=0
    )

    # During processing
    await sse_manager.send_agent_status(
        agent_id=agent_id,
        status=AgentStatusEnum.RUNNING,
        current_step="Thermal Analysis",
        progress_percent=33,
        metadata={
            "steam_flow": 25.5,  # kg/s
            "pressure": 8.5,      # bar
            "efficiency": 0.88
        }
    )

    # Agent completed
    await sse_manager.send_agent_status(
        agent_id=agent_id,
        status=AgentStatusEnum.COMPLETED,
        progress_percent=100
    )
```

### 3. Stream Job Progress

```python
async def execute_job(job_id: str, agent_id: str):
    total_steps = 5

    for step_num in range(1, total_steps + 1):
        # Perform step work...
        await asyncio.sleep(1)

        # Send progress
        progress_pct = int((step_num / total_steps) * 100)
        await sse_manager.send_job_progress(
            job_id=job_id,
            agent_id=agent_id,
            progress_percent=progress_pct,
            current_step=f"Step {step_num}: Thermal Analysis",
            step_number=step_num,
            total_steps=total_steps,
            elapsed_seconds=step_num,
            estimated_total_seconds=total_steps,
            details={
                "processed_records": step_num * 100,
                "current_temp": 150.0 + (step_num * 5),
                "efficiency_delta": 0.02 * step_num
            }
        )
```

### 4. Send Alarm Notifications

```python
from greenlang.infrastructure.api.sse_streaming import AlarmSeverityEnum

# Monitor for threshold exceedance
async def monitor_system(agent_id: str):
    pressure = 9.5  # bar

    if pressure > 9.0:
        await sse_manager.send_alarm_update(
            alarm_id="pressure_alarm_001",
            agent_id=agent_id,
            severity=AlarmSeverityEnum.CRITICAL,
            message="Pressure exceeded maximum safe operating limit",
            parameter="steam_pressure",
            current_value=pressure,
            threshold_value=9.0,
            metadata={
                "unit": "bar",
                "time_exceeded": "2 minutes",
                "rate_of_change": 0.1  # bar/min
            }
        )
```

### 5. Stream Live Metrics

```python
async def stream_metrics(agent_id: str):
    """Stream real-time metrics every 5 seconds."""
    while True:
        metrics = {
            "steam_temperature": 150.5,
            "steam_pressure": 8.5,
            "flow_rate": 25.3,
            "efficiency": 0.92,
            "power_output": 150.5,
            "heat_loss": 12.3
        }

        await sse_manager.send_metrics(
            source_id=agent_id,
            metrics=metrics,
            unit="SI",
            metadata={
                "measurement_interval": "1s",
                "data_quality": "GOOD"
            }
        )

        await asyncio.sleep(5)
```

## Client-Side Implementation

### JavaScript EventSource

```javascript
// Subscribe to global events
const eventSource = new EventSource('/api/v1/stream/events?event_types=agent.status,calculation.progress');

eventSource.addEventListener('agent.status', (event) => {
    const data = JSON.parse(event.data);
    console.log(`Agent ${data.agent_id}: ${data.status} (${data.progress_percent}%)`);
    updateAgentStatus(data.agent_id, data);
});

eventSource.addEventListener('calculation.progress', (event) => {
    const data = JSON.parse(event.data);
    console.log(`Job ${data.job_id}: ${data.current_step} (${data.progress_percent}%)`);
    updateProgressBar(data.job_id, data.progress_percent);
});

eventSource.addEventListener('alarm.update', (event) => {
    const data = JSON.parse(event.data);
    showAlarm(data.severity, data.message);
});

eventSource.addEventListener('metrics.update', (event) => {
    const data = JSON.parse(event.data);
    updateMetricsDashboard(data.metrics);
});

eventSource.addEventListener('heartbeat', (event) => {
    console.log('Connection alive:', event.data);
});

// Handle connection errors
eventSource.onerror = (event) => {
    if (event.target.readyState === EventSource.CLOSED) {
        console.log('Connection closed');
        // Reconnect after delay
        setTimeout(() => {
            window.location.reload();
        }, 5000);
    }
};
```

### React Hook Pattern

```javascript
import { useEffect, useState } from 'react';

function useSSEStream(url, eventTypes = []) {
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fullUrl = eventTypes.length > 0
            ? `${url}?event_types=${eventTypes.join(',')}`
            : url;

        const es = new EventSource(fullUrl);

        es.addEventListener('agent.status', (event) => {
            setData(JSON.parse(event.data));
        });

        es.addEventListener('calculation.progress', (event) => {
            setData(JSON.parse(event.data));
        });

        es.addEventListener('error', () => {
            setError('Connection error');
        });

        return () => es.close();
    }, [url, eventTypes]);

    return { data, error };
}

// Usage
function AgentMonitor({ agentId }) {
    const { data, error } = useSSEStream(
        `/api/v1/stream/agents/${agentId}`,
        ['agent.status', 'metrics.update']
    );

    return (
        <div>
            {error && <div>Error: {error}</div>}
            {data && (
                <div>
                    <h3>Agent: {data.agent_id}</h3>
                    <p>Status: {data.status}</p>
                    <progress value={data.progress_percent} max="100" />
                </div>
            )}
        </div>
    );
}
```

## API Endpoints

### 1. Global Events Stream

```
GET /api/v1/stream/events?event_types=agent.status,calculation.progress
```

Subscribe to all events globally. Optional filter by event types.

**Query Parameters:**
- `event_types`: Comma-separated list of event types to filter

**Response:**
```
Content-Type: text/event-stream
Cache-Control: no-cache

id: 8f4e3b2a-1c9d-4e7f-8a2b-3c4d5e6f7a8b
event: agent.status
data: {"agent_id":"agent:001","status":"RUNNING","progress_percent":50,"current_step":"thermal_analysis"}
retry: 3000

id: 9f5f4c3b-2d0e-5f8g-9b3c-4d5e6f7g8b9c
event: heartbeat
data: {"timestamp":"2025-12-07T10:30:45.123456"}
retry: 3000
```

### 2. Agent-Specific Stream

```
GET /api/v1/stream/agents/{agent_id}
```

Subscribe to updates for a specific agent.

**Path Parameters:**
- `agent_id`: Agent identifier

**Response:** Same as global stream, filtered to agent

### 3. Job Progress Stream

```
GET /api/v1/stream/jobs/{job_id}
```

Subscribe to progress updates for a specific job.

**Path Parameters:**
- `job_id`: Job identifier

**Response:** Calculation progress events only

### 4. Manager Status

```
GET /api/v1/stream/status
```

Get current streaming manager statistics.

**Response:**
```json
{
    "total_clients": 42,
    "total_streams": 15,
    "active_streams": {
        "global": {"subscribers": 5, "event_history_size": 45},
        "agent:001": {"subscribers": 3, "event_history_size": 50},
        "job:123": {"subscribers": 2, "event_history_size": 23}
    },
    "config": {
        "heartbeat_interval_seconds": 30,
        "client_timeout_seconds": 300,
        "max_queue_size": 500
    }
}
```

## Event Types Reference

### agent.status
```json
{
    "agent_id": "agent:001",
    "status": "RUNNING",
    "progress_percent": 50,
    "current_step": "thermal_analysis",
    "metadata": {
        "steam_flow": 25.5,
        "efficiency": 0.88
    }
}
```

### calculation.progress
```json
{
    "job_id": "job:123",
    "agent_id": "agent:001",
    "progress_percent": 75,
    "current_step": "Calculating emissions",
    "step_number": 3,
    "total_steps": 4,
    "elapsed_seconds": 120.5,
    "estimated_total_seconds": 160.0,
    "details": {
        "records_processed": 300,
        "current_temp": 155.0
    }
}
```

### alarm.update
```json
{
    "alarm_id": "alarm:001",
    "agent_id": "agent:001",
    "severity": "CRITICAL",
    "message": "Pressure exceeded maximum",
    "parameter": "steam_pressure",
    "current_value": 9.5,
    "threshold_value": 9.0,
    "metadata": {
        "unit": "bar",
        "time_exceeded": "2 minutes"
    }
}
```

### metrics.update
```json
{
    "source_id": "agent:001",
    "metrics": {
        "temperature": 150.5,
        "pressure": 8.5,
        "efficiency": 0.92
    },
    "unit": "SI"
}
```

## Reconnection Support

The SSE system supports automatic client reconnection using the Last-Event-ID header:

```javascript
// Browser automatically handles this
const es = new EventSource('/api/v1/stream/events');

// On reconnection, the server will replay missed events
// using the Last-Event-ID header
```

The server replays up to 100 recent events (configurable) when a client reconnects with a `Last-Event-ID` header.

## Configuration

```python
from greenlang.infrastructure.api.sse_streaming import SSEStreamConfig

config = SSEStreamConfig(
    heartbeat_interval_seconds=30,      # Heartbeat every 30s
    client_timeout_seconds=300,         # Remove clients after 5 min inactivity
    max_clients_per_stream=1000,        # Max clients per stream
    max_queue_size=500,                 # Max events buffered per client
    max_event_history=100,              # Keep last 100 events for replay
    enable_heartbeat=True,              # Send heartbeat events
    api_prefix="/api/v1/stream"         # API prefix
)

manager = SSEStreamManager(config)
```

## Performance Considerations

### Optimal Settings

- **Heartbeat**: 30s (typical HTTP timeout is 60-90s)
- **Client Timeout**: 300s (5 minutes)
- **Queue Size**: 500 events (balances memory vs latency)
- **Max Clients**: 1000 per stream (scale horizontally for more)

### Load Testing

```bash
# Test concurrent connections
hey -n 10000 -c 100 -m GET http://localhost:8000/api/v1/stream/events

# Monitor metrics stream
curl -i http://localhost:8000/api/v1/stream/agents/agent:001
```

### Memory Usage

- Per client: ~2-5 KB baseline
- 1000 clients: ~2-5 MB
- Event queue: 500 events * 1 KB average = 500 KB per client

## Error Handling

### Client-Side

```javascript
eventSource.onerror = (event) => {
    if (event.target.readyState === EventSource.CONNECTING) {
        console.log('Reconnecting...');
    } else if (event.target.readyState === EventSource.CLOSED) {
        console.log('Connection closed, will attempt to reconnect');
    }
};
```

### Server-Side

```python
# Automatic client cleanup on timeout
await manager.stop()  # Graceful shutdown

# Manual disconnect
manager.close_stream("client_id")

# Monitor connection health
stats = manager.get_statistics()
print(f"Active clients: {stats['total_clients']}")
```

## Security Considerations

- **CORS**: SSE respects CORS headers
- **Authentication**: Use middleware for auth (JWT tokens)
- **Rate Limiting**: Implement per-client rate limits
- **Payload Validation**: Always validate event data

## Troubleshooting

### Client Not Receiving Events

1. Check `/api/v1/stream/status` for active streams
2. Verify event type filter matches emitted events
3. Check browser console for JavaScript errors

### Connection Dropping

1. Increase `heartbeat_interval_seconds` if proxy/firewall timing out
2. Check server logs for cleanup messages
3. Monitor client network connectivity

### High Memory Usage

1. Reduce `max_queue_size` if many slow clients
2. Reduce `max_event_history` if replay not needed
3. Implement client-side batching of events

## Examples

See `/greenlang/agents/process_heat/*/sse_integration_examples.py` for complete agent integration examples.
