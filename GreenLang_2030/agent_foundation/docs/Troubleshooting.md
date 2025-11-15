# Troubleshooting Guide

## Common Issues and Solutions for GreenLang Agents

Quick reference guide for resolving common issues with GreenLang Agent Foundation.

---

## Quick Diagnosis

```bash
# Run diagnostics
greenlang diagnose

# Check system health
greenlang health-check --verbose

# View logs
greenlang logs --tail 100 --level error
```

---

## Common Issues

### Installation Issues

#### Issue: ImportError after installation

**Symptoms:**
```python
ImportError: No module named 'greenlang'
```

**Solution:**
```bash
# Verify installation
pip list | grep greenlang

# Reinstall
pip uninstall greenlang-ai
pip install greenlang-ai --upgrade

# Check Python path
python -c "import sys; print(sys.path)"
```

---

### Agent Initialization Issues

#### Issue: Agent fails to initialize

**Symptoms:**
```
InitializationError: Failed to setup resources
```

**Solutions:**
1. Check configuration:
```python
config.validate()  # Validate before use
```

2. Verify dependencies:
```bash
# Check Redis
redis-cli ping

# Check PostgreSQL
psql $DATABASE_URL -c "SELECT 1"
```

3. Check permissions:
```bash
chmod +x agent_startup.sh
chown greenlang:greenlang /app/data
```

---

### Memory Issues

#### Issue: Out of Memory Error

**Symptoms:**
```
MemoryError: Cannot allocate memory
```

**Solutions:**
1. Increase memory limits:
```yaml
resources:
  limits:
    memory: "8Gi"  # Increase from 4Gi
```

2. Enable memory pruning:
```python
memory_config = {
    'auto_prune': True,
    'prune_threshold': 0.8,  # Prune at 80% capacity
    'prune_strategy': 'least_recently_used'
}
```

3. Optimize memory usage:
```python
# Use generators instead of lists
for item in agent.process_stream(data):
    yield item

# Clear caches periodically
agent.clear_cache()
```

---

### LLM Issues

#### Issue: LLM API Rate Limit

**Symptoms:**
```
RateLimitError: Rate limit exceeded
```

**Solutions:**
1. Implement backoff:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_llm(prompt):
    return await llm.generate(prompt)
```

2. Use fallback providers:
```python
llm = LLMClient(
    primary='openai',
    fallback=['anthropic', 'google']
)
```

---

### Performance Issues

#### Issue: Slow agent response times

**Diagnostic Steps:**
```python
# Enable profiling
agent.enable_profiling()

# Check bottlenecks
stats = agent.get_performance_stats()
print(f"LLM time: {stats['llm_time']}ms")
print(f"Memory time: {stats['memory_time']}ms")
print(f"Processing time: {stats['processing_time']}ms")
```

**Solutions:**
1. Enable caching:
```python
agent.enable_cache(ttl=3600)
```

2. Optimize batch processing:
```python
results = await agent.process_batch(items, batch_size=50)
```

3. Use async operations:
```python
# Bad
for item in items:
    result = await process(item)

# Good
results = await asyncio.gather(*[process(item) for item in items])
```

---

### Communication Issues

#### Issue: Agent messages not received

**Diagnostic:**
```bash
# Check message bus
kafka-console-consumer --bootstrap-server localhost:9092 --topic agent.messages

# Check agent status
greenlang status agent-id
```

**Solutions:**
1. Verify network connectivity:
```bash
telnet kafka-broker 9092
```

2. Check firewall rules:
```bash
# Allow Kafka ports
sudo ufw allow 9092/tcp
```

3. Validate message format:
```python
message.validate()  # Ensure message is valid
```

---

### Database Issues

#### Issue: Connection pool exhausted

**Symptoms:**
```
PoolTimeout: Connection pool exhausted
```

**Solutions:**
1. Increase pool size:
```python
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
```

2. Close connections properly:
```python
async with agent.db.connection() as conn:
    # Use connection
    pass  # Auto-closes
```

3. Monitor connections:
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'greenlang';
```

---

### Kubernetes Issues

#### Issue: Pods crashing (CrashLoopBackOff)

**Diagnostic:**
```bash
kubectl logs greenlang-agent-xxx -n greenlang
kubectl describe pod greenlang-agent-xxx -n greenlang
```

**Solutions:**
1. Check resource limits:
```bash
kubectl top pod greenlang-agent-xxx -n greenlang
```

2. Verify config maps:
```bash
kubectl get configmap agent-config -n greenlang -o yaml
```

3. Check secrets:
```bash
kubectl get secret greenlang-secrets -n greenlang
```

---

## Error Codes Reference

| Code | Error | Solution |
|------|-------|----------|
| E1001 | ValidationError | Check input format |
| E1002 | TimeoutError | Increase timeout setting |
| E1003 | MemoryError | Increase memory or prune |
| E1004 | LLMError | Check API keys and quotas |
| E1005 | DatabaseError | Check connection and credentials |
| E1006 | NetworkError | Check connectivity and firewalls |

---

## Support Resources

### Get Help

- **Documentation**: https://docs.greenlang.ai
- **GitHub Issues**: https://github.com/greenlang/agent-foundation/issues
- **Discord**: https://discord.gg/greenlang
- **Email**: support@greenlang.ai

### Submit Bug Reports

Include the following:
1. GreenLang version (`greenlang --version`)
2. Python version
3. Operating system
4. Complete error traceback
5. Minimal reproducible example
6. Configuration (sanitized)

---

**Last Updated**: November 2024