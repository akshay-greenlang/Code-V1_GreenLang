# GreenLang Capabilities Security Model

## Overview

GreenLang implements a deny-by-default capability system to enforce strict security boundaries for pack execution. No pack can access network, filesystem, subprocess, or system clock resources unless explicitly granted through manifest declarations and policy approval.

## Threat Model

### 1. Network Threats

**Risks:**
- **Data Exfiltration**: Malicious packs could steal sensitive data and transmit it to external servers
- **SSRF (Server-Side Request Forgery)**: Packs could access internal services, metadata endpoints, or cloud provider APIs
- **Command & Control**: Compromised packs could establish persistent connections to attacker infrastructure
- **Resource Exhaustion**: Uncontrolled network access could lead to bandwidth consumption or API rate limit violations

**Mitigations:**
- Default deny all network access
- Explicit allowlist of permitted domains with glob patterns
- Block access to metadata endpoints (169.254.169.254, 169.254.170.2, 100.100.100.200)
- Block RFC1918 private network ranges unless explicitly allowed
- DNS resolution blocked when network capability not granted
- All HTTP client libraries patched (requests, urllib, http.client)

### 2. Filesystem Threats

**Risks:**
- **Credential Theft**: Reading SSH keys (~/.ssh/*), cloud credentials (~/.aws/*), tokens
- **System Tampering**: Modifying system files, installing backdoors, persistence mechanisms
- **Data Breach**: Accessing sensitive business data outside intended scope
- **Path Traversal**: Using ../ or symlinks to escape allowed directories
- **Information Disclosure**: Reading /etc/passwd, environment files, configuration

**Mitigations:**
- Default deny all filesystem access
- Read access limited to:
  - `${INPUT_DIR}`: Staged user input files
  - `${PACK_DATA_DIR}`: Pack's own bundled data
  - `${RUN_TMP}`: Ephemeral per-run workspace
- Write access limited to:
  - `${RUN_TMP}`: Temporary workspace (cleared after run)
- Path resolution to prevent traversal attacks
- Symlink target validation
- Block access to sensitive paths (/etc, /proc, /var/run, $HOME)

### 3. Subprocess Threats

**Risks:**
- **Command Injection**: Executing arbitrary system commands
- **Privilege Escalation**: Running SUID binaries or exploiting vulnerable programs
- **Container Escape**: Breaking out of sandboxed environment
- **Resource Exhaustion**: Fork bombs, CPU/memory consumption
- **Environment Variable Injection**: Manipulating PATH, LD_PRELOAD, PYTHONPATH

**Mitigations:**
- Default deny all subprocess execution
- Allowlist of permitted binaries (absolute paths only)
- Environment variable sanitization (minimal whitelist)
- Resource limits (CPU, memory, file descriptors)
- No shell interpretation (direct exec only)
- Close all file descriptors on exec
- Set no_new_privs flag (Linux)

### 4. Clock/Time Threats

**Risks:**
- **Non-Deterministic Behavior**: Results varying between runs due to time dependencies
- **Timing Attacks**: Exploiting time-based vulnerabilities
- **Rate Limit Bypass**: Manipulating timestamps to evade throttling
- **Cache Poisoning**: Time-based cache invalidation attacks
- **Audit Trail Manipulation**: Falsifying timestamps in logs

**Mitigations:**
- Default frozen logical clock (deterministic time)
- Monotonic counter for relative time measurements
- Seeded at pipeline start for reproducibility
- Real time access requires explicit capability grant
- All time functions patched (time.time, datetime.now, perf_counter)

## Capability Model

### Declaration

Capabilities are declared in the pack's `manifest.yaml`:

```yaml
name: my-pack
version: 1.0.0
capabilities:
  net:
    allow: false          # default false
    outbound:
      allowlist:          # only used if allow=true
        - https://api.company.com/*
        - https://*.gov/*
  fs:
    allow: false          # default false
    read:
      allowlist:
        - ${INPUT_DIR}/**
        - ${PACK_DATA_DIR}/**
    write:
      allowlist:
        - ${RUN_TMP}/**
  clock:
    allow: false          # default false (frozen time)
  subprocess:
    allow: false          # default false
    allowlist:
      - /usr/bin/exiftool
      - /usr/local/bin/ogr2ogr
```

### Enforcement Layers

1. **Manifest Validation** (Install Time)
   - Schema validation of capability declarations
   - Path pattern validation (no ../ escapes)
   - Binary path validation (absolute paths only)
   - Domain pattern validation

2. **Policy Gate** (Pre-Execution)
   - OPA policy evaluates requested capabilities
   - Organization policies can deny specific capabilities
   - User role-based capability limits
   - Audit log of capability requests

3. **Runtime Guard** (Execution Time)
   - Python import hooks and monkey-patching
   - Intercepts all capability-controlled operations
   - Validates against manifest allowlists
   - Raises CapabilityViolation on unauthorized access

4. **Audit Trail** (Post-Execution)
   - All capability checks logged
   - Violations recorded with full context
   - Successful grants tracked for compliance

## Security Principles

### 1. Principle of Least Privilege
Packs receive only the minimum capabilities required for their stated purpose. No capability is granted by default.

### 2. Defense in Depth
Multiple enforcement layers ensure a single bypass doesn't compromise security:
- Static analysis at install
- Policy evaluation pre-run
- Runtime enforcement
- Audit logging

### 3. Fail-Safe Defaults
All capabilities default to deny. Missing or malformed capability blocks result in zero privileges.

### 4. Complete Mediation
Every access to a controlled resource is checked. No bypass paths or special cases.

### 5. Separation of Privilege
Capability grants are separated by type (net, fs, subprocess, clock). Compromise of one doesn't grant others.

### 6. Psychological Acceptability
Clear error messages explain why operations failed and how to request capabilities properly.

## Implementation Details

### Runtime Guard Architecture

The runtime guard executes in a separate worker process to isolate monkey-patches from the parent orchestrator:

```
Orchestrator (Clean)
    ↓ spawn
Worker Process
    ↓ import guard.py
Guard Initialization
    ↓ monkey-patch
Pack Execution (Restricted)
```

### Patched Functions

**Network:**
- `socket.socket`
- `socket.connect`
- `http.client.HTTPConnection`
- `urllib.request.urlopen`
- `requests.Session.request`

**Filesystem:**
- `builtins.open`
- `os.open`
- `pathlib.Path.open`
- `os.remove`, `os.rename`
- `shutil.*` write operations

**Subprocess:**
- `subprocess.Popen`
- `subprocess.run`
- `os.system`
- `os.exec*`

**Clock:**
- `time.time`
- `time.perf_counter`
- `datetime.datetime.now`
- `time.monotonic`

## Migration Path

### Week 0 (Current Implementation)
- Python-level enforcement via monkey-patching
- Process isolation via subprocess
- Basic allowlist validation

### Q1 2025 (Planned)
- Linux seccomp filters for system call filtering
- Network namespaces for true network isolation
- cgroups for resource limits
- User namespaces for privilege separation

### Q2 2025 (Future)
- Full container/VM isolation
- Hardware security module integration
- Trusted execution environment support

## Developer Experience

### Local Development
Developers can use `--cap-override` flag for testing:
```bash
gl run pipeline.yaml --cap-override=net,fs,subprocess
```
This logs a security warning and is disabled in production.

### Capability Discovery
```bash
gl pack lint              # Shows required capabilities
gl pack capabilities      # Interactive capability wizard
```

### Error Messages
```
CapabilityViolation: Network access denied
  Pack requested: https://api.external.com
  Capability required: net

  To fix:
  1. Add to manifest.yaml:
     capabilities:
       net:
         allow: true
         outbound:
           allowlist:
             - https://api.external.com/*

  2. Request approval from security team
```

## Compliance & Audit

### Audit Events
All capability checks generate audit events:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event": "capability_check",
  "pack": "emissions-calculator",
  "capability": "net",
  "target": "https://api.emissions.gov",
  "result": "granted",
  "user": "user@company.com",
  "run_id": "run-12345"
}
```

### Compliance Reports
- Daily summary of capability usage
- Anomaly detection for unusual patterns
- Policy violation alerts
- Capability drift analysis

## Security Considerations

### Known Limitations (Week 0)
1. Python-level enforcement can be bypassed by:
   - Direct syscalls via ctypes
   - Binary extensions (.so files)
   - Memory manipulation

2. Process isolation limitations:
   - Shared filesystem (mitigated by path checks)
   - Network namespace not enforced
   - No kernel-level enforcement

These limitations are acceptable for Week 0 as they require sophisticated attacks and will be addressed in Q1 2025 with kernel-level enforcement.

### Security Boundaries
- Trust boundary is at the worker process level
- Parent orchestrator remains uncontaminated
- Each run gets fresh worker (no state leakage)

### Incident Response
1. Capability violation triggers immediate termination
2. Audit logs preserved for forensics
3. Pack automatically quarantined
4. Security team notified for repeated violations