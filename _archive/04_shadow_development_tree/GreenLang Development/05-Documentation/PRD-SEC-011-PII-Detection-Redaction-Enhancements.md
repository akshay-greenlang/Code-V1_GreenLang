# PRD-SEC-011: PII Detection/Redaction Enhancements

**Version:** 1.0
**Status:** APPROVED
**Author:** Security & Compliance Team
**Created:** 2026-02-06
**Last Updated:** 2026-02-06

---

## 1. Executive Summary

### 1.1 Background

GreenLang already has comprehensive PII detection and redaction capabilities implemented across multiple modules:

| Module | Location | Lines | Capabilities |
|--------|----------|-------|--------------|
| PII Redaction Agent | `greenlang/agents/foundation/pii_redaction.py` | 1,325 | 19 PII types, 6 redaction strategies, tokenization |
| PII Scanner | `greenlang/infrastructure/security_scanning/pii_scanner.py` | 796 | Regex + confidence scoring |
| PII ML Detection | `greenlang/infrastructure/security_scanning/pii_ml.py` | ~400 | Microsoft Presidio NER |
| PII Alert Router | `greenlang/infrastructure/security_scanning/pii_alerts.py` | ~300 | Classification-based routing |
| Log Redaction | `greenlang/infrastructure/logging/redaction.py` | 258 | Structlog processor |
| GDPR Data Discovery | `greenlang/infrastructure/compliance_automation/gdpr/data_discovery.py` | 827 | Cross-system scanning |

### 1.2 Purpose

SEC-011 enhances the existing PII infrastructure with:
1. **Real-time enforcement** - Block/quarantine PII, not just alert
2. **Secure encryption** - AES-256-GCM for token vault (replace XOR)
3. **Streaming integration** - Kafka/Kinesis real-time scanning
4. **Allowlist/whitelist** - Exclude test data and known safe patterns
5. **Automated remediation** - Auto-delete/purge detected PII
6. **Prometheus metrics** - Full observability for PII operations
7. **Multi-tenant isolation** - Per-tenant token vaults with encryption
8. **Unified PII Service** - Single API layer over all PII modules

### 1.3 Business Value

| Benefit | Impact |
|---------|--------|
| GDPR/CCPA compliance | Reduce breach risk, avoid €20M+ fines |
| Real-time protection | Prevent PII leakage before it occurs |
| Audit readiness | Comprehensive metrics and logging |
| Operational efficiency | Automated remediation reduces manual effort |
| Enterprise sales | Required for regulated customers |

### 1.4 Scope

| Component | Priority | Files | Lines (Est.) |
|-----------|----------|-------|--------------|
| Secure Token Vault | P0 | 3 | 800 |
| Enforcement Engine | P0 | 4 | 1,200 |
| Streaming Scanner | P1 | 4 | 1,000 |
| Allowlist Manager | P1 | 2 | 400 |
| Auto-Remediation | P1 | 3 | 800 |
| Metrics & Dashboard | P1 | 3 | 600 |
| Unified PII Service | P0 | 5 | 1,200 |
| Multi-Tenant Isolation | P0 | 2 | 500 |
| Tests | P2 | 10 | 2,500 |
| **Total** | - | **~36** | **~9,000** |

---

## 2. Component 1: Secure Token Vault

### 2.1 Problem Statement

The current PII redaction agent uses XOR encryption for the token vault:
```python
# Current implementation (pii_redaction.py line ~180)
encrypted_value = base64.b64encode(bytes([a ^ ord('K') for a in original]))  # XOR is NOT secure
```

This is cryptographically weak and unacceptable for production PII storage.

### 2.2 Solution

Replace with AES-256-GCM encryption integrated with the existing encryption service (SEC-003).

### 2.3 Module Structure

```
greenlang/infrastructure/pii_service/
├── __init__.py
├── secure_vault.py           # AES-256-GCM token vault
├── vault_config.py           # Vault configuration
└── vault_migration.py        # Migration from XOR to AES
```

### 2.4 Key Classes

```python
class SecureTokenVault:
    """AES-256-GCM encrypted token vault with tenant isolation"""

    def __init__(
        self,
        encryption_service: EncryptionService,  # From SEC-003
        tenant_id: str,
        config: VaultConfig
    ):
        self._encryption = encryption_service
        self._tenant_id = tenant_id
        self._tokens: Dict[str, EncryptedTokenEntry] = {}

    async def tokenize(
        self,
        value: str,
        pii_type: PIIType,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create encrypted, reversible token"""
        token_id = self._generate_token_id(value)

        # Encrypt with AES-256-GCM via SEC-003 encryption service
        encrypted_value = await self._encryption.encrypt(
            plaintext=value.encode(),
            context={"tenant_id": self._tenant_id, "pii_type": pii_type.value}
        )

        entry = EncryptedTokenEntry(
            token_id=token_id,
            pii_type=pii_type,
            original_hash=hashlib.sha256(value.encode()).hexdigest(),
            encrypted_value=encrypted_value,
            tenant_id=self._tenant_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=self._config.token_ttl_days),
            metadata=metadata or {}
        )

        self._tokens[token_id] = entry
        await self._persist_token(entry)

        return f"[TOKEN:{token_id}]"

    async def detokenize(
        self,
        token: str,
        requester_tenant_id: str,
        requester_user_id: str
    ) -> Optional[str]:
        """Decrypt token with authorization check"""
        token_id = self._extract_token_id(token)
        entry = await self._get_token(token_id)

        if not entry:
            raise TokenNotFoundError(token_id)

        # Tenant isolation check
        if entry.tenant_id != requester_tenant_id:
            await self._audit_access_denied(token_id, requester_tenant_id)
            raise UnauthorizedAccessError(token_id, requester_tenant_id)

        # Expiration check
        if entry.expires_at < datetime.utcnow():
            raise TokenExpiredError(token_id)

        # Decrypt
        plaintext = await self._encryption.decrypt(
            ciphertext=entry.encrypted_value,
            context={"tenant_id": entry.tenant_id, "pii_type": entry.pii_type.value}
        )

        # Audit log
        await self._audit_detokenization(token_id, requester_user_id)

        return plaintext.decode()

class VaultConfig(BaseSettings):
    """Token vault configuration"""
    token_ttl_days: int = 90
    max_tokens_per_tenant: int = 1_000_000
    enable_persistence: bool = True
    persistence_backend: str = "postgresql"  # postgresql, redis, s3
    encryption_key_id: str = "pii-vault-key"  # KMS key alias
```

---

## 3. Component 2: Enforcement Engine

### 3.1 Problem Statement

Current PII detection is alert-only. There's no mechanism to:
- Block requests containing PII
- Quarantine suspicious content
- Prevent PII from reaching storage

### 3.2 Solution

Add an enforcement engine that can block, quarantine, or transform PII in real-time.

### 3.3 Module Structure

```
greenlang/infrastructure/pii_service/
├── enforcement/
│   ├── __init__.py
│   ├── engine.py             # Core enforcement logic
│   ├── policies.py           # Enforcement policies
│   ├── middleware.py         # FastAPI middleware
│   └── actions.py            # Enforcement actions
```

### 3.4 Key Classes

```python
class EnforcementAction(str, Enum):
    """Actions when PII is detected"""
    ALLOW = "allow"           # Log only, allow through
    REDACT = "redact"         # Redact and allow
    BLOCK = "block"           # Block the request
    QUARANTINE = "quarantine" # Store for review, block
    TRANSFORM = "transform"   # Apply transformation (tokenize, hash)

class EnforcementPolicy(BaseModel):
    """Policy for a specific PII type"""
    pii_type: PIIType
    action: EnforcementAction
    min_confidence: float = 0.8
    contexts: List[str] = ["*"]  # Which contexts to apply (api, storage, logging)
    notify: bool = True
    quarantine_ttl_hours: int = 72

class PIIEnforcementEngine:
    """Real-time PII enforcement"""

    DEFAULT_POLICIES = {
        PIIType.SSN: EnforcementPolicy(pii_type=PIIType.SSN, action=EnforcementAction.BLOCK),
        PIIType.CREDIT_CARD: EnforcementPolicy(pii_type=PIIType.CREDIT_CARD, action=EnforcementAction.BLOCK),
        PIIType.PASSWORD: EnforcementPolicy(pii_type=PIIType.PASSWORD, action=EnforcementAction.BLOCK),
        PIIType.API_KEY: EnforcementPolicy(pii_type=PIIType.API_KEY, action=EnforcementAction.REDACT),
        PIIType.EMAIL: EnforcementPolicy(pii_type=PIIType.EMAIL, action=EnforcementAction.ALLOW),
        PIIType.PHONE: EnforcementPolicy(pii_type=PIIType.PHONE, action=EnforcementAction.ALLOW),
        # ...
    }

    async def enforce(
        self,
        content: str,
        context: EnforcementContext
    ) -> EnforcementResult:
        """
        Scan content and apply enforcement policies.

        Returns:
            EnforcementResult with action taken, modified content (if any),
            detections, and metadata.
        """
        # Detect PII
        detections = await self._scanner.scan(content)

        # Apply policies
        actions_taken = []
        modified_content = content
        blocked = False

        for detection in detections:
            policy = self._get_policy(detection.pii_type)

            if detection.confidence < policy.min_confidence:
                continue

            if policy.action == EnforcementAction.BLOCK:
                blocked = True
                actions_taken.append(ActionTaken(
                    detection=detection,
                    action=EnforcementAction.BLOCK,
                    reason=f"Blocked {detection.pii_type.value} (policy)"
                ))

            elif policy.action == EnforcementAction.REDACT:
                modified_content = self._redact(modified_content, detection)
                actions_taken.append(ActionTaken(
                    detection=detection,
                    action=EnforcementAction.REDACT,
                    reason=f"Redacted {detection.pii_type.value}"
                ))

            elif policy.action == EnforcementAction.QUARANTINE:
                await self._quarantine(content, detection, context)
                blocked = True
                actions_taken.append(ActionTaken(
                    detection=detection,
                    action=EnforcementAction.QUARANTINE,
                    reason=f"Quarantined for review"
                ))

            # Notify if configured
            if policy.notify:
                await self._notify(detection, policy.action, context)

        return EnforcementResult(
            blocked=blocked,
            original_content=content,
            modified_content=modified_content if not blocked else None,
            detections=detections,
            actions_taken=actions_taken,
            context=context
        )

class PIIEnforcementMiddleware:
    """FastAPI middleware for PII enforcement on request/response"""

    def __init__(
        self,
        engine: PIIEnforcementEngine,
        scan_requests: bool = True,
        scan_responses: bool = True,
        exclude_paths: List[str] = ["/health", "/metrics"]
    ):
        self._engine = engine
        self._scan_requests = scan_requests
        self._scan_responses = scan_responses
        self._exclude_paths = exclude_paths

    async def __call__(self, request: Request, call_next):
        if request.url.path in self._exclude_paths:
            return await call_next(request)

        # Scan request body
        if self._scan_requests and request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            result = await self._engine.enforce(
                content=body.decode(),
                context=EnforcementContext(
                    context_type="api_request",
                    path=request.url.path,
                    method=request.method,
                    tenant_id=request.state.tenant_id
                )
            )

            if result.blocked:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "PII_DETECTED",
                        "message": "Request blocked: sensitive data detected",
                        "detections": [d.pii_type.value for d in result.detections]
                    }
                )

        response = await call_next(request)

        # Scan response (for data exfiltration prevention)
        if self._scan_responses:
            # ... scan response body
            pass

        return response
```

---

## 4. Component 3: Streaming Scanner

### 4.1 Problem Statement

Current PII scanning is batch-oriented. No support for real-time streaming platforms like Kafka or Kinesis.

### 4.2 Solution

Add streaming consumers that scan messages in real-time.

### 4.3 Module Structure

```
greenlang/infrastructure/pii_service/
├── streaming/
│   ├── __init__.py
│   ├── kafka_scanner.py      # Kafka consumer/producer
│   ├── kinesis_scanner.py    # Kinesis consumer
│   ├── stream_processor.py   # Common processing logic
│   └── config.py             # Streaming configuration
```

### 4.4 Key Classes

```python
class KafkaPIIScanner:
    """Real-time PII scanning for Kafka streams"""

    def __init__(
        self,
        bootstrap_servers: List[str],
        input_topics: List[str],
        output_topic: str,
        dlq_topic: str,  # Dead letter queue for blocked messages
        enforcement_engine: PIIEnforcementEngine,
        consumer_group: str = "pii-scanner"
    ):
        self._consumer = AIOKafkaConsumer(
            *input_topics,
            bootstrap_servers=bootstrap_servers,
            group_id=consumer_group
        )
        self._producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
        self._enforcement = enforcement_engine
        self._output_topic = output_topic
        self._dlq_topic = dlq_topic

    async def start(self):
        """Start streaming scanner"""
        await self._consumer.start()
        await self._producer.start()

        try:
            async for message in self._consumer:
                await self._process_message(message)
        finally:
            await self._consumer.stop()
            await self._producer.stop()

    async def _process_message(self, message: ConsumerRecord):
        """Process a single Kafka message"""
        try:
            content = message.value.decode()

            result = await self._enforcement.enforce(
                content=content,
                context=EnforcementContext(
                    context_type="kafka_stream",
                    topic=message.topic,
                    partition=message.partition,
                    offset=message.offset
                )
            )

            if result.blocked:
                # Send to dead letter queue
                await self._producer.send(
                    self._dlq_topic,
                    key=message.key,
                    value=json.dumps({
                        "original_message": content,
                        "detections": [asdict(d) for d in result.detections],
                        "reason": "PII_BLOCKED",
                        "timestamp": datetime.utcnow().isoformat()
                    }).encode()
                )
                metrics.pii_stream_blocked_total.inc()
            else:
                # Send modified content to output topic
                await self._producer.send(
                    self._output_topic,
                    key=message.key,
                    value=result.modified_content.encode()
                )
                metrics.pii_stream_processed_total.inc()

        except Exception as e:
            metrics.pii_stream_errors_total.inc()
            logger.error(f"Error processing message: {e}")
```

---

## 5. Component 4: Allowlist Manager

### 5.1 Problem Statement

Cannot exclude known safe patterns (test data, example.com emails) from PII detection, leading to false positives.

### 5.2 Solution

Add configurable allowlists per PII type with pattern matching.

### 5.3 Module Structure

```
greenlang/infrastructure/pii_service/
├── allowlist/
│   ├── __init__.py
│   ├── manager.py            # Allowlist management
│   └── patterns.py           # Pattern definitions
```

### 5.4 Key Classes

```python
class AllowlistEntry(BaseModel):
    """Single allowlist entry"""
    id: UUID = Field(default_factory=uuid4)
    pii_type: PIIType
    pattern: str                    # Regex or exact match
    pattern_type: str = "regex"     # regex, exact, prefix, suffix, contains
    reason: str
    created_by: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    tenant_id: Optional[str] = None  # None = global

class AllowlistManager:
    """Manage PII detection allowlists"""

    DEFAULT_ALLOWLISTS = {
        PIIType.EMAIL: [
            AllowlistEntry(
                pii_type=PIIType.EMAIL,
                pattern=r".*@example\.(com|org|net)$",
                pattern_type="regex",
                reason="RFC 2606 reserved domain"
            ),
            AllowlistEntry(
                pii_type=PIIType.EMAIL,
                pattern=r".*@test\.(com|org|net)$",
                pattern_type="regex",
                reason="Test domain"
            ),
            AllowlistEntry(
                pii_type=PIIType.EMAIL,
                pattern=r"noreply@.*",
                pattern_type="regex",
                reason="No-reply addresses are not personal"
            ),
        ],
        PIIType.PHONE: [
            AllowlistEntry(
                pii_type=PIIType.PHONE,
                pattern=r"555-\d{4}",
                pattern_type="regex",
                reason="US fictional phone numbers (555)"
            ),
        ],
        PIIType.CREDIT_CARD: [
            AllowlistEntry(
                pii_type=PIIType.CREDIT_CARD,
                pattern="4111111111111111",
                pattern_type="exact",
                reason="Stripe test card"
            ),
            AllowlistEntry(
                pii_type=PIIType.CREDIT_CARD,
                pattern="4242424242424242",
                pattern_type="exact",
                reason="Stripe test card"
            ),
        ],
        PIIType.SSN: [
            AllowlistEntry(
                pii_type=PIIType.SSN,
                pattern=r"000-00-0000",
                pattern_type="exact",
                reason="Invalid SSN placeholder"
            ),
        ],
    }

    async def is_allowed(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: Optional[str] = None
    ) -> Tuple[bool, Optional[AllowlistEntry]]:
        """Check if a value is in the allowlist"""
        entries = await self._get_entries(pii_type, tenant_id)

        for entry in entries:
            if self._matches(value, entry):
                return True, entry

        return False, None

    def _matches(self, value: str, entry: AllowlistEntry) -> bool:
        """Check if value matches allowlist entry"""
        if entry.pattern_type == "exact":
            return value == entry.pattern
        elif entry.pattern_type == "regex":
            return bool(re.match(entry.pattern, value))
        elif entry.pattern_type == "prefix":
            return value.startswith(entry.pattern)
        elif entry.pattern_type == "suffix":
            return value.endswith(entry.pattern)
        elif entry.pattern_type == "contains":
            return entry.pattern in value
        return False
```

---

## 6. Component 5: Auto-Remediation

### 6.1 Problem Statement

Detected PII requires manual follow-up. No automated deletion/purge capabilities.

### 6.2 Solution

Add scheduled jobs for auto-remediation based on policies.

### 6.3 Module Structure

```
greenlang/infrastructure/pii_service/
├── remediation/
│   ├── __init__.py
│   ├── engine.py             # Remediation engine
│   ├── policies.py           # Remediation policies
│   └── jobs.py               # Scheduled jobs
```

### 6.4 Key Classes

```python
class RemediationPolicy(BaseModel):
    """Auto-remediation policy"""
    pii_type: PIIType
    action: str  # delete, anonymize, archive, notify_only
    delay_hours: int = 72  # Grace period before action
    requires_approval: bool = False
    notify_on_action: bool = True

class PIIRemediationEngine:
    """Automated PII remediation"""

    async def process_pending_remediations(self):
        """Process all pending remediation items"""
        pending = await self._get_pending_items()

        for item in pending:
            policy = self._get_policy(item.pii_type)

            # Check grace period
            if datetime.utcnow() < item.detected_at + timedelta(hours=policy.delay_hours):
                continue

            # Check approval if required
            if policy.requires_approval and not item.approved:
                continue

            # Execute remediation
            if policy.action == "delete":
                await self._delete_pii(item)
            elif policy.action == "anonymize":
                await self._anonymize_pii(item)
            elif policy.action == "archive":
                await self._archive_pii(item)

            # Notify
            if policy.notify_on_action:
                await self._notify_remediation(item, policy.action)

            # Update status
            await self._mark_remediated(item.id)

    async def _delete_pii(self, item: PIIRemediationItem):
        """Delete PII from source"""
        if item.source_type == "postgresql":
            await self._delete_from_postgresql(item)
        elif item.source_type == "s3":
            await self._delete_from_s3(item)
        elif item.source_type == "redis":
            await self._delete_from_redis(item)

        # Generate deletion certificate
        await self._generate_deletion_certificate(item)
```

---

## 7. Component 6: Prometheus Metrics

### 7.1 Metrics Definition

```python
# greenlang/infrastructure/pii_service/metrics.py

from prometheus_client import Counter, Gauge, Histogram

# Detection metrics
pii_detections_total = Counter(
    "gl_pii_detections_total",
    "Total PII detections",
    ["pii_type", "source", "confidence_level"]
)

pii_detection_latency_seconds = Histogram(
    "gl_pii_detection_latency_seconds",
    "PII detection latency",
    ["scanner_type"],  # regex, ml, hybrid
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Enforcement metrics
pii_enforcement_actions_total = Counter(
    "gl_pii_enforcement_actions_total",
    "Total enforcement actions",
    ["action", "pii_type", "context"]
)

pii_blocked_requests_total = Counter(
    "gl_pii_blocked_requests_total",
    "Total blocked requests due to PII",
    ["pii_type", "endpoint"]
)

# Token vault metrics
pii_tokens_total = Gauge(
    "gl_pii_tokens_total",
    "Total tokens in vault",
    ["tenant_id", "pii_type"]
)

pii_tokenization_total = Counter(
    "gl_pii_tokenization_total",
    "Total tokenization operations",
    ["pii_type", "status"]  # success, failed
)

pii_detokenization_total = Counter(
    "gl_pii_detokenization_total",
    "Total detokenization operations",
    ["pii_type", "status"]  # success, failed, denied, expired
)

# Streaming metrics
pii_stream_processed_total = Counter(
    "gl_pii_stream_processed_total",
    "Total stream messages processed",
    ["topic", "action"]
)

pii_stream_blocked_total = Counter(
    "gl_pii_stream_blocked_total",
    "Total stream messages blocked",
    ["topic", "pii_type"]
)

# Remediation metrics
pii_remediation_total = Counter(
    "gl_pii_remediation_total",
    "Total remediation actions",
    ["action", "pii_type", "source"]
)

pii_quarantine_items = Gauge(
    "gl_pii_quarantine_items",
    "Current items in quarantine",
    ["pii_type"]
)

# Allowlist metrics
pii_allowlist_matches_total = Counter(
    "gl_pii_allowlist_matches_total",
    "Total allowlist matches (false positives avoided)",
    ["pii_type", "pattern"]
)
```

---

## 8. Component 7: Unified PII Service

### 8.1 Module Structure

```
greenlang/infrastructure/pii_service/
├── __init__.py               # Public API
├── service.py                # Unified service facade
├── config.py                 # Service configuration
├── secure_vault.py           # Component 1
├── enforcement/              # Component 2
├── streaming/                # Component 3
├── allowlist/                # Component 4
├── remediation/              # Component 5
├── metrics.py                # Component 6
├── api/
│   ├── __init__.py
│   └── pii_routes.py         # REST API endpoints
└── models.py                 # Shared models
```

### 8.2 Unified Service

```python
class PIIService:
    """Unified PII detection, redaction, and management service"""

    def __init__(
        self,
        config: PIIServiceConfig,
        encryption_service: EncryptionService,
        audit_service: AuditService
    ):
        self._config = config
        self._scanner = PIIScanner(config.scanner_config)
        self._ml_scanner = PIIMLScanner(config.ml_config) if config.enable_ml else None
        self._vault = SecureTokenVault(encryption_service, config.vault_config)
        self._enforcement = PIIEnforcementEngine(config.enforcement_config)
        self._allowlist = AllowlistManager(config.allowlist_config)
        self._remediation = PIIRemediationEngine(config.remediation_config)
        self._audit = audit_service

    async def detect(
        self,
        content: str,
        options: Optional[DetectionOptions] = None
    ) -> List[PIIDetection]:
        """Detect PII in content"""
        options = options or DetectionOptions()

        # Regex detection
        detections = await self._scanner.scan(content)

        # ML detection (if enabled)
        if self._ml_scanner and options.use_ml:
            ml_detections = await self._ml_scanner.scan(content)
            detections = self._merge_detections(detections, ml_detections)

        # Filter by allowlist
        if options.apply_allowlist:
            detections = await self._filter_allowlisted(detections, options.tenant_id)

        # Record metrics
        for d in detections:
            metrics.pii_detections_total.labels(
                pii_type=d.pii_type.value,
                source=options.source,
                confidence_level=self._confidence_level(d.confidence)
            ).inc()

        return detections

    async def redact(
        self,
        content: str,
        options: Optional[RedactionOptions] = None
    ) -> RedactionResult:
        """Detect and redact PII"""
        options = options or RedactionOptions()

        detections = await self.detect(content, options.detection_options)

        redacted_content = content
        for detection in sorted(detections, key=lambda d: d.start, reverse=True):
            policy = self._get_redaction_policy(detection.pii_type)
            redacted_content = self._apply_redaction(
                redacted_content,
                detection,
                policy
            )

        return RedactionResult(
            original_content=content,
            redacted_content=redacted_content,
            detections=detections
        )

    async def enforce(
        self,
        content: str,
        context: EnforcementContext
    ) -> EnforcementResult:
        """Enforce PII policies (block/allow/redact)"""
        return await self._enforcement.enforce(content, context)

    async def tokenize(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: str
    ) -> str:
        """Create reversible token"""
        return await self._vault.tokenize(value, pii_type, tenant_id)

    async def detokenize(
        self,
        token: str,
        tenant_id: str,
        user_id: str
    ) -> str:
        """Retrieve original value from token"""
        return await self._vault.detokenize(token, tenant_id, user_id)
```

---

## 9. API Endpoints

### 9.1 PII Service Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pii/detect` | Detect PII in content |
| POST | `/api/v1/pii/redact` | Redact PII from content |
| POST | `/api/v1/pii/tokenize` | Create reversible token |
| POST | `/api/v1/pii/detokenize` | Retrieve original from token |
| GET | `/api/v1/pii/policies` | List enforcement policies |
| PUT | `/api/v1/pii/policies/{pii_type}` | Update policy |
| GET | `/api/v1/pii/allowlist` | List allowlist entries |
| POST | `/api/v1/pii/allowlist` | Add allowlist entry |
| DELETE | `/api/v1/pii/allowlist/{id}` | Remove allowlist entry |
| GET | `/api/v1/pii/quarantine` | List quarantined items |
| POST | `/api/v1/pii/quarantine/{id}/release` | Release from quarantine |
| POST | `/api/v1/pii/quarantine/{id}/delete` | Delete quarantined item |
| GET | `/api/v1/pii/metrics` | Get PII metrics |

---

## 10. Database Schema

### 10.1 Migration: V018__pii_service.sql

```sql
-- PII Service Schema
CREATE SCHEMA IF NOT EXISTS pii_service;

-- Token vault
CREATE TABLE pii_service.token_vault (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_id VARCHAR(64) NOT NULL UNIQUE,
    pii_type VARCHAR(30) NOT NULL,
    original_hash VARCHAR(64) NOT NULL,
    encrypted_value BYTEA NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX idx_token_vault_tenant ON pii_service.token_vault(tenant_id);
CREATE INDEX idx_token_vault_expires ON pii_service.token_vault(expires_at);

-- Allowlist
CREATE TABLE pii_service.allowlist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    pattern TEXT NOT NULL,
    pattern_type VARCHAR(20) NOT NULL,
    reason TEXT,
    tenant_id VARCHAR(50),  -- NULL = global
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    enabled BOOLEAN DEFAULT true
);

CREATE INDEX idx_allowlist_pii_type ON pii_service.allowlist(pii_type);
CREATE INDEX idx_allowlist_tenant ON pii_service.allowlist(tenant_id);

-- Quarantine
CREATE TABLE pii_service.quarantine (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(64) NOT NULL,
    pii_type VARCHAR(30) NOT NULL,
    detection_confidence DECIMAL(3,2) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT,
    tenant_id VARCHAR(50) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, released, deleted
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX idx_quarantine_status ON pii_service.quarantine(status);
CREATE INDEX idx_quarantine_tenant ON pii_service.quarantine(tenant_id);

-- Remediation log
CREATE TABLE pii_service.remediation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pii_type VARCHAR(30) NOT NULL,
    action VARCHAR(20) NOT NULL,
    source_type VARCHAR(30) NOT NULL,
    source_location TEXT,
    tenant_id VARCHAR(50) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    executed_by VARCHAR(50),  -- 'system' or user_id
    deletion_certificate_id UUID,
    metadata JSONB
);

SELECT create_hypertable('pii_service.remediation_log', 'executed_at');

-- Audit log
CREATE TABLE pii_service.audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action VARCHAR(30) NOT NULL,
    pii_type VARCHAR(30),
    tenant_id VARCHAR(50) NOT NULL,
    user_id UUID,
    content_hash VARCHAR(64),
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('pii_service.audit_log', 'created_at');

-- Permissions
INSERT INTO security.permissions (name, resource, action, description) VALUES
    ('pii:detect', 'pii', 'detect', 'Detect PII in content'),
    ('pii:redact', 'pii', 'redact', 'Redact PII from content'),
    ('pii:tokenize', 'pii', 'tokenize', 'Create PII tokens'),
    ('pii:detokenize', 'pii', 'detokenize', 'Retrieve original from tokens'),
    ('pii:policies:read', 'pii_policies', 'read', 'View PII policies'),
    ('pii:policies:write', 'pii_policies', 'write', 'Modify PII policies'),
    ('pii:allowlist:read', 'pii_allowlist', 'read', 'View PII allowlist'),
    ('pii:allowlist:write', 'pii_allowlist', 'write', 'Modify PII allowlist'),
    ('pii:quarantine:read', 'pii_quarantine', 'read', 'View quarantine'),
    ('pii:quarantine:manage', 'pii_quarantine', 'manage', 'Manage quarantine'),
    ('pii:audit:read', 'pii_audit', 'read', 'View PII audit logs');
```

---

## 11. Monitoring & Alerting

### 11.1 Grafana Dashboard

- PII detections by type (time series)
- Enforcement actions (allow/block/redact)
- Token vault usage by tenant
- Quarantine items pending review
- Streaming throughput and blocked messages
- Detection latency (P50/P95/P99)
- Remediation actions timeline

### 11.2 Prometheus Alerts

```yaml
groups:
  - name: pii_service_alerts
    rules:
      - alert: PIIHighBlockRate
        expr: rate(gl_pii_blocked_requests_total[5m]) > 10
        labels:
          severity: warning
        annotations:
          summary: "High PII block rate detected"

      - alert: PIIQuarantineBacklog
        expr: gl_pii_quarantine_items > 100
        labels:
          severity: warning
        annotations:
          summary: "Quarantine backlog exceeds 100 items"

      - alert: PIITokenVaultFull
        expr: gl_pii_tokens_total > 900000
        labels:
          severity: critical
        annotations:
          summary: "Token vault approaching capacity"

      - alert: PIIStreamProcessingLag
        expr: rate(gl_pii_stream_processed_total[1m]) < 100
        labels:
          severity: warning
        annotations:
          summary: "PII stream processing throughput degraded"
```

---

## 12. Implementation Phases

### Phase 1 (P0) - Core
- [ ] Secure Token Vault (AES-256-GCM)
- [ ] Unified PII Service
- [ ] Multi-tenant isolation
- [ ] Database migration

### Phase 2 (P1) - Enforcement
- [ ] Enforcement Engine
- [ ] FastAPI Middleware
- [ ] Allowlist Manager
- [ ] API endpoints

### Phase 3 (P1) - Streaming
- [ ] Kafka Scanner
- [ ] Kinesis Scanner
- [ ] Stream processor

### Phase 4 (P1) - Operations
- [ ] Auto-Remediation Engine
- [ ] Prometheus Metrics
- [ ] Grafana Dashboard
- [ ] Alert rules

### Phase 5 (P2) - Testing
- [ ] Unit tests (200+)
- [ ] Integration tests
- [ ] Load tests

---

## 13. Success Metrics

| Metric | Target |
|--------|--------|
| Detection accuracy | >99% for regex patterns |
| False positive rate | <2% |
| Enforcement latency | <10ms P99 |
| Tokenization throughput | >10,000/sec |
| Stream processing | >1,000 msg/sec |
| Quarantine review SLA | <24 hours |

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Security Team | Initial version |
