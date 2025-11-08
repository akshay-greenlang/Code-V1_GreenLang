# GreenLang Version Compatibility Matrix

**Last Updated:** 2025-11-08
**Document Version:** 1.0.0

---

## Feature Compatibility Across Versions

| Feature | v0.1 | v0.2 | v0.3 | v0.4 (Planned) |
|---------|------|------|------|----------------|
| **Agent Specifications** |
| AgentSpec v1 | ✓ | ✓ | Deprecated¹ | ✗ |
| AgentSpec v2 | ✗ | Beta | ✓ | ✓ |
| **Execution** |
| Synchronous Agents | ✓ | ✓ | ✓ | ✓ |
| Asynchronous Agents | ✗ | ✗ | ✓ | ✓ |
| Parallel Execution | ✗ | Beta | ✓ | ✓ |
| **APIs** |
| REST API v1 | ✓ | ✓ | Deprecated² | ✗ |
| REST API v2 | ✗ | ✗ | ✓ | ✓ |
| GraphQL API | ✗ | ✗ | Beta | ✓ |
| WebSocket API | ✗ | ✗ | ✗ | ✓ |
| **Authentication** |
| API Keys | ✓ | ✓ | Deprecated³ | ✗ |
| JWT Tokens | ✗ | ✗ | ✓ | ✓ |
| OAuth 2.0 | ✗ | ✗ | Beta | ✓ |
| SAML | ✗ | ✗ | ✗ | ✓ |
| **Security** |
| MFA (Multi-Factor Auth) | ✗ | ✗ | ✓ | ✓ |
| Encryption at Rest | ✗ | ✗ | ✓ | ✓ |
| TLS 1.2 | ✓ | ✓ | ✓ | Deprecated |
| TLS 1.3 | ✗ | Beta | ✓ | ✓ |
| **Features** |
| Agent Marketplace | ✗ | ✗ | ✗ | ✓ |
| Version Control | ✗ | Beta | ✓ | ✓ |
| Audit Logging | Basic | ✓ | ✓ | ✓ |
| Multi-Tenancy | ✗ | ✗ | Beta | ✓ |
| **Compliance** |
| SOC 2 Type II | ✗ | ✗ | ✓ | ✓ |
| ISO 27001 | ✗ | ✗ | ✓ | ✓ |
| GDPR | ✗ | ✗ | ✓ | ✓ |
| HIPAA | ✗ | ✗ | ✓ | ✓ |
| **Python Support** |
| Python 3.7 | ✓ | ✓ | ✗ | ✗ |
| Python 3.8 | ✓ | ✓ | ✗ | ✗ |
| Python 3.9 | ✗ | ✓ | ✓ | ✓ |
| Python 3.10 | ✗ | ✓ | ✓ | ✓ |
| Python 3.11 | ✗ | Beta | ✓ | ✓ |
| Python 3.12 | ✗ | ✗ | Beta | ✓ |

**Legend:**
- ✓ = Fully Supported
- Beta = Beta Support (may have limitations)
- Deprecated = Available but deprecated (will be removed)
- ✗ = Not Available

**Footnotes:**
1. AgentSpec v1 deprecated in v0.3, removed in v0.4 (Q2 2026)
2. REST API v1 deprecated in v0.3, removed in v0.4 (Q2 2026)
3. API Keys deprecated in v0.3, removed in v0.4 (Q2 2026)

---

## API Endpoint Compatibility

### REST API v1 (Deprecated in v0.3)

| Endpoint | v0.1 | v0.2 | v0.3 | v0.4 |
|----------|------|------|------|------|
| `/api/v1/agents/register` | ✓ | ✓ | Deprecated | ✗ |
| `/api/v1/workflows/execute` | ✓ | ✓ | Deprecated | ✗ |
| `/api/v1/workflows/list` | ✓ | ✓ | Deprecated | ✗ |
| `/api/v1/health` | ✓ | ✓ | Deprecated | ✗ |

**Migration Path:** All v1 endpoints have v2 equivalents. See [API Migration Guide](MIGRATION_GUIDE_v0.2_to_v0.3.md#api-endpoint-changes).

### REST API v2 (Introduced in v0.3)

| Endpoint | v0.3 | v0.4 |
|----------|------|------|
| `/api/v2/agents` | ✓ | ✓ |
| `/api/v2/workflows/execute` | ✓ | ✓ |
| `/api/v2/workflows` | ✓ | ✓ |
| `/api/v2/health` | ✓ | ✓ |
| `/api/v2/auth/token` | ✓ | ✓ |

---

## Database Schema Compatibility

| Schema Version | v0.1 | v0.2 | v0.3 | v0.4 |
|----------------|------|------|------|------|
| Schema v1 | ✓ | ✓ | Migration Required | ✗ |
| Schema v2 | ✗ | ✗ | ✓ | ✓ |

### Schema Migrations

- **v0.1 → v0.2:** No migration required (schema v1 compatible)
- **v0.2 → v0.3:** Migration required (schema v1 → v2)
  - New tables: `audit_logs`, `user_sessions`, `agent_versions`
  - Modified: `agents` (add `spec_version`), `workflows` (add `async_enabled`)
- **v0.3 → v0.4:** Minor migration (schema v2.1)

**Migration Tool:** Use `greenlang migrate execute` for automated migration.

---

## Configuration File Compatibility

### greenlang.yaml

| Config Key | v0.1 | v0.2 | v0.3 | v0.4 |
|------------|------|------|------|------|
| `database.url` | ✓ | ✓ | Deprecated | ✗ |
| `database.connection_string` | ✗ | ✗ | ✓ | ✓ |
| `security.api_keys_enabled` | ✓ | ✓ | Deprecated | ✗ |
| `security.jwt_enabled` | ✗ | ✗ | ✓ | ✓ |
| `security.encryption` | ✗ | ✗ | Required | Required |
| `logging.level` (lowercase) | ✓ | ✓ | Deprecated | ✗ |
| `logging.level` (uppercase) | ✗ | ✗ | ✓ | ✓ |

### Environment Variables

| Variable | v0.1 | v0.2 | v0.3 | v0.4 |
|----------|------|------|------|------|
| `GREENLANG_ENV` | ✓ | ✓ | Deprecated | ✗ |
| `GL_ENV` | ✗ | ✗ | ✓ | ✓ |
| `GREENLANG_DB_URL` | ✓ | ✓ | Deprecated | ✗ |
| `GL_DATABASE_URL` | ✗ | ✗ | ✓ | ✓ |
| `GL_SECRET_KEY` | ✗ | ✗ | Required | Required |
| `GL_ENCRYPTION_KEY` | ✗ | ✗ | Required | Required |

---

## Package Dependencies Compatibility

### Core Dependencies

| Package | v0.1 | v0.2 | v0.3 | v0.4 |
|---------|------|------|------|------|
| pydantic | 1.8.x | 1.10.x | 2.x | 2.x |
| sqlalchemy | 1.4.x | 1.4.x | 2.0.x | 2.0.x |
| click | 8.0.x | 8.0.x | 8.1.x | 8.1.x |
| aiohttp | - | - | 3.9.x | 3.9.x |
| cryptography | - | - | 41.x | 42.x |
| alembic | 1.7.x | 1.8.x | 1.12.x | 1.13.x |

### Optional Dependencies

| Package | v0.1 | v0.2 | v0.3 | v0.4 |
|---------|------|------|------|------|
| numpy | 1.21.x | 1.23.x | 1.24.x | 1.26.x |
| pandas | 1.3.x | 1.5.x | 2.0.x | 2.1.x |
| fastapi | - | 0.95.x | 0.104.x | 0.109.x |
| uvicorn | - | 0.20.x | 0.24.x | 0.27.x |

---

## Upgrade Paths

### Direct Upgrades

```
v0.1 → v0.2 (Direct)
v0.2 → v0.3 (Migration Tool Required)
v0.3 → v0.4 (Direct - Planned)
```

### Multi-Step Upgrades

```
v0.1 → v0.3 (Two-Step)
  Step 1: v0.1 → v0.2
  Step 2: v0.2 → v0.3 (with migration tool)

v0.1 → v0.4 (Three-Step - Planned)
  Step 1: v0.1 → v0.2
  Step 2: v0.2 → v0.3
  Step 3: v0.3 → v0.4
```

**Recommendation:** Always upgrade one major version at a time.

---

## Breaking Changes Timeline

### Already Deprecated (v0.3)

| Feature | Deprecated In | Removed In | Replacement |
|---------|--------------|-----------|-------------|
| AgentSpec v1 | v0.3.0 | v0.4.0 (Q2 2026) | AgentSpec v2 |
| REST API v1 | v0.3.0 | v0.4.0 (Q2 2026) | REST API v2 |
| API Keys | v0.3.0 | v0.4.0 (Q2 2026) | JWT Tokens |
| `database.url` | v0.3.0 | v0.4.0 (Q2 2026) | `database.connection_string` |
| `GREENLANG_*` env vars | v0.3.0 | v0.4.0 (Q2 2026) | `GL_*` env vars |

### Planned Deprecations (v0.4)

| Feature | Deprecate In | Remove In | Replacement |
|---------|-------------|-----------|-------------|
| TLS 1.2 | v0.4.0 (Q2 2026) | v0.5.0 (Q4 2026) | TLS 1.3 |
| Sync-Only Agents | v0.4.0 (Q2 2026) | v0.5.0 (Q4 2026) | Async Agents |

---

## Backward Compatibility Policy

### General Policy

- **Major versions** (0.x → 1.x): May include breaking changes
- **Minor versions** (0.2 → 0.3): Breaking changes with deprecation period (6 months)
- **Patch versions** (0.3.0 → 0.3.1): No breaking changes

### Deprecation Process

1. **Announcement:** Feature marked as deprecated in release notes
2. **Deprecation Period:** Minimum 6 months of support
3. **Warnings:** Runtime deprecation warnings in logs
4. **Migration Guide:** Documentation provided for migration
5. **Removal:** Feature removed in subsequent major/minor version

### Support Windows

- **Current Version (v0.3):** Full support + new features
- **Previous Version (v0.2):** Security fixes only (6 months)
- **Older Versions (v0.1):** End of life

---

## Testing Compatibility

### Test Framework Versions

| Framework | v0.1 | v0.2 | v0.3 | v0.4 |
|-----------|------|------|------|------|
| pytest | 7.0.x | 7.2.x | 7.4.x | 8.0.x |
| pytest-cov | 3.0.x | 4.0.x | 4.1.x | 4.1.x |
| pytest-asyncio | - | - | 0.21.x | 0.23.x |

### Compatibility Test Matrix

GreenLang is tested against:

**Python Versions:**
- v0.3: Python 3.9, 3.10, 3.11
- v0.4: Python 3.10, 3.11, 3.12 (planned)

**Operating Systems:**
- Linux (Ubuntu 20.04, 22.04)
- macOS (12, 13, 14)
- Windows (10, 11, Server 2019, 2022)

**Databases:**
- PostgreSQL (12, 13, 14, 15)
- MySQL (8.0)
- SQLite (3.35+)

---

## Migration Support

### Automated Migration

Use the migration CLI tool for automated migration:

```bash
greenlang migrate analyze    # Check compatibility
greenlang migrate plan       # Generate migration plan
greenlang migrate execute    # Execute migration
greenlang migrate verify     # Verify success
```

### Manual Migration Resources

- [Migration Guide](MIGRATION_GUIDE_v0.2_to_v0.3.md) - Complete step-by-step guide
- [Breaking Changes](BREAKING_CHANGES.md) - Detailed breaking changes
- [API Reference](../API_REFERENCE.md) - API v2 documentation
- [AgentSpec v2](../specs/agentspec_v2.md) - New agent specification

---

## Version Support Matrix

| Version | Release Date | End of Support | Status |
|---------|-------------|----------------|--------|
| v0.1.x | 2024-01-15 | 2025-01-15 | End of Life |
| v0.2.x | 2024-09-01 | 2025-09-01 | Security Only |
| v0.3.x | 2025-03-01 | 2026-03-01 | Supported |
| v0.4.x | 2026-06-01 (planned) | 2027-06-01 | Planned |

---

## Compatibility FAQs

### Q: Can I use AgentSpec v1 packs in v0.3?

**A:** Yes, with deprecation warnings. AgentSpec v1 packs are supported in v0.3 but will be removed in v0.4. We recommend converting to AgentSpec v2 using the migration tool:

```bash
greenlang migrate execute
```

### Q: Will my v0.2 API clients work with v0.3?

**A:** Yes. REST API v1 endpoints are still available in v0.3 with deprecation headers. However, we recommend migrating to API v2 before v0.4 release.

### Q: Can I rollback from v0.3 to v0.2?

**A:** Yes, if you used the migration tool with backups:

```bash
greenlang migrate rollback
```

Manual rollback is also possible using database and configuration backups.

### Q: What Python version should I use?

**A:** For v0.3, use Python 3.9, 3.10, or 3.11. Python 3.8 and earlier are not supported.

### Q: Are there any breaking changes in configuration?

**A:** Yes. The main changes are:
- `database.url` → `database.connection_string`
- New required: `security.encryption`
- Logging levels must be uppercase

See the [Migration Guide](MIGRATION_GUIDE_v0.2_to_v0.3.md#configuration-format-changes) for details.

---

## Support & Resources

### Documentation
- [Migration Guide](MIGRATION_GUIDE_v0.2_to_v0.3.md)
- [API Reference](../API_REFERENCE.md)
- [AgentSpec v2 Specification](../specs/agentspec_v2.md)

### Tools
- Migration CLI: `greenlang migrate`
- Compatibility Checker: `greenlang migrate analyze`

### Community
- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Discord: https://discord.gg/greenlang
- Forum: https://forum.greenlang.io

### Enterprise Support
- Email: support@greenlang.io
- Migration Assistance: migration@greenlang.io
- Phone: +1 (555) 123-4567

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-08
**Maintained By:** GreenLang Team
