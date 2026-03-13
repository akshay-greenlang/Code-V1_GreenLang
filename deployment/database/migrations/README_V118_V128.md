# Database Migrations V118-V128 - Quick Reference

## TL;DR

Apply EUDR agent migrations V118-V128 for Documentation & Reporting agents (030-040).

**Status**: ✅ Ready to apply
**Current Version**: V117
**Target Version**: V128
**Migrations**: 11 files (V118-V128)

---

## Quick Start

### Windows (PowerShell)
```powershell
# Start Docker Desktop first, then:
.\deployment\database\migrations\apply_v118_v128.ps1
```

### Linux/Mac/Git Bash
```bash
# Start Docker Desktop first, then:
./deployment/database/migrations/apply_v118_v128.sh dev
```

---

## What Gets Migrated

11 EUDR agents for Documentation & Reporting:

| Version | Agent | Tables |
|---------|-------|--------|
| V118 | Documentation Generator | 9 |
| V119 | Stakeholder Engagement | 10 |
| V120 | Grievance Mechanism | 9 |
| V121 | Continuous Monitoring | 9 |
| V122 | Annual Review Scheduler | 9 |
| V123 | Improvement Plan Creator | 9 |
| V124 | EU Information System | 8 |
| V125 | DDS Creator | 9 |
| V126 | Reference Number Generator | 9 |
| V127 | Customs Declaration | 9 |
| V128 | Authority Communication | 10 |

**Total**: ~97 tables, ~1,200 indexes, 19 TimescaleDB hypertables

---

## Prerequisites

1. Docker Desktop running
2. PostgreSQL database accessible
3. Current database version: V117

---

## Files

| File | Purpose |
|------|---------|
| `apply_v118_v128.sh` | Bash migration script |
| `apply_v118_v128.ps1` | PowerShell migration script |
| `MIGRATION_STATUS_V118_V128.md` | Detailed status and procedures |
| `MIGRATION_VERIFICATION_REPORT.md` | Complete verification report |
| `README_V118_V128.md` | This quick reference |

---

## Verification

After migration:
```sql
-- Check version
SELECT MAX(version) FROM flyway_schema_history;
-- Expected: V128

-- Check tables
SELECT COUNT(*) FROM information_schema.tables
WHERE table_name ~ '^eudr_(dgn|set|gmm|cmt|ars|ipc|eis|dsc|rng|cds|acm)_';
-- Expected: ~97
```

---

## Support

- **Detailed Docs**: See `MIGRATION_STATUS_V118_V128.md`
- **Verification**: See `MIGRATION_VERIFICATION_REPORT.md`
- **Issues**: Check Flyway logs

---

**Last Updated**: 2026-03-13
