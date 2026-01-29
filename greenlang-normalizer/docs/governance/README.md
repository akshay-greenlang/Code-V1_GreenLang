# Governance Documentation

This directory contains governance documentation for vocabulary management, policy configuration, and approval workflows.

## Contents

- `vocabulary-governance.md` - Vocabulary change management
- `policy-management.md` - Policy configuration guide
- `approval-workflows.md` - Change approval processes
- `audit-requirements.md` - Audit trail requirements

## Vocabulary Governance

### Vocabulary Lifecycle

1. **Draft**: New vocabulary entries submitted for review
2. **Review**: Subject matter expert review
3. **Approved**: Entry approved and active
4. **Deprecated**: Entry marked for replacement
5. **Archived**: Entry removed from active use

### Change Request Process

1. Submit change request via Review Console
2. Automatic validation checks
3. SME review and approval
4. Staging deployment for testing
5. Production deployment
6. Audit log entry created

### Versioning

Vocabularies follow semantic versioning:
- **Major**: Breaking changes (removed entries, changed IDs)
- **Minor**: New entries, new aliases
- **Patch**: Metadata updates, corrections

## Policy Management

### Policy Hierarchy

```
System Defaults
    └── Compliance Profile
            └── Organization Policy
                    └── Request Overrides
```

### Policy Modes

- **STRICT**: Fail on missing required context
- **LENIENT**: Apply defaults with warnings

### Compliance Profiles

- GHG Protocol
- ISO 14064
- CSRD/ESRS
- SEC Climate Disclosure
- ISSB Standards

## Audit Requirements

All operations must:
- Generate audit events
- Include provenance hash
- Be immutable after creation
- Support 7-year retention
