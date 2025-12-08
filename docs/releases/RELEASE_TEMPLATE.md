# GreenLang Release Notes Template

## Template Instructions

This template provides a standardized format for GreenLang release notes. Copy this template and replace the placeholders with actual release information.

---

# GreenLang v{MAJOR}.{MINOR}.{PATCH}

**Release Date:** {YYYY-MM-DD}
**Release Type:** {Major|Minor|Patch|Security}
**Supported Upgrade Paths:** {Previous versions that can upgrade}

---

## Highlights

{Brief 2-3 sentence summary of the most important changes in this release. This should capture the essence of the release for readers who only skim the document.}

---

## New Features

### Feature 1: {Feature Name}

{Description of the feature and its value to users}

**How to use:**
```python
# Code example if applicable
```

**Configuration:**
```yaml
# Configuration example if applicable
```

**Documentation:** [Link to detailed documentation]

### Feature 2: {Feature Name}

{Repeat format for each new feature}

---

## Improvements

### {Category 1: e.g., Performance}

- **{Improvement title}:** {Brief description of the improvement and its impact}
- **{Improvement title}:** {Brief description}

### {Category 2: e.g., User Experience}

- **{Improvement title}:** {Brief description}
- **{Improvement title}:** {Brief description}

### {Category 3: e.g., API}

- **{Improvement title}:** {Brief description}

---

## Bug Fixes

### Critical

- **{BUG-XXX}:** {Bug description and fix summary} ([Issue #{number}]({link}))

### High

- **{BUG-XXX}:** {Bug description and fix summary}
- **{BUG-XXX}:** {Bug description and fix summary}

### Medium

- **{BUG-XXX}:** {Bug description and fix summary}
- **{BUG-XXX}:** {Bug description and fix summary}

### Low

- **{BUG-XXX}:** {Bug description and fix summary}

---

## Breaking Changes

{If no breaking changes, state: "This release contains no breaking changes."}

### Change 1: {Description of breaking change}

**What changed:**
{Explain what was changed and why}

**Impact:**
{Explain who is affected and how}

**Migration:**
```python
# Before (v{OLD})
old_code_example()

# After (v{NEW})
new_code_example()
```

### Change 2: {Description}

{Repeat format for each breaking change}

---

## Migration Guide

### Upgrading from v{PREVIOUS} to v{CURRENT}

**Estimated time:** {X minutes/hours}
**Downtime required:** {Yes/No, and duration if yes}

#### Pre-upgrade Steps

1. {Step 1}
2. {Step 2}
3. {Step 3}

#### Upgrade Steps

```bash
# Command examples for upgrade
```

#### Post-upgrade Steps

1. {Verification step 1}
2. {Verification step 2}
3. {Configuration updates if needed}

#### Rollback Procedure

{Instructions for rolling back if issues occur}

```bash
# Rollback commands
```

---

## Deprecations

### Deprecated in This Release

| Feature/API | Replacement | Removal Version |
|-------------|-------------|-----------------|
| {Old feature} | {New feature} | v{X.Y} |
| {Old API endpoint} | {New endpoint} | v{X.Y} |

### Previously Deprecated (Now Removed)

| Feature/API | Removed In | Migration Guide |
|-------------|------------|-----------------|
| {Feature} | This release | [Link] |

---

## Known Issues

### Issue 1: {Brief description}

**Symptoms:** {What users may experience}

**Workaround:** {Steps to work around the issue}

**Status:** {Under investigation / Fix planned for vX.Y / Won't fix}

**Tracking:** [Issue #{number}]({link})

### Issue 2: {Brief description}

{Repeat format}

---

## Security Updates

{If no security updates, state: "This release contains no security updates."}

### {CVE-YYYY-XXXXX}: {Brief description}

**Severity:** {Critical/High/Medium/Low}
**CVSS Score:** {X.X}
**Affected versions:** v{X.Y.Z} - v{X.Y.Z}
**Fixed in:** This release

**Description:**
{Technical description of the vulnerability}

**Mitigation:**
{For users who cannot upgrade immediately}

**Credit:**
{Credit to reporter if applicable}

---

## Compatibility

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | {X cores} | {X+ cores} |
| RAM | {X GB} | {X+ GB} |
| Storage | {X GB} | {X+ GB} |

### Software Dependencies

| Software | Required Version | Notes |
|----------|------------------|-------|
| Docker | {X.X+} | |
| Kubernetes | {X.X+} | Optional |
| PostgreSQL | {X+} | |
| Python | {X.X+} | For SDK |

### Browser Support

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | {XX} |
| Firefox | {XX} |
| Safari | {XX} |
| Edge | {XX} |

### API Compatibility

- REST API: v{X} (no changes)
- GraphQL: v{X} (additions, no breaking changes)
- WebSocket: v{X} (no changes)

---

## Contributors

{Thank contributors to this release}

We thank the following contributors for their work on this release:

- @{username} - {contribution summary}
- @{username} - {contribution summary}
- {Company/Organization} - {contribution summary}

---

## Checksums

```
SHA256 (greenlang-{version}.tar.gz) = {checksum}
SHA256 (greenlang-{version}.zip) = {checksum}
SHA256 (greenlang-docker-{version}.tar) = {checksum}
```

---

## Getting Help

- **Documentation:** https://docs.greenlang.io
- **Release Announcement:** https://greenlang.io/blog/{release-post}
- **Upgrade Support:** support@greenlang.io
- **Community Forum:** https://community.greenlang.io
- **Security Issues:** security@greenlang.io

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| {YYYY-MM-DD} | 1.0 | Initial release |
| {YYYY-MM-DD} | 1.1 | {Update description} |

---

*Release Notes Version: 1.0*
*Last Updated: {YYYY-MM-DD}*
