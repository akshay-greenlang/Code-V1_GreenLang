# GreenLang v0.2.0 Release Rollback Plan

## Overview
This document outlines the rollback procedures for GreenLang v0.2.0 release to ensure rapid recovery in case of critical issues.

## Rollback Triggers
The following conditions will trigger an immediate rollback:
1. **Critical Installation Failure**: >5% of users unable to install via pip
2. **Entry Point Failure**: `gl` command not found after installation
3. **Data Loss**: Any scenario causing user data corruption
4. **Security Breach**: Discovery of exposed credentials or vulnerabilities
5. **Breaking API Changes**: Undocumented breaking changes affecting >10% of users

## Rollback Procedures

### Phase 1: Immediate Actions (0-15 minutes)
1. **Alert Team**
   ```
   - Notify: Release Manager, CTO, DevOps Lead
   - Channel: #release-emergency on Discord
   - Include: Issue description, impact assessment, user reports
   ```

2. **Yank from PyPI**
   ```bash
   # DO NOT DELETE - Only yank to preserve history
   twine yank greenlang==0.2.0 --reason "Critical issue: [DESCRIPTION]"
   ```

3. **Update Status Page**
   - Post incident notice on status.greenlang.io
   - Update Discord #announcements channel
   - Tweet from @greenlang_official

### Phase 2: Recovery Actions (15-60 minutes)

#### Option A: Hotfix (Preferred for minor issues)
1. Create hotfix branch
   ```bash
   git checkout -b hotfix/0.2.1 v0.2.0
   git cherry-pick [fix-commits]
   ```

2. Fast-track testing
   ```bash
   python -m pytest tests/unit/ -x
   python -m pytest tests/e2e/ -x
   ```

3. Release v0.2.1
   ```bash
   # Update version to 0.2.1
   python -m build
   twine upload dist/greenlang-0.2.1*
   ```

#### Option B: Full Rollback (For major issues)
1. Re-promote previous stable version
   ```bash
   # Ensure v0.1.x artifacts are available
   pip install greenlang==0.1.9  # Previous stable
   ```

2. Update documentation
   - Revert README.md install instructions
   - Update docs.greenlang.io to show v0.1.9
   - Pin GitHub release as "Latest"

### Phase 3: Post-Incident (1-24 hours)

1. **Root Cause Analysis**
   - Collect all logs and error reports
   - Identify gap in testing/validation
   - Document in `incident-reports/YYYY-MM-DD-v0.2.0.md`

2. **Communication**
   - Email registered users about the issue
   - Post detailed explanation on blog
   - Schedule post-mortem meeting

3. **Prevention Measures**
   - Add regression tests for the specific issue
   - Update CI/CD pipeline checks
   - Review and update release checklist

## Recovery Verification

### Success Metrics
- [ ] Users can install previous version successfully
- [ ] No new error reports in 2 hours
- [ ] CI/CD pipeline green for rollback version
- [ ] Documentation reflects correct version

### Testing Commands
```bash
# Verify rollback success
pip uninstall greenlang -y
pip install greenlang  # Should install 0.1.9 or 0.2.1
gl --version  # Should show correct version
gl doctor  # Should pass all checks
```

## Contact Information

| Role | Name | Contact | Backup |
|------|------|---------|--------|
| Release Manager | TBD | release@greenlang.io | - |
| CTO | TBD | cto@greenlang.io | - |
| DevOps Lead | TBD | devops@greenlang.io | - |
| Security Lead | TBD | security@greenlang.io | - |

## Pre-Approved Actions
The following actions are pre-approved and can be executed immediately without additional authorization:
1. Yanking package from PyPI
2. Reverting documentation to previous version
3. Posting status updates on official channels
4. Creating hotfix branches
5. Fast-tracking CI/CD for hotfixes

## Rollback Checklist

### Before Release
- [x] Backup v0.1.9 artifacts stored safely
- [x] Rollback plan reviewed by team
- [x] Contact list updated and verified
- [x] Status page access confirmed
- [ ] PyPI yank permissions verified

### During Incident
- [ ] Team notified via Discord
- [ ] Package yanked from PyPI
- [ ] Status page updated
- [ ] Hotfix/rollback decision made
- [ ] Recovery actions initiated

### After Recovery
- [ ] Users can install working version
- [ ] Post-mortem scheduled
- [ ] Incident report drafted
- [ ] Regression tests added
- [ ] Documentation updated

## Notes
- **Never delete packages** from PyPI - always use yank
- **Communicate early and often** - transparency builds trust
- **Document everything** - for future reference
- **Test the rollback plan** - before you need it

---
*Last Updated: 2025-09-22*
*Next Review: Before v0.3.0 release*