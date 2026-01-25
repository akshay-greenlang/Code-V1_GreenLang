# GreenLang: Reach A- Grade PRD

## Current Status: B+
## Target Status: A-

## CRITICAL CONSTRAINTS
- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory

---

## Task 1: Fix 10 Remaining CI Workflows

Fix all GitHub workflows that still reference old paths.

### Files to Fix:

1. **gl-001-cd.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-001` → `applications/GL Agents/GL-001_Thermalcommand`

2. **gl-002-cd.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-002` → `applications/GL Agents/GL-002_Flameguard`

3. **gl-002-scheduled.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-002` → `applications/GL Agents/GL-002_Flameguard`

4. **gl-003-ci.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-003` → `applications/GL Agents/GL-003*`

5. **gl-003-scheduled.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-003` → `applications/GL Agents/GL-003*`

6. **gl-004-tests.yml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-004` → `applications/GL Agents/GL-004*`

7. **gl-005-ci.yaml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-005` → `applications/GL Agents/GL-005*`

8. **gl-006-tests.yml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-006` → `applications/GL Agents/GL-006*`

9. **gl-007-tests.yml**
   - Change: `GreenLang_2030/agent_foundation/agents/GL-007` → `applications/GL Agents/GL-007*`

10. **security-audit.yml**
    - Change: `GreenLang_2030/agent_foundation` → `applications/`

---

## Task 2: Consolidate GitHub Workflows (72 → 15-20)

### Workflows to Keep (Core):
1. ci.yml - Main CI pipeline
2. tests.yml - Test execution
3. security.yml - Security scanning
4. lint.yml - Code quality
5. release.yml - Release management
6. deploy-staging.yml - Staging deployment
7. deploy-production.yml - Production deployment
8. docs.yml - Documentation build
9. dependency-review.yml - Dependency audit
10. codeql.yml - Code analysis

### Workflows to Archive:
Move to `.github/workflows/archive/`:
- All gl-00X-*.yaml files (merge functionality into main ci.yml)
- Duplicate security scans
- Old deployment workflows
- Agent-specific workflows that can be parameterized

---

## Task 3: Address TODO/FIXME Markers

1. Find all markers:
   ```bash
   grep -r "TODO\|FIXME\|HACK\|XXX" greenlang/ --include="*.py" -c
   ```

2. For each marker:
   - Security-related: Fix immediately
   - Import-related: Fix as part of consolidation
   - Deprecated code: Remove or add deprecation warning
   - Feature requests: Create GitHub issue

3. Create `docs/TECHNICAL_DEBT.md` documenting remaining items

4. Target: Reduce from 96 to <20 active markers

---

## Success Criteria for A-

- [x] CI workflows with old paths: 0 (currently 10)
- [x] GitHub workflows: 15-20 (currently 72)
- [x] TODO/FIXME markers: <20 (currently 96)
- [ ] All tests pass
- [ ] All imports work
- [ ] Clean git history

## Target Grade: A-
