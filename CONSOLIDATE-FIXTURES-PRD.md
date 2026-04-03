# CONSOLIDATE-FIXTURES-PRD

## Task 1: Verify root conftest.py consolidation
- Verify tests/conftest.py exists and contains merged fixtures from conftest_enhanced.py and conftest_v2.py
- Run: `python -c "import ast; tree = ast.parse(open('tests/conftest.py').read()); fixtures = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and any(isinstance(d, ast.Name) and d.id == 'fixture' or isinstance(d, ast.Attribute) and d.attr == 'fixture' for d in getattr(n, 'decorator_list', []))]  ; print(f'Fixtures: {len(fixtures)}')" `
- Expected: 40+ fixtures (was 26, added ~15+ from enhanced)
- Verify conftest_enhanced.py and conftest_v2.py are deleted

## Task 2: Verify tests/fixtures/ package structure
- Check tests/fixtures/__init__.py exists
- Check tests/fixtures/mocks.py exists with MockPrometheusRegistry, MockRedisClient, MockS3Client, MockHTTPClient classes
- Check tests/fixtures/generators.py exists with EmissionDataGenerator, SupplyChainDataGenerator classes
- Check tests/fixtures/helpers.py exists with assert_valid_provenance_hash, assert_decimal_close functions
- Check tests/fixtures/constants.py exists with TEST_TENANT_ID, GWP constants

## Task 3: Verify cleanup script
- Check scripts/consolidate_conftest.py exists
- Run: `python scripts/consolidate_conftest.py --help`
- Verify it supports --audit, --clean, --overrides modes
- Run audit: `python scripts/consolidate_conftest.py`
- Verify audit report is generated without errors

## Task 4: Verify imports work
- Run: `python -c "from tests.fixtures.mocks import MockPrometheusRegistry, MockRedisClient; print('mocks OK')"`
- Run: `python -c "from tests.fixtures.generators import EmissionDataGenerator; print('generators OK')"`
- Run: `python -c "from tests.fixtures.helpers import assert_valid_provenance_hash; print('helpers OK')"`
- Run: `python -c "from tests.fixtures.constants import TEST_TENANT_ID, GWP_CO2; print('constants OK')"`

## Task 5: Verify no test breakage
- Run a sample of existing tests to ensure root conftest changes don't break anything
- Run: `python -m pytest tests/conftest.py --collect-only -q 2>&1 | head -20`
- Verify fixtures are collectable without import errors

## Task 6: Verify conftest_emission_factors.py status
- Check if conftest_emission_factors.py is properly handled (moved to fixtures/ or kept as reference)
- Verify emission factor test data is accessible

## Task 7: Run conftest audit report
- Run: `python scripts/consolidate_conftest.py`
- Verify the audit identifies duplicate fixtures
- Document the top 10 most duplicated fixtures
- Verify recommendations section provides actionable items
