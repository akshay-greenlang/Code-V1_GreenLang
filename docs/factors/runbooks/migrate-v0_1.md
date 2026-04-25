# Operator Runbook — Apply Factors v0.1 Schema

**Audience:** SRE / DBA on-call for the FY27 Factors launch.
**Wave:** C / TaskCreate #3 / WS1-T3.
**Schema:** `factors_v0_1` (canonical DDL = `V500__factors_v0_1_canonical.sql`).

This runbook describes how to bring an empty Postgres database up to the
frozen v0.1 alpha schema, how to verify the result, how to roll back, and
what smoke-checks to run before declaring the database operational.

---

## 1. Pre-flight

* Postgres 16+ reachable from the deploy host.
* The `factors_v0_1` schema does **not** already exist (or is at a known
  earlier revision).
* You have *write* DDL grants on the target database.
* `alembic` is installed in the active virtualenv:
  `python -c "import alembic; print(alembic.__version__)"` should print
  `1.13+`.
* Set the database URL via env var (preferred over editing `alembic.ini`):

  ```bash
  export ALEMBIC_SQLALCHEMY_URL="postgresql+psycopg://gl_app:***@<host>:5432/<db>"
  ```

  `DATABASE_URL` is honoured as a fallback if `ALEMBIC_SQLALCHEMY_URL`
  is not set. Both are read by `migrations/env.py`.

## 2. Apply

From the repo root:

```bash
# Apply every revision in the factors tree (currently just 0001):
alembic upgrade head

# Or, pin the explicit revision id:
alembic upgrade 0001_factors_v0_1_initial
```

`upgrade head` executes
`deployment/database/migrations/sql/V500__factors_v0_1_canonical.sql` in a
single transaction (the revision splits the file into top-level
statements while preserving the `$$ ... $$` block that wraps the
`factor_immutable_trigger` function).

## 3. Verify

```bash
# 3a. Alembic version row landed in the factors schema:
psql "$ALEMBIC_SQLALCHEMY_URL" \
  -c "SELECT version_num FROM factors_v0_1.alembic_version;"
# Expect: 0001_factors_v0_1_initial
```

```bash
# 3b. Schema and key tables exist:
psql "$ALEMBIC_SQLALCHEMY_URL" -c "\dn factors_v0_1"
psql "$ALEMBIC_SQLALCHEMY_URL" -c "\dt factors_v0_1.*"
# Expect 7 tables: source, methodology, geography, unit,
# factor_pack, factor, factor_publish_log
```

```bash
# 3c. Immutability trigger function is defined:
psql "$ALEMBIC_SQLALCHEMY_URL" -c "
SELECT n.nspname, p.proname
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'factors_v0_1'
  AND p.proname = 'factor_immutable_trigger';"
# Expect 1 row: factors_v0_1 | factor_immutable_trigger
```

## 4. Smoke check

```bash
# 4a. URN pattern check actually rejects bad input:
psql "$ALEMBIC_SQLALCHEMY_URL" -c "
INSERT INTO factors_v0_1.source (urn, source_id, name, organization,
  primary_url, licence, license_class, update_cadence, source_owner,
  parser_module, parser_function, parser_version, source_version,
  publication_url, trust_tier)
VALUES ('not-a-urn', 'x', 'x', 'x', 'http://x', 'x', 'cc_by',
  'annual', 'x', 'x', 'x', '0', '0', 'http://x', 'tier_1');"
# Expect: ERROR: new row for relation "source" violates check constraint
#         "source_urn_pattern"
```

```bash
# 4b. Indexes are present:
psql "$ALEMBIC_SQLALCHEMY_URL" -c "
SELECT indexname FROM pg_indexes
WHERE schemaname = 'factors_v0_1'
ORDER BY indexname;"
# Expect at least: factor_active_idx, factor_alias_idx,
# factor_category_idx, factor_fts_idx, factor_geo_vintage_idx,
# factor_pack_idx, factor_pack_source_idx, factor_published_at_idx,
# factor_publish_log_edition_idx, factor_publish_log_factor_idx,
# factor_source_idx, factor_tags_idx, source_alpha_v0_1_idx
```

## 5. Rollback

```bash
# Roll back exactly one revision (drops everything in factors_v0_1):
alembic downgrade -1

# Or pin the empty target:
alembic downgrade base
```

The revision invokes
`V500__factors_v0_1_canonical_DOWN.sql`, which drops in dependency order:
`factor_publish_log` -> trigger -> trigger function -> `factor` ->
`factor_pack` -> `unit` / `geography` / `methodology` / `source` ->
`DROP SCHEMA factors_v0_1 CASCADE`.

## 6. If something goes wrong

* **Mid-upgrade error:** Alembic wraps the upgrade in a transaction; on
  failure the schema is left untouched. Re-read the error, fix the
  underlying problem (commonly: insufficient grants, schema already
  partially populated by a previous manual `psql` run, or extension
  missing), and re-run `alembic upgrade head`.
* **State drift (DBA ran V500 manually):** stamp the existing schema:
  `alembic stamp 0001_factors_v0_1_initial`. This records the version
  row without re-applying DDL.
* **Total wedge:** drop the schema manually
  (`DROP SCHEMA factors_v0_1 CASCADE`), drop the alembic_version row if
  it survived, then re-run `alembic upgrade head`.

## References

* Forward DDL: `deployment/database/migrations/sql/V500__factors_v0_1_canonical.sql`
* Reverse DDL: `deployment/database/migrations/sql/V500__factors_v0_1_canonical_DOWN.sql`
* Alembic env: `migrations/env.py`
* First revision: `migrations/versions/0001_factors_v0_1_initial.py`
* Operator README: `migrations/README.md`
