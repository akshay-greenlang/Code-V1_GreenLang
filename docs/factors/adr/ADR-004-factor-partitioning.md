# ADR-004: Factor Table Partitioning Strategy

| Field      | Value |
| ---------- | ----- |
| Status     | Accepted (implementation deferred to v1.0) |
| Date       | 2026-04-27 |
| Owner      | CTO (Akshay) |
| Author     | GL-BackendDeveloper (under CTO direction) |
| Related    | `ADR-001-greenlang-factors-source-of-truth.md`, `docs/factors/PHASE_2_PLAN.md` §2.4 |
| Phase      | 2 / TaskCreate #7 / WS7-T1 |

> **Numbering note**: The Phase 2 brief refers to this ADR as
> "ADR-003-factor-partitioning". Slot ADR-003 is already in use for
> `ADR-003-ghgp-method-references-citation-vs-data-split.md` (committed
> 2026-04-26 under the methodology workstream). To preserve the linear
> ADR numbering convention, this partitioning decision is recorded as
> ADR-004.

## Context

`factors_v0_1.factor` (V500 canonical schema) carries the central
immutable record set for the GreenLang Factors product. v0.1 alpha
ships ~1,500 rows; the v1.0 catalog expansion is projected at >100k
rows once the long-tail commercial connectors (eGRID intra-year
revisions, CBAM updates, ecoinvent cuts) land.

Two partitioning pressures matter:

1. **Range scan latency** — the most common API filter is
   `vintage_start >= ? AND vintage_end <= ?`. As the table grows,
   sequential scans across all vintage years degrade P95 lookup
   latency. The v0.1 SLO is P95 < 100 ms for `GET /factors/{urn}`
   and < 150 ms for filtered list queries; once raw-row count crosses
   ~100k the unpartitioned table is projected to breach the list-query
   SLO under realistic ingestion-time write load.
2. **Hot-writer contention** — the same calendar window can see large
   bulk inserts from a single source (eGRID annual cut, CBAM quarterly
   update). A single-node B-tree heap forces those writes to share the
   same heap relation, increasing WAL contention and vacuum pressure.

The V500 schema is **frozen**; partitioning therefore cannot retro-fit
the existing table. A v1.0 schema evolution will introduce a new
partitioned table and migrate rows in-place.

## Decision

When **either** trigger condition is met, partition `factors_v0_1.factor`:

| Trigger                                 | Threshold                  |
|-----------------------------------------|----------------------------|
| Total `factor` row count                | > 100,000                  |
| `GET /factors` filtered-list P95        | > 100 ms (warm cache)      |

Use **two-level Postgres declarative partitioning**:

1. **First level: RANGE on `vintage_start` (yearly buckets).**
   Each bucket holds factors whose vintage year matches the bucket
   year (`PARTITION FOR VALUES FROM ('YYYY-01-01') TO ('YYYY+1-01-01')`).
   New years get a fresh bucket via a quarterly cron; the
   `default` partition catches anything outside the configured range
   so writes never fail.

   ```sql
   CREATE TABLE factors_v0_1.factor_partitioned (
       LIKE factors_v0_1.factor INCLUDING ALL
   ) PARTITION BY RANGE (vintage_start);

   CREATE TABLE factors_v0_1.factor_2024
     PARTITION OF factors_v0_1.factor_partitioned
     FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')
     PARTITION BY HASH (source_urn);
   ```

2. **Second level: HASH on `source_urn` (4 buckets).**
   Each yearly bucket is itself sub-partitioned by hash of
   `source_urn` into four leaves. This evenly distributes the hot
   writers (eGRID, CBAM, IPCC AR6) across separate heaps, removing
   single-relation WAL contention and letting parallel sequential
   scans light up multiple workers for filtered list queries.

   ```sql
   CREATE TABLE factors_v0_1.factor_2024_h0
     PARTITION OF factors_v0_1.factor_2024
     FOR VALUES WITH (MODULUS 4, REMAINDER 0);
   -- ... h1, h2, h3
   ```

Indexes (`urn` UNIQUE, `geography_urn`+`vintage_*`, FTS on
`name||description`, GIN on `tags`) are recreated per leaf as inherited
local indexes; the `unique (urn)` index is enforced via a global
partitioned-index using `CREATE UNIQUE INDEX ... ON ... (urn)` at the
parent level (Postgres 16 supports this when `urn` is part of every
partition key, which it is — every leaf's row carries a urn).

## Rationale

- **Vintage-year is the dominant range filter** in real-world traffic
  (compliance teams pull "all 2024 emission factors for grid X" or
  "all factors with `vintage_end >= 2023-01-01`"). RANGE on
  `vintage_start` lets the planner partition-prune to a single year
  bucket, which slashes the working set by 7-10x for typical queries.
- **`source_urn` distributes hot writers evenly.** A naive HASH-only
  scheme would still hit every bucket on a yearly bulk insert; a
  RANGE-only scheme would still hot-spot one writer onto a single
  yearly heap. The two-level scheme decouples the two pressures.
- **4 hash buckets** balances index cardinality (each leaf stays small
  enough for B-tree to fit in shared_buffers comfortably) against
  partition-management overhead (a 4-way fan-out keeps `pg_class`
  bloat manageable across 10+ year buckets).
- **`PARTITION BY RANGE (vintage_start)` over `LIST (date_trunc('year', vintage_start))`**:
  RANGE partitions are more flexible (we can keep adding years without
  altering the parent) and Postgres can prune them faster because the
  bound check is direct comparison rather than function-result
  comparison. The Phase 2 brief mentions LIST + `date_trunc`; we adopt
  RANGE because it composes cleanly with sub-partitioning by HASH and
  avoids wrapping every bound check in a function call.

## Out of scope

This ADR captures the **strategy and trigger conditions only**. The
following are explicitly NOT delivered by V500/V501/V503/V504:

- The actual `CREATE TABLE ... PARTITION BY ...` migration.
- Online row migration tooling (logical replication or
  `INSERT ... SELECT` cutover).
- Index re-creation per leaf.
- API/SDK changes (none — clients keep using `factors_v0_1.factor`
  via a view-or-rename pattern at cutover time).

These ship in the v1.0 release once a trigger condition is hit.

## Consequences

**Positive**
- Filtered list-query P95 stays < 100 ms past the 100k-row mark.
- Hot writers (eGRID, CBAM) no longer serialize on a single heap.
- New yearly buckets are added via a quarterly cron — zero on-call
  burden after the v1.0 cutover.

**Negative**
- v1.0 cutover requires a maintenance window or logical replication
  migration; both are non-trivial.
- Cross-partition queries (e.g. "every factor for `urn:gl:source:eGRID`
  across all years") now span 10+ leaves; the planner handles this
  fine but operators must remember partition_pruning is per-query.
- `urn` UNIQUE enforcement requires Postgres 16+ partitioned global
  indexes; pre-16 deployments cannot adopt this strategy as-stated.

## Validation triggers

The Phase 2 KPI dashboard
(`docs/factors/PHASE_2_KPI_DASHBOARD.md`) tracks both triggers:

- `factor_row_count` (Postgres `pg_stat_user_tables` reltuples)
- `factor_list_query_p95_ms` (Prometheus histogram from the alpha API)

When either crosses the threshold defined above, the v1.0 partitioning
migration becomes a P0 task and this ADR becomes the implementation
spec.
