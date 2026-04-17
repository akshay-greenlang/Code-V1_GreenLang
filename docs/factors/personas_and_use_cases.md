# Personas and core use cases (FY27)

## Personas

1. **API Developer** — embeds factor lookup in SaaS; needs pinning and OpenAPI.
2. **Data Engineer** — runs ingestion pipelines; needs artifacts and checksums.
3. **Sustainability Consultant** — defends methodology; needs citations and audit export.
4. **Climate Platform PM** — scopes integration; needs licensing clarity.
5. **Methodology Lead** — approves certified promotions; needs review queue.
6. **Enterprise Security** — reviews data residency and connector mode.

## Twenty core use cases

1. Pin API responses to a named edition for audit reproducibility.
2. List editions and read changelog before upgrading pin.
3. Search factors by fuel, geography, scope, boundary, and **status**.
4. Retrieve factor detail with ETag for caching.
5. Retrieve provenance and license for a factor.
6. Compare two editions for regression testing (added/removed/changed).
7. Walk deprecation chain to current replacement factor.
8. Ingest public EPA/DESNZ-style tables into preview, then promote to certified.
9. Attach connector-only dataset (IEA/ecoinvent) without open redistribution.
10. Record raw artifact with SHA-256 and retrieval timestamp.
11. Run QA validators on a batch before release.
12. Open methodology review ticket for borderline factors.
13. Match natural-language activity to ranked candidates with explanations.
14. Run offline eval (recall@5, precision@1) on gold set.
15. Receive alert when upstream source file hash changes.
16. Classify change as numeric vs policy vs parser break.
17. Generate draft changelog; human approves before stable publish.
18. Hotfix wrong default factor in under 24h with rollback edition env.
19. Export audit bundle (raw + parser + normalized + QA) for one factor.
20. Report API usage by key for enterprise billing.
