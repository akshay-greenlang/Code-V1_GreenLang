# Factors Watch Pipeline — Operations

> **What.** The per-source watch pipeline monitors each upstream data source (EPA, eGRID, DESNZ, IPCC, Green-e, TCR, CBAM) and emits a change signal when a new release is detected.
> **Where.** Daily Kubernetes `CronJob` at 06:00 UTC invokes `python -m greenlang.factors.cli watch-run`.
> **Status signal.** Public `GET /api/v1/factors/watch/status`.

---

## 1. Components

| Component | Purpose | Source |
|---|---|---|
| Source registry | Declarative list of sources + cadences + watch URLs + license terms | `greenlang/factors/data/source_registry.yaml` |
| Watch scheduler | Nightly pass that HTTP-HEADs each source + classifies change | `greenlang/factors/watch/scheduler.py` |
| Change detector | SHA-256 comparison against previous_hash | `greenlang/factors/watch/change_detector.py` |
| Change classifier | Bucketises: metadata-only / content-patch / new-edition | `greenlang/factors/watch/change_classification.py` |
| Release orchestrator | Kicks the new-edition pipeline (parse → QA → signoff → promote) | `greenlang/factors/watch/release_orchestrator.py` |
| Results table | Postgres `factors_catalog.watch_results` (migration V430) | `deployment/database/migrations/sql/V430__factors_watch_results.sql` |
| Public status API | `/api/v1/factors/watch/status` (Phase 5.4, unauthenticated) | `greenlang/integration/api/routes/factors.py` |
| Status helper | Reads the results table + classifies per-source health | `greenlang/factors/watch/status_api.py` |

## 2. Per-source cadence matrix

Pulled from `source_registry.yaml` as of 2026-04-20:

| source_id | Display name | Cadence | Watch mechanism | Connector-only |
|---|---|---|---|---|
| `epa_hub` | EPA GHG Emission Factors Hub | quarterly | `http_head` | no |
| `egrid` | eGRID | annual | `http_head` | no |
| `desnz_ghg_conversion` | DESNZ GHG conversion factors | annual | `http_head` | no |
| `tcr_grp_defaults` | TCR / GRP default factors | annual | `http_head` | no |
| `ipcc` | IPCC AR6 / guidelines | major-releases | manual | no |
| `green_e` | Green-e residual mix | annual | `http_head` | no |
| `cbam_registry` | EU CBAM covered-goods register | quarterly | `http_get` | no |
| `electricity_maps` | Electricity Maps grid intensity | daily | `http_head` | **yes** |

**Rule of thumb.** The CronJob runs *daily*. Sources whose registered
cadence is slower than daily are still probed daily (to detect emergency
out-of-band revisions), but change-classification suppresses downstream
work unless a real content delta is detected.

## 3. The CronJob

File: `deployment/k8s/factors-watch-cronjob.yaml`.

Key settings:

| Setting | Value | Why |
|---|---|---|
| `schedule` | `"0 6 * * *"` (06:00 UTC) | Off the US business day, before EU morning |
| `concurrencyPolicy` | `Forbid` | No overlapping runs |
| `startingDeadlineSeconds` | 600 | Skip if controller is > 10 min late |
| `activeDeadlineSeconds` | 1800 | Hard-kill at 30 minutes |
| `backoffLimit` | 2 | Retry twice before failing the Job |
| `successfulJobsHistoryLimit` | 3 | Keep 3 green runs for debugging |
| `failedJobsHistoryLimit` | 7 | Keep a week of red runs |
| `restartPolicy` | `OnFailure` | Container restarts on transient errors |
| `runAsNonRoot` | `true` | Namespace-wide security baseline |
| `readOnlyRootFilesystem` | `true` | Only `/tmp` is writable (emptyDir, 128 Mi) |

### Deploy

```bash
kubectl apply -f deployment/k8s/factors-watch-cronjob.yaml -n greenlang-factors
kubectl describe cronjob factors-watch-daily -n greenlang-factors
```

### Trigger an ad-hoc run

```bash
kubectl create job --from=cronjob/factors-watch-daily \
  factors-watch-$(date +%Y%m%d-%H%M) -n greenlang-factors
```

## 4. Local execution

No cluster required for local dev:

```bash
export GL_FACTORS_WATCH_SQLITE="./out/factors-watch.sqlite"
python -m greenlang.factors.cli watch-run
```

Results land in `watch_results` of the same SQLite file. The public
status endpoint reads from this SQLite when running against the local
dev server.

## 5. Public status endpoint

### Contract

```
GET /api/v1/factors/watch/status?limit_per_source=10

200 OK
{
  "generated_at": "2026-04-20T06:15:32+00:00",
  "source_count": 8,
  "health_counts": {
    "healthy": 6, "stale": 1, "error": 1, "unknown": 0
  },
  "sources": [
    {
      "source_id": "egrid",
      "display_name": "eGRID",
      "cadence": "annual",
      "connector_only": false,
      "health": "healthy",
      "recent_checks": [
        {
          "id": 42, "check_timestamp": "2026-04-20T06:00:07+00:00",
          "watch_mechanism": "http_head", "url": "https://www.epa.gov/egrid",
          "http_status": 200, "file_hash": "...", "change_detected": false
        },
        ...
      ],
      "latest_timestamp": "2026-04-20T06:00:07+00:00",
      "checks_in_window": 10
    },
    ...
  ],
  "limit_per_source": 10
}
```

### Health classification

Per `status_api._classify_source`:

| Classification | Rule |
|---|---|
| `healthy` | Latest check is 2xx and ≤ 7 days old |
| `stale` | Latest check is 2xx but > 7 days old |
| `error` | Latest check carried an error message OR returned non-2xx |
| `unknown` | No rows for this source yet |

### Cache

The endpoint sets `Cache-Control: public, max-age=300`. A CDN edge can cache the response for 5 minutes without compromising freshness.

## 6. Alerting

Prometheus rules ship with the main Factors Helm chart (`deployment/helm/greenlang-factors/templates/prometheusrule.yaml`). The watch CronJob contributes three alert signals:

| Alert | Rule |
|---|---|
| `WatchCronJobFailed` | `kube_cronjob_status_last_schedule_time{cronjob="factors-watch-daily"}` older than 30 hours |
| `WatchSourceError` | `health_counts.error > 0` for > 2 consecutive runs |
| `WatchSourceStale` | Any `source` with health `stale` for > 14 days |

PagerDuty integration is owned by the central alertmanager config; see
`deployment/helm/monitoring/`.

## 7. Runbook

| Symptom | Steps |
|---|---|
| All sources `error` | Check cluster egress. Most likely a NetworkPolicy drift or an egress proxy outage. |
| One source `error` | Open the URL in a browser. If the source has moved, edit `source_registry.yaml` + commit. Re-run the CronJob. |
| `stale` without recent commit | The CronJob controller may be behind. `kubectl get cronjob -n greenlang-factors` + confirm `LAST SCHEDULE` is recent. |
| A source changed but we didn't publish | Open the issue in the release queue. Check `change_classification` output for whether it was suppressed. |

## 8. References

- `deployment/k8s/factors-watch-cronjob.yaml` — the K8s manifest.
- `deployment/helm/greenlang-factors/` — the full Helm chart (API + worker).
- `greenlang/factors/watch/scheduler.py::run_watch` — the one-shot entry point.
- `greenlang/factors/watch/status_api.py::collect_watch_status` — the public-endpoint aggregator.
- `greenlang/factors/data/source_registry.yaml` — source-of-truth for cadences + URLs.
- [`docs/factors/hosted_api.md`](hosted_api.md) — full hosted deployment guide.
- [`docs/factors/explorer-ui-spec.md`](explorer-ui-spec.md) — the companion frontend page.

---

*Last updated: 2026-04-20. Source: FY27_vs_Reality_Analysis.md §5.4 + source registry.*
