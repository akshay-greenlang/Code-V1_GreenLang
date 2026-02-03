# TimescaleDB HA Helm Chart

Production-ready TimescaleDB with High Availability using Patroni.

## Features

- **High Availability**: Automatic failover using Patroni with Kubernetes DCS
- **Synchronous Replication**: Optional synchronous mode for zero data loss
- **Connection Pooling**: Built-in PgBouncer sidecar
- **Monitoring**: Prometheus metrics with ServiceMonitor and PrometheusRules
- **Backup**: pgBackRest integration for continuous archiving and PITR
- **Security**: NetworkPolicies, TLS support, non-root containers
- **Auto-scaling**: Horizontal Pod Autoscaler ready

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+
- PV provisioner support (for persistent storage)
- cert-manager (optional, for TLS certificates)

## Installation

### Add the repository (if published)

```bash
helm repo add greenlang https://charts.greenlang.io
helm repo update
```

### Install the chart

```bash
# Default installation
helm install timescaledb-ha ./timescaledb-ha -n database --create-namespace

# Production installation
helm install timescaledb-ha ./timescaledb-ha -n database \
  -f values-prod.yaml \
  --set postgresql.password=<secure-password> \
  --set postgresql.replication.password=<replication-password>
```

## Configuration

See `values.yaml` for all configuration options.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cluster.name` | Cluster name | `timescaledb-ha` |
| `cluster.replicas` | Number of pods (1 primary + N-1 replicas) | `3` |
| `postgresql.database` | Default database | `greenlang` |
| `postgresql.username` | Superuser name | `postgres` |
| `persistence.size` | Data volume size | `500Gi` |
| `patroni.synchronous.enabled` | Enable sync replication | `false` |
| `pgbouncer.enabled` | Enable connection pooling | `true` |
| `backup.enabled` | Enable pgBackRest backups | `true` |
| `metrics.enabled` | Enable Prometheus metrics | `true` |

### Production Configuration

For production deployments, use `values-prod.yaml`:

```bash
helm install timescaledb-ha ./timescaledb-ha -f values-prod.yaml
```

Key production settings:
- 3 replicas across availability zones
- Synchronous replication enabled
- Higher resource limits
- Strict pod security context
- Network policies enabled

## Connecting

### Get credentials

```bash
export PGPASSWORD=$(kubectl get secret timescaledb-ha-credentials -n database \
  -o jsonpath="{.data.postgres-password}" | base64 -d)
```

### Connect to primary

```bash
kubectl run psql --rm -it --image=postgres:15 -- \
  psql -h timescaledb-ha-primary.database.svc.cluster.local \
  -U postgres -d greenlang
```

### Application connection string

Via PgBouncer (recommended):
```
postgresql://postgres:${PGPASSWORD}@timescaledb-ha-pgbouncer.database.svc.cluster.local:6432/greenlang
```

## Patroni Operations

### Check cluster status

```bash
kubectl exec -n database timescaledb-ha-0 -- patronictl list
```

### Perform switchover

```bash
kubectl exec -n database timescaledb-ha-0 -- patronictl switchover
```

### Restart cluster

```bash
kubectl exec -n database timescaledb-ha-0 -- patronictl restart timescaledb-ha
```

## Backup and Recovery

### Manual backup

```bash
kubectl exec -n database timescaledb-ha-0 -- \
  pgbackrest --stanza=main backup --type=full
```

### List backups

```bash
kubectl exec -n database timescaledb-ha-0 -- \
  pgbackrest --stanza=main info
```

### Point-in-time recovery

See documentation for PITR procedures.

## Monitoring

Prometheus metrics are available at `:9187/metrics` on each pod.

Grafana dashboards available in the monitoring directory.

## Upgrading

```bash
helm upgrade timescaledb-ha ./timescaledb-ha -f values-prod.yaml
```

The pre-upgrade hook will:
1. Check cluster health
2. Verify replication lag
3. Create a backup (if enabled)
4. Validate Patroni cluster state

## Uninstalling

```bash
helm uninstall timescaledb-ha -n database
```

Note: PVCs are retained by default. To delete:
```bash
kubectl delete pvc -l cluster-name=timescaledb-ha -n database
```

## Troubleshooting

### Pod not starting

Check events:
```bash
kubectl describe pod timescaledb-ha-0 -n database
```

### Replication issues

Check replication status:
```bash
kubectl exec -n database timescaledb-ha-0 -- \
  psql -c "SELECT * FROM pg_stat_replication;"
```

### Patroni issues

Check Patroni logs:
```bash
kubectl logs -n database timescaledb-ha-0 -c timescaledb | grep patroni
```

## License

Apache 2.0
