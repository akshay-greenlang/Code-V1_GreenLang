# Factors Status Page

This directory holds the declarative config for the public status page.

## Files

- `statuspage.yaml` — dual-format config (statuspage.io + Cachet).

## Bootstrap

Apply the config with whichever tool matches your platform:

### statuspage.io

```bash
# api token in kv/factors/ops/statuspage
python scripts/statuspage_sync.py \
  --config deployment/status_page/statuspage.yaml \
  --page-id <page-id>
```

### Cachet (self-host)

```bash
helm upgrade --install factors-status \
  cachethq/cachet \
  -f deployment/status_page/statuspage.yaml \
  -n factors-status --create-namespace
```

## Update playbook

See `docs/runbooks/status_page_updates.md` for the who/when/voice rules.
