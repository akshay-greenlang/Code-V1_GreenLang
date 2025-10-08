# gl init agent

Initialize a new GreenLang agent with AgentSpec v2 compliance.

## Synopsis

```bash
gl init agent <name> [OPTIONS]
```

## Description

The `gl init agent` command scaffolds a production-ready GreenLang agent pack with AgentSpec v2 manifest, Python implementation, comprehensive tests, documentation, and security defaults.

## Arguments

### name (required)
Agent name in kebab-case (e.g., `boiler-efficiency`, `climate-advisor`).

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--template, -t` | choice | `compute` | Template: compute\|ai\|industry |
| `--from-spec` | path | - | Load from existing spec.yaml |
| `--dir` | path | `.` | Output directory |
| `--force, -f` | flag | false | Overwrite existing files |
| `--license` | choice | `apache-2.0` | License: apache-2.0\|mit\|none |
| `--author` | string | - | Author name and email |
| `--no-git` | flag | false | Skip git initialization |
| `--no-precommit` | flag | false | Skip pre-commit hooks |
| `--runtimes` | csv | `local` | Runtimes: local,docker,k8s |
| `--realtime` | flag | false | Include realtime connectors |
| `--with-ci` | flag | false | Generate GitHub Actions workflow |

## Examples

### Basic compute agent:
```bash
gl init agent boiler-efficiency
```

### AI agent with CI:
```bash
gl init agent climate-advisor --template ai --with-ci
```

### From existing spec:
```bash
gl init agent my-agent --from-spec ./spec.yaml --force
```

### Full customization:
```bash
gl init agent industry-tracker \
  --template industry \
  --license apache-2.0 \
  --author "Your Name <email@example.com>" \
  --realtime \
  --with-ci \
  --dir ./agents
```

## Generated Structure

```
<agent-name>/
├── pack.yaml                    # AgentSpec v2 manifest
├── src/<python_pkg>/
│   ├── agent.py                 # Agent implementation
│   ├── schemas.py               # Pydantic models
│   ├── provenance.py            # Audit trail
│   └── realtime.py              # (if --realtime)
├── tests/
│   └── test_agent.py            # Golden + property + spec tests
├── examples/
│   ├── pipeline.gl.yaml
│   └── input.sample.json
├── docs/
│   ├── README.md
│   └── CHANGELOG.md
├── .github/workflows/ci.yml     # (if --with-ci)
├── .pre-commit-config.yaml      # (unless --no-precommit)
├── LICENSE
└── pyproject.toml
```

## Next Steps

1. `cd <agent-name>`
2. `pip install -e ".[dev,test]"`
3. `pre-commit install` (if not --no-precommit)
4. `pytest`
5. `gl run examples/pipeline.gl.yaml`

## See Also

- [AgentSpec v2](../specs/agentspec_v2.md)
- [gl pack](pack.md)
- [gl run](run.md)

Added in v0.3.0 (FRMW-202)
