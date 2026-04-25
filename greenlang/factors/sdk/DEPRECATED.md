# Legacy `greenlang-factors-sdk==1.1.0` — DEPRECATED

**Status**: DEPRECATED — DO NOT REPUBLISH
**Replacement**: `greenlang-factors==0.1.0` (alpha) — see `python/` directory in this folder
**Decision date**: 2026-04-25
**Decision authority**: CTO doc §19.1 (Python SDK `greenlang-factors` v0.1.0)

The `greenlang-factors-sdk` distribution (declared at `greenlang/factors/sdk/pyproject.toml`, package dir `greenlang_factors_sdk`) is the predecessor of the unified alpha SDK and is retained only for historical reference. It will not be republished to PyPI.

If you have it installed:
```
pip uninstall greenlang-factors-sdk
pip install greenlang-factors==0.1.0
```

The new client class lives at `greenlang.factors.sdk.python.FactorsClient`. Migration is a one-line import change for the alpha contract; the surface in 0.1.0 is intentionally narrower than what 1.1.0 attempted.
