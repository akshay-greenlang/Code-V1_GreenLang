---
title: Changelog
description: SDK + API release history.
---

# Changelog

The Python SDK (`greenlang-factors`), the TypeScript SDK (`@greenlang/factors`), and the API server share a single version line.

For the full per-release detail, see the [SDK CHANGELOG](https://github.com/greenlang/greenlang/blob/master/greenlang/factors/sdk/CHANGELOG.md) and the [v1.0.0 release notes](https://github.com/greenlang/greenlang/blob/master/greenlang/factors/sdk/RELEASE_NOTES_v1.0.0.md).

## v1.0.0 -- 2026-05-01 (planned)

**General Availability** of the GreenLang Factors SDK.

* Edition pinning helpers (`pin_edition`, `with_edition`).
* Offline signed-receipt verification (HMAC-SHA256 + Ed25519).
* `gl-factors verify-receipt` standalone CLI.
* New typed exceptions: `LicensingGapError`, `EditionPinError`, `EntitlementError`.
* Rate-limit-aware backoff (Retry-After honoured + surfaced on `RateLimitError`).
* Python 3.10+, Node 18+.
* PyPI distribution renamed `greenlang-factors-sdk` -> `greenlang-factors`.

## v1.1.0 (pre-release) -- 2026-04-20

Pre-release shipped to early-access customers; consolidated into v1.0.0 GA. See the [archived 1.1.0 changelog entry](https://github.com/greenlang/greenlang/blob/master/greenlang/factors/sdk/CHANGELOG.md#110--2026-04-20) for detail.

## v1.0.0 (early-access) -- 2026-02

First publishable release shipped to design partners. Superseded by the GA v1.0.0.
