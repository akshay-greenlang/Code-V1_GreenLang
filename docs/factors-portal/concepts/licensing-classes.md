---
title: "Concept: licensing classes"
description: How GreenLang exposes upstream-data licences and which tiers can use which classes.
---

# Licensing classes

Every emission factor in the catalog is tagged with a **redistribution class** that captures what the upstream data publisher licensed us (and you) to do with it.

## The five classes

| Class                  | What it means                                                              |
|------------------------|----------------------------------------------------------------------------|
| `redistribute_open`    | Public-domain or open-licensed data. You can publish, copy, and redistribute freely. |
| `redistribute_restricted` | Licensed redistribution permitted under specific terms (attribution, no-derivatives, etc.). |
| `connector_only`       | We can compute against it on your behalf, but you cannot extract the raw factor value. |
| `customer_private`     | Customer-specific (your overrides, your private supplier data).            |
| `internal_only`        | GreenLang-internal vintage; not exposed via any public endpoint.           |

The `redistribution_class` field is on every Canonical Factor Record, and the API enforces it at the boundary -- a Community-tier client asking for a `connector_only` factor gets a `403 LicenseError`, not a stripped record that pretends to be open.

## Which tier sees which classes

| Tier                  | Classes accessible                                                 |
|-----------------------|--------------------------------------------------------------------|
| Community             | `redistribute_open`                                                |
| Developer Pro         | `redistribute_open` + `redistribute_restricted`                    |
| Consulting / Platform | `redistribute_open` + `redistribute_restricted` + `customer_private` |
| Enterprise            | All five (including `connector_only` + `internal_only` for OEM cases) |

Plans on Pricing -> https://greenlang.ai/pricing.

## What if I ask for something my tier cannot see?

The server returns `403` with one of two SDK-level exceptions, depending on which side of the boundary the gap falls:

* `LicenseError` -- the factor itself is `connector_only` (the SDK-side analogue of the server's "we'd love to but the upstream contract says no").
* `LicensingGapError` -- you asked for a Premium pack your contract does not include (e.g. CBAM Premium on a Community plan).

Both are subclasses of `FactorsAPIError`; you can catch them individually or generically.

## Connector-only: how it works

Some upstream data publishers (most ecoinvent / Sphera / GaBi factors) license us to **compute** but not to **redistribute** the raw value. For those factors we operate as a connector:

* You send a request (`/resolve` with a `customer_private` activity context).
* We do the math against the licensed factor on our infrastructure.
* You receive the *result* (kg CO2e for your activity), not the raw factor.
* You are billed normally; the upstream publisher receives their attribution share.

This is the model that makes Premium packs (Product LCI, Finance Proxy / PCAF, etc.) commercially viable inside the catalog without forcing every customer to negotiate their own ecoinvent / PCAF licence.

## Including overrides

Customer-private overrides (Consulting / Platform / Enterprise tiers) are also a licensing class -- they are tagged `customer_private` and visible only to your tenant. See the [Override manager](https://developers.greenlang.ai/operator/overrides) docs for how to create them.
