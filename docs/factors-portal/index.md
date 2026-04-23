---
title: GreenLang Factors -- Developer Portal
description: The global emission-factor catalog and resolution engine.
---

# GreenLang Factors

GreenLang Factors is the global emission-factor catalog and resolution engine that powers reproducible carbon accounting at scale.

Every API call returns:

* A fully provenanced factor record (source, edition, license class, uncertainty band).
* An optional signed receipt your auditor can verify months later, entirely offline.
* A pinnable edition id so the same query returns the same factor on the same edition every time.

## Where to start

| If you want to                                            | Read this                                          |
|-----------------------------------------------------------|----------------------------------------------------|
| Make your first API call in 5 minutes                     | [Quickstart](./quickstart.md)                      |
| Understand what a factor record actually contains         | [Factor record](./concepts/factor-record.md)       |
| Pin an edition for reproducible reporting                 | [Editions](./concepts/editions.md)                 |
| Decide which licensed packs your tier can call            | [Licensing classes](./concepts/licensing-classes.md) |
| Read the OpenAPI reference                                | [API: resolve](./api/resolve.md), [API: explain](./api/explain.md) |
| Install the typed Python or TypeScript SDK                | [SDK -- Python](./sdk/python.md), [SDK -- TypeScript](./sdk/typescript.md) |
| Run the published gold-set against your own integration   | [Gold set](./gold-set.md)                          |
| See what changed between SDK versions                     | [Changelog](./changelog.md)                        |

## What is the right plan?

Visit [greenlang.ai/pricing](https://greenlang.ai/pricing) for the four canonical SKUs:

| Plan                  | Price                | Best for                                              |
|-----------------------|----------------------|-------------------------------------------------------|
| Community             | Free                 | Individuals, students, OSS maintainers                |
| Developer Pro         | $499 / month         | Startups + consultants shipping into production       |
| Consulting / Platform | $2,500 / month       | Consulting firms + ESG SaaS platforms                 |
| Enterprise            | Contact sales        | Fortune 500, regulated reporting, OEM redistribution  |

## Support

* Docs: you are reading them.
* Slack (Community): [join the GreenLang community](https://greenlang.ai/slack).
* Email (Pro+): support@greenlang.io.
* Customer-success TAM (Enterprise): contact your account manager.
