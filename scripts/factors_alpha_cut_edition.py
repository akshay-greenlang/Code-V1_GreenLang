#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wave E / TaskCreate #24 / WS9-T2 — alpha-edition cut script.

Builds a v0.1 Alpha factor catalog edition manifest, signs it with the
Ed25519 key from ``GL_FACTORS_ED25519_PRIVATE_KEY`` (PEM), and writes the
release artefacts to ``releases/<edition-id>/``::

    releases/factors-v0.1.0-alpha-2026-04-25/
        manifest.json                    canonical-JSON manifest
        manifest.json.sig                Ed25519 signature (or *.placeholder)
        RELEASE_NOTES.md                 auto-generated release notes
        MANIFEST_HASH.txt                single line: sha256:<hex>

Usage::

    python scripts/factors_alpha_cut_edition.py                          \\
        --edition-id factors-v0.1.0-alpha-2026-04-25                     \\
        --out releases/                                                  \\
        --approver human:methodology-lead@greenlang.io

If ``--edition-id`` is omitted the script uses today's UTC date. If the
``GL_FACTORS_ED25519_PRIVATE_KEY`` env var is empty the script writes a
``manifest.json.sig.placeholder`` instead — the cut still completes (so CI
keeps green) but the artefact is marked unsigned.

The script is **idempotent** at the file-system layer: running twice with
the same inputs produces the same canonical bytes (and therefore the same
``manifest_sha256``).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the repo root is on PYTHONPATH when invoked as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from greenlang.factors.release.alpha_edition_manifest import (  # noqa: E402
    DEFAULT_METHODOLOGY_LEAD,
    PLACEHOLDER_SUFFIX,
    build_manifest,
    write_manifest,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut a v0.1 Alpha GreenLang Factors edition manifest.",
    )
    parser.add_argument(
        "--edition-id",
        default=None,
        help=(
            "Edition id (must match 'factors-v0.1.0-alpha-YYYY-MM-DD'). "
            "Defaults to today's UTC date."
        ),
    )
    parser.add_argument(
        "--out",
        default="releases",
        help=(
            "Output directory root. The cut writes into "
            "<out>/<edition-id>/."
        ),
    )
    parser.add_argument(
        "--approver",
        default=DEFAULT_METHODOLOGY_LEAD,
        help=(
            "Methodology-lead approver string of the form "
            "'human:<email>' or 'bot:<email>'."
        ),
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help=(
            "Skip Ed25519 signing even if GL_FACTORS_ED25519_PRIVATE_KEY is "
            "set. Always writes a *.placeholder file."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("factors_alpha_cut_edition")

    manifest = build_manifest(
        edition_id=args.edition_id,
        methodology_lead_approver=args.approver,
    )

    out_root = Path(args.out)
    edition_dir = out_root / manifest.edition_id
    edition_dir.mkdir(parents=True, exist_ok=True)

    pem = None
    if args.no_sign:
        pem = ""  # explicit empty -> no key resolved, falls through to placeholder
        # Use environment isolation by temporarily clearing the env var.
        prior = os.environ.pop("GL_FACTORS_ED25519_PRIVATE_KEY", None)
        try:
            manifest_path, sig_path = write_manifest(
                manifest, edition_dir, private_key_pem=pem
            )
        finally:
            if prior is not None:
                os.environ["GL_FACTORS_ED25519_PRIVATE_KEY"] = prior
    else:
        manifest_path, sig_path = write_manifest(manifest, edition_dir)

    signed = not sig_path.name.endswith(PLACEHOLDER_SUFFIX)
    log.info("Edition cut complete:")
    log.info("  edition_id        : %s", manifest.edition_id)
    log.info("  factors           : %d", len(manifest.factors))
    log.info("  sources           : %d", len(manifest.sources))
    log.info("  manifest_sha256   : %s", manifest.manifest_sha256)
    log.info("  manifest path     : %s", manifest_path)
    log.info("  signature         : %s (%s)", sig_path, "signed" if signed else "PLACEHOLDER")
    log.info("  release notes     : %s", edition_dir / "RELEASE_NOTES.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
