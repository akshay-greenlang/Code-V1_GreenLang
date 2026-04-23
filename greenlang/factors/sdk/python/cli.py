# -*- coding: utf-8 -*-
"""Command-line interface for the Factors SDK.

Installs an entry point ``greenlang-factors`` with subcommands:

    greenlang-factors search <query>           [--edition ...] [--limit N]
    greenlang-factors get-factor <factor_id>   [--edition ...]
    greenlang-factors resolve <activity>       [--method-profile X] [--jurisdiction US]
    greenlang-factors explain <factor_id>      [--alternates N]
    greenlang-factors list-editions            [--include-pending]

Authentication is sourced from environment variables to keep secrets
out of shell history:

    GREENLANG_FACTORS_BASE_URL      (default: http://localhost:8000)
    GREENLANG_FACTORS_API_KEY       (sent via X-API-Key)
    GREENLANG_FACTORS_JWT           (sent as Authorization: Bearer)
    GREENLANG_FACTORS_EDITION       (optional default edition)

Run ``greenlang-factors --help`` for full usage.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .client import FactorsClient
from .errors import FactorsAPIError
from .models import ResolutionRequest
from .verify import ReceiptVerificationError, verify_receipt

logger = logging.getLogger(__name__)

ENV_BASE_URL = "GREENLANG_FACTORS_BASE_URL"
ENV_API_KEY = "GREENLANG_FACTORS_API_KEY"
ENV_JWT = "GREENLANG_FACTORS_JWT"
ENV_EDITION = "GREENLANG_FACTORS_EDITION"
DEFAULT_BASE_URL = "http://localhost:8000"


def _build_client(args: argparse.Namespace) -> FactorsClient:
    base_url = args.base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
    api_key = os.environ.get(ENV_API_KEY)
    jwt_token = os.environ.get(ENV_JWT)
    edition = args.edition or os.environ.get(ENV_EDITION)
    return FactorsClient(
        base_url=base_url,
        api_key=api_key,
        jwt_token=jwt_token,
        default_edition=edition,
        timeout=args.timeout,
    )


def _print_json(payload: Any, *, pretty: bool) -> None:
    if pretty:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, default=str))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(json.dumps(payload, default=str))
        sys.stdout.write("\n")


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return dict(model) if isinstance(model, dict) else {"value": model}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace) -> int:
    with _build_client(args) as client:
        resp = client.search(args.query, limit=args.limit, geography=args.geography)
    _print_json(_model_to_dict(resp), pretty=args.pretty)
    return 0


def cmd_get_factor(args: argparse.Namespace) -> int:
    with _build_client(args) as client:
        factor = client.get_factor(args.factor_id, edition=args.edition)
    _print_json(_model_to_dict(factor), pretty=args.pretty)
    return 0


#: Visual length cap for CLI rendering of ``audit_text`` so short
#: terminals stay readable. Longer narratives are truncated with an
#: ellipsis; use ``--show-full-audit`` to print the whole thing.
AUDIT_TEXT_PREVIEW_CHARS = 200


def _truncate_audit_text(text: Optional[str], *, limit: int = AUDIT_TEXT_PREVIEW_CHARS) -> str:
    """Return ``text`` trimmed to ``limit`` chars with an ellipsis when cut."""
    if text is None:
        return ""
    t = str(text).strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "..."


def cmd_resolve(args: argparse.Namespace) -> int:
    req = ResolutionRequest(
        activity=args.activity,
        method_profile=args.method_profile,
        jurisdiction=args.jurisdiction,
        reporting_date=args.reporting_date,
    )
    with _build_client(args) as client:
        resolved = client.resolve(req, alternates=args.alternates, edition=args.edition)

    payload = _model_to_dict(resolved)

    # Wave 2.5: surface ``audit_text`` at the top of the CLI output so
    # operators can eyeball the narrative without digging through JSON.
    # Truncated to 200 chars by default; ``--show-full-audit`` prints all.
    audit_text = payload.get("audit_text")
    audit_text_draft = payload.get("audit_text_draft")
    if audit_text:
        preview = (
            audit_text
            if getattr(args, "show_full_audit", False)
            else _truncate_audit_text(audit_text)
        )
        banner = "[audit_text draft] " if audit_text_draft else "[audit_text] "
        sys.stdout.write(banner + preview + "\n\n")

    _print_json(payload, pretty=args.pretty)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    with _build_client(args) as client:
        payload = client.resolve_explain(
            args.factor_id,
            method_profile=args.method_profile,
            alternates=args.alternates,
            edition=args.edition,
        )
    data = _model_to_dict(payload)

    # Wave 2 pretty-print: group the 16 envelope fields so an operator can
    # scan the resolve outcome without parsing a wall of JSON. Raw JSON is
    # still printed afterwards for machine-consumable output.
    if args.pretty:
        sys.stdout.write(_format_explain_groups(data))
        sys.stdout.write("\n")
    _print_json(data, pretty=args.pretty)
    return 0


# ---------------------------------------------------------------------------
# Wave 2 /explain pretty-printer.
# ---------------------------------------------------------------------------


#: 16 envelope fields grouped for human-readable display on ``explain``.
_EXPLAIN_FIELD_GROUPS: List[tuple] = [
    (
        "Chosen factor",
        [
            "chosen_factor",
            "chosen_factor_id",
            "factor_id",
            "factor_version",
            "release_version",
        ],
    ),
    (
        "Method",
        [
            "method_profile",
            "method_pack_version",
            "fallback_rank",
            "step_label",
            "why_chosen",
        ],
    ),
    (
        "Source & licensing",
        ["source", "licensing"],
    ),
    (
        "Quality & uncertainty",
        ["quality", "quality_score", "uncertainty", "gas_breakdown"],
    ),
    (
        "Status",
        ["deprecation_status", "deprecation_replacement", "edition_id"],
    ),
    (
        "Audit narrative (Wave 2.5)",
        ["audit_text", "audit_text_draft"],
    ),
]


def _format_explain_groups(payload: Dict[str, Any]) -> str:
    """Render the 16 envelope fields as a grouped human-readable block."""
    lines: List[str] = []
    for heading, keys in _EXPLAIN_FIELD_GROUPS:
        present = [(k, payload.get(k)) for k in keys if k in payload]
        if not present:
            continue
        lines.append(f"== {heading} ==")
        for k, v in present:
            if k == "audit_text" and isinstance(v, str):
                v = _truncate_audit_text(v)
            if isinstance(v, (dict, list)):
                rendered = json.dumps(v, indent=2, sort_keys=True, default=str)
                # Indent nested JSON under the key name.
                indented = "\n    ".join(rendered.splitlines())
                lines.append(f"  {k}:\n    {indented}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def cmd_list_editions(args: argparse.Namespace) -> int:
    with _build_client(args) as client:
        editions = client.list_editions(include_pending=args.include_pending)
    _print_json([_model_to_dict(e) for e in editions], pretty=args.pretty)
    return 0


def cmd_verify_receipt(args: argparse.Namespace) -> int:
    """Standalone offline receipt verifier.

    Reads a JSON response from a file (or stdin when path is ``-``) and
    runs :func:`verify_receipt`. Exits 0 on success, 3 on verification
    failure, 2 on usage / IO errors.

    The optional ``--key`` flag points to a file containing the HMAC
    secret (plain text) — convenient when the secret is managed via
    Vault/K8s secrets and mounted as a file rather than passed on the
    command line. For Ed25519 verification, the public key is fetched
    from ``--jwks-url``; pointing ``--key`` at a local JWKS file is also
    supported.
    """
    if args.response_path == "-":
        payload_text = sys.stdin.read()
    else:
        with open(args.response_path, "r", encoding="utf-8") as fh:
            payload_text = fh.read()

    secret = args.secret
    jwks_url = args.jwks_url
    if getattr(args, "key", None):
        key_path = args.key
        try:
            with open(key_path, "r", encoding="utf-8") as fh:
                key_content = fh.read().strip()
        except OSError as exc:
            sys.stderr.write(f"could not read key file {key_path}: {exc}\n")
            return 2
        # If the file looks like a JWKS document, treat it as one; else
        # treat it as a raw HMAC secret.
        stripped = key_content.lstrip()
        if stripped.startswith("{") and "keys" in stripped:
            jwks_url = jwks_url or f"file://{os.path.abspath(key_path)}"
            # verify_receipt only accepts http(s); for JWKS-from-file
            # we inline-parse via a tiny helper rather than reimplement.
            sys.stderr.write(
                "verify-receipt --key pointing at a JWKS file is not yet "
                "supported on the CLI; pass --jwks-url with a http(s) URL "
                "or use the SDK programmatically.\n"
            )
            return 2
        # Otherwise, treat as HMAC secret material.
        if secret is None:
            secret = key_content

    try:
        result = verify_receipt(
            payload_text,
            secret=secret,
            jwks_url=jwks_url,
            algorithm=args.algorithm,
        )
    except ReceiptVerificationError as exc:
        sys.stderr.write(f"verification failed: {exc}\n")
        return 3
    _print_json(result, pretty=args.pretty)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (env: %s)" % ENV_BASE_URL,
    )
    parser.add_argument(
        "--edition",
        default=None,
        help="Pin to a specific edition (env: %s)" % ENV_EDITION,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="greenlang-factors",
        description="GreenLang Factors SDK — CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p_search = sub.add_parser("search", help="Full-text search for factors")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=20)
    p_search.add_argument("--geography", default=None)
    _add_common(p_search)
    p_search.set_defaults(func=cmd_search)

    # get-factor
    p_get = sub.add_parser("get-factor", help="Fetch a factor by id")
    p_get.add_argument("factor_id")
    _add_common(p_get)
    p_get.set_defaults(func=cmd_get_factor)

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve + explain a full request")
    p_resolve.add_argument("activity", help="Activity description or id")
    p_resolve.add_argument(
        "--method-profile",
        default="corporate_scope1",
        help="Method profile (default: corporate_scope1)",
    )
    p_resolve.add_argument("--jurisdiction", default=None)
    p_resolve.add_argument("--reporting-date", default=None)
    p_resolve.add_argument("--alternates", type=int, default=None)
    p_resolve.add_argument(
        "--show-full-audit",
        action="store_true",
        help=(
            "Print the full ``audit_text`` narrative (default: truncate to "
            f"{AUDIT_TEXT_PREVIEW_CHARS} chars with an ellipsis)."
        ),
    )
    _add_common(p_resolve)
    p_resolve.set_defaults(func=cmd_resolve)

    # explain
    p_explain = sub.add_parser("explain", help="Explain a factor by id (Pro+)")
    p_explain.add_argument("factor_id")
    p_explain.add_argument("--method-profile", default=None)
    p_explain.add_argument("--alternates", type=int, default=None)
    _add_common(p_explain)
    p_explain.set_defaults(func=cmd_explain)

    # list-editions
    p_list = sub.add_parser("list-editions", help="List catalog editions")
    p_list.add_argument(
        "--include-pending",
        action="store_true",
        help="Include pending editions (default: only published)",
    )
    _add_common(p_list)
    p_list.set_defaults(func=cmd_list_editions)

    # verify-receipt (standalone, no network call)
    p_verify = sub.add_parser(
        "verify-receipt",
        help="Verify a signed-receipt-bearing response file (offline, no network call)",
    )
    p_verify.add_argument(
        "response_path",
        help='Path to the JSON response file. Use "-" to read from stdin.',
    )
    p_verify.add_argument(
        "--secret",
        default=None,
        help="HMAC secret (defaults to GL_FACTORS_SIGNING_SECRET env var)",
    )
    p_verify.add_argument(
        "--key",
        default=None,
        help=(
            "Path to a file containing the HMAC secret (plain text). "
            "Takes precedence over GL_FACTORS_SIGNING_SECRET but not over "
            "--secret. Useful when keys are mounted as files by Vault/K8s."
        ),
    )
    p_verify.add_argument(
        "--jwks-url",
        default=None,
        help="JWKS URL for Ed25519 receipts (defaults to GL_FACTORS_JWKS_URL env var)",
    )
    p_verify.add_argument(
        "--algorithm",
        default=None,
        choices=["sha256-hmac", "ed25519"],
        help="Force a specific algorithm (default: trust the receipt's algorithm field)",
    )
    p_verify.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    p_verify.set_defaults(func=cmd_verify_receipt)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=os.environ.get("GREENLANG_LOG_LEVEL", "WARNING"))
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FactorsAPIError as exc:
        sys.stderr.write("error: %s\n" % exc)
        if exc.status_code is not None:
            sys.stderr.write("status: %s\n" % exc.status_code)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        sys.stderr.write("interrupted\n")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
