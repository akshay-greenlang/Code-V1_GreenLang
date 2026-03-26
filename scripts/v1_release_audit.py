#!/usr/bin/env python
"""Run GreenLang v1 release gate audit locally."""

from __future__ import annotations

from greenlang.v1.conformance import release_gate_checks


def main() -> int:
    checks = release_gate_checks()
    failed = [check for check in checks if not check.ok]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}")
        for detail in check.details:
            print(f"  - {detail}")
    if failed:
        print(f"\nRelease audit failed: {len(failed)} check(s) failing")
        return 1
    print("\nRelease audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

