"""Compatibility shim for GreenLang CLI package.

Canonical CLI wiring lives in ``greenlang.cli.main``.
This module re-exports the canonical Typer app to avoid split command surfaces.
"""

from .main import app, main


def deprecated_main() -> None:
    """Backward-compatible entrypoint that forwards to canonical main."""
    main()


if __name__ == "__main__":
    main()
