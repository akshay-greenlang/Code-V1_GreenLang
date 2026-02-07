# -*- coding: utf-8 -*-
"""
Agent CLI Command Group - GreenLang Agent Factory CLI (INFRA-010 Phase 4)

Provides a unified command group for managing GreenLang agents through the
entire lifecycle: create, test, deploy, rollback, status, logs, inspect,
pack, and publish.

Usage:
    gl agent create --name my-agent --template deterministic
    gl agent test --agent-dir ./agents/my-agent --coverage
    gl agent deploy --agent-key my-agent --env staging
    gl agent status --all
    gl agent logs --agent-key my-agent --tail 200
    gl agent inspect --agent-key my-agent --deps
    gl agent pack --agent-dir ./agents/my-agent
    gl agent publish --package ./dist/my-agent.glpack
    gl agent rollback --agent-key my-agent --version 1.0.0

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="agent",
    help="Agent Factory: Full lifecycle management for GreenLang agents.",
    no_args_is_help=True,
)

# ---- Register all subcommands ------------------------------------------------

from greenlang.cli.commands.agent.create import create  # noqa: E402
from greenlang.cli.commands.agent.test import test  # noqa: E402
from greenlang.cli.commands.agent.deploy import deploy  # noqa: E402
from greenlang.cli.commands.agent.rollback import rollback  # noqa: E402
from greenlang.cli.commands.agent.status import status  # noqa: E402
from greenlang.cli.commands.agent.logs import logs  # noqa: E402
from greenlang.cli.commands.agent.inspect import inspect_agent  # noqa: E402
from greenlang.cli.commands.agent.pack import pack  # noqa: E402
from greenlang.cli.commands.agent.publish import publish  # noqa: E402

app.command(name="create")(create)
app.command(name="test")(test)
app.command(name="deploy")(deploy)
app.command(name="rollback")(rollback)
app.command(name="status")(status)
app.command(name="logs")(logs)
app.command(name="inspect")(inspect_agent)
app.command(name="pack")(pack)
app.command(name="publish")(publish)

__all__ = ["app"]
