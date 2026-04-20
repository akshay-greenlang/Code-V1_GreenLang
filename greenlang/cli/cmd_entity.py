# -*- coding: utf-8 -*-
"""
CLI: Entity Graph (v3 L1 Data Foundation)
===========================================

Subcommands::

    gl entity register   Register a node (organization, facility, asset, meter, …)
    gl entity link       Link two nodes with a typed edge
    gl entity tree       Print a subgraph rooted at a node
    gl entity query      List nodes filtered by type / geography / name substring
    gl entity delete     Soft-delete a node (or use --hard)

Phase 2.3 of the FY27 plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from greenlang.entity_graph import EntityGraph
from greenlang.entity_graph.models import EntityEdge, EntityNode
from greenlang.entity_graph.types import EdgeType, NodeType

app = typer.Typer(
    help="Entity Graph operations (register / link / tree / query / delete)",
    no_args_is_help=True,
)
console = Console()


def _make_graph(graph_id: str, sqlite: Optional[str]) -> EntityGraph:
    if sqlite is not None:
        return EntityGraph(
            graph_id=graph_id, storage_backend="sqlite", sqlite_path=Path(sqlite)
        )
    return EntityGraph(graph_id=graph_id, storage_backend="memory")


@app.command("register")
def register(
    node_id: str = typer.Argument(..., help="Unique node identifier"),
    node_type: str = typer.Argument(
        ..., help="One of: " + ", ".join(NodeType.ALL)
    ),
    name: str = typer.Argument(..., help="Human-readable name"),
    graph_id: str = typer.Option("default", "--graph-id"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite graph file"),
    geography: Optional[str] = typer.Option(None, "--geography", help="ISO country/region"),
    attributes: Optional[str] = typer.Option(
        None, "--attributes", help="JSON object of extra metadata"
    ),
) -> None:
    """Register a node in the entity graph."""
    attrs = {}
    if attributes:
        try:
            attrs = json.loads(attributes)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--attributes must be JSON: {exc}[/red]")
            raise typer.Exit(2)

    g = _make_graph(graph_id, sqlite)
    try:
        g.add_node(
            EntityNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                geography=geography,
                attributes=attrs,
            )
        )
    finally:
        g.close()
    console.print(f"[green][OK][/green] Registered {node_type}/{node_id}: {name}")


@app.command("link")
def link(
    source_id: str = typer.Argument(..., help="Source node id"),
    target_id: str = typer.Argument(..., help="Target node id"),
    edge_type: str = typer.Argument(
        ..., help="One of: " + ", ".join(EdgeType.ALL)
    ),
    edge_id: Optional[str] = typer.Option(None, "--edge-id"),
    graph_id: str = typer.Option("default", "--graph-id"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite"),
) -> None:
    """Link two nodes with a typed directed edge."""
    g = _make_graph(graph_id, sqlite)
    try:
        eid = g.add_edge(
            EntityEdge(
                edge_id=edge_id or "",
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
            )
        )
    finally:
        g.close()
    console.print(
        f"[green][OK][/green] Linked {source_id} -[{edge_type}]-> {target_id} "
        f"(edge_id={eid})"
    )


@app.command("tree")
def tree(
    root_id: str = typer.Argument(..., help="Root node id"),
    depth: int = typer.Option(2, "--depth", help="BFS depth"),
    graph_id: str = typer.Option("default", "--graph-id"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite"),
) -> None:
    """Print a subgraph rooted at ``root_id`` as a tree."""
    g = _make_graph(graph_id, sqlite)
    try:
        sub = g.get_subgraph(root_id, depth=depth)
    finally:
        g.close()

    root_label = next(
        (
            f"{n['node_type']}/{n['node_id']} — {n['name']}"
            for n in sub["nodes"]
            if n["node_id"] == root_id
        ),
        root_id,
    )
    rich_tree = Tree(f"[bold]{root_label}[/bold]")
    node_index = {n["node_id"]: n for n in sub["nodes"]}
    children_by_source: dict[str, list[dict]] = {}
    for e in sub["edges"]:
        children_by_source.setdefault(e["source_id"], []).append(e)

    def _walk(branch: Tree, node_id: str, seen: set[str]) -> None:
        for e in children_by_source.get(node_id, []):
            if e["target_id"] in seen:
                continue
            target = node_index.get(e["target_id"])
            if target is None:
                continue
            label = (
                f"[{e['edge_type']}] {target['node_type']}/{target['node_id']} "
                f"— {target['name']}"
            )
            sub_branch = branch.add(label)
            _walk(sub_branch, target["node_id"], seen | {target["node_id"]})

    _walk(rich_tree, root_id, {root_id})
    console.print(rich_tree)


@app.command("query")
def query(
    node_type: Optional[str] = typer.Option(None, "--type", help="Filter by node_type"),
    geography: Optional[str] = typer.Option(None, "--geography", help="Filter by geography"),
    name_contains: Optional[str] = typer.Option(None, "--name", help="Substring match"),
    graph_id: str = typer.Option("default", "--graph-id"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite"),
) -> None:
    """Find nodes by optional filters (AND logic)."""
    g = _make_graph(graph_id, sqlite)
    try:
        nodes = g.find_nodes(
            node_type=node_type,
            geography=geography,
            name_contains=name_contains,
        )
    finally:
        g.close()

    table = Table(title=f"Graph '{graph_id}'")
    table.add_column("node_id")
    table.add_column("type")
    table.add_column("name")
    table.add_column("geography")
    for n in nodes:
        table.add_row(n.node_id, n.node_type, n.name, n.geography or "-")
    console.print(table)
    console.print(f"[green]{len(nodes)} node(s)[/green]")


@app.command("delete")
def delete(
    node_id: str = typer.Argument(..., help="Node to delete"),
    graph_id: str = typer.Option("default", "--graph-id"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite"),
    hard: bool = typer.Option(False, "--hard", help="Hard delete (removes row)"),
) -> None:
    """Soft-delete (default) or hard-delete a node."""
    g = _make_graph(graph_id, sqlite)
    try:
        ok = g.delete_node(node_id, soft=not hard)
    finally:
        g.close()
    if ok:
        console.print(f"[green][OK][/green] Deleted {node_id} (hard={hard})")
    else:
        console.print(f"[yellow]No such node: {node_id}[/yellow]")
        raise typer.Exit(1)
