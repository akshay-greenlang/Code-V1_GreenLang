"""
GL-003 UNIFIEDSTEAM - Causal Graph Management

Manages causal graphs for steam system root cause analysis:
- Site-specific template generation
- Node and edge management
- Graph traversal (ancestors/descendants)
- Version control for causal graphs

Graph covers:
- Boilers, headers, PRVs, desuperheaters
- Steam users, traps, condensate return
- Condenser/cooling water, exogenous drivers
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import logging
import uuid
import copy

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the causal graph."""
    # Equipment nodes
    BOILER = "boiler"
    HEADER = "header"
    PRV = "prv"
    DESUPERHEATER = "desuperheater"
    TURBINE = "turbine"
    STEAM_USER = "steam_user"
    STEAM_TRAP = "steam_trap"
    CONDENSATE_TANK = "condensate_tank"
    CONDENSATE_PUMP = "condensate_pump"
    CONDENSER = "condenser"
    COOLING_TOWER = "cooling_tower"
    DEAERATOR = "deaerator"
    FEEDWATER_PUMP = "feedwater_pump"

    # Process variable nodes
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW = "flow"
    LEVEL = "level"
    ENTHALPY = "enthalpy"
    QUALITY = "quality"

    # Control nodes
    CONTROLLER = "controller"
    SETPOINT = "setpoint"
    VALVE = "valve"

    # Exogenous drivers
    AMBIENT_CONDITION = "ambient_condition"
    PRODUCTION_DEMAND = "production_demand"
    FUEL_SUPPLY = "fuel_supply"
    WATER_SUPPLY = "water_supply"

    # Derived/calculated nodes
    EFFICIENCY = "efficiency"
    LOSS = "loss"
    KPI = "kpi"


class RelationshipType(Enum):
    """Types of causal relationships."""
    # Physical relationships
    CAUSES = "causes"  # Direct causation
    AFFECTS = "affects"  # Influences
    TRANSFERS_MASS = "transfers_mass"  # Mass flow
    TRANSFERS_ENERGY = "transfers_energy"  # Energy flow
    CONSTRAINS = "constrains"  # Physical constraint

    # Control relationships
    CONTROLS = "controls"  # Controller -> process
    MEASURES = "measures"  # Sensor -> variable
    SETS = "sets"  # Setpoint -> controller

    # Temporal relationships
    PRECEDES = "precedes"  # Temporal ordering
    LAGS = "lags"  # Time-delayed effect

    # Conditional relationships
    ENABLES = "enables"  # Enables when condition met
    INHIBITS = "inhibits"  # Prevents when condition met


@dataclass
class CausalNode:
    """A node in the causal graph."""
    node_id: str
    node_type: NodeType
    name: str
    description: str

    # Location in hierarchy
    parent_system: Optional[str] = None
    subsystem: Optional[str] = None

    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)

    # State
    current_value: Optional[float] = None
    normal_range: Optional[Tuple[float, float]] = None
    unit: str = ""

    # Metadata
    is_observable: bool = True  # Can be measured
    is_controllable: bool = False  # Can be manipulated
    is_exogenous: bool = False  # External driver

    # Tags for filtering
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "description": self.description,
            "parent_system": self.parent_system,
            "subsystem": self.subsystem,
            "properties": self.properties,
            "current_value": self.current_value,
            "normal_range": list(self.normal_range) if self.normal_range else None,
            "unit": self.unit,
            "is_observable": self.is_observable,
            "is_controllable": self.is_controllable,
            "is_exogenous": self.is_exogenous,
            "tags": self.tags,
        }


@dataclass
class CausalEdge:
    """An edge in the causal graph."""
    edge_id: str
    source: str  # Source node ID
    target: str  # Target node ID
    relationship_type: RelationshipType

    # Strength of relationship
    strength: float = 1.0  # 0-1, default full strength
    confidence: float = 0.9  # Confidence in this relationship

    # Timing
    time_lag_seconds: float = 0.0  # Delay in causal effect
    is_instantaneous: bool = True

    # Directionality
    is_positive: bool = True  # True = same direction, False = inverse

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)  # When relationship applies

    # Evidence
    evidence_source: str = ""  # "physics", "data", "expert"
    evidence_description: str = ""

    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "relationship_type": self.relationship_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "time_lag_seconds": self.time_lag_seconds,
            "is_instantaneous": self.is_instantaneous,
            "is_positive": self.is_positive,
            "conditions": self.conditions,
            "evidence_source": self.evidence_source,
            "evidence_description": self.evidence_description,
        }


@dataclass
class CausalGraphTemplate:
    """Template for generating site-specific causal graphs."""
    template_id: str
    template_name: str
    description: str
    version: str

    # Template structure
    node_templates: List[Dict[str, Any]] = field(default_factory=list)
    edge_templates: List[Dict[str, Any]] = field(default_factory=list)

    # Configuration
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""

    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "template_name": self.template_name,
            "description": self.description,
            "version": self.version,
            "node_templates": self.node_templates,
            "edge_templates": self.edge_templates,
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


class CausalGraph:
    """
    Manages causal graphs for steam system analysis.

    Features:
    - Site-specific template generation
    - Node and edge management
    - Graph traversal (ancestors/descendants)
    - Version control
    """

    def __init__(
        self,
        graph_id: Optional[str] = None,
        site_id: str = "default",
        version: str = "1.0.0",
    ) -> None:
        self.graph_id = graph_id or str(uuid.uuid4())
        self.site_id = site_id
        self.version = version

        # Graph structure
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: Dict[str, CausalEdge] = {}

        # Adjacency lists for efficient traversal
        self._outgoing: Dict[str, Set[str]] = {}  # node_id -> set of target node_ids
        self._incoming: Dict[str, Set[str]] = {}  # node_id -> set of source node_ids
        self._edge_lookup: Dict[Tuple[str, str], str] = {}  # (source, target) -> edge_id

        # Version history
        self._version_history: List[Dict[str, Any]] = []
        self._current_version = version

        # Metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_modified = self.created_at

        logger.info(f"CausalGraph initialized: {self.graph_id} for site {site_id}")

    def define_site_template(
        self,
        site_config: Dict[str, Any],
    ) -> CausalGraphTemplate:
        """
        Generate a causal graph template from site configuration.

        Args:
            site_config: Site configuration including equipment lists

        Returns:
            CausalGraphTemplate for the site
        """
        template_id = str(uuid.uuid4())

        # Extract equipment from config
        boilers = site_config.get("boilers", [])
        headers = site_config.get("headers", [])
        prvs = site_config.get("prvs", [])
        desuperheaters = site_config.get("desuperheaters", [])
        turbines = site_config.get("turbines", [])
        users = site_config.get("steam_users", [])
        traps = site_config.get("steam_traps", [])

        # Generate node templates
        node_templates = []

        # Boiler nodes
        for boiler in boilers:
            node_templates.append({
                "node_type": NodeType.BOILER.value,
                "name_pattern": f"Boiler_{boiler.get('id', '{id}')}",
                "properties": boiler,
            })
            # Associated pressure and temperature nodes
            node_templates.append({
                "node_type": NodeType.PRESSURE.value,
                "name_pattern": f"Boiler_{boiler.get('id', '{id}')}_Outlet_Pressure",
                "parent_ref": f"Boiler_{boiler.get('id', '{id}')}",
            })
            node_templates.append({
                "node_type": NodeType.TEMPERATURE.value,
                "name_pattern": f"Boiler_{boiler.get('id', '{id}')}_Outlet_Temp",
                "parent_ref": f"Boiler_{boiler.get('id', '{id}')}",
            })

        # Header nodes
        for header in headers:
            node_templates.append({
                "node_type": NodeType.HEADER.value,
                "name_pattern": f"Header_{header.get('id', '{id}')}",
                "properties": header,
            })
            node_templates.append({
                "node_type": NodeType.PRESSURE.value,
                "name_pattern": f"Header_{header.get('id', '{id}')}_Pressure",
                "parent_ref": f"Header_{header.get('id', '{id}')}",
            })

        # PRV nodes
        for prv in prvs:
            node_templates.append({
                "node_type": NodeType.PRV.value,
                "name_pattern": f"PRV_{prv.get('id', '{id}')}",
                "properties": prv,
            })

        # Desuperheater nodes
        for dsh in desuperheaters:
            node_templates.append({
                "node_type": NodeType.DESUPERHEATER.value,
                "name_pattern": f"Desuperheater_{dsh.get('id', '{id}')}",
                "properties": dsh,
            })

        # Exogenous drivers
        node_templates.extend([
            {"node_type": NodeType.AMBIENT_CONDITION.value, "name_pattern": "Ambient_Temperature"},
            {"node_type": NodeType.PRODUCTION_DEMAND.value, "name_pattern": "Steam_Demand"},
            {"node_type": NodeType.FUEL_SUPPLY.value, "name_pattern": "Fuel_Supply"},
        ])

        # Generate edge templates
        edge_templates = self._generate_edge_templates(site_config)

        template = CausalGraphTemplate(
            template_id=template_id,
            template_name=f"Site_{self.site_id}_Template",
            description=f"Causal graph template for site {self.site_id}",
            version="1.0.0",
            node_templates=node_templates,
            edge_templates=edge_templates,
            required_inputs=["boilers", "headers"],
            optional_inputs=["prvs", "desuperheaters", "turbines", "steam_users", "steam_traps"],
            created_by="CausalGraph.define_site_template",
        )

        logger.info(f"Created site template: {template_id}")
        return template

    def _generate_edge_templates(
        self,
        site_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate edge templates based on site configuration."""
        edge_templates = []

        # Standard causal relationships for steam systems

        # Boiler -> Header (mass/energy transfer)
        edge_templates.append({
            "source_pattern": "Boiler_*",
            "target_pattern": "Header_HP",
            "relationship_type": RelationshipType.TRANSFERS_MASS.value,
            "evidence_source": "physics",
        })

        # Header pressure -> PRV (controls)
        edge_templates.append({
            "source_pattern": "Header_*_Pressure",
            "target_pattern": "PRV_*",
            "relationship_type": RelationshipType.CONTROLS.value,
            "evidence_source": "physics",
        })

        # PRV -> Downstream Header (mass transfer)
        edge_templates.append({
            "source_pattern": "PRV_*",
            "target_pattern": "Header_*",
            "relationship_type": RelationshipType.TRANSFERS_MASS.value,
            "evidence_source": "physics",
        })

        # PRV -> Desuperheater (causes superheat)
        edge_templates.append({
            "source_pattern": "PRV_*",
            "target_pattern": "Desuperheater_*",
            "relationship_type": RelationshipType.CAUSES.value,
            "evidence_source": "physics",
            "evidence_description": "PRV opening causes superheat increase requiring spray water",
        })

        # Ambient -> Boiler Efficiency (affects)
        edge_templates.append({
            "source_pattern": "Ambient_Temperature",
            "target_pattern": "Boiler_*",
            "relationship_type": RelationshipType.AFFECTS.value,
            "evidence_source": "physics",
        })

        # Steam Demand -> Header Pressure (affects)
        edge_templates.append({
            "source_pattern": "Steam_Demand",
            "target_pattern": "Header_*_Pressure",
            "relationship_type": RelationshipType.AFFECTS.value,
            "is_positive": False,  # Higher demand -> lower pressure
            "evidence_source": "physics",
        })

        return edge_templates

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        properties: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: str = "",
        **kwargs,
    ) -> CausalNode:
        """
        Add a node to the causal graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of node
            properties: Node properties
            name: Human-readable name
            description: Description of the node
            **kwargs: Additional node attributes

        Returns:
            The created CausalNode
        """
        if node_id in self._nodes:
            logger.warning(f"Node {node_id} already exists, updating")

        node = CausalNode(
            node_id=node_id,
            node_type=node_type,
            name=name or node_id,
            description=description,
            properties=properties or {},
            **kwargs,
        )

        self._nodes[node_id] = node

        # Initialize adjacency lists
        if node_id not in self._outgoing:
            self._outgoing[node_id] = set()
        if node_id not in self._incoming:
            self._incoming[node_id] = set()

        self.last_modified = datetime.now(timezone.utc)
        logger.debug(f"Added node: {node_id} ({node_type.value})")

        return node

    def add_edge(
        self,
        source: str,
        target: str,
        relationship_type: RelationshipType,
        edge_id: Optional[str] = None,
        **kwargs,
    ) -> CausalEdge:
        """
        Add an edge to the causal graph.

        Args:
            source: Source node ID
            target: Target node ID
            relationship_type: Type of causal relationship
            edge_id: Optional edge ID (auto-generated if not provided)
            **kwargs: Additional edge attributes

        Returns:
            The created CausalEdge
        """
        # Validate nodes exist
        if source not in self._nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self._nodes:
            raise ValueError(f"Target node {target} does not exist")

        edge_id = edge_id or str(uuid.uuid4())[:8]

        # Check for existing edge
        existing = self._edge_lookup.get((source, target))
        if existing:
            logger.warning(f"Edge from {source} to {target} already exists: {existing}")

        edge = CausalEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            relationship_type=relationship_type,
            **kwargs,
        )

        self._edges[edge_id] = edge
        self._outgoing[source].add(target)
        self._incoming[target].add(source)
        self._edge_lookup[(source, target)] = edge_id

        self.last_modified = datetime.now(timezone.utc)
        logger.debug(f"Added edge: {source} -> {target} ({relationship_type.value})")

        return edge

    def get_node(self, node_id: str) -> Optional[CausalNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Optional[CausalEdge]:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def get_edge_between(
        self,
        source: str,
        target: str,
    ) -> Optional[CausalEdge]:
        """Get edge between two nodes."""
        edge_id = self._edge_lookup.get((source, target))
        if edge_id:
            return self._edges.get(edge_id)
        return None

    def get_ancestors(
        self,
        node_id: str,
        max_depth: Optional[int] = None,
    ) -> List[CausalNode]:
        """
        Get all ancestor nodes (causes) of a node.

        Args:
            node_id: Node to find ancestors for
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of ancestor CausalNodes
        """
        if node_id not in self._nodes:
            return []

        ancestors = []
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            if max_depth is not None and depth > max_depth:
                continue

            for parent in self._incoming.get(current, set()):
                if parent not in visited:
                    visited.add(parent)
                    ancestors.append(self._nodes[parent])
                    queue.append((parent, depth + 1))

        return ancestors

    def get_descendants(
        self,
        node_id: str,
        max_depth: Optional[int] = None,
    ) -> List[CausalNode]:
        """
        Get all descendant nodes (effects) of a node.

        Args:
            node_id: Node to find descendants for
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of descendant CausalNodes
        """
        if node_id not in self._nodes:
            return []

        descendants = []
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)
            if max_depth is not None and depth > max_depth:
                continue

            for child in self._outgoing.get(current, set()):
                if child not in visited:
                    visited.add(child)
                    descendants.append(self._nodes[child])
                    queue.append((child, depth + 1))

        return descendants

    def get_direct_causes(self, node_id: str) -> List[CausalNode]:
        """Get direct causes (parent nodes) of a node."""
        if node_id not in self._nodes:
            return []

        return [
            self._nodes[parent]
            for parent in self._incoming.get(node_id, set())
        ]

    def get_direct_effects(self, node_id: str) -> List[CausalNode]:
        """Get direct effects (child nodes) of a node."""
        if node_id not in self._nodes:
            return []

        return [
            self._nodes[child]
            for child in self._outgoing.get(node_id, set())
        ]

    def get_nodes_by_type(self, node_type: NodeType) -> List[CausalNode]:
        """Get all nodes of a specific type."""
        return [
            node for node in self._nodes.values()
            if node.node_type == node_type
        ]

    def get_path(
        self,
        source: str,
        target: str,
    ) -> List[List[str]]:
        """
        Find all causal paths from source to target.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if source not in self._nodes or target not in self._nodes:
            return []

        paths = []
        stack = [(source, [source])]

        while stack:
            current, path = stack.pop()

            if current == target:
                paths.append(path)
                continue

            for child in self._outgoing.get(current, set()):
                if child not in path:  # Avoid cycles
                    stack.append((child, path + [child]))

        return paths

    def get_markov_blanket(self, node_id: str) -> Set[str]:
        """
        Get the Markov blanket of a node.

        The Markov blanket includes:
        - Parents
        - Children
        - Parents of children (co-parents)

        Args:
            node_id: Node to find Markov blanket for

        Returns:
            Set of node IDs in the Markov blanket
        """
        if node_id not in self._nodes:
            return set()

        blanket = set()

        # Parents
        parents = self._incoming.get(node_id, set())
        blanket.update(parents)

        # Children
        children = self._outgoing.get(node_id, set())
        blanket.update(children)

        # Co-parents (parents of children)
        for child in children:
            co_parents = self._incoming.get(child, set())
            blanket.update(co_parents)

        # Remove the node itself
        blanket.discard(node_id)

        return blanket

    def create_version(self, description: str = "") -> str:
        """
        Create a new version of the graph.

        Args:
            description: Description of changes

        Returns:
            New version string
        """
        # Parse current version
        parts = self._current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Increment patch version
        new_version = f"{major}.{minor}.{patch + 1}"

        # Store version history
        self._version_history.append({
            "version": self._current_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        })

        self._current_version = new_version
        self.version = new_version
        self.last_modified = datetime.now(timezone.utc)

        logger.info(f"Created graph version: {new_version}")
        return new_version

    def clone(self) -> "CausalGraph":
        """Create a deep copy of the graph."""
        new_graph = CausalGraph(
            graph_id=str(uuid.uuid4()),
            site_id=self.site_id,
            version=self.version,
        )

        # Deep copy nodes
        for node_id, node in self._nodes.items():
            new_graph._nodes[node_id] = copy.deepcopy(node)

        # Deep copy edges
        for edge_id, edge in self._edges.items():
            new_graph._edges[edge_id] = copy.deepcopy(edge)

        # Copy adjacency lists
        new_graph._outgoing = {k: set(v) for k, v in self._outgoing.items()}
        new_graph._incoming = {k: set(v) for k, v in self._incoming.items()}
        new_graph._edge_lookup = dict(self._edge_lookup)

        return new_graph

    def to_dict(self) -> Dict:
        """Convert graph to dictionary representation."""
        return {
            "graph_id": self.graph_id,
            "site_id": self.site_id,
            "version": self.version,
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "edges": {k: v.to_dict() for k, v in self._edges.items()},
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "version_history": self._version_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalGraph":
        """Create graph from dictionary representation."""
        graph = cls(
            graph_id=data.get("graph_id"),
            site_id=data.get("site_id", "default"),
            version=data.get("version", "1.0.0"),
        )

        # Restore nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = CausalNode(
                node_id=node_data["node_id"],
                node_type=NodeType(node_data["node_type"]),
                name=node_data["name"],
                description=node_data.get("description", ""),
                properties=node_data.get("properties", {}),
            )
            graph._nodes[node_id] = node
            graph._outgoing[node_id] = set()
            graph._incoming[node_id] = set()

        # Restore edges
        for edge_id, edge_data in data.get("edges", {}).items():
            edge = CausalEdge(
                edge_id=edge_data["edge_id"],
                source=edge_data["source"],
                target=edge_data["target"],
                relationship_type=RelationshipType(edge_data["relationship_type"]),
                strength=edge_data.get("strength", 1.0),
                confidence=edge_data.get("confidence", 0.9),
            )
            graph._edges[edge_id] = edge
            graph._outgoing[edge.source].add(edge.target)
            graph._incoming[edge.target].add(edge.source)
            graph._edge_lookup[(edge.source, edge.target)] = edge_id

        return graph

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": {
                t.value: len([n for n in self._nodes.values() if n.node_type == t])
                for t in NodeType
                if any(n.node_type == t for n in self._nodes.values())
            },
            "relationship_types": {
                r.value: len([e for e in self._edges.values() if e.relationship_type == r])
                for r in RelationshipType
                if any(e.relationship_type == r for e in self._edges.values())
            },
            "avg_in_degree": (
                sum(len(s) for s in self._incoming.values()) / len(self._nodes)
                if self._nodes else 0
            ),
            "avg_out_degree": (
                sum(len(s) for s in self._outgoing.values()) / len(self._nodes)
                if self._nodes else 0
            ),
        }


def create_steam_system_graph(site_config: Dict[str, Any]) -> CausalGraph:
    """
    Create a standard steam system causal graph.

    Args:
        site_config: Site configuration dictionary

    Returns:
        Populated CausalGraph
    """
    graph = CausalGraph(
        site_id=site_config.get("site_id", "default"),
    )

    # Add exogenous drivers
    graph.add_node(
        "ambient_temp",
        NodeType.AMBIENT_CONDITION,
        name="Ambient Temperature",
        description="External ambient temperature",
        is_exogenous=True,
    )

    graph.add_node(
        "steam_demand",
        NodeType.PRODUCTION_DEMAND,
        name="Steam Demand",
        description="Total steam demand from users",
        is_exogenous=True,
    )

    # Add boilers
    for boiler_id in site_config.get("boiler_ids", ["B1"]):
        graph.add_node(
            f"boiler_{boiler_id}",
            NodeType.BOILER,
            name=f"Boiler {boiler_id}",
            is_controllable=True,
        )
        graph.add_node(
            f"boiler_{boiler_id}_steam_flow",
            NodeType.FLOW,
            name=f"Boiler {boiler_id} Steam Flow",
            unit="klb/hr",
        )
        graph.add_edge(
            f"boiler_{boiler_id}",
            f"boiler_{boiler_id}_steam_flow",
            RelationshipType.CAUSES,
        )

    # Add headers
    for header in site_config.get("headers", [{"id": "HP", "pressure": 600}]):
        header_id = header.get("id", "HP")
        graph.add_node(
            f"header_{header_id}",
            NodeType.HEADER,
            name=f"{header_id} Steam Header",
            properties=header,
        )
        graph.add_node(
            f"header_{header_id}_pressure",
            NodeType.PRESSURE,
            name=f"{header_id} Header Pressure",
            unit="psig",
        )
        graph.add_edge(
            f"header_{header_id}",
            f"header_{header_id}_pressure",
            RelationshipType.CAUSES,
        )

    # Add standard causal relationships
    # Steam demand affects header pressure (inverse)
    if "header_HP_pressure" in graph._nodes:
        graph.add_edge(
            "steam_demand",
            "header_HP_pressure",
            RelationshipType.AFFECTS,
            is_positive=False,
        )

    # Ambient temp affects boiler efficiency
    for boiler_id in site_config.get("boiler_ids", ["B1"]):
        graph.add_edge(
            "ambient_temp",
            f"boiler_{boiler_id}",
            RelationshipType.AFFECTS,
        )

    logger.info(f"Created steam system graph with {len(graph._nodes)} nodes")
    return graph
