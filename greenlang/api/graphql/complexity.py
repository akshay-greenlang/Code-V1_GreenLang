"""
GraphQL Query Complexity Analysis
Prevents expensive queries and enforces depth/cost limits
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import logging
from graphql import (
    GraphQLField,
    GraphQLSchema,
    GraphQLObjectType,
    FieldNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    DocumentNode,
    OperationDefinitionNode,
    SelectionSetNode,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ComplexityConfig:
    """Query complexity configuration"""

    max_depth: int = 10
    max_complexity: int = 1000
    default_field_cost: int = 1
    default_list_multiplier: int = 10

    # Field-specific costs
    field_costs: Dict[str, int] = field(default_factory=dict)

    # Type-specific multipliers
    type_multipliers: Dict[str, int] = field(default_factory=dict)

    # Enable/disable features
    enable_depth_limit: bool = True
    enable_complexity_limit: bool = True
    enable_introspection_limit: bool = True

    # Introspection limits
    max_introspection_depth: int = 5


# ==============================================================================
# Complexity Calculator
# ==============================================================================

class ComplexityCalculator:
    """
    Calculate query complexity

    Complexity is calculated by:
    1. Each field has a base cost (default: 1)
    2. List fields are multiplied by estimated result count
    3. Nested fields add their costs together
    4. Total complexity = sum of all field costs
    """

    def __init__(self, config: Optional[ComplexityConfig] = None):
        """
        Initialize calculator

        Args:
            config: Complexity configuration
        """
        self.config = config or ComplexityConfig()
        self._field_cost_cache: Dict[str, int] = {}

    def calculate_complexity(
        self,
        document: DocumentNode,
        schema: GraphQLSchema,
        variables: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Calculate total query complexity

        Args:
            document: GraphQL query document
            schema: GraphQL schema
            variables: Query variables

        Returns:
            Total complexity score
        """
        total_complexity = 0

        # Process each operation
        for definition in document.definitions:
            if isinstance(definition, OperationDefinitionNode):
                complexity = self._calculate_operation_complexity(
                    definition,
                    schema,
                    variables or {},
                )
                total_complexity += complexity

        return total_complexity

    def _calculate_operation_complexity(
        self,
        operation: OperationDefinitionNode,
        schema: GraphQLSchema,
        variables: Dict[str, Any],
    ) -> int:
        """Calculate complexity for single operation"""
        # Get root type
        if operation.operation == "query":
            root_type = schema.query_type
        elif operation.operation == "mutation":
            root_type = schema.mutation_type
        elif operation.operation == "subscription":
            root_type = schema.subscription_type
        else:
            return 0

        if not root_type:
            return 0

        # Calculate selection set complexity
        return self._calculate_selection_set_complexity(
            operation.selection_set,
            root_type,
            schema,
            variables,
            depth=0,
        )

    def _calculate_selection_set_complexity(
        self,
        selection_set: SelectionSetNode,
        parent_type: GraphQLObjectType,
        schema: GraphQLSchema,
        variables: Dict[str, Any],
        depth: int,
        multiplier: int = 1,
    ) -> int:
        """Calculate complexity for selection set"""
        if not selection_set or not selection_set.selections:
            return 0

        total_complexity = 0

        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                complexity = self._calculate_field_complexity(
                    selection,
                    parent_type,
                    schema,
                    variables,
                    depth,
                    multiplier,
                )
                total_complexity += complexity

            elif isinstance(selection, FragmentSpreadNode):
                # Handle fragment spreads (would need fragment definitions)
                pass

            elif isinstance(selection, InlineFragmentNode):
                # Handle inline fragments
                complexity = self._calculate_selection_set_complexity(
                    selection.selection_set,
                    parent_type,
                    schema,
                    variables,
                    depth,
                    multiplier,
                )
                total_complexity += complexity

        return total_complexity

    def _calculate_field_complexity(
        self,
        field_node: FieldNode,
        parent_type: GraphQLObjectType,
        schema: GraphQLSchema,
        variables: Dict[str, Any],
        depth: int,
        parent_multiplier: int,
    ) -> int:
        """Calculate complexity for single field"""
        field_name = field_node.name.value

        # Skip introspection fields if limited
        if self.config.enable_introspection_limit:
            if field_name.startswith("__"):
                if depth > self.config.max_introspection_depth:
                    return 0

        # Get field definition
        field_def = parent_type.fields.get(field_name)
        if not field_def:
            return 0

        # Get base field cost
        field_cost = self._get_field_cost(parent_type.name, field_name)

        # Get multiplier for this field
        field_multiplier = self._get_field_multiplier(
            field_node,
            field_def,
            variables,
        )

        # Total multiplier is parent * field
        total_multiplier = parent_multiplier * field_multiplier

        # Calculate cost for this field
        cost = field_cost * total_multiplier

        # Add complexity of nested fields
        if field_node.selection_set:
            # Get field return type
            return_type = field_def.type

            # Unwrap non-null and list types
            while hasattr(return_type, "of_type"):
                return_type = return_type.of_type

            if isinstance(return_type, GraphQLObjectType):
                nested_complexity = self._calculate_selection_set_complexity(
                    field_node.selection_set,
                    return_type,
                    schema,
                    variables,
                    depth + 1,
                    total_multiplier,
                )
                cost += nested_complexity

        return cost

    def _get_field_cost(self, type_name: str, field_name: str) -> int:
        """Get cost for specific field"""
        # Check cache
        cache_key = f"{type_name}.{field_name}"
        if cache_key in self._field_cost_cache:
            return self._field_cost_cache[cache_key]

        # Check configuration
        cost = self.config.field_costs.get(cache_key)
        if cost is None:
            cost = self.config.field_costs.get(field_name)
        if cost is None:
            cost = self.config.default_field_cost

        # Cache and return
        self._field_cost_cache[cache_key] = cost
        return cost

    def _get_field_multiplier(
        self,
        field_node: FieldNode,
        field_def: GraphQLField,
        variables: Dict[str, Any],
    ) -> int:
        """Get multiplier for field based on arguments"""
        # Check if field returns a list
        return_type = field_def.type
        is_list = False

        while hasattr(return_type, "of_type"):
            if hasattr(return_type, "__class__"):
                if "List" in return_type.__class__.__name__:
                    is_list = True
                    break
            return_type = return_type.of_type

        if not is_list:
            return 1

        # Try to extract limit from arguments
        multiplier = self.config.default_list_multiplier

        if field_node.arguments:
            for arg in field_node.arguments:
                arg_name = arg.name.value

                # Common pagination argument names
                if arg_name in ["first", "last", "limit", "pageSize", "page_size"]:
                    # Get argument value
                    value = self._get_argument_value(arg.value, variables)
                    if isinstance(value, int) and value > 0:
                        multiplier = min(value, multiplier)

        return multiplier

    def _get_argument_value(self, value_node: Any, variables: Dict[str, Any]) -> Any:
        """Extract argument value from node"""
        # Handle different value node types
        if hasattr(value_node, "value"):
            return value_node.value
        elif hasattr(value_node, "name"):
            # Variable reference
            var_name = value_node.name.value
            return variables.get(var_name)
        return None


# ==============================================================================
# Depth Analyzer
# ==============================================================================

class DepthAnalyzer:
    """Analyze query depth"""

    def __init__(self, max_depth: int = 10):
        """
        Initialize analyzer

        Args:
            max_depth: Maximum allowed query depth
        """
        self.max_depth = max_depth

    def calculate_depth(
        self,
        document: DocumentNode,
        schema: GraphQLSchema,
    ) -> int:
        """
        Calculate maximum query depth

        Args:
            document: GraphQL query document
            schema: GraphQL schema

        Returns:
            Maximum depth
        """
        max_depth = 0

        for definition in document.definitions:
            if isinstance(definition, OperationDefinitionNode):
                depth = self._calculate_operation_depth(definition, schema)
                max_depth = max(max_depth, depth)

        return max_depth

    def _calculate_operation_depth(
        self,
        operation: OperationDefinitionNode,
        schema: GraphQLSchema,
    ) -> int:
        """Calculate depth for operation"""
        # Get root type
        if operation.operation == "query":
            root_type = schema.query_type
        elif operation.operation == "mutation":
            root_type = schema.mutation_type
        elif operation.operation == "subscription":
            root_type = schema.subscription_type
        else:
            return 0

        if not root_type:
            return 0

        return self._calculate_selection_set_depth(
            operation.selection_set,
            root_type,
            depth=0,
        )

    def _calculate_selection_set_depth(
        self,
        selection_set: SelectionSetNode,
        parent_type: GraphQLObjectType,
        depth: int,
    ) -> int:
        """Calculate depth for selection set"""
        if not selection_set or not selection_set.selections:
            return depth

        max_depth = depth

        for selection in selection_set.selections:
            if isinstance(selection, FieldNode):
                field_depth = self._calculate_field_depth(
                    selection,
                    parent_type,
                    depth,
                )
                max_depth = max(max_depth, field_depth)

            elif isinstance(selection, InlineFragmentNode):
                fragment_depth = self._calculate_selection_set_depth(
                    selection.selection_set,
                    parent_type,
                    depth,
                )
                max_depth = max(max_depth, fragment_depth)

        return max_depth

    def _calculate_field_depth(
        self,
        field_node: FieldNode,
        parent_type: GraphQLObjectType,
        depth: int,
    ) -> int:
        """Calculate depth for field"""
        field_name = field_node.name.value

        # Skip introspection fields
        if field_name.startswith("__"):
            return depth

        # Get field definition
        field_def = parent_type.fields.get(field_name)
        if not field_def:
            return depth

        current_depth = depth + 1

        # Check nested selections
        if field_node.selection_set:
            return_type = field_def.type

            # Unwrap non-null and list types
            while hasattr(return_type, "of_type"):
                return_type = return_type.of_type

            if isinstance(return_type, GraphQLObjectType):
                return self._calculate_selection_set_depth(
                    field_node.selection_set,
                    return_type,
                    current_depth,
                )

        return current_depth


# ==============================================================================
# Complexity Validator
# ==============================================================================

class ComplexityValidator:
    """
    Validate query complexity and depth

    Raises exceptions for queries that exceed limits
    """

    def __init__(self, config: Optional[ComplexityConfig] = None):
        """
        Initialize validator

        Args:
            config: Complexity configuration
        """
        self.config = config or ComplexityConfig()
        self.complexity_calculator = ComplexityCalculator(config)
        self.depth_analyzer = DepthAnalyzer(config.max_depth)

    def validate(
        self,
        document: DocumentNode,
        schema: GraphQLSchema,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate query complexity and depth

        Args:
            document: GraphQL query document
            schema: GraphQL schema
            variables: Query variables

        Returns:
            Validation result with complexity and depth

        Raises:
            ValueError: If query exceeds limits
        """
        result = {
            "valid": True,
            "complexity": 0,
            "depth": 0,
            "errors": [],
        }

        # Calculate depth
        if self.config.enable_depth_limit:
            depth = self.depth_analyzer.calculate_depth(document, schema)
            result["depth"] = depth

            if depth > self.config.max_depth:
                result["valid"] = False
                result["errors"].append(
                    f"Query depth {depth} exceeds maximum {self.config.max_depth}"
                )

        # Calculate complexity
        if self.config.enable_complexity_limit:
            complexity = self.complexity_calculator.calculate_complexity(
                document, schema, variables
            )
            result["complexity"] = complexity

            if complexity > self.config.max_complexity:
                result["valid"] = False
                result["errors"].append(
                    f"Query complexity {complexity} exceeds maximum {self.config.max_complexity}"
                )

        # Raise if invalid
        if not result["valid"]:
            raise ValueError("; ".join(result["errors"]))

        return result

    def estimate_cost(
        self,
        document: DocumentNode,
        schema: GraphQLSchema,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Estimate query cost without validation

        Args:
            document: GraphQL query document
            schema: GraphQL schema
            variables: Query variables

        Returns:
            Cost estimation with complexity and depth
        """
        return {
            "complexity": self.complexity_calculator.calculate_complexity(
                document, schema, variables
            ),
            "depth": self.depth_analyzer.calculate_depth(document, schema),
        }


# ==============================================================================
# Default Configuration
# ==============================================================================

# Default field costs (expensive operations)
DEFAULT_FIELD_COSTS = {
    "agents": 5,
    "workflows": 5,
    "executions": 10,
    "users": 5,
    "roles": 3,
    "metrics": 15,
    "systemHealth": 10,
    "executeWorkflow": 50,
    "executeSingleAgent": 50,
    "batchCreateAgents": 100,
}

# Create default configuration
default_config = ComplexityConfig(
    max_depth=10,
    max_complexity=1000,
    default_field_cost=1,
    default_list_multiplier=10,
    field_costs=DEFAULT_FIELD_COSTS,
)

# Create default validator
default_validator = ComplexityValidator(default_config)
