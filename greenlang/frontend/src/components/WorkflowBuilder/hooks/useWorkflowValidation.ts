/**
 * Custom React hook for workflow validation
 *
 * Provides comprehensive validation for workflows including:
 * - Cycle detection in DAGs
 * - Type compatibility checking
 * - Required field validation
 * - Connection rules validation
 * - Orphaned node detection
 */

import { useCallback, useMemo } from 'react';
import {
  WorkflowNode,
  WorkflowEdge,
  ValidationError,
  ValidationErrorType,
  ValidationSeverity,
  DataType,
  ConnectionRule,
  DEFAULT_CONNECTION_RULES,
} from '../types';

/**
 * Hook for validating workflows
 */
export function useWorkflowValidation(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  connectionRules: ConnectionRule[] = DEFAULT_CONNECTION_RULES
) {
  /**
   * Build adjacency list for graph traversal
   */
  const buildAdjacencyList = useCallback(
    (nodeList: WorkflowNode[], edgeList: WorkflowEdge[]) => {
      const adjacency = new Map<string, string[]>();

      // Initialize all nodes
      nodeList.forEach((node) => {
        adjacency.set(node.id, []);
      });

      // Build connections
      edgeList.forEach((edge) => {
        const neighbors = adjacency.get(edge.source) || [];
        neighbors.push(edge.target);
        adjacency.set(edge.source, neighbors);
      });

      return adjacency;
    },
    []
  );

  /**
   * Detect cycles using DFS with color-based detection
   * White (0) = unvisited, Gray (1) = visiting, Black (2) = visited
   */
  const detectCycles = useCallback(
    (nodeList: WorkflowNode[], edgeList: WorkflowEdge[]): ValidationError[] => {
      const errors: ValidationError[] = [];
      const adjacency = buildAdjacencyList(nodeList, edgeList);
      const colors = new Map<string, number>();
      const path: string[] = [];

      // Initialize all nodes as white (unvisited)
      nodeList.forEach((node) => colors.set(node.id, 0));

      const dfs = (nodeId: string): boolean => {
        colors.set(nodeId, 1); // Mark as gray (visiting)
        path.push(nodeId);

        const neighbors = adjacency.get(nodeId) || [];
        for (const neighbor of neighbors) {
          const color = colors.get(neighbor);

          if (color === 1) {
            // Back edge detected - cycle found
            const cycleStart = path.indexOf(neighbor);
            const cycle = path.slice(cycleStart);
            cycle.push(neighbor);

            const node = nodeList.find((n) => n.id === nodeId);

            errors.push({
              id: `cycle-${nodeId}-${neighbor}`,
              nodeId,
              severity: ValidationSeverity.ERROR,
              message: `Cycle detected: ${cycle.join(' → ')}`,
              type: ValidationErrorType.CYCLE_DETECTED,
              position: node?.position,
              details: { cycle },
              suggestedFix: 'Remove one of the connections in the cycle to break it',
            });

            return true;
          }

          if (color === 0 && dfs(neighbor)) {
            return true;
          }
        }

        path.pop();
        colors.set(nodeId, 2); // Mark as black (visited)
        return false;
      };

      // Check all nodes for cycles
      nodeList.forEach((node) => {
        if (colors.get(node.id) === 0) {
          dfs(node.id);
        }
      });

      return errors;
    },
    [buildAdjacencyList]
  );

  /**
   * Validate type compatibility between connected ports
   */
  const validateTypeCompatibility = useCallback(
    (nodeList: WorkflowNode[], edgeList: WorkflowEdge[]): ValidationError[] => {
      const errors: ValidationError[] = [];

      edgeList.forEach((edge) => {
        const sourceNode = nodeList.find((n) => n.id === edge.source);
        const targetNode = nodeList.find((n) => n.id === edge.target);

        if (!sourceNode || !targetNode) {
          errors.push({
            id: `invalid-edge-${edge.id}`,
            edgeId: edge.id,
            severity: ValidationSeverity.ERROR,
            message: 'Edge connects to non-existent node',
            type: ValidationErrorType.INVALID_CONNECTION,
            suggestedFix: 'Remove this edge',
          });
          return;
        }

        // Get source output port
        const sourcePortId = edge.data?.sourcePort;
        const sourcePort = sourceNode.data.agent.outputs.find(
          (p) => p.id === sourcePortId
        );

        // Get target input port
        const targetPortId = edge.data?.targetPort;
        const targetPort = targetNode.data.agent.inputs.find(
          (p) => p.id === targetPortId
        );

        if (!sourcePort) {
          errors.push({
            id: `invalid-source-port-${edge.id}`,
            edgeId: edge.id,
            nodeId: sourceNode.id,
            severity: ValidationSeverity.ERROR,
            message: `Invalid source port: ${sourcePortId}`,
            type: ValidationErrorType.INVALID_PORT,
            position: sourceNode.position,
            suggestedFix: 'Reconnect the edge to a valid output port',
          });
          return;
        }

        if (!targetPort) {
          errors.push({
            id: `invalid-target-port-${edge.id}`,
            edgeId: edge.id,
            nodeId: targetNode.id,
            severity: ValidationSeverity.ERROR,
            message: `Invalid target port: ${targetPortId}`,
            type: ValidationErrorType.INVALID_PORT,
            position: targetNode.position,
            suggestedFix: 'Reconnect the edge to a valid input port',
          });
          return;
        }

        // Check type compatibility
        const rule = connectionRules.find(
          (r) =>
            r.sourceType === sourcePort.type && r.targetType === targetPort.type
        );

        // If no specific rule, check for ANY type
        const anyRule =
          rule ||
          connectionRules.find(
            (r) =>
              (r.sourceType === DataType.ANY && r.targetType === targetPort.type) ||
              (r.sourceType === sourcePort.type && r.targetType === DataType.ANY) ||
              (r.sourceType === DataType.ANY && r.targetType === DataType.ANY)
          );

        if (!anyRule || !anyRule.allowed) {
          errors.push({
            id: `type-mismatch-${edge.id}`,
            edgeId: edge.id,
            severity: ValidationSeverity.ERROR,
            message: `Type mismatch: Cannot connect ${sourcePort.type} to ${targetPort.type}`,
            type: ValidationErrorType.TYPE_MISMATCH,
            details: {
              sourceType: sourcePort.type,
              targetType: targetPort.type,
              sourceNode: sourceNode.data.label,
              targetNode: targetNode.data.label,
            },
            suggestedFix: 'Add a type conversion node or change the connection',
          });
        } else if (anyRule.autoConvert) {
          // Add warning for auto-conversion
          errors.push({
            id: `auto-convert-${edge.id}`,
            edgeId: edge.id,
            severity: ValidationSeverity.WARNING,
            message: `Automatic conversion from ${sourcePort.type} to ${targetPort.type}`,
            type: ValidationErrorType.TYPE_MISMATCH,
            details: {
              sourceType: sourcePort.type,
              targetType: targetPort.type,
              autoConvert: true,
            },
          });
        }
      });

      return errors;
    },
    [connectionRules]
  );

  /**
   * Validate required inputs are connected
   */
  const validateRequiredInputs = useCallback(
    (nodeList: WorkflowNode[], edgeList: WorkflowEdge[]): ValidationError[] => {
      const errors: ValidationError[] = [];

      nodeList.forEach((node) => {
        const requiredInputs = node.data.agent.inputs.filter((input) => input.required);

        requiredInputs.forEach((input) => {
          // Check if input is connected
          const isConnected = edgeList.some(
            (edge) =>
              edge.target === node.id && edge.data?.targetPort === input.id
          );

          // Check if input has a configured value
          const hasValue = node.data.inputs[input.id] !== undefined;

          // Check if input has a default value
          const hasDefault = input.defaultValue !== undefined;

          if (!isConnected && !hasValue && !hasDefault) {
            errors.push({
              id: `required-input-${node.id}-${input.id}`,
              nodeId: node.id,
              severity: ValidationSeverity.ERROR,
              message: `Required input '${input.name}' is not connected or configured`,
              type: ValidationErrorType.REQUIRED_INPUT_MISSING,
              position: node.position,
              details: {
                inputId: input.id,
                inputName: input.name,
                inputType: input.type,
              },
              suggestedFix: 'Connect an output to this input or configure a value',
            });
          }
        });
      });

      return errors;
    },
    []
  );

  /**
   * Validate node configuration
   */
  const validateNodeConfiguration = useCallback(
    (nodeList: WorkflowNode[]): ValidationError[] => {
      const errors: ValidationError[] = [];

      nodeList.forEach((node) => {
        const config = node.data.config;
        const agentConfig = node.data.agent.configuration;

        if (!agentConfig) return;

        // Check required configuration fields
        Object.entries(agentConfig).forEach(([key, value]) => {
          if (typeof value === 'object' && value !== null && 'required' in value) {
            if (value.required && !(key in config)) {
              errors.push({
                id: `missing-config-${node.id}-${key}`,
                nodeId: node.id,
                severity: ValidationSeverity.ERROR,
                message: `Missing required configuration: ${key}`,
                type: ValidationErrorType.MISSING_CONFIGURATION,
                position: node.position,
                details: { configKey: key },
                suggestedFix: `Configure the '${key}' field in the node settings`,
              });
            }
          }
        });
      });

      return errors;
    },
    []
  );

  /**
   * Detect orphaned nodes (nodes with no connections)
   */
  const detectOrphanedNodes = useCallback(
    (nodeList: WorkflowNode[], edgeList: WorkflowEdge[]): ValidationError[] => {
      const errors: ValidationError[] = [];

      nodeList.forEach((node) => {
        const hasIncomingEdge = edgeList.some((edge) => edge.target === node.id);
        const hasOutgoingEdge = edgeList.some((edge) => edge.source === node.id);

        if (!hasIncomingEdge && !hasOutgoingEdge) {
          errors.push({
            id: `orphaned-${node.id}`,
            nodeId: node.id,
            severity: ValidationSeverity.WARNING,
            message: `Node '${node.data.label}' is not connected to the workflow`,
            type: ValidationErrorType.ORPHANED_NODE,
            position: node.position,
            suggestedFix: 'Connect this node to the workflow or remove it',
          });
        }
      });

      return errors;
    },
    []
  );

  /**
   * Detect duplicate connections between same ports
   */
  const detectDuplicateConnections = useCallback(
    (edgeList: WorkflowEdge[]): ValidationError[] => {
      const errors: ValidationError[] = [];
      const connections = new Set<string>();

      edgeList.forEach((edge) => {
        const connectionKey = `${edge.source}-${edge.data?.sourcePort}-${edge.target}-${edge.data?.targetPort}`;

        if (connections.has(connectionKey)) {
          errors.push({
            id: `duplicate-${edge.id}`,
            edgeId: edge.id,
            severity: ValidationSeverity.WARNING,
            message: 'Duplicate connection detected',
            type: ValidationErrorType.DUPLICATE_CONNECTION,
            suggestedFix: 'Remove one of the duplicate connections',
          });
        }

        connections.add(connectionKey);
      });

      return errors;
    },
    []
  );

  /**
   * Run all validations
   */
  const validate = useCallback((): ValidationError[] => {
    const allErrors: ValidationError[] = [];

    // Run all validation checks
    allErrors.push(...detectCycles(nodes, edges));
    allErrors.push(...validateTypeCompatibility(nodes, edges));
    allErrors.push(...validateRequiredInputs(nodes, edges));
    allErrors.push(...validateNodeConfiguration(nodes));
    allErrors.push(...detectOrphanedNodes(nodes, edges));
    allErrors.push(...detectDuplicateConnections(edges));

    return allErrors;
  }, [
    nodes,
    edges,
    detectCycles,
    validateTypeCompatibility,
    validateRequiredInputs,
    validateNodeConfiguration,
    detectOrphanedNodes,
    detectDuplicateConnections,
  ]);

  /**
   * Check if workflow is valid
   */
  const isValid = useMemo(() => {
    const errors = validate();
    return !errors.some((error) => error.severity === ValidationSeverity.ERROR);
  }, [validate]);

  /**
   * Get errors for a specific node
   */
  const getNodeErrors = useCallback(
    (nodeId: string): ValidationError[] => {
      const errors = validate();
      return errors.filter((error) => error.nodeId === nodeId);
    },
    [validate]
  );

  /**
   * Get errors for a specific edge
   */
  const getEdgeErrors = useCallback(
    (edgeId: string): ValidationError[] => {
      const errors = validate();
      return errors.filter((error) => error.edgeId === edgeId);
    },
    [validate]
  );

  /**
   * Check if a specific connection is valid
   */
  const isConnectionValid = useCallback(
    (
      sourceNodeId: string,
      sourcePortId: string,
      targetNodeId: string,
      targetPortId: string
    ): boolean => {
      const sourceNode = nodes.find((n) => n.id === sourceNodeId);
      const targetNode = nodes.find((n) => n.id === targetNodeId);

      if (!sourceNode || !targetNode) return false;

      const sourcePort = sourceNode.data.agent.outputs.find(
        (p) => p.id === sourcePortId
      );
      const targetPort = targetNode.data.agent.inputs.find(
        (p) => p.id === targetPortId
      );

      if (!sourcePort || !targetPort) return false;

      // Check if connection would create a cycle
      const tempEdge: WorkflowEdge = {
        id: 'temp',
        source: sourceNodeId,
        target: targetNodeId,
        data: { sourcePort: sourcePortId, targetPort: targetPortId },
      };

      const cycleErrors = detectCycles(nodes, [...edges, tempEdge]);
      if (cycleErrors.length > 0) return false;

      // Check type compatibility
      const rule = connectionRules.find(
        (r) =>
          r.sourceType === sourcePort.type && r.targetType === targetPort.type
      );

      const anyRule =
        rule ||
        connectionRules.find(
          (r) =>
            (r.sourceType === DataType.ANY && r.targetType === targetPort.type) ||
            (r.sourceType === sourcePort.type && r.targetType === DataType.ANY) ||
            (r.sourceType === DataType.ANY && r.targetType === DataType.ANY)
        );

      return anyRule?.allowed ?? false;
    },
    [nodes, edges, connectionRules, detectCycles]
  );

  return {
    validate,
    isValid,
    getNodeErrors,
    getEdgeErrors,
    isConnectionValid,
    detectCycles,
    validateTypeCompatibility,
    validateRequiredInputs,
    validateNodeConfiguration,
    detectOrphanedNodes,
    detectDuplicateConnections,
  };
}
