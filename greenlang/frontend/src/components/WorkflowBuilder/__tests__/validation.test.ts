/**
 * Tests for workflow validation
 *
 * Coverage:
 * - Cycle detection
 * - Type compatibility checking
 * - Required field validation
 * - Complex workflow scenarios
 * - Connection validation
 * - Orphaned node detection
 */

import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useWorkflowValidation } from '../hooks/useWorkflowValidation';
import {
  WorkflowNode,
  WorkflowEdge,
  AgentMetadata,
  AgentCategory,
  DataType,
  ExecutionStatus,
  ValidationSeverity,
  ValidationErrorType,
} from '../types';

// Helper to create a mock agent
function createMockAgent(
  inputs: Array<{ id: string; type: DataType; required: boolean }>,
  outputs: Array<{ id: string; type: DataType }>
): AgentMetadata {
  return {
    id: 'test-agent',
    name: 'Test Agent',
    description: 'Test',
    category: AgentCategory.UTILITIES,
    version: '1.0.0',
    tags: [],
    inputs: inputs.map((i) => ({
      ...i,
      name: i.id,
      required: i.required,
      type: i.type,
    })),
    outputs: outputs.map((o) => ({
      ...o,
      name: o.id,
      required: true,
      type: o.type,
    })),
  };
}

// Helper to create a workflow node
function createNode(
  id: string,
  agent: AgentMetadata,
  position = { x: 0, y: 0 }
): WorkflowNode {
  return {
    id,
    type: 'custom',
    position,
    data: {
      agent,
      label: id,
      status: ExecutionStatus.IDLE,
      config: {},
      inputs: {},
      outputs: {},
    },
  };
}

// Helper to create an edge
function createEdge(
  id: string,
  source: string,
  target: string,
  sourcePort?: string,
  targetPort?: string
): WorkflowEdge {
  return {
    id,
    source,
    target,
    data: {
      sourcePort,
      targetPort,
      status: ExecutionStatus.IDLE,
    },
  };
}

describe('useWorkflowValidation', () => {
  describe('Cycle Detection', () => {
    it('should detect a simple cycle', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node2', 'node3', 'out', 'in'),
        createEdge('e3', 'node3', 'node1', 'out', 'in'), // Creates cycle
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors.length).toBeGreaterThan(0);
      expect(cycleErrors[0].severity).toBe(ValidationSeverity.ERROR);
    });

    it('should not detect cycles in a DAG', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node1', 'node3', 'out', 'in'),
        createEdge('e3', 'node2', 'node3', 'out', 'in'),
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors).toHaveLength(0);
    });

    it('should detect self-loop', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [createNode('node1', agent)];

      const edges = [createEdge('e1', 'node1', 'node1', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors.length).toBeGreaterThan(0);
    });

    it('should detect complex cycles', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
        createNode('node4', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node2', 'node3', 'out', 'in'),
        createEdge('e3', 'node3', 'node4', 'out', 'in'),
        createEdge('e4', 'node4', 'node2', 'out', 'in'), // Creates cycle: 2->3->4->2
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors.length).toBeGreaterThan(0);
    });
  });

  describe('Type Compatibility', () => {
    it('should allow compatible type connections', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const typeErrors = errors.filter(
        (e) => e.type === ValidationErrorType.TYPE_MISMATCH
      );

      expect(typeErrors).toHaveLength(0);
    });

    it('should detect type mismatch', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.NUMBER }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.BOOLEAN, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const typeErrors = errors.filter(
        (e) => e.type === ValidationErrorType.TYPE_MISMATCH
      );

      expect(typeErrors.length).toBeGreaterThan(0);
    });

    it('should allow ANY type to connect to any type', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.ANY }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const typeErrors = errors.filter(
        (e) =>
          e.type === ValidationErrorType.TYPE_MISMATCH &&
          e.severity === ValidationSeverity.ERROR
      );

      expect(typeErrors).toHaveLength(0);
    });

    it('should warn on auto-conversion', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.NUMBER, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const warnings = errors.filter(
        (e) =>
          e.type === ValidationErrorType.TYPE_MISMATCH &&
          e.severity === ValidationSeverity.WARNING
      );

      expect(warnings.length).toBeGreaterThan(0);
    });
  });

  describe('Required Input Validation', () => {
    it('should detect missing required inputs', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [createNode('node1', agent)];
      const edges: WorkflowEdge[] = [];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const inputErrors = errors.filter(
        (e) => e.type === ValidationErrorType.REQUIRED_INPUT_MISSING
      );

      expect(inputErrors.length).toBeGreaterThan(0);
      expect(inputErrors[0].severity).toBe(ValidationSeverity.ERROR);
    });

    it('should not error if required input is connected', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const inputErrors = errors.filter(
        (e) => e.type === ValidationErrorType.REQUIRED_INPUT_MISSING
      );

      expect(inputErrors).toHaveLength(0);
    });

    it('should allow optional inputs to be unconnected', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [createNode('node1', agent)];
      const edges: WorkflowEdge[] = [];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const inputErrors = errors.filter(
        (e) => e.type === ValidationErrorType.REQUIRED_INPUT_MISSING
      );

      expect(inputErrors).toHaveLength(0);
    });
  });

  describe('Orphaned Node Detection', () => {
    it('should detect orphaned nodes', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
      ];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const orphanErrors = errors.filter(
        (e) => e.type === ValidationErrorType.ORPHANED_NODE
      );

      expect(orphanErrors.length).toBe(1); // node3 is orphaned
      expect(orphanErrors[0].severity).toBe(ValidationSeverity.WARNING);
    });

    it('should not detect orphans in fully connected workflow', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node2', 'node3', 'out', 'in'),
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const orphanErrors = errors.filter(
        (e) => e.type === ValidationErrorType.ORPHANED_NODE
      );

      expect(orphanErrors).toHaveLength(0);
    });
  });

  describe('Invalid Port Detection', () => {
    it('should detect invalid source port', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'invalid', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const portErrors = errors.filter(
        (e) => e.type === ValidationErrorType.INVALID_PORT
      );

      expect(portErrors.length).toBeGreaterThan(0);
    });

    it('should detect invalid target port', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'invalid')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const portErrors = errors.filter(
        (e) => e.type === ValidationErrorType.INVALID_PORT
      );

      expect(portErrors.length).toBeGreaterThan(0);
    });
  });

  describe('Connection Validation', () => {
    it('should validate if connection is allowed', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];
      const edges: WorkflowEdge[] = [];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const isValid = result.current.isConnectionValid(
        'node1',
        'out',
        'node2',
        'in'
      );

      expect(isValid).toBe(true);
    });

    it('should reject connection that would create cycle', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node2', 'node3', 'out', 'in'),
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const isValid = result.current.isConnectionValid(
        'node3',
        'out',
        'node1',
        'in'
      );

      expect(isValid).toBe(false);
    });

    it('should reject incompatible type connection', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.NUMBER }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.BOOLEAN, required: true }],
        []
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];
      const edges: WorkflowEdge[] = [];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const isValid = result.current.isConnectionValid(
        'node1',
        'out',
        'node2',
        'in'
      );

      expect(isValid).toBe(false);
    });
  });

  describe('Complex Workflow Scenarios', () => {
    it('should validate a complex branching workflow', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
        createNode('node4', agent),
        createNode('node5', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node1', 'node3', 'out', 'in'),
        createEdge('e3', 'node2', 'node4', 'out', 'in'),
        createEdge('e4', 'node3', 'node5', 'out', 'in'),
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors).toHaveLength(0);
      expect(result.current.isValid).toBe(true);
    });

    it('should validate a diamond-shaped workflow', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: false }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
        createNode('node3', agent),
        createNode('node4', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node1', 'node3', 'out', 'in'),
        createEdge('e3', 'node2', 'node4', 'out', 'in'),
        createEdge('e4', 'node3', 'node4', 'out', 'in'),
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();
      const cycleErrors = errors.filter(
        (e) => e.type === ValidationErrorType.CYCLE_DETECTED
      );

      expect(cycleErrors).toHaveLength(0);
    });

    it('should detect multiple independent errors', () => {
      const agent1 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.NUMBER }]
      );
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.BOOLEAN, required: true }],
        []
      );

      const nodes = [
        createNode('node1', agent1),
        createNode('node2', agent2),
        createNode('node3', agent1),
      ];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      const errors = result.current.validate();

      // Should detect: missing input on node1, type mismatch on e1, missing input on node3, orphaned node3
      expect(errors.length).toBeGreaterThan(2);
    });
  });

  describe('Overall Workflow Validation', () => {
    it('should return isValid = true for valid workflow', () => {
      const agent1 = createMockAgent([], [{ id: 'out', type: DataType.STRING }]);
      const agent2 = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [createNode('node1', agent1), createNode('node2', agent2)];

      const edges = [createEdge('e1', 'node1', 'node2', 'out', 'in')];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      expect(result.current.isValid).toBe(true);
    });

    it('should return isValid = false for invalid workflow', () => {
      const agent = createMockAgent(
        [{ id: 'in', type: DataType.STRING, required: true }],
        [{ id: 'out', type: DataType.STRING }]
      );

      const nodes = [
        createNode('node1', agent),
        createNode('node2', agent),
      ];

      const edges = [
        createEdge('e1', 'node1', 'node2', 'out', 'in'),
        createEdge('e2', 'node2', 'node1', 'out', 'in'), // Creates cycle
      ];

      const { result } = renderHook(() => useWorkflowValidation(nodes, edges));

      expect(result.current.isValid).toBe(false);
    });
  });
});
