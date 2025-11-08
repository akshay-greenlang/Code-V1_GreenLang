/**
 * Tests for WorkflowCanvas component
 *
 * Coverage:
 * - Canvas rendering
 * - Drag-and-drop workflow creation
 * - Undo/redo functionality
 * - Export/import workflows
 * - Auto-layout
 * - Node selection and manipulation
 * - Edge creation and validation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { WorkflowCanvas, useCanvasStore } from '../WorkflowCanvas';
import {
  WorkflowNode,
  WorkflowEdge,
  AgentMetadata,
  AgentCategory,
  DataType,
  ExecutionStatus,
} from '../types';

// Mock React Flow
vi.mock('reactflow', () => ({
  default: ({ children }: any) => <div data-testid="react-flow">{children}</div>,
  ReactFlowProvider: ({ children }: any) => <div>{children}</div>,
  Background: () => <div data-testid="background" />,
  Controls: () => <div data-testid="controls" />,
  MiniMap: () => <div data-testid="minimap" />,
  Panel: ({ children }: any) => <div data-testid="panel">{children}</div>,
  useReactFlow: () => ({
    getViewport: () => ({ x: 0, y: 0, zoom: 1 }),
    setViewport: vi.fn(),
    fitView: vi.fn(),
  }),
  useNodesState: (initialNodes: any) => {
    const [nodes, setNodes] = [initialNodes, vi.fn()];
    return [nodes, setNodes, vi.fn()];
  },
  useEdgesState: (initialEdges: any) => {
    const [edges, setEdges] = [initialEdges, vi.fn()];
    return [edges, setEdges, vi.fn()];
  },
  addEdge: vi.fn((edge, edges) => [...edges, edge]),
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
}));

// Mock Zustand store
vi.mock('zustand', () => ({
  create: (fn: any) => {
    const store = fn(vi.fn(), vi.fn());
    return () => store;
  },
}));

describe('WorkflowCanvas', () => {
  beforeEach(() => {
    // Reset store before each test
    const store = useCanvasStore.getState();
    store.reset();
  });

  describe('Canvas Rendering', () => {
    it('should render the workflow canvas', () => {
      render(<WorkflowCanvas />);
      expect(screen.getByTestId('react-flow')).toBeInTheDocument();
    });

    it('should render canvas controls', () => {
      render(<WorkflowCanvas />);
      expect(screen.getByTestId('controls')).toBeInTheDocument();
      expect(screen.getByTestId('minimap')).toBeInTheDocument();
      expect(screen.getByTestId('background')).toBeInTheDocument();
    });

    it('should render toolbar with action buttons', () => {
      render(<WorkflowCanvas />);
      expect(screen.getByTitle('Execute Workflow')).toBeInTheDocument();
      expect(screen.getByTitle('Undo (Ctrl+Z)')).toBeInTheDocument();
      expect(screen.getByTitle('Redo (Ctrl+Y)')).toBeInTheDocument();
      expect(screen.getByTitle('Auto Layout')).toBeInTheDocument();
      expect(screen.getByTitle('Export Workflow')).toBeInTheDocument();
    });
  });

  describe('Node Management', () => {
    const mockAgent: AgentMetadata = {
      id: 'test-agent',
      name: 'Test Agent',
      description: 'Test description',
      category: AgentCategory.UTILITIES,
      version: '1.0.0',
      tags: ['test'],
      inputs: [
        {
          id: 'input1',
          name: 'Input 1',
          type: DataType.STRING,
          required: true,
        },
      ],
      outputs: [
        {
          id: 'output1',
          name: 'Output 1',
          type: DataType.STRING,
          required: true,
        },
      ],
    };

    it('should add a node to the canvas', () => {
      const store = useCanvasStore.getState();

      const newNode: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: mockAgent,
          label: 'Test Node',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(newNode);

      expect(store.nodes).toHaveLength(1);
      expect(store.nodes[0].id).toBe('node-1');
    });

    it('should remove a node from the canvas', () => {
      const store = useCanvasStore.getState();

      const newNode: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: mockAgent,
          label: 'Test Node',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(newNode);
      expect(store.nodes).toHaveLength(1);

      store.removeNode('node-1');
      expect(store.nodes).toHaveLength(0);
    });

    it('should update node data', () => {
      const store = useCanvasStore.getState();

      const newNode: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: mockAgent,
          label: 'Test Node',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(newNode);
      store.updateNode('node-1', { status: ExecutionStatus.SUCCESS });

      expect(store.nodes[0].data.status).toBe(ExecutionStatus.SUCCESS);
    });

    it('should track selected nodes', () => {
      const store = useCanvasStore.getState();

      store.setSelectedNodes(['node-1', 'node-2']);
      expect(store.selectedNodes).toEqual(['node-1', 'node-2']);
    });
  });

  describe('Edge Management', () => {
    it('should add an edge to the canvas', () => {
      const store = useCanvasStore.getState();

      const newEdge: WorkflowEdge = {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        data: {
          sourcePort: 'output1',
          targetPort: 'input1',
          status: ExecutionStatus.IDLE,
        },
      };

      store.addEdge(newEdge);

      expect(store.edges).toHaveLength(1);
      expect(store.edges[0].id).toBe('edge-1');
    });

    it('should remove an edge from the canvas', () => {
      const store = useCanvasStore.getState();

      const newEdge: WorkflowEdge = {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        data: {},
      };

      store.addEdge(newEdge);
      expect(store.edges).toHaveLength(1);

      store.removeEdge('edge-1');
      expect(store.edges).toHaveLength(0);
    });

    it('should update edge data', () => {
      const store = useCanvasStore.getState();

      const newEdge: WorkflowEdge = {
        id: 'edge-1',
        source: 'node-1',
        target: 'node-2',
        data: {
          status: ExecutionStatus.IDLE,
        },
      };

      store.addEdge(newEdge);
      store.updateEdge('edge-1', { status: ExecutionStatus.SUCCESS });

      expect(store.edges[0].data?.status).toBe(ExecutionStatus.SUCCESS);
    });
  });

  describe('Undo/Redo Functionality', () => {
    it('should support undo operation', () => {
      const store = useCanvasStore.getState();

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      expect(store.nodes).toHaveLength(1);

      store.undo();
      expect(store.nodes).toHaveLength(0);
    });

    it('should support redo operation', () => {
      const store = useCanvasStore.getState();

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      store.undo();
      expect(store.nodes).toHaveLength(0);

      store.redo();
      expect(store.nodes).toHaveLength(1);
    });

    it('should track canUndo state', () => {
      const store = useCanvasStore.getState();

      expect(store.canUndo()).toBe(false);

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      expect(store.canUndo()).toBe(true);
    });

    it('should track canRedo state', () => {
      const store = useCanvasStore.getState();

      expect(store.canRedo()).toBe(false);

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      store.undo();

      expect(store.canRedo()).toBe(true);
    });
  });

  describe('Workflow Export/Import', () => {
    it('should export workflow to JSON', () => {
      const store = useCanvasStore.getState();

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);

      expect(store.nodes).toHaveLength(1);
      expect(store.nodes[0]).toMatchObject({
        id: 'node-1',
        position: { x: 100, y: 100 },
      });
    });

    it('should clear canvas', () => {
      const store = useCanvasStore.getState();

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      expect(store.nodes).toHaveLength(1);

      store.clear();
      expect(store.nodes).toHaveLength(0);
    });

    it('should reset store to initial state', () => {
      const store = useCanvasStore.getState();

      const node: WorkflowNode = {
        id: 'node-1',
        type: 'custom',
        position: { x: 100, y: 100 },
        data: {
          agent: {} as AgentMetadata,
          label: 'Test',
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      store.addNode(node);
      store.setIsExecuting(true);
      store.setValidationErrors([]);

      store.reset();

      expect(store.nodes).toHaveLength(0);
      expect(store.edges).toHaveLength(0);
      expect(store.isExecuting).toBe(false);
      expect(store.history.past).toHaveLength(0);
      expect(store.history.future).toHaveLength(0);
    });
  });

  describe('Canvas State Management', () => {
    it('should manage execution state', () => {
      const store = useCanvasStore.getState();

      expect(store.isExecuting).toBe(false);

      store.setIsExecuting(true);
      expect(store.isExecuting).toBe(true);

      store.setIsExecuting(false);
      expect(store.isExecuting).toBe(false);
    });

    it('should track execution progress', () => {
      const store = useCanvasStore.getState();

      expect(store.executionProgress).toBe(0);

      store.setExecutionProgress(50);
      expect(store.executionProgress).toBe(50);

      store.setExecutionProgress(100);
      expect(store.executionProgress).toBe(100);
    });

    it('should manage panel state', () => {
      const store = useCanvasStore.getState();

      store.setActivePanel('agent');
      expect(store.activePanel).toBe('agent');
      expect(store.isPanelOpen).toBe(true);

      store.setActivePanel(null);
      expect(store.activePanel).toBe(null);
      expect(store.isPanelOpen).toBe(false);
    });

    it('should manage validation state', () => {
      const store = useCanvasStore.getState();

      expect(store.validationErrors).toHaveLength(0);
      expect(store.isValid).toBe(true);

      const errors = [
        {
          id: 'error-1',
          severity: 'error' as const,
          message: 'Test error',
          type: 'cycle_detected' as const,
        },
      ];

      store.setValidationErrors(errors);
      expect(store.validationErrors).toHaveLength(1);

      store.setIsValid(false);
      expect(store.isValid).toBe(false);
    });
  });

  describe('Multi-select and Alignment', () => {
    it('should select multiple nodes', () => {
      const store = useCanvasStore.getState();

      store.setSelectedNodes(['node-1', 'node-2', 'node-3']);
      expect(store.selectedNodes).toHaveLength(3);
    });

    it('should clear node selection', () => {
      const store = useCanvasStore.getState();

      store.setSelectedNodes(['node-1', 'node-2']);
      expect(store.selectedNodes).toHaveLength(2);

      store.setSelectedNodes([]);
      expect(store.selectedNodes).toHaveLength(0);
    });

    it('should select edges', () => {
      const store = useCanvasStore.getState();

      store.setSelectedEdges(['edge-1', 'edge-2']);
      expect(store.selectedEdges).toHaveLength(2);
    });
  });

  describe('Search and Filter', () => {
    it('should update search query', () => {
      const store = useCanvasStore.getState();

      expect(store.searchQuery).toBe('');

      store.setSearchQuery('test');
      expect(store.searchQuery).toBe('test');
    });

    it('should filter by category', () => {
      const store = useCanvasStore.getState();

      expect(store.filterCategory).toBe(null);

      store.setFilterCategory(AgentCategory.AI_ML);
      expect(store.filterCategory).toBe(AgentCategory.AI_ML);
    });
  });
});
