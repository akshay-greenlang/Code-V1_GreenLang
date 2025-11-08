/**
 * WorkflowCanvas - Main React Flow canvas component for workflow building
 *
 * Features:
 * - Drag-and-drop interface for workflow creation
 * - Custom node styling with execution status
 * - Pan, zoom, and minimap controls
 * - Auto-layout with Dagre.js
 * - Export/Import workflows
 * - Undo/Redo functionality
 * - Multi-select and alignment tools
 */

import React, { useCallback, useRef, useEffect, useMemo, memo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  OnConnect,
  OnNodesChange,
  OnEdgesChange,
  ReactFlowProvider,
  Panel,
  useReactFlow,
  ReactFlowInstance,
  NodeTypes,
  EdgeTypes,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { nanoid } from 'nanoid';
import {
  Play,
  Save,
  FolderOpen,
  Undo,
  Redo,
  Grid,
  Maximize2,
  Download,
  Upload,
  Trash2,
  Copy,
  Scissors,
  AlignLeft,
  AlignRight,
  AlignTop,
  AlignBottom,
  AlignHorizontalJustifyCenter,
  AlignVerticalJustifyCenter,
} from 'lucide-react';

import {
  WorkflowNode,
  WorkflowEdge,
  WorkflowNodeData,
  WorkflowEdgeData,
  ExecutionStatus,
  Workflow,
  CanvasState,
  AgentMetadata,
  ExportOptions,
  ImportOptions,
} from './types';
import { useWorkflowValidation } from './hooks/useWorkflowValidation';
import {
  calculateLayout,
  alignToGrid,
  alignLeft,
  alignRight,
  alignTop,
  alignBottom,
  centerHorizontally,
  centerVertically,
  distributeHorizontally,
  distributeVertically,
} from './utils/layoutEngine';

/**
 * Zustand store for canvas state management
 */
export const useCanvasStore = create<CanvasState>()(
  immer((set, get) => ({
    nodes: [],
    edges: [],
    selectedNodes: [],
    selectedEdges: [],
    validationErrors: [],
    isValid: true,
    isExecuting: false,
    executionProgress: 0,
    isPanelOpen: true,
    activePanel: 'agent',
    searchQuery: '',
    filterCategory: null,
    history: {
      past: [],
      future: [],
    },

    setNodes: (nodes) =>
      set((state) => {
        state.nodes = nodes;
      }),

    setEdges: (edges) =>
      set((state) => {
        state.edges = edges;
      }),

    addNode: (node) =>
      set((state) => {
        // Save to history
        state.history.past.push({
          nodes: [...state.nodes],
          edges: [...state.edges],
        });
        state.history.future = [];

        state.nodes.push(node);
      }),

    removeNode: (nodeId) =>
      set((state) => {
        // Save to history
        state.history.past.push({
          nodes: [...state.nodes],
          edges: [...state.edges],
        });
        state.history.future = [];

        state.nodes = state.nodes.filter((n) => n.id !== nodeId);
        state.edges = state.edges.filter(
          (e) => e.source !== nodeId && e.target !== nodeId
        );
      }),

    updateNode: (nodeId, data) =>
      set((state) => {
        const node = state.nodes.find((n) => n.id === nodeId);
        if (node) {
          node.data = { ...node.data, ...data };
        }
      }),

    addEdge: (edge) =>
      set((state) => {
        // Save to history
        state.history.past.push({
          nodes: [...state.nodes],
          edges: [...state.edges],
        });
        state.history.future = [];

        state.edges.push(edge);
      }),

    removeEdge: (edgeId) =>
      set((state) => {
        // Save to history
        state.history.past.push({
          nodes: [...state.nodes],
          edges: [...state.edges],
        });
        state.history.future = [];

        state.edges = state.edges.filter((e) => e.id !== edgeId);
      }),

    updateEdge: (edgeId, data) =>
      set((state) => {
        const edge = state.edges.find((e) => e.id === edgeId);
        if (edge && edge.data) {
          edge.data = { ...edge.data, ...data };
        }
      }),

    setSelectedNodes: (nodeIds) =>
      set((state) => {
        state.selectedNodes = nodeIds;
      }),

    setSelectedEdges: (edgeIds) =>
      set((state) => {
        state.selectedEdges = edgeIds;
      }),

    setValidationErrors: (errors) =>
      set((state) => {
        state.validationErrors = errors;
      }),

    setIsValid: (isValid) =>
      set((state) => {
        state.isValid = isValid;
      }),

    setIsExecuting: (isExecuting) =>
      set((state) => {
        state.isExecuting = isExecuting;
      }),

    setExecutionProgress: (progress) =>
      set((state) => {
        state.executionProgress = progress;
      }),

    setActivePanel: (panel) =>
      set((state) => {
        state.activePanel = panel;
        state.isPanelOpen = panel !== null;
      }),

    setSearchQuery: (query) =>
      set((state) => {
        state.searchQuery = query;
      }),

    setFilterCategory: (category) =>
      set((state) => {
        state.filterCategory = category;
      }),

    undo: () =>
      set((state) => {
        if (state.history.past.length === 0) return;

        const previous = state.history.past.pop();
        if (previous) {
          state.history.future.push({
            nodes: [...state.nodes],
            edges: [...state.edges],
          });
          state.nodes = previous.nodes;
          state.edges = previous.edges;
        }
      }),

    redo: () =>
      set((state) => {
        if (state.history.future.length === 0) return;

        const next = state.history.future.pop();
        if (next) {
          state.history.past.push({
            nodes: [...state.nodes],
            edges: [...state.edges],
          });
          state.nodes = next.nodes;
          state.edges = next.edges;
        }
      }),

    canUndo: () => get().history.past.length > 0,
    canRedo: () => get().history.future.length > 0,

    clear: () =>
      set((state) => {
        state.history.past.push({
          nodes: [...state.nodes],
          edges: [...state.edges],
        });
        state.history.future = [];
        state.nodes = [];
        state.edges = [];
        state.selectedNodes = [];
        state.selectedEdges = [];
      }),

    reset: () =>
      set((state) => {
        state.nodes = [];
        state.edges = [];
        state.selectedNodes = [];
        state.selectedEdges = [];
        state.validationErrors = [];
        state.isValid = true;
        state.isExecuting = false;
        state.executionProgress = 0;
        state.history = { past: [], future: [] };
      }),
  }))
);

/**
 * Custom node component with execution status
 */
const CustomNode = memo(({ data, selected }: { data: WorkflowNodeData; selected: boolean }) => {
  const statusColors = {
    idle: 'bg-gray-100 border-gray-300',
    running: 'bg-blue-100 border-blue-500 animate-pulse',
    success: 'bg-green-100 border-green-500',
    error: 'bg-red-100 border-red-500',
    warning: 'bg-yellow-100 border-yellow-500',
    skipped: 'bg-gray-200 border-gray-400',
  };

  const statusColor = statusColors[data.status] || statusColors.idle;

  return (
    <div
      className={`px-4 py-3 rounded-lg border-2 shadow-md min-w-[200px] ${statusColor} ${
        selected ? 'ring-2 ring-blue-400' : ''
      }`}
    >
      <div className="flex items-center gap-2 mb-2">
        {data.agent.icon && (
          <span className="text-lg">{data.agent.icon}</span>
        )}
        <div className="font-semibold text-sm truncate">{data.label}</div>
      </div>

      {data.errors && data.errors.length > 0 && (
        <div className="text-xs text-red-600 mb-1">
          {data.errors.length} error(s)
        </div>
      )}

      {data.executionTime && (
        <div className="text-xs text-gray-500">
          {data.executionTime}ms
        </div>
      )}
    </div>
  );
});

CustomNode.displayName = 'CustomNode';

/**
 * Custom edge component with status indicators
 */
const CustomEdge = memo(({ data }: { data?: WorkflowEdgeData }) => {
  const edgeColors = {
    idle: '#94a3b8',
    running: '#3b82f6',
    success: '#10b981',
    error: '#ef4444',
    warning: '#f59e0b',
    skipped: '#6b7280',
  };

  const color = data?.status ? edgeColors[data.status] : edgeColors.idle;

  return (
    <g>
      <path
        stroke={color}
        strokeWidth={2}
        fill="none"
        className={data?.animated ? 'animate-pulse' : ''}
      />
      {data?.label && (
        <text>
          <tspan x="0" dy="0" className="text-xs">
            {data.label}
          </tspan>
        </text>
      )}
    </g>
  );
});

CustomEdge.displayName = 'CustomEdge';

/**
 * Toolbar component with workflow actions
 */
const Toolbar: React.FC<{
  onAutoLayout: () => void;
  onFitView: () => void;
  onExport: () => void;
  onImport: () => void;
  onClear: () => void;
  onExecute: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  selectedNodes: WorkflowNode[];
  onAlign: (type: string) => void;
}> = ({
  onAutoLayout,
  onFitView,
  onExport,
  onImport,
  onClear,
  onExecute,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  selectedNodes,
  onAlign,
}) => {
  return (
    <Panel position="top-left" className="flex gap-2 bg-white p-2 rounded-lg shadow-md">
      <button
        onClick={onExecute}
        className="p-2 hover:bg-gray-100 rounded"
        title="Execute Workflow"
      >
        <Play size={20} />
      </button>

      <div className="w-px bg-gray-300" />

      <button
        onClick={onUndo}
        disabled={!canUndo}
        className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
        title="Undo (Ctrl+Z)"
      >
        <Undo size={20} />
      </button>

      <button
        onClick={onRedo}
        disabled={!canRedo}
        className="p-2 hover:bg-gray-100 rounded disabled:opacity-50"
        title="Redo (Ctrl+Y)"
      >
        <Redo size={20} />
      </button>

      <div className="w-px bg-gray-300" />

      <button
        onClick={onAutoLayout}
        className="p-2 hover:bg-gray-100 rounded"
        title="Auto Layout"
      >
        <Grid size={20} />
      </button>

      <button
        onClick={onFitView}
        className="p-2 hover:bg-gray-100 rounded"
        title="Fit View"
      >
        <Maximize2 size={20} />
      </button>

      <div className="w-px bg-gray-300" />

      <button
        onClick={onExport}
        className="p-2 hover:bg-gray-100 rounded"
        title="Export Workflow"
      >
        <Download size={20} />
      </button>

      <button
        onClick={onImport}
        className="p-2 hover:bg-gray-100 rounded"
        title="Import Workflow"
      >
        <Upload size={20} />
      </button>

      <button
        onClick={onClear}
        className="p-2 hover:bg-gray-100 rounded text-red-600"
        title="Clear Canvas"
      >
        <Trash2 size={20} />
      </button>

      {selectedNodes.length > 1 && (
        <>
          <div className="w-px bg-gray-300" />
          <button
            onClick={() => onAlign('left')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Align Left"
          >
            <AlignLeft size={20} />
          </button>
          <button
            onClick={() => onAlign('right')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Align Right"
          >
            <AlignRight size={20} />
          </button>
          <button
            onClick={() => onAlign('top')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Align Top"
          >
            <AlignTop size={20} />
          </button>
          <button
            onClick={() => onAlign('bottom')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Align Bottom"
          >
            <AlignBottom size={20} />
          </button>
          <button
            onClick={() => onAlign('horizontal')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Center Horizontally"
          >
            <AlignHorizontalJustifyCenter size={20} />
          </button>
          <button
            onClick={() => onAlign('vertical')}
            className="p-2 hover:bg-gray-100 rounded"
            title="Center Vertically"
          >
            <AlignVerticalJustifyCenter size={20} />
          </button>
        </>
      )}
    </Panel>
  );
};

/**
 * Main canvas component
 */
const WorkflowCanvasInner: React.FC = () => {
  const reactFlowInstance = useReactFlow();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Store state
  const {
    nodes,
    edges,
    setNodes,
    setEdges,
    addNode,
    addEdge,
    removeNode,
    removeEdge,
    updateNode,
    undo,
    redo,
    canUndo,
    canRedo,
    clear,
    selectedNodes: selectedNodeIds,
    setSelectedNodes,
    setValidationErrors,
    setIsValid,
    isExecuting,
    setIsExecuting,
    setExecutionProgress,
  } = useCanvasStore();

  // React Flow state
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState(nodes);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState(edges);

  // Validation
  const { validate, isValid, isConnectionValid } = useWorkflowValidation(nodes, edges);

  // Sync store with React Flow
  useEffect(() => {
    setRfNodes(nodes);
  }, [nodes, setRfNodes]);

  useEffect(() => {
    setRfEdges(edges);
  }, [edges, setRfEdges]);

  // Run validation on changes
  useEffect(() => {
    const errors = validate();
    setValidationErrors(errors);
    setIsValid(isValid);
  }, [nodes, edges, validate, isValid, setValidationErrors, setIsValid]);

  // Handle connection
  const onConnect: OnConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return;

      // Validate connection
      const isValid = isConnectionValid(
        connection.source,
        connection.sourceHandle || '',
        connection.target,
        connection.targetHandle || ''
      );

      if (!isValid) {
        alert('Invalid connection: Type mismatch or would create a cycle');
        return;
      }

      const newEdge: WorkflowEdge = {
        id: `edge-${nanoid()}`,
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle || undefined,
        targetHandle: connection.targetHandle || undefined,
        data: {
          sourcePort: connection.sourceHandle || undefined,
          targetPort: connection.targetHandle || undefined,
          status: ExecutionStatus.IDLE,
        },
      };

      addEdge(newEdge);
    },
    [addEdge, isConnectionValid]
  );

  // Handle node selection
  const handleSelectionChange = useCallback(
    ({ nodes }: { nodes: Node[] }) => {
      setSelectedNodes(nodes.map((n) => n.id));
    },
    [setSelectedNodes]
  );

  // Auto layout
  const handleAutoLayout = useCallback(() => {
    const layoutedNodes = calculateLayout(nodes, edges);
    setNodes(layoutedNodes);
  }, [nodes, edges, setNodes]);

  // Fit view
  const handleFitView = useCallback(() => {
    reactFlowInstance.fitView({ padding: 0.2 });
  }, [reactFlowInstance]);

  // Export workflow
  const handleExport = useCallback(() => {
    const workflow: Workflow = {
      metadata: {
        id: nanoid(),
        name: 'Untitled Workflow',
        version: '1.0.0',
        created: new Date(),
        modified: new Date(),
        tags: [],
      },
      nodes,
      edges,
      viewport: reactFlowInstance.getViewport(),
    };

    const json = JSON.stringify(workflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `workflow-${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);
  }, [nodes, edges, reactFlowInstance]);

  // Import workflow
  const handleImport = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const workflow: Workflow = JSON.parse(e.target?.result as string);
          setNodes(workflow.nodes);
          setEdges(workflow.edges);

          if (workflow.viewport) {
            reactFlowInstance.setViewport(workflow.viewport);
          }
        } catch (error) {
          alert('Failed to import workflow: Invalid file format');
        }
      };
      reader.readAsText(file);
    },
    [setNodes, setEdges, reactFlowInstance]
  );

  // Clear canvas
  const handleClear = useCallback(() => {
    if (confirm('Are you sure you want to clear the canvas?')) {
      clear();
    }
  }, [clear]);

  // Execute workflow
  const handleExecute = useCallback(async () => {
    if (!isValid) {
      alert('Cannot execute: Workflow has validation errors');
      return;
    }

    setIsExecuting(true);
    setExecutionProgress(0);

    // Simulate execution
    for (let i = 0; i < nodes.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      updateNode(nodes[i].id, { status: ExecutionStatus.RUNNING });
      await new Promise((resolve) => setTimeout(resolve, 1000));
      updateNode(nodes[i].id, {
        status: ExecutionStatus.SUCCESS,
        executionTime: Math.random() * 1000,
      });
      setExecutionProgress(((i + 1) / nodes.length) * 100);
    }

    setIsExecuting(false);
  }, [nodes, isValid, setIsExecuting, setExecutionProgress, updateNode]);

  // Alignment functions
  const handleAlign = useCallback(
    (type: string) => {
      const selected = nodes.filter((n) => selectedNodeIds.includes(n.id));
      if (selected.length < 2) return;

      let alignedNodes: WorkflowNode[];

      switch (type) {
        case 'left':
          alignedNodes = alignLeft(selected);
          break;
        case 'right':
          alignedNodes = alignRight(selected);
          break;
        case 'top':
          alignedNodes = alignTop(selected);
          break;
        case 'bottom':
          alignedNodes = alignBottom(selected);
          break;
        case 'horizontal':
          alignedNodes = centerHorizontally(selected);
          break;
        case 'vertical':
          alignedNodes = centerVertically(selected);
          break;
        default:
          return;
      }

      // Update nodes
      const updatedNodes = nodes.map((node) => {
        const aligned = alignedNodes.find((n) => n.id === node.id);
        return aligned || node;
      });

      setNodes(updatedNodes);
    },
    [nodes, selectedNodeIds, setNodes]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey) {
        switch (event.key.toLowerCase()) {
          case 'z':
            event.preventDefault();
            if (event.shiftKey) {
              redo();
            } else {
              undo();
            }
            break;
          case 'y':
            event.preventDefault();
            redo();
            break;
          case 's':
            event.preventDefault();
            handleExport();
            break;
          case 'o':
            event.preventDefault();
            handleImport();
            break;
        }
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        selectedNodeIds.forEach((id) => removeNode(id));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, handleExport, handleImport, selectedNodeIds, removeNode]);

  // Node types
  const nodeTypes = useMemo<NodeTypes>(
    () => ({
      custom: CustomNode,
    }),
    []
  );

  const selectedNodes = nodes.filter((n) => selectedNodeIds.includes(n.id));

  return (
    <div className="w-full h-screen">
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={handleSelectionChange}
        nodeTypes={nodeTypes}
        fitView
        snapToGrid
        snapGrid={[15, 15]}
        defaultEdgeOptions={{
          type: 'smoothstep',
          animated: false,
        }}
      >
        <Background />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const data = node.data as WorkflowNodeData;
            const colors = {
              idle: '#e5e7eb',
              running: '#3b82f6',
              success: '#10b981',
              error: '#ef4444',
              warning: '#f59e0b',
              skipped: '#6b7280',
            };
            return colors[data.status] || colors.idle;
          }}
        />

        <Toolbar
          onAutoLayout={handleAutoLayout}
          onFitView={handleFitView}
          onExport={handleExport}
          onImport={handleImport}
          onClear={handleClear}
          onExecute={handleExecute}
          canUndo={canUndo()}
          canRedo={canRedo()}
          onUndo={undo}
          onRedo={redo}
          selectedNodes={selectedNodes}
          onAlign={handleAlign}
        />

        {isExecuting && (
          <Panel position="bottom-center">
            <div className="bg-white p-4 rounded-lg shadow-md min-w-[300px]">
              <div className="text-sm font-semibold mb-2">Executing Workflow...</div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${useCanvasStore.getState().executionProgress}%` }}
                />
              </div>
            </div>
          </Panel>
        )}
      </ReactFlow>

      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
};

/**
 * WorkflowCanvas component with ReactFlowProvider
 */
export const WorkflowCanvas: React.FC = () => {
  return (
    <ReactFlowProvider>
      <WorkflowCanvasInner />
    </ReactFlowProvider>
  );
};

export default WorkflowCanvas;
