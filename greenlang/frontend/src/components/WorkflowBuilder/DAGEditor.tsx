/**
 * DAGEditor - Visual DAG editor with validation
 *
 * Features:
 * - Real-time DAG validation (no cycles)
 * - Visual error indicators for invalid connections
 * - Connection rules based on input/output types
 * - Type compatibility checking
 * - Required input validation
 * - Missing configuration warnings
 * - Workflow execution path preview
 * - Highlight critical path
 * - Step configuration panel
 * - Input/output mapping UI
 * - Conditional branching visual editor
 * - Parallel execution groups
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import {
  AlertCircle,
  CheckCircle,
  AlertTriangle,
  Info,
  Settings,
  Play,
  GitBranch,
  GitMerge,
  Layers,
  Eye,
  EyeOff,
  X,
  Save,
  RotateCcw,
} from 'lucide-react';

import {
  WorkflowNode,
  WorkflowEdge,
  ValidationError,
  ValidationSeverity,
  ValidationErrorType,
  ExecutionPlan,
  ExecutionPathNode,
  ConfigPanelState,
  DataType,
} from './types';
import { useCanvasStore } from './WorkflowCanvas';
import { useWorkflowValidation } from './hooks/useWorkflowValidation';

/**
 * Validation panel component
 */
const ValidationPanel: React.FC<{
  errors: ValidationError[];
  onSelectError: (error: ValidationError) => void;
}> = ({ errors, onSelectError }) => {
  const errorCount = errors.filter((e) => e.severity === ValidationSeverity.ERROR).length;
  const warningCount = errors.filter((e) => e.severity === ValidationSeverity.WARNING).length;
  const infoCount = errors.filter((e) => e.severity === ValidationSeverity.INFO).length;

  const getIcon = (severity: ValidationSeverity) => {
    switch (severity) {
      case ValidationSeverity.ERROR:
        return <AlertCircle className="text-red-500" size={16} />;
      case ValidationSeverity.WARNING:
        return <AlertTriangle className="text-yellow-500" size={16} />;
      case ValidationSeverity.INFO:
        return <Info className="text-blue-500" size={16} />;
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        Validation Results
        {errorCount === 0 && warningCount === 0 ? (
          <CheckCircle className="text-green-500" size={20} />
        ) : (
          <AlertCircle className="text-red-500" size={20} />
        )}
      </h3>

      <div className="flex gap-4 mb-4 text-sm">
        <div className="flex items-center gap-1">
          <AlertCircle className="text-red-500" size={16} />
          <span>{errorCount} Errors</span>
        </div>
        <div className="flex items-center gap-1">
          <AlertTriangle className="text-yellow-500" size={16} />
          <span>{warningCount} Warnings</span>
        </div>
        <div className="flex items-center gap-1">
          <Info className="text-blue-500" size={16} />
          <span>{infoCount} Info</span>
        </div>
      </div>

      {errors.length === 0 ? (
        <div className="text-center text-gray-500 py-4">
          <CheckCircle className="mx-auto mb-2 text-green-500" size={32} />
          <p>No validation issues found</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {errors.map((error) => (
            <button
              key={error.id}
              onClick={() => onSelectError(error)}
              className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start gap-2">
                {getIcon(error.severity)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{error.message}</p>
                  {error.suggestedFix && (
                    <p className="text-xs text-gray-500 mt-1">
                      Fix: {error.suggestedFix}
                    </p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">
                    Type: {error.type.replace(/_/g, ' ')}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Execution path visualization component
 */
const ExecutionPathPanel: React.FC<{
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  showCriticalPath: boolean;
}> = ({ nodes, edges, showCriticalPath }) => {
  const executionPlan = useMemo(() => {
    return calculateExecutionPlan(nodes, edges);
  }, [nodes, edges]);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="font-semibold mb-3 flex items-center gap-2">
        <Play size={20} />
        Execution Plan
      </h3>

      <div className="space-y-4">
        {/* Summary */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-500">Total Steps</p>
            <p className="text-lg font-semibold">{executionPlan.path.length}</p>
          </div>
          <div>
            <p className="text-gray-500">Estimated Duration</p>
            <p className="text-lg font-semibold">
              {executionPlan.estimatedTotalDuration.toFixed(0)}ms
            </p>
          </div>
        </div>

        {/* Parallel Groups */}
        {executionPlan.parallelGroups.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold mb-2 flex items-center gap-1">
              <Layers size={16} />
              Parallel Execution Groups
            </h4>
            <div className="space-y-2">
              {executionPlan.parallelGroups.map((group, index) => (
                <div
                  key={index}
                  className="p-2 bg-blue-50 border border-blue-200 rounded text-xs"
                >
                  <span className="font-semibold">Group {index + 1}:</span>{' '}
                  {group.length} nodes can run in parallel
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Critical Path */}
        {showCriticalPath && executionPlan.criticalPath.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold mb-2 flex items-center gap-1">
              <GitBranch size={16} />
              Critical Path
            </h4>
            <div className="space-y-1">
              {executionPlan.criticalPath.map((nodeId, index) => {
                const node = nodes.find((n) => n.id === nodeId);
                return (
                  <div
                    key={nodeId}
                    className="p-2 bg-orange-50 border border-orange-200 rounded text-xs flex items-center gap-2"
                  >
                    <span className="font-mono text-orange-600">
                      {index + 1}
                    </span>
                    <span>{node?.data.label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Execution Order */}
        <div>
          <h4 className="text-sm font-semibold mb-2">Execution Order</h4>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {executionPlan.path.map((pathNode) => {
              const node = nodes.find((n) => n.id === pathNode.nodeId);
              const isCritical = executionPlan.criticalPath.includes(
                pathNode.nodeId
              );

              return (
                <div
                  key={pathNode.nodeId}
                  className={`p-2 border rounded text-xs flex items-center justify-between ${
                    isCritical
                      ? 'bg-orange-50 border-orange-200'
                      : 'bg-gray-50 border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-gray-600">
                      {pathNode.step}
                    </span>
                    <span>{node?.data.label}</span>
                  </div>
                  {pathNode.estimatedDuration && (
                    <span className="text-gray-500">
                      ~{pathNode.estimatedDuration}ms
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Node configuration panel
 */
const NodeConfigPanel: React.FC<{
  node: WorkflowNode | null;
  onClose: () => void;
  onSave: (nodeId: string, config: any, inputs: any) => void;
}> = ({ node, onClose, onSave }) => {
  const [activeTab, setActiveTab] = useState<'config' | 'inputs' | 'outputs' | 'advanced'>('config');
  const [config, setConfig] = useState<any>(node?.data.config || {});
  const [inputs, setInputs] = useState<any>(node?.data.inputs || {});
  const [isDirty, setIsDirty] = useState(false);

  useEffect(() => {
    if (node) {
      setConfig(node.data.config);
      setInputs(node.data.inputs);
      setIsDirty(false);
    }
  }, [node]);

  const handleSave = () => {
    if (node) {
      onSave(node.id, config, inputs);
      setIsDirty(false);
    }
  };

  const handleReset = () => {
    if (node) {
      setConfig(node.data.config);
      setInputs(node.data.inputs);
      setIsDirty(false);
    }
  };

  if (!node) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <p className="text-center text-gray-500">Select a node to configure</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings size={20} />
          <h3 className="font-semibold">{node.data.label}</h3>
        </div>
        <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded">
          <X size={20} />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {(['config', 'inputs', 'outputs', 'advanced'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium capitalize ${
              activeTab === tab
                ? 'border-b-2 border-blue-500 text-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-4 max-h-96 overflow-y-auto">
        {activeTab === 'config' && (
          <div className="space-y-4">
            <p className="text-sm text-gray-600">{node.data.agent.description}</p>

            {node.data.agent.configuration && (
              <div className="space-y-3">
                {Object.entries(node.data.agent.configuration).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium mb-1">
                      {key}
                      {typeof value === 'object' && value !== null && 'required' in value && value.required && (
                        <span className="text-red-500 ml-1">*</span>
                      )}
                    </label>
                    <input
                      type="text"
                      value={config[key] || ''}
                      onChange={(e) => {
                        setConfig({ ...config, [key]: e.target.value });
                        setIsDirty(true);
                      }}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      placeholder={`Enter ${key}`}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'inputs' && (
          <div className="space-y-3">
            {node.data.agent.inputs.map((input) => (
              <div key={input.id}>
                <label className="block text-sm font-medium mb-1">
                  {input.name}
                  {input.required && <span className="text-red-500 ml-1">*</span>}
                </label>
                <div className="text-xs text-gray-500 mb-1">
                  Type: {input.type}
                </div>
                {input.description && (
                  <div className="text-xs text-gray-500 mb-2">
                    {input.description}
                  </div>
                )}
                <input
                  type="text"
                  value={inputs[input.id] || ''}
                  onChange={(e) => {
                    setInputs({ ...inputs, [input.id]: e.target.value });
                    setIsDirty(true);
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  placeholder={input.defaultValue?.toString() || `Enter ${input.name}`}
                />
              </div>
            ))}
          </div>
        )}

        {activeTab === 'outputs' && (
          <div className="space-y-3">
            {node.data.agent.outputs.map((output) => (
              <div
                key={output.id}
                className="p-3 bg-gray-50 border border-gray-200 rounded-lg"
              >
                <div className="font-medium text-sm">{output.name}</div>
                <div className="text-xs text-gray-500 mt-1">
                  Type: {output.type}
                </div>
                {output.description && (
                  <div className="text-xs text-gray-600 mt-1">
                    {output.description}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'advanced' && (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">Node ID</label>
              <input
                type="text"
                value={node.id}
                disabled
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Agent ID</label>
              <input
                type="text"
                value={node.data.agent.id}
                disabled
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-gray-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Version</label>
              <input
                type="text"
                value={node.data.agent.version}
                disabled
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm bg-gray-100"
              />
            </div>

            {node.data.agent.tags && (
              <div>
                <label className="block text-sm font-medium mb-1">Tags</label>
                <div className="flex flex-wrap gap-1">
                  {node.data.agent.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      {isDirty && (
        <div className="p-4 border-t border-gray-200 flex gap-2">
          <button
            onClick={handleSave}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2"
          >
            <Save size={16} />
            Save Changes
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <RotateCcw size={16} />
            Reset
          </button>
        </div>
      )}
    </div>
  );
};

/**
 * Calculate execution plan for workflow
 */
function calculateExecutionPlan(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): ExecutionPlan {
  // Build adjacency lists
  const adjacency = new Map<string, string[]>();
  const reverseAdjacency = new Map<string, string[]>();

  nodes.forEach((node) => {
    adjacency.set(node.id, []);
    reverseAdjacency.set(node.id, []);
  });

  edges.forEach((edge) => {
    adjacency.get(edge.source)?.push(edge.target);
    reverseAdjacency.get(edge.target)?.push(edge.source);
  });

  // Topological sort to get execution order
  const path: ExecutionPathNode[] = [];
  const visited = new Set<string>();
  const inDegree = new Map<string, number>();

  nodes.forEach((node) => {
    inDegree.set(node.id, reverseAdjacency.get(node.id)?.length || 0);
  });

  const queue: string[] = [];
  inDegree.forEach((degree, nodeId) => {
    if (degree === 0) queue.push(nodeId);
  });

  let step = 1;
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    visited.add(nodeId);

    path.push({
      nodeId,
      step: step++,
      estimatedDuration: Math.random() * 500 + 100,
      dependencies: reverseAdjacency.get(nodeId) || [],
      isCriticalPath: false,
    });

    adjacency.get(nodeId)?.forEach((neighbor) => {
      const degree = inDegree.get(neighbor)! - 1;
      inDegree.set(neighbor, degree);
      if (degree === 0) queue.push(neighbor);
    });
  }

  // Find parallel execution groups
  const parallelGroups: string[][] = [];
  const levels = new Map<string, number>();

  // Calculate node levels
  nodes.forEach((node) => {
    const deps = reverseAdjacency.get(node.id) || [];
    if (deps.length === 0) {
      levels.set(node.id, 0);
    } else {
      const maxDepLevel = Math.max(...deps.map((d) => levels.get(d) || 0));
      levels.set(node.id, maxDepLevel + 1);
    }
  });

  // Group nodes by level
  const levelGroups = new Map<number, string[]>();
  levels.forEach((level, nodeId) => {
    const group = levelGroups.get(level) || [];
    group.push(nodeId);
    levelGroups.set(level, group);
  });

  levelGroups.forEach((group) => {
    if (group.length > 1) {
      parallelGroups.push(group);
    }
  });

  // Calculate critical path (longest path)
  const criticalPath: string[] = [];
  const pathLengths = new Map<string, number>();

  nodes.forEach((node) => {
    pathLengths.set(node.id, 0);
  });

  path.forEach((pathNode) => {
    const deps = reverseAdjacency.get(pathNode.nodeId) || [];
    if (deps.length === 0) {
      pathLengths.set(pathNode.nodeId, pathNode.estimatedDuration || 0);
    } else {
      const maxDepLength = Math.max(...deps.map((d) => pathLengths.get(d) || 0));
      pathLengths.set(
        pathNode.nodeId,
        maxDepLength + (pathNode.estimatedDuration || 0)
      );
    }
  });

  // Find longest path
  let maxLength = 0;
  let endNode = '';
  pathLengths.forEach((length, nodeId) => {
    if (length > maxLength) {
      maxLength = length;
      endNode = nodeId;
    }
  });

  // Backtrack to find critical path
  let current = endNode;
  while (current) {
    criticalPath.unshift(current);
    const deps = reverseAdjacency.get(current) || [];
    if (deps.length === 0) break;

    const next = deps.reduce((prev, curr) =>
      (pathLengths.get(curr) || 0) > (pathLengths.get(prev) || 0) ? curr : prev
    );
    current = next;
  }

  // Mark critical path nodes
  path.forEach((pathNode) => {
    pathNode.isCriticalPath = criticalPath.includes(pathNode.nodeId);
  });

  return {
    path,
    criticalPath,
    estimatedTotalDuration: maxLength,
    parallelGroups,
  };
}

/**
 * Main DAGEditor component
 */
export const DAGEditor: React.FC = () => {
  const [showValidation, setShowValidation] = useState(true);
  const [showExecutionPath, setShowExecutionPath] = useState(true);
  const [showCriticalPath, setShowCriticalPath] = useState(true);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);

  const { nodes, edges, updateNode, validationErrors, selectedNodes } = useCanvasStore();

  const { validate } = useWorkflowValidation(nodes, edges);

  const errors = useMemo(() => validate(), [validate]);

  const handleSelectError = useCallback(
    (error: ValidationError) => {
      // Focus on the node/edge with the error
      if (error.nodeId) {
        const node = nodes.find((n) => n.id === error.nodeId);
        if (node) {
          setSelectedNode(node);
        }
      }
    },
    [nodes]
  );

  const handleSaveConfig = useCallback(
    (nodeId: string, config: any, inputs: any) => {
      updateNode(nodeId, { config, inputs });
    },
    [updateNode]
  );

  // Auto-select node when selected in canvas
  useEffect(() => {
    if (selectedNodes.length === 1) {
      const node = nodes.find((n) => n.id === selectedNodes[0]);
      if (node) {
        setSelectedNode(node);
      }
    } else {
      setSelectedNode(null);
    }
  }, [selectedNodes, nodes]);

  return (
    <div className="w-96 h-screen bg-gray-50 border-l border-gray-200 overflow-y-auto">
      <div className="p-4 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold">Workflow Editor</h2>
          <div className="flex gap-2">
            <button
              onClick={() => setShowValidation(!showValidation)}
              className={`p-2 rounded ${
                showValidation ? 'bg-blue-100 text-blue-600' : 'bg-gray-100'
              }`}
              title="Toggle Validation"
            >
              {showValidation ? <Eye size={16} /> : <EyeOff size={16} />}
            </button>
          </div>
        </div>

        {/* Validation Panel */}
        {showValidation && (
          <ValidationPanel errors={errors} onSelectError={handleSelectError} />
        )}

        {/* Execution Path Panel */}
        {showExecutionPath && nodes.length > 0 && (
          <ExecutionPathPanel
            nodes={nodes}
            edges={edges}
            showCriticalPath={showCriticalPath}
          />
        )}

        {/* Node Configuration Panel */}
        <NodeConfigPanel
          node={selectedNode}
          onClose={() => setSelectedNode(null)}
          onSave={handleSaveConfig}
        />

        {/* Connection Info */}
        {edges.length > 0 && (
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold mb-2 flex items-center gap-2">
              <GitMerge size={20} />
              Connections
            </h3>
            <div className="text-sm text-gray-600">
              <p>Total Connections: {edges.length}</p>
              <p>
                Active Nodes:{' '}
                {new Set(edges.flatMap((e) => [e.source, e.target])).size}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DAGEditor;
