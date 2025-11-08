/**
 * ExecutionMonitor Component
 *
 * Real-time workflow execution monitoring dashboard with:
 * - Live execution progress overlay
 * - Node status indicators
 * - Performance metrics
 * - Error tracking
 * - Execution controls (pause/resume/kill)
 * - Historical comparisons
 * - Export functionality
 *
 * @module ExecutionMonitor
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useMutation } from '@tantml/react-query';
import jsPDF from 'jspdf';

// Type definitions
interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, any>;
}

interface ExecutionStatus {
  executionId: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'killed';
  startTime: string;
  endTime?: string;
  duration?: number;
  nodeStatuses: Map<string, NodeExecutionStatus>;
  totalNodes: number;
  completedNodes: number;
  failedNodes: number;
  error?: ExecutionError;
}

interface NodeExecutionStatus {
  nodeId: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  startTime?: string;
  endTime?: string;
  duration?: number;
  metrics: PerformanceMetrics;
  logs: ExecutionLog[];
  error?: NodeError;
  retryCount: number;
  maxRetries: number;
}

interface PerformanceMetrics {
  executionTime: number;
  cpuUsage: number;
  memoryUsage: number;
  inputDataSize: number;
  outputDataSize: number;
}

interface ExecutionLog {
  timestamp: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
  metadata?: Record<string, any>;
}

interface NodeError {
  message: string;
  stackTrace?: string;
  code?: string;
  recoverable: boolean;
}

interface ExecutionError {
  message: string;
  failedNodeId: string;
  stackTrace?: string;
  timestamp: string;
}

interface ExecutionMonitorProps {
  workflowId: string;
  executionId: string;
  nodes: WorkflowNode[];
  onNodeClick?: (nodeId: string) => void;
  showTimeline?: boolean;
  showMetrics?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

/**
 * Main ExecutionMonitor component
 */
export const ExecutionMonitor: React.FC<ExecutionMonitorProps> = ({
  workflowId,
  executionId,
  nodes,
  onNodeClick,
  showTimeline = true,
  showMetrics = true,
  autoRefresh = true,
  refreshInterval = 2000,
}) => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [showErrorDetails, setShowErrorDetails] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [viewMode, setViewMode] = useState<'overlay' | 'panel'>('overlay');
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch execution status
  const { data: executionStatus, refetch } = useQuery<ExecutionStatus>(
    ['execution-status', executionId],
    async () => {
      const response = await fetch(`/api/v1/executions/${executionId}`);
      if (!response.ok) throw new Error('Failed to fetch execution status');
      return response.json();
    },
    {
      refetchInterval: autoRefresh ? refreshInterval : false,
      refetchIntervalInBackground: true,
    }
  );

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!executionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/executions/${executionId}/ws`;

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onmessage = (event) => {
      const update = JSON.parse(event.data);
      refetch();
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [executionId, refetch]);

  // Control mutations
  const pauseExecution = useMutation(async () => {
    const response = await fetch(`/api/v1/executions/${executionId}/pause`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to pause execution');
    return response.json();
  });

  const resumeExecution = useMutation(async () => {
    const response = await fetch(`/api/v1/executions/${executionId}/resume`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to resume execution');
    return response.json();
  });

  const killExecution = useMutation(async () => {
    const response = await fetch(`/api/v1/executions/${executionId}/kill`, {
      method: 'POST',
    });
    if (!response.ok) throw new Error('Failed to kill execution');
    return response.json();
  });

  const retryNode = useMutation(async (nodeId: string) => {
    const response = await fetch(
      `/api/v1/executions/${executionId}/nodes/${nodeId}/retry`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to retry node');
    return response.json();
  });

  // Get node status
  const getNodeStatus = useCallback((nodeId: string): NodeExecutionStatus | undefined => {
    return executionStatus?.nodeStatuses.get(nodeId);
  }, [executionStatus]);

  // Get status color
  const getStatusColor = (status: NodeExecutionStatus['status']): string => {
    const colors = {
      pending: '#9ca3af',
      running: '#3b82f6',
      success: '#10b981',
      failed: '#ef4444',
      skipped: '#f59e0b',
    };
    return colors[status];
  };

  // Get status icon
  const getStatusIcon = (status: NodeExecutionStatus['status']): string => {
    const icons = {
      pending: '○',
      running: '◉',
      success: '✓',
      failed: '✗',
      skipped: '⊘',
    };
    return icons[status];
  };

  // Calculate progress percentage
  const progressPercentage = useMemo(() => {
    if (!executionStatus) return 0;
    return (executionStatus.completedNodes / executionStatus.totalNodes) * 100;
  }, [executionStatus]);

  // Filter nodes by status
  const filteredNodes = useMemo(() => {
    if (filterStatus === 'all') return nodes;

    return nodes.filter(node => {
      const status = getNodeStatus(node.id);
      return status?.status === filterStatus;
    });
  }, [nodes, filterStatus, getNodeStatus]);

  // Export execution report
  const exportReport = useCallback(async (format: 'json' | 'pdf') => {
    if (!executionStatus) return;

    if (format === 'json') {
      const data = JSON.stringify(executionStatus, null, 2);
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `execution-${executionId}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } else if (format === 'pdf') {
      const pdf = new jsPDF();

      // Title
      pdf.setFontSize(20);
      pdf.text('Workflow Execution Report', 20, 20);

      // Summary
      pdf.setFontSize(12);
      pdf.text(`Execution ID: ${executionId}`, 20, 40);
      pdf.text(`Workflow ID: ${workflowId}`, 20, 50);
      pdf.text(`Status: ${executionStatus.status}`, 20, 60);
      pdf.text(`Start Time: ${new Date(executionStatus.startTime).toLocaleString()}`, 20, 70);

      if (executionStatus.endTime) {
        pdf.text(`End Time: ${new Date(executionStatus.endTime).toLocaleString()}`, 20, 80);
        pdf.text(`Duration: ${formatDuration(executionStatus.duration || 0)}`, 20, 90);
      }

      // Statistics
      pdf.text('Statistics:', 20, 110);
      pdf.text(`Total Nodes: ${executionStatus.totalNodes}`, 30, 120);
      pdf.text(`Completed: ${executionStatus.completedNodes}`, 30, 130);
      pdf.text(`Failed: ${executionStatus.failedNodes}`, 30, 140);

      // Node details
      let yPos = 160;
      pdf.text('Node Execution Details:', 20, yPos);

      executionStatus.nodeStatuses.forEach((status, nodeId) => {
        yPos += 10;
        if (yPos > 270) {
          pdf.addPage();
          yPos = 20;
        }
        pdf.text(`${nodeId}: ${status.status} (${formatDuration(status.duration || 0)})`, 30, yPos);
      });

      pdf.save(`execution-${executionId}.pdf`);
    }
  }, [executionStatus, executionId, workflowId]);

  // Format duration
  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!executionStatus) {
    return (
      <div className="execution-monitor-loading">
        <div className="spinner" />
        <p>Loading execution status...</p>
      </div>
    );
  }

  return (
    <div className="execution-monitor">
      {/* Header */}
      <div className="execution-monitor-header">
        <div className="execution-info">
          <h2>Execution Monitor</h2>
          <span className={`execution-status ${executionStatus.status}`}>
            {executionStatus.status.toUpperCase()}
          </span>
        </div>

        <div className="execution-controls">
          {executionStatus.status === 'running' && (
            <>
              <button
                onClick={() => pauseExecution.mutate()}
                disabled={pauseExecution.isLoading}
                className="control-button pause"
              >
                ⏸ Pause
              </button>
              <button
                onClick={() => killExecution.mutate()}
                disabled={killExecution.isLoading}
                className="control-button kill"
              >
                ⏹ Kill
              </button>
            </>
          )}
          {executionStatus.status === 'paused' && (
            <button
              onClick={() => resumeExecution.mutate()}
              disabled={resumeExecution.isLoading}
              className="control-button resume"
            >
              ▶ Resume
            </button>
          )}
          <button onClick={() => exportReport('json')} className="control-button export">
            Export JSON
          </button>
          <button onClick={() => exportReport('pdf')} className="control-button export">
            Export PDF
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="execution-progress">
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
        <div className="progress-stats">
          <span>{executionStatus.completedNodes} / {executionStatus.totalNodes} nodes</span>
          <span>{progressPercentage.toFixed(1)}% complete</span>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="view-mode-toggle">
        <button
          className={viewMode === 'overlay' ? 'active' : ''}
          onClick={() => setViewMode('overlay')}
        >
          Overlay View
        </button>
        <button
          className={viewMode === 'panel' ? 'active' : ''}
          onClick={() => setViewMode('panel')}
        >
          Panel View
        </button>
      </div>

      {/* Status Filter */}
      <div className="status-filter">
        <label>Filter by status:</label>
        <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
          <option value="all">All</option>
          <option value="pending">Pending</option>
          <option value="running">Running</option>
          <option value="success">Success</option>
          <option value="failed">Failed</option>
          <option value="skipped">Skipped</option>
        </select>
      </div>

      {/* Main Content */}
      <div className={`execution-content ${viewMode}`}>
        {viewMode === 'overlay' ? (
          <ExecutionOverlay
            nodes={filteredNodes}
            getNodeStatus={getNodeStatus}
            getStatusColor={getStatusColor}
            getStatusIcon={getStatusIcon}
            onNodeClick={(nodeId) => {
              setSelectedNode(nodeId);
              if (onNodeClick) onNodeClick(nodeId);
            }}
          />
        ) : (
          <ExecutionPanel
            executionStatus={executionStatus}
            nodes={filteredNodes}
            getNodeStatus={getNodeStatus}
            getStatusColor={getStatusColor}
            getStatusIcon={getStatusIcon}
            formatDuration={formatDuration}
            onNodeClick={(nodeId) => {
              setSelectedNode(nodeId);
              if (onNodeClick) onNodeClick(nodeId);
            }}
            onRetry={(nodeId) => retryNode.mutate(nodeId)}
          />
        )}
      </div>

      {/* Timeline */}
      {showTimeline && (
        <ExecutionTimeline
          executionStatus={executionStatus}
          nodes={nodes}
          formatDuration={formatDuration}
        />
      )}

      {/* Selected Node Details */}
      {selectedNode && (
        <NodeDetailsPanel
          nodeId={selectedNode}
          nodeStatus={getNodeStatus(selectedNode)}
          onClose={() => setSelectedNode(null)}
          onRetry={() => retryNode.mutate(selectedNode)}
          formatDuration={formatDuration}
          showMetrics={showMetrics}
        />
      )}

      {/* Error Details Modal */}
      {executionStatus.error && showErrorDetails && (
        <ErrorDetailsModal
          error={executionStatus.error}
          onClose={() => setShowErrorDetails(false)}
        />
      )}
    </div>
  );
};

/**
 * Execution Overlay Component
 */
interface ExecutionOverlayProps {
  nodes: WorkflowNode[];
  getNodeStatus: (nodeId: string) => NodeExecutionStatus | undefined;
  getStatusColor: (status: NodeExecutionStatus['status']) => string;
  getStatusIcon: (status: NodeExecutionStatus['status']) => string;
  onNodeClick: (nodeId: string) => void;
}

const ExecutionOverlay: React.FC<ExecutionOverlayProps> = ({
  nodes,
  getNodeStatus,
  getStatusColor,
  getStatusIcon,
  onNodeClick,
}) => {
  return (
    <div className="execution-overlay">
      <svg className="execution-canvas" width="100%" height="100%">
        {nodes.map(node => {
          const status = getNodeStatus(node.id);
          if (!status) return null;

          return (
            <g key={node.id} onClick={() => onNodeClick(node.id)}>
              <rect
                x={node.position.x}
                y={node.position.y}
                width={200}
                height={80}
                fill={getStatusColor(status.status)}
                fillOpacity={0.2}
                stroke={getStatusColor(status.status)}
                strokeWidth={2}
                rx={8}
              />
              <text
                x={node.position.x + 100}
                y={node.position.y + 40}
                textAnchor="middle"
                fontSize={24}
                fill={getStatusColor(status.status)}
              >
                {getStatusIcon(status.status)}
              </text>
              {status.status === 'running' && (
                <circle
                  cx={node.position.x + 100}
                  cy={node.position.y + 40}
                  r={30}
                  fill="none"
                  stroke={getStatusColor(status.status)}
                  strokeWidth={2}
                  opacity={0.5}
                >
                  <animate
                    attributeName="r"
                    from="20"
                    to="40"
                    dur="1s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    from="0.8"
                    to="0"
                    dur="1s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

/**
 * Execution Panel Component
 */
interface ExecutionPanelProps {
  executionStatus: ExecutionStatus;
  nodes: WorkflowNode[];
  getNodeStatus: (nodeId: string) => NodeExecutionStatus | undefined;
  getStatusColor: (status: NodeExecutionStatus['status']) => string;
  getStatusIcon: (status: NodeExecutionStatus['status']) => string;
  formatDuration: (ms: number) => string;
  onNodeClick: (nodeId: string) => void;
  onRetry: (nodeId: string) => void;
}

const ExecutionPanel: React.FC<ExecutionPanelProps> = ({
  executionStatus,
  nodes,
  getNodeStatus,
  getStatusColor,
  getStatusIcon,
  formatDuration,
  onNodeClick,
  onRetry,
}) => {
  return (
    <div className="execution-panel">
      <table className="execution-table">
        <thead>
          <tr>
            <th>Node</th>
            <th>Status</th>
            <th>Duration</th>
            <th>CPU</th>
            <th>Memory</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {nodes.map(node => {
            const status = getNodeStatus(node.id);
            if (!status) return null;

            return (
              <tr
                key={node.id}
                onClick={() => onNodeClick(node.id)}
                className="clickable"
              >
                <td>{node.id}</td>
                <td>
                  <span
                    className="status-badge"
                    style={{ backgroundColor: getStatusColor(status.status) }}
                  >
                    {getStatusIcon(status.status)} {status.status}
                  </span>
                </td>
                <td>{status.duration ? formatDuration(status.duration) : '-'}</td>
                <td>{status.metrics.cpuUsage.toFixed(1)}%</td>
                <td>{(status.metrics.memoryUsage / 1024 / 1024).toFixed(1)} MB</td>
                <td>
                  {status.status === 'failed' && status.retryCount < status.maxRetries && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onRetry(node.id);
                      }}
                      className="retry-button"
                    >
                      Retry ({status.retryCount}/{status.maxRetries})
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

/**
 * Execution Timeline Component (imported from separate file)
 */
interface ExecutionTimelineProps {
  executionStatus: ExecutionStatus;
  nodes: WorkflowNode[];
  formatDuration: (ms: number) => string;
}

const ExecutionTimeline: React.FC<ExecutionTimelineProps> = ({
  executionStatus,
  nodes,
  formatDuration,
}) => {
  return (
    <div className="execution-timeline">
      <h3>Execution Timeline</h3>
      <div className="timeline-container">
        {/* Timeline visualization - simplified version */}
        <div className="timeline-bar">
          {Array.from(executionStatus.nodeStatuses.entries()).map(([nodeId, status]) => {
            if (!status.startTime) return null;

            const start = new Date(status.startTime).getTime();
            const execStart = new Date(executionStatus.startTime).getTime();
            const offset = ((start - execStart) / 1000) * 10; // 10px per second

            return (
              <div
                key={nodeId}
                className="timeline-item"
                style={{
                  left: `${offset}px`,
                  width: status.duration ? `${(status.duration / 1000) * 10}px` : '10px',
                }}
                title={`${nodeId}: ${formatDuration(status.duration || 0)}`}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
};

/**
 * Node Details Panel Component
 */
interface NodeDetailsPanelProps {
  nodeId: string;
  nodeStatus?: NodeExecutionStatus;
  onClose: () => void;
  onRetry: () => void;
  formatDuration: (ms: number) => string;
  showMetrics: boolean;
}

const NodeDetailsPanel: React.FC<NodeDetailsPanelProps> = ({
  nodeId,
  nodeStatus,
  onClose,
  onRetry,
  formatDuration,
  showMetrics,
}) => {
  if (!nodeStatus) return null;

  return (
    <div className="node-details-panel">
      <div className="panel-header">
        <h3>Node: {nodeId}</h3>
        <button onClick={onClose} className="close-button">×</button>
      </div>

      <div className="panel-content">
        {/* Status */}
        <div className="detail-section">
          <h4>Status</h4>
          <p className={`status ${nodeStatus.status}`}>{nodeStatus.status}</p>
        </div>

        {/* Timing */}
        <div className="detail-section">
          <h4>Timing</h4>
          <p>Duration: {formatDuration(nodeStatus.duration || 0)}</p>
          {nodeStatus.startTime && (
            <p>Started: {new Date(nodeStatus.startTime).toLocaleString()}</p>
          )}
          {nodeStatus.endTime && (
            <p>Ended: {new Date(nodeStatus.endTime).toLocaleString()}</p>
          )}
        </div>

        {/* Performance Metrics */}
        {showMetrics && (
          <div className="detail-section">
            <h4>Performance Metrics</h4>
            <div className="metrics-grid">
              <div className="metric">
                <label>Execution Time</label>
                <span>{formatDuration(nodeStatus.metrics.executionTime)}</span>
              </div>
              <div className="metric">
                <label>CPU Usage</label>
                <span>{nodeStatus.metrics.cpuUsage.toFixed(1)}%</span>
              </div>
              <div className="metric">
                <label>Memory Usage</label>
                <span>{(nodeStatus.metrics.memoryUsage / 1024 / 1024).toFixed(1)} MB</span>
              </div>
              <div className="metric">
                <label>Input Size</label>
                <span>{(nodeStatus.metrics.inputDataSize / 1024).toFixed(1)} KB</span>
              </div>
              <div className="metric">
                <label>Output Size</label>
                <span>{(nodeStatus.metrics.outputDataSize / 1024).toFixed(1)} KB</span>
              </div>
            </div>
          </div>
        )}

        {/* Logs */}
        <div className="detail-section">
          <h4>Execution Logs</h4>
          <div className="logs-container">
            {nodeStatus.logs.map((log, index) => (
              <div key={index} className={`log-entry ${log.level}`}>
                <span className="log-time">
                  {new Date(log.timestamp).toLocaleTimeString()}
                </span>
                <span className="log-level">{log.level}</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Error Details */}
        {nodeStatus.error && (
          <div className="detail-section error">
            <h4>Error Details</h4>
            <p className="error-message">{nodeStatus.error.message}</p>
            {nodeStatus.error.code && (
              <p className="error-code">Code: {nodeStatus.error.code}</p>
            )}
            {nodeStatus.error.stackTrace && (
              <pre className="stack-trace">{nodeStatus.error.stackTrace}</pre>
            )}
            {nodeStatus.error.recoverable && nodeStatus.retryCount < nodeStatus.maxRetries && (
              <button onClick={onRetry} className="retry-button">
                Retry ({nodeStatus.retryCount}/{nodeStatus.maxRetries})
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Error Details Modal Component
 */
const ErrorDetailsModal: React.FC<{
  error: ExecutionError;
  onClose: () => void;
}> = ({ error, onClose }) => {
  return (
    <div className="error-details-modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h3>Execution Error</h3>
          <button onClick={onClose} className="close-button">×</button>
        </div>
        <div className="modal-body">
          <p className="error-message">{error.message}</p>
          <p className="failed-node">Failed at node: {error.failedNodeId}</p>
          <p className="error-time">
            Occurred at: {new Date(error.timestamp).toLocaleString()}
          </p>
          {error.stackTrace && (
            <div className="stack-trace-section">
              <h4>Stack Trace</h4>
              <pre>{error.stackTrace}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExecutionMonitor;
