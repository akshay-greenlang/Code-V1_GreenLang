/**
 * ExecutionTimeline Component
 *
 * Interactive Gantt chart visualization for workflow execution timeline:
 * - Zoom and pan controls
 * - Critical path highlighting
 * - Parallel execution groups
 * - Step dependencies
 * - Time distribution charts
 * - Status filtering
 *
 * @module ExecutionTimeline
 */

import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import * as d3 from 'd3';

// Type definitions
interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, any>;
}

interface NodeExecutionStatus {
  nodeId: string;
  status: 'pending' | 'running' | 'success' | 'failed' | 'skipped';
  startTime?: string;
  endTime?: string;
  duration?: number;
  dependencies: string[];
  parallelGroup?: number;
  isCriticalPath?: boolean;
}

interface ExecutionStatus {
  executionId: string;
  startTime: string;
  endTime?: string;
  nodeStatuses: Map<string, NodeExecutionStatus>;
}

interface TimelineBar {
  nodeId: string;
  startTime: number;
  duration: number;
  status: NodeExecutionStatus['status'];
  dependencies: string[];
  parallelGroup?: number;
  isCriticalPath: boolean;
}

interface ExecutionTimelineProps {
  executionStatus: ExecutionStatus;
  nodes: WorkflowNode[];
  formatDuration: (ms: number) => string;
  height?: number;
  showCriticalPath?: boolean;
  showDependencies?: boolean;
  onNodeClick?: (nodeId: string) => void;
}

/**
 * Main ExecutionTimeline component
 */
export const ExecutionTimeline: React.FC<ExecutionTimelineProps> = ({
  executionStatus,
  nodes,
  formatDuration,
  height = 600,
  showCriticalPath = true,
  showDependencies = true,
  onNodeClick,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [highlightedNode, setHighlightedNode] = useState<string | null>(null);
  const [showDistribution, setShowDistribution] = useState(false);

  // Calculate timeline data
  const timelineData = useMemo(() => {
    const bars: TimelineBar[] = [];
    const executionStart = new Date(executionStatus.startTime).getTime();

    executionStatus.nodeStatuses.forEach((status, nodeId) => {
      if (!status.startTime) return;

      const startTime = new Date(status.startTime).getTime() - executionStart;
      const duration = status.duration || 0;

      bars.push({
        nodeId,
        startTime,
        duration,
        status: status.status,
        dependencies: status.dependencies || [],
        parallelGroup: status.parallelGroup,
        isCriticalPath: status.isCriticalPath || false,
      });
    });

    return bars.sort((a, b) => a.startTime - b.startTime);
  }, [executionStatus]);

  // Filter timeline data
  const filteredData = useMemo(() => {
    if (filterStatus === 'all') return timelineData;
    return timelineData.filter(bar => bar.status === filterStatus);
  }, [timelineData, filterStatus]);

  // Calculate critical path
  const criticalPath = useMemo(() => {
    if (!showCriticalPath) return [];

    const criticalNodes = timelineData
      .filter(bar => bar.isCriticalPath)
      .map(bar => bar.nodeId);

    return criticalNodes;
  }, [timelineData, showCriticalPath]);

  // Calculate parallel groups
  const parallelGroups = useMemo(() => {
    const groups = new Map<number, TimelineBar[]>();

    timelineData.forEach(bar => {
      if (bar.parallelGroup !== undefined) {
        if (!groups.has(bar.parallelGroup)) {
          groups.set(bar.parallelGroup, []);
        }
        groups.get(bar.parallelGroup)!.push(bar);
      }
    });

    return groups;
  }, [timelineData]);

  // Calculate time distribution
  const timeDistribution = useMemo(() => {
    const distribution: Record<string, number> = {
      pending: 0,
      running: 0,
      success: 0,
      failed: 0,
      skipped: 0,
    };

    timelineData.forEach(bar => {
      distribution[bar.status] += bar.duration;
    });

    return distribution;
  }, [timelineData]);

  // Render Gantt chart
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || filteredData.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth;
    const barHeight = 30;
    const barPadding = 10;
    const margin = { top: 50, right: 50, bottom: 50, left: 150 };

    // Clear previous content
    svg.selectAll('*').remove();

    // Calculate scales
    const maxTime = Math.max(...filteredData.map(d => d.startTime + d.duration));
    const xScale = d3.scaleLinear()
      .domain([0, maxTime])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleBand()
      .domain(filteredData.map(d => d.nodeId))
      .range([margin.top, margin.top + filteredData.length * (barHeight + barPadding)])
      .padding(0.1);

    // Create main group with zoom/pan transform
    const g = svg.append('g')
      .attr('transform', `translate(${pan.x}, ${pan.y}) scale(${zoom})`);

    // Draw grid lines
    const gridLines = g.append('g')
      .attr('class', 'grid-lines')
      .attr('opacity', 0.1);

    const tickCount = 10;
    const tickValues = d3.range(0, maxTime, maxTime / tickCount);

    gridLines.selectAll('line')
      .data(tickValues)
      .enter()
      .append('line')
      .attr('x1', d => xScale(d))
      .attr('x2', d => xScale(d))
      .attr('y1', margin.top)
      .attr('y2', margin.top + filteredData.length * (barHeight + barPadding))
      .attr('stroke', '#888')
      .attr('stroke-width', 1);

    // Draw parallel group backgrounds
    if (parallelGroups.size > 0) {
      parallelGroups.forEach((group, groupId) => {
        const nodeIds = group.map(bar => bar.nodeId);
        const firstY = yScale(nodeIds[0]) || 0;
        const lastY = yScale(nodeIds[nodeIds.length - 1]) || 0;
        const groupHeight = lastY - firstY + barHeight;

        g.append('rect')
          .attr('x', margin.left)
          .attr('y', firstY)
          .attr('width', width - margin.left - margin.right)
          .attr('height', groupHeight)
          .attr('fill', `hsl(${groupId * 60}, 70%, 90%)`)
          .attr('opacity', 0.2)
          .attr('rx', 5);
      });
    }

    // Draw dependency lines
    if (showDependencies) {
      const dependencyLines = g.append('g').attr('class', 'dependency-lines');

      filteredData.forEach(bar => {
        bar.dependencies.forEach(depId => {
          const depBar = filteredData.find(b => b.nodeId === depId);
          if (!depBar) return;

          const y1 = (yScale(depId) || 0) + barHeight / 2;
          const y2 = (yScale(bar.nodeId) || 0) + barHeight / 2;
          const x1 = xScale(depBar.startTime + depBar.duration);
          const x2 = xScale(bar.startTime);

          dependencyLines.append('path')
            .attr('d', `M ${x1} ${y1} L ${x2} ${y2}`)
            .attr('stroke', '#888')
            .attr('stroke-width', 2)
            .attr('fill', 'none')
            .attr('marker-end', 'url(#arrow)')
            .attr('opacity', 0.3);
        });
      });

      // Define arrow marker
      svg.append('defs')
        .append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8)
        .attr('refY', 5)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', '#888');
    }

    // Draw timeline bars
    const bars = g.append('g').attr('class', 'timeline-bars');

    bars.selectAll('rect')
      .data(filteredData)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.startTime))
      .attr('y', d => yScale(d.nodeId) || 0)
      .attr('width', d => Math.max(xScale(d.duration) - xScale(0), 2))
      .attr('height', barHeight)
      .attr('fill', d => getStatusColor(d.status))
      .attr('stroke', d => d.isCriticalPath ? '#f59e0b' : 'none')
      .attr('stroke-width', d => d.isCriticalPath ? 3 : 0)
      .attr('rx', 4)
      .attr('opacity', d => highlightedNode === null || highlightedNode === d.nodeId ? 1 : 0.3)
      .on('mouseover', function(event, d) {
        setHighlightedNode(d.nodeId);
        d3.select(this).attr('opacity', 1);
      })
      .on('mouseout', function() {
        setHighlightedNode(null);
        d3.select(this).attr('opacity', 1);
      })
      .on('click', function(event, d) {
        if (onNodeClick) onNodeClick(d.nodeId);
      })
      .append('title')
      .text(d => `${d.nodeId}\nDuration: ${formatDuration(d.duration)}\nStatus: ${d.status}`);

    // Draw node labels
    const labels = g.append('g').attr('class', 'node-labels');

    labels.selectAll('text')
      .data(filteredData)
      .enter()
      .append('text')
      .attr('x', margin.left - 10)
      .attr('y', d => (yScale(d.nodeId) || 0) + barHeight / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#333')
      .text(d => d.nodeId);

    // Draw time axis
    const xAxis = d3.axisTop(xScale)
      .tickFormat(d => formatDuration(d as number))
      .ticks(10);

    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0, ${margin.top})`)
      .call(xAxis);

    // Draw critical path indicator
    if (showCriticalPath && criticalPath.length > 0) {
      g.append('g')
        .attr('class', 'critical-path-label')
        .append('text')
        .attr('x', width - margin.right)
        .attr('y', margin.top - 20)
        .attr('text-anchor', 'end')
        .attr('font-size', 14)
        .attr('font-weight', 'bold')
        .attr('fill', '#f59e0b')
        .text('⚡ Critical Path');
    }

  }, [filteredData, zoom, pan, showDependencies, showCriticalPath, criticalPath, highlightedNode, formatDuration, onNodeClick, parallelGroups]);

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

  // Handle zoom
  const handleZoomIn = useCallback(() => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom(prev => Math.max(prev / 1.2, 0.5));
  }, []);

  const handleResetZoom = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  // Handle pan (simplified - in production, use mouse drag)
  const handlePanLeft = useCallback(() => {
    setPan(prev => ({ ...prev, x: prev.x + 50 }));
  }, []);

  const handlePanRight = useCallback(() => {
    setPan(prev => ({ ...prev, x: prev.x - 50 }));
  }, []);

  return (
    <div className="execution-timeline">
      {/* Header */}
      <div className="timeline-header">
        <h3>Execution Timeline</h3>

        {/* Controls */}
        <div className="timeline-controls">
          {/* Zoom Controls */}
          <div className="zoom-controls">
            <button onClick={handleZoomOut} title="Zoom Out">−</button>
            <span>{(zoom * 100).toFixed(0)}%</span>
            <button onClick={handleZoomIn} title="Zoom In">+</button>
            <button onClick={handleResetZoom} title="Reset">⟲</button>
          </div>

          {/* Pan Controls */}
          <div className="pan-controls">
            <button onClick={handlePanLeft} title="Pan Left">←</button>
            <button onClick={handlePanRight} title="Pan Right">→</button>
          </div>

          {/* Filter Controls */}
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="status-filter"
          >
            <option value="all">All statuses</option>
            <option value="pending">Pending</option>
            <option value="running">Running</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="skipped">Skipped</option>
          </select>

          {/* View Options */}
          <label className="view-option">
            <input
              type="checkbox"
              checked={showDistribution}
              onChange={(e) => setShowDistribution(e.target.checked)}
            />
            Show Distribution
          </label>
        </div>
      </div>

      {/* Gantt Chart */}
      <div ref={containerRef} className="timeline-container">
        <svg
          ref={svgRef}
          width="100%"
          height={height}
          className="timeline-svg"
        />
      </div>

      {/* Time Distribution Chart */}
      {showDistribution && (
        <div className="time-distribution">
          <h4>Time Distribution by Status</h4>
          <div className="distribution-chart">
            {Object.entries(timeDistribution).map(([status, duration]) => {
              const totalDuration = Object.values(timeDistribution).reduce((a, b) => a + b, 0);
              const percentage = totalDuration > 0 ? (duration / totalDuration) * 100 : 0;

              return (
                <div key={status} className="distribution-item">
                  <div className="distribution-label">
                    <span className="status-name">{status}</span>
                    <span className="status-duration">{formatDuration(duration)}</span>
                    <span className="status-percentage">{percentage.toFixed(1)}%</span>
                  </div>
                  <div className="distribution-bar">
                    <div
                      className="distribution-fill"
                      style={{
                        width: `${percentage}%`,
                        backgroundColor: getStatusColor(status as NodeExecutionStatus['status']),
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="timeline-legend">
        <h4>Legend</h4>
        <div className="legend-items">
          {(['pending', 'running', 'success', 'failed', 'skipped'] as const).map(status => (
            <div key={status} className="legend-item">
              <div
                className="legend-color"
                style={{ backgroundColor: getStatusColor(status) }}
              />
              <span>{status}</span>
            </div>
          ))}
          {showCriticalPath && (
            <div className="legend-item">
              <div
                className="legend-color critical-path"
                style={{ border: '3px solid #f59e0b' }}
              />
              <span>Critical Path</span>
            </div>
          )}
        </div>
      </div>

      {/* Statistics */}
      <div className="timeline-statistics">
        <div className="stat-item">
          <label>Total Nodes:</label>
          <span>{timelineData.length}</span>
        </div>
        <div className="stat-item">
          <label>Parallel Groups:</label>
          <span>{parallelGroups.size}</span>
        </div>
        {criticalPath.length > 0 && (
          <div className="stat-item">
            <label>Critical Path Length:</label>
            <span>{criticalPath.length} nodes</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExecutionTimeline;
