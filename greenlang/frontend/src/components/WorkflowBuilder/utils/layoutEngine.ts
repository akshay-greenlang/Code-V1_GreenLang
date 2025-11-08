/**
 * Auto-layout engine using Dagre for workflow graphs
 *
 * Provides automatic node positioning with hierarchical layout,
 * edge routing with minimal crossings, and optimization for complex workflows.
 */

import dagre from 'dagre';
import { WorkflowNode, WorkflowEdge, LayoutConfig } from '../types';
import { Position } from 'reactflow';

/**
 * Default layout configuration
 */
export const DEFAULT_LAYOUT_CONFIG: LayoutConfig = {
  direction: 'TB', // Top to Bottom
  nodeSpacing: 50,
  rankSpacing: 100,
  edgeSpacing: 10,
  align: 'UL', // Up-Left
  ranker: 'network-simplex',
};

/**
 * Node dimensions for layout calculation
 */
interface NodeDimensions {
  width: number;
  height: number;
}

/**
 * Default node dimensions
 */
const DEFAULT_NODE_DIMENSIONS: NodeDimensions = {
  width: 200,
  height: 100,
};

/**
 * Calculate layout for workflow nodes using Dagre
 *
 * @param nodes - Array of workflow nodes
 * @param edges - Array of workflow edges
 * @param config - Layout configuration
 * @returns Array of nodes with updated positions
 */
export function calculateLayout(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  config: Partial<LayoutConfig> = {}
): WorkflowNode[] {
  const layoutConfig = { ...DEFAULT_LAYOUT_CONFIG, ...config };

  // Create a new directed graph
  const graph = new dagre.graphlib.Graph();

  // Set graph configuration
  graph.setGraph({
    rankdir: layoutConfig.direction,
    nodesep: layoutConfig.nodeSpacing,
    ranksep: layoutConfig.rankSpacing,
    edgesep: layoutConfig.edgeSpacing,
    align: layoutConfig.align,
    ranker: layoutConfig.ranker,
    marginx: 50,
    marginy: 50,
  });

  // Default edge configuration
  graph.setDefaultEdgeLabel(() => ({}));

  // Add nodes to the graph
  nodes.forEach((node) => {
    const dimensions = getNodeDimensions(node);
    graph.setNode(node.id, {
      width: dimensions.width,
      height: dimensions.height,
    });
  });

  // Add edges to the graph
  edges.forEach((edge) => {
    graph.setEdge(edge.source, edge.target);
  });

  // Run the layout algorithm
  dagre.layout(graph);

  // Update node positions
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = graph.node(node.id);

    // Calculate position (Dagre returns center position, we need top-left)
    const position = {
      x: nodeWithPosition.x - nodeWithPosition.width / 2,
      y: nodeWithPosition.y - nodeWithPosition.height / 2,
    };

    return {
      ...node,
      position,
      targetPosition: getTargetPosition(layoutConfig.direction),
      sourcePosition: getSourcePosition(layoutConfig.direction),
    };
  });

  return layoutedNodes;
}

/**
 * Get node dimensions based on content
 *
 * @param node - Workflow node
 * @returns Node dimensions
 */
function getNodeDimensions(node: WorkflowNode): NodeDimensions {
  // Base dimensions
  let width = DEFAULT_NODE_DIMENSIONS.width;
  let height = DEFAULT_NODE_DIMENSIONS.height;

  // Adjust width based on label length
  const labelLength = node.data.label.length;
  width = Math.max(width, labelLength * 8 + 60);

  // Adjust height based on number of inputs/outputs
  const maxPorts = Math.max(
    node.data.agent.inputs.length,
    node.data.agent.outputs.length
  );
  height = Math.max(height, maxPorts * 30 + 60);

  return { width, height };
}

/**
 * Get target position based on layout direction
 */
function getTargetPosition(direction: LayoutConfig['direction']): Position {
  switch (direction) {
    case 'TB':
      return Position.Top;
    case 'BT':
      return Position.Bottom;
    case 'LR':
      return Position.Left;
    case 'RL':
      return Position.Right;
    default:
      return Position.Top;
  }
}

/**
 * Get source position based on layout direction
 */
function getSourcePosition(direction: LayoutConfig['direction']): Position {
  switch (direction) {
    case 'TB':
      return Position.Bottom;
    case 'BT':
      return Position.Top;
    case 'LR':
      return Position.Right;
    case 'RL':
      return Position.Left;
    default:
      return Position.Bottom;
  }
}

/**
 * Optimize layout for better visual presentation
 *
 * @param nodes - Array of workflow nodes
 * @param edges - Array of workflow edges
 * @returns Optimized nodes
 */
export function optimizeLayout(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): WorkflowNode[] {
  // First, apply basic layout
  let optimizedNodes = calculateLayout(nodes, edges);

  // Reduce edge crossings by trying different rankers
  const rankers: LayoutConfig['ranker'][] = [
    'network-simplex',
    'tight-tree',
    'longest-path',
  ];

  let bestLayout = optimizedNodes;
  let minCrossings = calculateEdgeCrossings(optimizedNodes, edges);

  rankers.forEach((ranker) => {
    const testLayout = calculateLayout(nodes, edges, { ranker });
    const crossings = calculateEdgeCrossings(testLayout, edges);

    if (crossings < minCrossings) {
      bestLayout = testLayout;
      minCrossings = crossings;
    }
  });

  return bestLayout;
}

/**
 * Calculate number of edge crossings (simplified heuristic)
 *
 * @param nodes - Array of workflow nodes
 * @param edges - Array of workflow edges
 * @returns Number of edge crossings
 */
function calculateEdgeCrossings(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): number {
  let crossings = 0;

  // Create position map
  const positions = new Map(nodes.map((n) => [n.id, n.position]));

  // Check each pair of edges
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const edge1 = edges[i];
      const edge2 = edges[j];

      const p1 = positions.get(edge1.source);
      const p2 = positions.get(edge1.target);
      const p3 = positions.get(edge2.source);
      const p4 = positions.get(edge2.target);

      if (p1 && p2 && p3 && p4) {
        if (doEdgesCross(p1, p2, p3, p4)) {
          crossings++;
        }
      }
    }
  }

  return crossings;
}

/**
 * Check if two line segments cross
 */
function doEdgesCross(
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  p3: { x: number; y: number },
  p4: { x: number; y: number }
): boolean {
  const det =
    (p2.x - p1.x) * (p4.y - p3.y) - (p4.x - p3.x) * (p2.y - p1.y);

  if (det === 0) return false; // Parallel lines

  const lambda =
    ((p4.y - p3.y) * (p4.x - p1.x) + (p3.x - p4.x) * (p4.y - p1.y)) / det;
  const gamma =
    ((p1.y - p2.y) * (p4.x - p1.x) + (p2.x - p1.x) * (p4.y - p1.y)) / det;

  return lambda > 0 && lambda < 1 && gamma > 0 && gamma < 1;
}

/**
 * Align nodes to grid
 *
 * @param nodes - Array of workflow nodes
 * @param gridSize - Grid size in pixels
 * @returns Nodes aligned to grid
 */
export function alignToGrid(
  nodes: WorkflowNode[],
  gridSize: number = 20
): WorkflowNode[] {
  return nodes.map((node) => ({
    ...node,
    position: {
      x: Math.round(node.position.x / gridSize) * gridSize,
      y: Math.round(node.position.y / gridSize) * gridSize,
    },
  }));
}

/**
 * Distribute nodes evenly in horizontal direction
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function distributeHorizontally(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 3) return nodes;

  // Sort by x position
  const sorted = [...nodes].sort((a, b) => a.position.x - b.position.x);

  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const totalWidth = last.position.x - first.position.x;
  const spacing = totalWidth / (sorted.length - 1);

  return nodes.map((node) => {
    const index = sorted.findIndex((n) => n.id === node.id);
    if (index === 0 || index === sorted.length - 1) return node;

    return {
      ...node,
      position: {
        x: first.position.x + spacing * index,
        y: node.position.y,
      },
    };
  });
}

/**
 * Distribute nodes evenly in vertical direction
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function distributeVertically(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 3) return nodes;

  // Sort by y position
  const sorted = [...nodes].sort((a, b) => a.position.y - b.position.y);

  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const totalHeight = last.position.y - first.position.y;
  const spacing = totalHeight / (sorted.length - 1);

  return nodes.map((node) => {
    const index = sorted.findIndex((n) => n.id === node.id);
    if (index === 0 || index === sorted.length - 1) return node;

    return {
      ...node,
      position: {
        x: node.position.x,
        y: first.position.y + spacing * index,
      },
    };
  });
}

/**
 * Align nodes to the left edge of the leftmost node
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function alignLeft(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const minX = Math.min(...nodes.map((n) => n.position.x));

  return nodes.map((node) => ({
    ...node,
    position: {
      x: minX,
      y: node.position.y,
    },
  }));
}

/**
 * Align nodes to the right edge of the rightmost node
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function alignRight(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const maxX = Math.max(...nodes.map((n) => n.position.x));

  return nodes.map((node) => ({
    ...node,
    position: {
      x: maxX,
      y: node.position.y,
    },
  }));
}

/**
 * Align nodes to the top edge of the topmost node
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function alignTop(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const minY = Math.min(...nodes.map((n) => n.position.y));

  return nodes.map((node) => ({
    ...node,
    position: {
      x: node.position.x,
      y: minY,
    },
  }));
}

/**
 * Align nodes to the bottom edge of the bottommost node
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function alignBottom(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const maxY = Math.max(...nodes.map((n) => n.position.y));

  return nodes.map((node) => ({
    ...node,
    position: {
      x: node.position.x,
      y: maxY,
    },
  }));
}

/**
 * Center nodes horizontally
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function centerHorizontally(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const positions = nodes.map((n) => n.position.x);
  const center = (Math.min(...positions) + Math.max(...positions)) / 2;

  return nodes.map((node) => ({
    ...node,
    position: {
      x: center,
      y: node.position.y,
    },
  }));
}

/**
 * Center nodes vertically
 *
 * @param nodes - Array of selected workflow nodes
 * @returns Nodes with updated positions
 */
export function centerVertically(nodes: WorkflowNode[]): WorkflowNode[] {
  if (nodes.length < 2) return nodes;

  const positions = nodes.map((n) => n.position.y);
  const center = (Math.min(...positions) + Math.max(...positions)) / 2;

  return nodes.map((node) => ({
    ...node,
    position: {
      x: node.position.x,
      y: center,
    },
  }));
}
