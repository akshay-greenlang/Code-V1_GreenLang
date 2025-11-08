/**
 * Type definitions for the GreenLang Visual Workflow Builder
 *
 * This module contains all TypeScript interfaces and types used across the workflow builder components.
 * It defines the structure for nodes, edges, agents, validation errors, and canvas state.
 */

import { Node, Edge, XYPosition } from 'reactflow';

/**
 * Agent categories for organization in the palette
 */
export enum AgentCategory {
  DATA_PROCESSING = 'Data Processing',
  AI_ML = 'AI/ML',
  INTEGRATION = 'Integration',
  UTILITIES = 'Utilities',
  CUSTOM = 'Custom'
}

/**
 * Data types for input/output validation
 */
export enum DataType {
  STRING = 'string',
  NUMBER = 'number',
  BOOLEAN = 'boolean',
  OBJECT = 'object',
  ARRAY = 'array',
  FILE = 'file',
  ANY = 'any'
}

/**
 * Node execution status for visual indicators
 */
export enum ExecutionStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  SUCCESS = 'success',
  ERROR = 'error',
  WARNING = 'warning',
  SKIPPED = 'skipped'
}

/**
 * Validation error severity levels
 */
export enum ValidationSeverity {
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info'
}

/**
 * Input/Output port definition
 */
export interface Port {
  id: string;
  name: string;
  type: DataType;
  required: boolean;
  description?: string;
  defaultValue?: any;
  validation?: {
    min?: number;
    max?: number;
    pattern?: string;
    enum?: any[];
  };
}

/**
 * Agent metadata describing capabilities and configuration
 */
export interface AgentMetadata {
  id: string;
  name: string;
  description: string;
  category: AgentCategory;
  version: string;
  author?: string;
  icon?: string;
  color?: string;
  tags: string[];
  inputs: Port[];
  outputs: Port[];
  configuration?: Record<string, any>;
  examples?: Array<{
    title: string;
    description: string;
    config: Record<string, any>;
  }>;
  documentation?: string;
  deprecated?: boolean;
  usageCount?: number;
  lastUsed?: Date;
}

/**
 * Custom data stored in workflow nodes
 */
export interface WorkflowNodeData {
  agent: AgentMetadata;
  label: string;
  status: ExecutionStatus;
  config: Record<string, any>;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  errors?: ValidationError[];
  executionTime?: number;
  metadata?: Record<string, any>;
}

/**
 * Extended Node type with workflow-specific data
 */
export type WorkflowNode = Node<WorkflowNodeData>;

/**
 * Custom data stored in workflow edges
 */
export interface WorkflowEdgeData {
  sourcePort?: string;
  targetPort?: string;
  status?: ExecutionStatus;
  dataType?: DataType;
  label?: string;
  animated?: boolean;
  conditional?: {
    condition: string;
    value: any;
  };
}

/**
 * Extended Edge type with workflow-specific data
 */
export type WorkflowEdge = Edge<WorkflowEdgeData>;

/**
 * Validation error with position information
 */
export interface ValidationError {
  id: string;
  nodeId?: string;
  edgeId?: string;
  severity: ValidationSeverity;
  message: string;
  type: ValidationErrorType;
  position?: XYPosition;
  details?: Record<string, any>;
  suggestedFix?: string;
}

/**
 * Types of validation errors
 */
export enum ValidationErrorType {
  CYCLE_DETECTED = 'cycle_detected',
  TYPE_MISMATCH = 'type_mismatch',
  REQUIRED_INPUT_MISSING = 'required_input_missing',
  INVALID_CONNECTION = 'invalid_connection',
  MISSING_CONFIGURATION = 'missing_configuration',
  ORPHANED_NODE = 'orphaned_node',
  DUPLICATE_CONNECTION = 'duplicate_connection',
  INVALID_PORT = 'invalid_port',
  EXECUTION_PATH_ERROR = 'execution_path_error'
}

/**
 * Workflow metadata and configuration
 */
export interface WorkflowMetadata {
  id: string;
  name: string;
  description?: string;
  version: string;
  author?: string;
  created: Date;
  modified: Date;
  tags: string[];
  variables?: Record<string, any>;
  settings?: {
    autoLayout?: boolean;
    snapToGrid?: boolean;
    gridSize?: number;
    executionMode?: 'sequential' | 'parallel' | 'hybrid';
    errorHandling?: 'stop' | 'continue' | 'retry';
    maxRetries?: number;
  };
}

/**
 * Complete workflow definition
 */
export interface Workflow {
  metadata: WorkflowMetadata;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  viewport?: {
    x: number;
    y: number;
    zoom: number;
  };
}

/**
 * Canvas state managed by Zustand
 */
export interface CanvasState {
  // Workflow data
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];

  // Selection state
  selectedNodes: string[];
  selectedEdges: string[];

  // Validation state
  validationErrors: ValidationError[];
  isValid: boolean;

  // Execution state
  isExecuting: boolean;
  executionProgress: number;

  // UI state
  isPanelOpen: boolean;
  activePanel: 'agent' | 'config' | 'validation' | null;
  searchQuery: string;
  filterCategory: AgentCategory | null;

  // History for undo/redo
  history: {
    past: Array<{ nodes: WorkflowNode[]; edges: WorkflowEdge[] }>;
    future: Array<{ nodes: WorkflowNode[]; edges: WorkflowEdge[] }>;
  };

  // Actions
  setNodes: (nodes: WorkflowNode[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  addNode: (node: WorkflowNode) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, data: Partial<WorkflowNodeData>) => void;
  addEdge: (edge: WorkflowEdge) => void;
  removeEdge: (edgeId: string) => void;
  updateEdge: (edgeId: string, data: Partial<WorkflowEdgeData>) => void;

  setSelectedNodes: (nodeIds: string[]) => void;
  setSelectedEdges: (edgeIds: string[]) => void;

  setValidationErrors: (errors: ValidationError[]) => void;
  setIsValid: (isValid: boolean) => void;

  setIsExecuting: (isExecuting: boolean) => void;
  setExecutionProgress: (progress: number) => void;

  setActivePanel: (panel: 'agent' | 'config' | 'validation' | null) => void;
  setSearchQuery: (query: string) => void;
  setFilterCategory: (category: AgentCategory | null) => void;

  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  clear: () => void;
  reset: () => void;
}

/**
 * Agent library item for the palette
 */
export interface AgentLibraryItem extends AgentMetadata {
  isFavorite: boolean;
  usageCount: number;
  lastUsed?: Date;
}

/**
 * Search and filter options for agent palette
 */
export interface AgentSearchOptions {
  query: string;
  category?: AgentCategory;
  tags?: string[];
  favorites?: boolean;
  sortBy?: 'name' | 'usage' | 'recent';
  sortOrder?: 'asc' | 'desc';
}

/**
 * Layout configuration for auto-layout
 */
export interface LayoutConfig {
  direction: 'TB' | 'BT' | 'LR' | 'RL'; // Top-Bottom, Bottom-Top, Left-Right, Right-Left
  nodeSpacing: number;
  rankSpacing: number;
  edgeSpacing: number;
  align?: 'UL' | 'UR' | 'DL' | 'DR'; // Up-Left, Up-Right, Down-Left, Down-Right
  ranker?: 'network-simplex' | 'tight-tree' | 'longest-path';
}

/**
 * Connection rules for validation
 */
export interface ConnectionRule {
  sourceType: DataType;
  targetType: DataType;
  allowed: boolean;
  autoConvert?: boolean;
  conversionFunction?: string;
}

/**
 * Execution path node for preview
 */
export interface ExecutionPathNode {
  nodeId: string;
  step: number;
  estimatedDuration?: number;
  dependencies: string[];
  isCriticalPath: boolean;
}

/**
 * Workflow execution plan
 */
export interface ExecutionPlan {
  path: ExecutionPathNode[];
  criticalPath: string[];
  estimatedTotalDuration: number;
  parallelGroups: string[][];
}

/**
 * Keyboard shortcut definition
 */
export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  action: () => void;
  description: string;
}

/**
 * Export options for workflow
 */
export interface ExportOptions {
  format: 'json' | 'yaml' | 'python' | 'javascript';
  includeMetadata: boolean;
  minify?: boolean;
  comments?: boolean;
}

/**
 * Import options for workflow
 */
export interface ImportOptions {
  format: 'json' | 'yaml';
  validate: boolean;
  merge?: boolean;
  autoLayout?: boolean;
}

/**
 * Node configuration panel state
 */
export interface ConfigPanelState {
  nodeId: string | null;
  activeTab: 'config' | 'inputs' | 'outputs' | 'advanced';
  isDirty: boolean;
  validationErrors: ValidationError[];
}

/**
 * Type guard to check if a value is a WorkflowNode
 */
export function isWorkflowNode(value: any): value is WorkflowNode {
  return (
    value &&
    typeof value === 'object' &&
    'id' in value &&
    'data' in value &&
    'agent' in value.data
  );
}

/**
 * Type guard to check if a value is a WorkflowEdge
 */
export function isWorkflowEdge(value: any): value is WorkflowEdge {
  return (
    value &&
    typeof value === 'object' &&
    'id' in value &&
    'source' in value &&
    'target' in value
  );
}

/**
 * Default connection rules
 */
export const DEFAULT_CONNECTION_RULES: ConnectionRule[] = [
  { sourceType: DataType.ANY, targetType: DataType.ANY, allowed: true },
  { sourceType: DataType.STRING, targetType: DataType.STRING, allowed: true },
  { sourceType: DataType.NUMBER, targetType: DataType.NUMBER, allowed: true },
  { sourceType: DataType.STRING, targetType: DataType.NUMBER, allowed: true, autoConvert: true },
  { sourceType: DataType.NUMBER, targetType: DataType.STRING, allowed: true, autoConvert: true },
  { sourceType: DataType.BOOLEAN, targetType: DataType.STRING, allowed: true, autoConvert: true },
  { sourceType: DataType.OBJECT, targetType: DataType.OBJECT, allowed: true },
  { sourceType: DataType.ARRAY, targetType: DataType.ARRAY, allowed: true },
  { sourceType: DataType.FILE, targetType: DataType.FILE, allowed: true },
  { sourceType: DataType.FILE, targetType: DataType.STRING, allowed: true },
];
