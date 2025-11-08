/**
 * GreenLang Visual Workflow Builder
 *
 * Main exports for the workflow builder components
 */

// Main components
export { WorkflowCanvas, useCanvasStore } from './WorkflowCanvas';
export { AgentPalette } from './AgentPalette';
export { DAGEditor } from './DAGEditor';

// Hooks
export { useWorkflowValidation } from './hooks/useWorkflowValidation';

// Utils
export {
  calculateLayout,
  optimizeLayout,
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

// Types
export type {
  WorkflowNode,
  WorkflowEdge,
  WorkflowNodeData,
  WorkflowEdgeData,
  AgentMetadata,
  AgentLibraryItem,
  Workflow,
  WorkflowMetadata,
  ValidationError,
  ExecutionPlan,
  ExecutionPathNode,
  CanvasState,
  Port,
  LayoutConfig,
  ConfigPanelState,
  ConnectionRule,
  AgentSearchOptions,
  ExportOptions,
  ImportOptions,
} from './types';

export {
  AgentCategory,
  DataType,
  ExecutionStatus,
  ValidationSeverity,
  ValidationErrorType,
  DEFAULT_CONNECTION_RULES,
} from './types';
