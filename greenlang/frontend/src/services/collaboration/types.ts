/**
 * Collaboration Types
 *
 * Type definitions for collaborative editing features including:
 * - Message types
 * - User presence
 * - Operational Transform operations
 * - Comment types
 * - Permission types
 *
 * @module CollaborationTypes
 */

// User types
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  color: string;
}

export interface UserPresence {
  userId: string;
  user: User;
  cursorPosition?: { x: number; y: number };
  selectedNodeId?: string;
  editingNodeId?: string;
  lastActive: string;
  status: 'active' | 'idle' | 'offline';
}

// Permission types
export type PermissionLevel = 'owner' | 'editor' | 'viewer';

export interface Permission {
  userId: string;
  level: PermissionLevel;
  grantedBy: string;
  grantedAt: string;
}

export interface WorkflowPermissions {
  workflowId: string;
  owner: User;
  permissions: Permission[];
}

// Message types
export enum MessageType {
  // Connection lifecycle
  JOIN = 'join',
  LEAVE = 'leave',
  HEARTBEAT = 'heartbeat',

  // Presence updates
  CURSOR_MOVE = 'cursor_move',
  NODE_SELECT = 'node_select',
  NODE_EDIT_START = 'node_edit_start',
  NODE_EDIT_END = 'node_edit_end',

  // Operations
  OPERATION = 'operation',
  OPERATION_ACK = 'operation_ack',

  // Comments
  COMMENT_ADD = 'comment_add',
  COMMENT_REPLY = 'comment_reply',
  COMMENT_RESOLVE = 'comment_resolve',
  COMMENT_DELETE = 'comment_delete',

  // Sync
  SYNC_REQUEST = 'sync_request',
  SYNC_RESPONSE = 'sync_response',

  // Errors
  ERROR = 'error',
}

export interface BaseMessage {
  type: MessageType;
  userId: string;
  timestamp: string;
  workflowId: string;
}

export interface JoinMessage extends BaseMessage {
  type: MessageType.JOIN;
  user: User;
}

export interface LeaveMessage extends BaseMessage {
  type: MessageType.LEAVE;
}

export interface HeartbeatMessage extends BaseMessage {
  type: MessageType.HEARTBEAT;
  cursorPosition?: { x: number; y: number };
  selectedNodeId?: string;
}

export interface CursorMoveMessage extends BaseMessage {
  type: MessageType.CURSOR_MOVE;
  position: { x: number; y: number };
}

export interface NodeSelectMessage extends BaseMessage {
  type: MessageType.NODE_SELECT;
  nodeId: string | null;
}

export interface NodeEditStartMessage extends BaseMessage {
  type: MessageType.NODE_EDIT_START;
  nodeId: string;
}

export interface NodeEditEndMessage extends BaseMessage {
  type: MessageType.NODE_EDIT_END;
  nodeId: string;
}

export interface OperationMessage extends BaseMessage {
  type: MessageType.OPERATION;
  operation: Operation;
  clientId: string;
  version: number;
}

export interface OperationAckMessage extends BaseMessage {
  type: MessageType.OPERATION_ACK;
  operationId: string;
  version: number;
}

export interface CommentMessage extends BaseMessage {
  type: MessageType.COMMENT_ADD | MessageType.COMMENT_REPLY;
  comment: Comment;
}

export interface CommentResolveMessage extends BaseMessage {
  type: MessageType.COMMENT_RESOLVE;
  commentId: string;
}

export interface CommentDeleteMessage extends BaseMessage {
  type: MessageType.COMMENT_DELETE;
  commentId: string;
}

export interface SyncRequestMessage extends BaseMessage {
  type: MessageType.SYNC_REQUEST;
  lastKnownVersion: number;
}

export interface SyncResponseMessage extends BaseMessage {
  type: MessageType.SYNC_RESPONSE;
  currentVersion: number;
  operations: Operation[];
  state: WorkflowState;
}

export interface ErrorMessage extends BaseMessage {
  type: MessageType.ERROR;
  error: {
    code: string;
    message: string;
    details?: any;
  };
}

export type CollaborationMessage =
  | JoinMessage
  | LeaveMessage
  | HeartbeatMessage
  | CursorMoveMessage
  | NodeSelectMessage
  | NodeEditStartMessage
  | NodeEditEndMessage
  | OperationMessage
  | OperationAckMessage
  | CommentMessage
  | CommentResolveMessage
  | CommentDeleteMessage
  | SyncRequestMessage
  | SyncResponseMessage
  | ErrorMessage;

// Operational Transform types
export enum OperationType {
  INSERT_NODE = 'insert_node',
  DELETE_NODE = 'delete_node',
  UPDATE_NODE = 'update_node',
  MOVE_NODE = 'move_node',
  INSERT_EDGE = 'insert_edge',
  DELETE_EDGE = 'delete_edge',
  UPDATE_EDGE = 'update_edge',
}

export interface BaseOperation {
  id: string;
  type: OperationType;
  timestamp: string;
  userId: string;
  version: number;
}

export interface InsertNodeOperation extends BaseOperation {
  type: OperationType.INSERT_NODE;
  node: {
    id: string;
    type: string;
    position: { x: number; y: number };
    data: Record<string, any>;
  };
}

export interface DeleteNodeOperation extends BaseOperation {
  type: OperationType.DELETE_NODE;
  nodeId: string;
}

export interface UpdateNodeOperation extends BaseOperation {
  type: OperationType.UPDATE_NODE;
  nodeId: string;
  changes: {
    path: string[];
    oldValue: any;
    newValue: any;
  }[];
}

export interface MoveNodeOperation extends BaseOperation {
  type: OperationType.MOVE_NODE;
  nodeId: string;
  oldPosition: { x: number; y: number };
  newPosition: { x: number; y: number };
}

export interface InsertEdgeOperation extends BaseOperation {
  type: OperationType.INSERT_EDGE;
  edge: {
    id: string;
    source: string;
    target: string;
    type?: string;
  };
}

export interface DeleteEdgeOperation extends BaseOperation {
  type: OperationType.DELETE_EDGE;
  edgeId: string;
}

export interface UpdateEdgeOperation extends BaseOperation {
  type: OperationType.UPDATE_EDGE;
  edgeId: string;
  changes: {
    path: string[];
    oldValue: any;
    newValue: any;
  }[];
}

export type Operation =
  | InsertNodeOperation
  | DeleteNodeOperation
  | UpdateNodeOperation
  | MoveNodeOperation
  | InsertEdgeOperation
  | DeleteEdgeOperation
  | UpdateEdgeOperation;

// Workflow state
export interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, any>;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
}

export interface WorkflowState {
  version: number;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  metadata: {
    lastModified: string;
    modifiedBy: string;
  };
}

// Comment types
export interface Comment {
  id: string;
  workflowId: string;
  nodeId: string;
  author: User;
  content: string;
  createdAt: string;
  updatedAt?: string;
  resolved: boolean;
  resolvedBy?: User;
  resolvedAt?: string;
  parentId?: string; // For replies
  mentions: string[]; // User IDs mentioned in the comment
  reactions: CommentReaction[];
}

export interface CommentReaction {
  emoji: string;
  userId: string;
  timestamp: string;
}

export interface CommentThread {
  id: string;
  nodeId: string;
  rootComment: Comment;
  replies: Comment[];
  resolved: boolean;
  participants: User[];
}

// Access token for public links
export interface AccessToken {
  token: string;
  workflowId: string;
  permission: PermissionLevel;
  expiresAt?: string;
  createdBy: string;
  createdAt: string;
}

// Activity feed
export enum ActivityType {
  NODE_ADDED = 'node_added',
  NODE_DELETED = 'node_deleted',
  NODE_UPDATED = 'node_updated',
  NODE_MOVED = 'node_moved',
  EDGE_ADDED = 'edge_added',
  EDGE_DELETED = 'edge_deleted',
  COMMENT_ADDED = 'comment_added',
  COMMENT_RESOLVED = 'comment_resolved',
  USER_JOINED = 'user_joined',
  USER_LEFT = 'user_left',
  PERMISSION_CHANGED = 'permission_changed',
}

export interface Activity {
  id: string;
  type: ActivityType;
  workflowId: string;
  user: User;
  timestamp: string;
  metadata: Record<string, any>;
  description: string;
}

// Conflict resolution
export interface Conflict {
  id: string;
  operationId: string;
  conflictingOperationId: string;
  type: 'concurrent_edit' | 'node_delete' | 'edge_delete' | 'data_mismatch';
  description: string;
  resolution?: ConflictResolution;
}

export interface ConflictResolution {
  strategy: 'accept_local' | 'accept_remote' | 'manual_merge';
  resolvedBy: string;
  resolvedAt: string;
  mergedOperation?: Operation;
}

// Session info
export interface CollaborationSession {
  sessionId: string;
  workflowId: string;
  userId: string;
  connectedAt: string;
  lastHeartbeat: string;
  clientId: string;
}

// Connection state
export enum ConnectionState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  DISCONNECTED = 'disconnected',
  ERROR = 'error',
}

// Export all types
export type {
  CollaborationMessage as Message,
};
