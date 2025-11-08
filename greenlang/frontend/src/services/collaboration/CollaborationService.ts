/**
 * CollaborationService
 *
 * WebSocket-based collaboration service with:
 * - Real-time communication
 * - Operational Transform
 * - Presence protocol
 * - Offline support
 * - Auto-reconnection
 * - State synchronization
 *
 * @module CollaborationService
 */

import {
  User,
  UserPresence,
  Operation,
  WorkflowState,
  CollaborationMessage,
  MessageType,
  ConnectionState,
  CollaborationSession,
  Conflict,
  OperationType,
} from './types';

/**
 * Operational Transform implementation
 */
class OperationalTransform {
  /**
   * Transform operation against another operation
   * Returns the transformed version of op1 that can be applied after op2
   */
  static transform(op1: Operation, op2: Operation, priority: 'left' | 'right'): Operation {
    // Handle different operation type combinations
    if (op1.type === OperationType.INSERT_NODE && op2.type === OperationType.INSERT_NODE) {
      return op1; // Both inserts are independent
    }

    if (op1.type === OperationType.DELETE_NODE && op2.type === OperationType.DELETE_NODE) {
      const deleteOp1 = op1 as any;
      const deleteOp2 = op2 as any;

      if (deleteOp1.nodeId === deleteOp2.nodeId) {
        // Both trying to delete same node - drop one based on priority
        if (priority === 'left') {
          return op1;
        } else {
          // Make op1 a no-op since op2 already deleted it
          return { ...op1, type: OperationType.DELETE_NODE, nodeId: '' } as Operation;
        }
      }
      return op1;
    }

    if (op1.type === OperationType.UPDATE_NODE && op2.type === OperationType.UPDATE_NODE) {
      const updateOp1 = op1 as any;
      const updateOp2 = op2 as any;

      if (updateOp1.nodeId === updateOp2.nodeId) {
        // Both updating same node - merge changes
        const mergedChanges = this.mergeChanges(updateOp1.changes, updateOp2.changes, priority);
        return { ...op1, changes: mergedChanges } as Operation;
      }
      return op1;
    }

    if (op1.type === OperationType.UPDATE_NODE && op2.type === OperationType.DELETE_NODE) {
      const updateOp = op1 as any;
      const deleteOp = op2 as any;

      if (updateOp.nodeId === deleteOp.nodeId) {
        // Update on deleted node - drop the update
        return { ...op1, type: OperationType.UPDATE_NODE, nodeId: '', changes: [] } as Operation;
      }
      return op1;
    }

    if (op1.type === OperationType.MOVE_NODE && op2.type === OperationType.MOVE_NODE) {
      const moveOp1 = op1 as any;
      const moveOp2 = op2 as any;

      if (moveOp1.nodeId === moveOp2.nodeId) {
        // Both moving same node - use priority
        if (priority === 'right') {
          // Adjust op1 to move from op2's new position
          return {
            ...op1,
            oldPosition: moveOp2.newPosition,
          } as Operation;
        }
      }
      return op1;
    }

    // Default: operations are independent
    return op1;
  }

  /**
   * Merge changes from two update operations
   */
  private static mergeChanges(
    changes1: any[],
    changes2: any[],
    priority: 'left' | 'right'
  ): any[] {
    const mergedMap = new Map<string, any>();

    // Add all changes from first operation
    changes1.forEach(change => {
      const key = change.path.join('.');
      mergedMap.set(key, change);
    });

    // Merge changes from second operation
    changes2.forEach(change => {
      const key = change.path.join('.');
      const existing = mergedMap.get(key);

      if (existing) {
        // Conflict - use priority
        if (priority === 'right') {
          mergedMap.set(key, {
            ...existing,
            oldValue: existing.newValue,
            newValue: change.newValue,
          });
        }
      } else {
        mergedMap.set(key, change);
      }
    });

    return Array.from(mergedMap.values());
  }

  /**
   * Apply operation to workflow state
   */
  static apply(state: WorkflowState, operation: Operation): WorkflowState {
    const newState = JSON.parse(JSON.stringify(state)); // Deep clone

    switch (operation.type) {
      case OperationType.INSERT_NODE: {
        const op = operation as any;
        newState.nodes.push(op.node);
        break;
      }

      case OperationType.DELETE_NODE: {
        const op = operation as any;
        newState.nodes = newState.nodes.filter((n: any) => n.id !== op.nodeId);
        // Also remove edges connected to this node
        newState.edges = newState.edges.filter(
          (e: any) => e.source !== op.nodeId && e.target !== op.nodeId
        );
        break;
      }

      case OperationType.UPDATE_NODE: {
        const op = operation as any;
        const nodeIndex = newState.nodes.findIndex((n: any) => n.id === op.nodeId);
        if (nodeIndex >= 0) {
          op.changes.forEach((change: any) => {
            this.applyChange(newState.nodes[nodeIndex], change);
          });
        }
        break;
      }

      case OperationType.MOVE_NODE: {
        const op = operation as any;
        const node = newState.nodes.find((n: any) => n.id === op.nodeId);
        if (node) {
          node.position = op.newPosition;
        }
        break;
      }

      case OperationType.INSERT_EDGE: {
        const op = operation as any;
        newState.edges.push(op.edge);
        break;
      }

      case OperationType.DELETE_EDGE: {
        const op = operation as any;
        newState.edges = newState.edges.filter((e: any) => e.id !== op.edgeId);
        break;
      }

      case OperationType.UPDATE_EDGE: {
        const op = operation as any;
        const edgeIndex = newState.edges.findIndex((e: any) => e.id === op.edgeId);
        if (edgeIndex >= 0) {
          op.changes.forEach((change: any) => {
            this.applyChange(newState.edges[edgeIndex], change);
          });
        }
        break;
      }
    }

    newState.version++;
    newState.metadata.lastModified = new Date().toISOString();
    newState.metadata.modifiedBy = operation.userId;

    return newState;
  }

  /**
   * Apply a single change to an object
   */
  private static applyChange(obj: any, change: any): void {
    const { path, newValue } = change;
    let current = obj;

    for (let i = 0; i < path.length - 1; i++) {
      if (!current[path[i]]) {
        current[path[i]] = {};
      }
      current = current[path[i]];
    }

    current[path[path.length - 1]] = newValue;
  }
}

/**
 * Main CollaborationService class
 */
export class CollaborationService {
  private ws: WebSocket | null = null;
  private workflowId: string;
  private userId: string;
  private user: User;
  private clientId: string;
  private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // Start with 1 second
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageQueue: CollaborationMessage[] = [];
  private pendingOperations: Map<string, Operation> = new Map();
  private operationHistory: Operation[] = [];
  private currentVersion = 0;
  private listeners: Map<string, Set<Function>> = new Map();

  // Presence tracking
  private activeUsers: Map<string, UserPresence> = new Map();

  constructor(workflowId: string, userId: string, user: User) {
    this.workflowId = workflowId;
    this.userId = userId;
    this.user = user;
    this.clientId = this.generateClientId();
  }

  /**
   * Connect to collaboration server
   */
  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    this.setConnectionState(ConnectionState.CONNECTING);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/collaboration/${this.workflowId}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.onConnect();
      };

      this.ws.onmessage = (event) => {
        this.onMessage(event);
      };

      this.ws.onerror = (error) => {
        this.onError(error);
      };

      this.ws.onclose = () => {
        this.onDisconnect();
      };
    } catch (error) {
      console.error('Failed to connect:', error);
      this.setConnectionState(ConnectionState.ERROR);
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from collaboration server
   */
  disconnect(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.ws) {
      this.sendMessage({
        type: MessageType.LEAVE,
        userId: this.userId,
        workflowId: this.workflowId,
        timestamp: new Date().toISOString(),
      } as any);

      this.ws.close();
      this.ws = null;
    }

    this.setConnectionState(ConnectionState.DISCONNECTED);
  }

  /**
   * Send an operation
   */
  async sendOperation(operation: Operation): Promise<void> {
    // Add to pending operations
    this.pendingOperations.set(operation.id, operation);

    // Send via WebSocket
    this.sendMessage({
      type: MessageType.OPERATION,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      operation,
      clientId: this.clientId,
      version: this.currentVersion,
    } as any);
  }

  /**
   * Update cursor position
   */
  updateCursor(position: { x: number; y: number }): void {
    this.sendMessage({
      type: MessageType.CURSOR_MOVE,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      position,
    } as any);
  }

  /**
   * Select a node
   */
  selectNode(nodeId: string | null): void {
    this.sendMessage({
      type: MessageType.NODE_SELECT,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      nodeId,
    } as any);
  }

  /**
   * Start editing a node
   */
  startEditingNode(nodeId: string): void {
    this.sendMessage({
      type: MessageType.NODE_EDIT_START,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      nodeId,
    } as any);
  }

  /**
   * Stop editing a node
   */
  stopEditingNode(nodeId: string): void {
    this.sendMessage({
      type: MessageType.NODE_EDIT_END,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      nodeId,
    } as any);
  }

  /**
   * Request state synchronization
   */
  requestSync(): void {
    this.sendMessage({
      type: MessageType.SYNC_REQUEST,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      lastKnownVersion: this.currentVersion,
    } as any);
  }

  /**
   * Get active users
   */
  getActiveUsers(): UserPresence[] {
    return Array.from(this.activeUsers.values());
  }

  /**
   * Get connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Add event listener
   */
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  /**
   * Remove event listener
   */
  off(event: string, callback: Function): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.delete(callback);
    }
  }

  /**
   * Emit event to listeners
   */
  private emit(event: string, ...args: any[]): void {
    if (this.listeners.has(event)) {
      this.listeners.get(event)!.forEach(callback => callback(...args));
    }
  }

  /**
   * Handle connection open
   */
  private onConnect(): void {
    console.log('Connected to collaboration server');
    this.setConnectionState(ConnectionState.CONNECTED);
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;

    // Send join message
    this.sendMessage({
      type: MessageType.JOIN,
      userId: this.userId,
      workflowId: this.workflowId,
      timestamp: new Date().toISOString(),
      user: this.user,
    } as any);

    // Start heartbeat
    this.startHeartbeat();

    // Flush message queue
    this.flushMessageQueue();

    // Request sync
    this.requestSync();

    this.emit('connected');
  }

  /**
   * Handle disconnection
   */
  private onDisconnect(): void {
    console.log('Disconnected from collaboration server');
    this.setConnectionState(ConnectionState.DISCONNECTED);

    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    this.emit('disconnected');

    // Schedule reconnect
    this.scheduleReconnect();
  }

  /**
   * Handle WebSocket error
   */
  private onError(error: Event): void {
    console.error('WebSocket error:', error);
    this.setConnectionState(ConnectionState.ERROR);
    this.emit('error', error);
  }

  /**
   * Handle incoming message
   */
  private onMessage(event: MessageEvent): void {
    try {
      const message: CollaborationMessage = JSON.parse(event.data);

      switch (message.type) {
        case MessageType.JOIN:
          this.handleJoin(message as any);
          break;

        case MessageType.LEAVE:
          this.handleLeave(message as any);
          break;

        case MessageType.CURSOR_MOVE:
          this.handleCursorMove(message as any);
          break;

        case MessageType.NODE_SELECT:
          this.handleNodeSelect(message as any);
          break;

        case MessageType.NODE_EDIT_START:
          this.handleNodeEditStart(message as any);
          break;

        case MessageType.NODE_EDIT_END:
          this.handleNodeEditEnd(message as any);
          break;

        case MessageType.OPERATION:
          this.handleOperation(message as any);
          break;

        case MessageType.OPERATION_ACK:
          this.handleOperationAck(message as any);
          break;

        case MessageType.SYNC_RESPONSE:
          this.handleSyncResponse(message as any);
          break;

        case MessageType.ERROR:
          this.handleError(message as any);
          break;

        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to process message:', error);
    }
  }

  /**
   * Handle user join
   */
  private handleJoin(message: any): void {
    if (message.userId === this.userId) return; // Ignore own join

    const presence: UserPresence = {
      userId: message.userId,
      user: message.user,
      lastActive: message.timestamp,
      status: 'active',
    };

    this.activeUsers.set(message.userId, presence);
    this.emit('user_joined', presence);
  }

  /**
   * Handle user leave
   */
  private handleLeave(message: any): void {
    this.activeUsers.delete(message.userId);
    this.emit('user_left', message.userId);
  }

  /**
   * Handle cursor move
   */
  private handleCursorMove(message: any): void {
    const presence = this.activeUsers.get(message.userId);
    if (presence) {
      presence.cursorPosition = message.position;
      presence.lastActive = message.timestamp;
      this.activeUsers.set(message.userId, presence);
      this.emit('cursor_move', message.userId, message.position);
    }
  }

  /**
   * Handle node select
   */
  private handleNodeSelect(message: any): void {
    const presence = this.activeUsers.get(message.userId);
    if (presence) {
      presence.selectedNodeId = message.nodeId;
      presence.lastActive = message.timestamp;
      this.activeUsers.set(message.userId, presence);
      this.emit('node_select', message.userId, message.nodeId);
    }
  }

  /**
   * Handle node edit start
   */
  private handleNodeEditStart(message: any): void {
    const presence = this.activeUsers.get(message.userId);
    if (presence) {
      presence.editingNodeId = message.nodeId;
      presence.lastActive = message.timestamp;
      this.activeUsers.set(message.userId, presence);
      this.emit('node_edit_start', message.userId, message.nodeId);
    }
  }

  /**
   * Handle node edit end
   */
  private handleNodeEditEnd(message: any): void {
    const presence = this.activeUsers.get(message.userId);
    if (presence) {
      presence.editingNodeId = undefined;
      presence.lastActive = message.timestamp;
      this.activeUsers.set(message.userId, presence);
      this.emit('node_edit_end', message.userId, message.nodeId);
    }
  }

  /**
   * Handle incoming operation
   */
  private handleOperation(message: any): void {
    const { operation, clientId, version } = message;

    // Transform against pending operations
    let transformedOp = operation;
    this.pendingOperations.forEach(pendingOp => {
      transformedOp = OperationalTransform.transform(transformedOp, pendingOp, 'right');
    });

    // Add to operation history
    this.operationHistory.push(transformedOp);
    this.currentVersion = version;

    // Emit the transformed operation
    this.emit('operation', transformedOp);
  }

  /**
   * Handle operation acknowledgment
   */
  private handleOperationAck(message: any): void {
    const { operationId, version } = message;

    // Remove from pending operations
    this.pendingOperations.delete(operationId);
    this.currentVersion = version;

    this.emit('operation_ack', operationId);
  }

  /**
   * Handle sync response
   */
  private handleSyncResponse(message: any): void {
    const { currentVersion, operations, state } = message;

    this.currentVersion = currentVersion;

    // Apply missing operations
    operations.forEach((op: Operation) => {
      this.operationHistory.push(op);
    });

    this.emit('sync', state, operations);
  }

  /**
   * Handle error message
   */
  private handleError(message: any): void {
    console.error('Server error:', message.error);
    this.emit('server_error', message.error);
  }

  /**
   * Send message via WebSocket
   */
  private sendMessage(message: CollaborationMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for later
      this.messageQueue.push(message);
    }
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.sendMessage(message);
      }
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    this.heartbeatInterval = setInterval(() => {
      this.sendMessage({
        type: MessageType.HEARTBEAT,
        userId: this.userId,
        workflowId: this.workflowId,
        timestamp: new Date().toISOString(),
      } as any);
    }, 30000); // Every 30 seconds
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      this.setConnectionState(ConnectionState.ERROR);
      return;
    }

    this.reconnectAttempts++;
    this.setConnectionState(ConnectionState.RECONNECTING);

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Set connection state
   */
  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.emit('connection_state_change', state);
  }

  /**
   * Generate unique client ID
   */
  private generateClientId(): string {
    return `${this.userId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

export default CollaborationService;
