/**
 * useCollaboration Hook
 *
 * Custom hook for managing real-time collaboration including:
 * - WebSocket connection management
 * - Presence tracking
 * - Operational Transform handling
 * - Comment management
 * - Permission management
 * - State synchronization
 *
 * @module useCollaboration
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { CollaborationService } from '../services/collaboration/CollaborationService';
import {
  User,
  UserPresence,
  Operation,
  Comment,
  Activity,
  PermissionLevel,
  ConnectionState,
  WorkflowState,
  OperationType,
} from '../services/collaboration/types';

interface UseCollaborationResult {
  connectionState: ConnectionState;
  activeUsers: UserPresence[];
  comments: Comment[];
  activities: Activity[];
  permissions: any[];
  connect: () => Promise<void>;
  disconnect: () => void;
  updateCursorPosition: (position: { x: number; y: number }) => void;
  selectNode: (nodeId: string | null) => void;
  startEditingNode: (nodeId: string) => void;
  stopEditingNode: (nodeId: string) => void;
  sendOperation: (operation: Operation) => Promise<void>;
  addComment: (nodeId: string, content: string) => Promise<void>;
  replyToComment: (commentId: string, content: string) => Promise<void>;
  resolveComment: (commentId: string) => Promise<void>;
  shareWorkflow: (email: string, permission: PermissionLevel) => Promise<void>;
  updatePermission: (userId: string, level: PermissionLevel) => Promise<void>;
  addReaction: (commentId: string, emoji: string) => Promise<void>;
}

/**
 * useCollaboration Hook
 */
export function useCollaboration(
  workflowId: string,
  currentUser: User
): UseCollaborationResult {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    ConnectionState.DISCONNECTED
  );
  const [activeUsers, setActiveUsers] = useState<UserPresence[]>([]);
  const [comments, setComments] = useState<Comment[]>([]);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [permissions, setPermissions] = useState<any[]>([]);

  const serviceRef = useRef<CollaborationService | null>(null);

  // Initialize collaboration service
  useEffect(() => {
    if (!serviceRef.current) {
      serviceRef.current = new CollaborationService(workflowId, currentUser.id, currentUser);
    }

    return () => {
      if (serviceRef.current) {
        serviceRef.current.disconnect();
      }
    };
  }, [workflowId, currentUser]);

  // Set up event listeners
  useEffect(() => {
    const service = serviceRef.current;
    if (!service) return;

    // Connection state changes
    const handleConnectionStateChange = (state: ConnectionState) => {
      setConnectionState(state);
    };

    // User joined
    const handleUserJoined = (presence: UserPresence) => {
      setActiveUsers(prev => {
        const existing = prev.find(p => p.userId === presence.userId);
        if (existing) {
          return prev.map(p => p.userId === presence.userId ? presence : p);
        }
        return [...prev, presence];
      });

      // Add activity
      addActivity({
        id: `${Date.now()}-${Math.random()}`,
        type: 'user_joined' as any,
        workflowId,
        user: presence.user,
        timestamp: new Date().toISOString(),
        metadata: {},
        description: 'joined the workflow',
      });
    };

    // User left
    const handleUserLeft = (userId: string) => {
      setActiveUsers(prev => prev.filter(p => p.userId !== userId));
    };

    // Cursor move
    const handleCursorMove = (userId: string, position: { x: number; y: number }) => {
      setActiveUsers(prev =>
        prev.map(p =>
          p.userId === userId
            ? { ...p, cursorPosition: position, lastActive: new Date().toISOString() }
            : p
        )
      );
    };

    // Node select
    const handleNodeSelect = (userId: string, nodeId: string | null) => {
      setActiveUsers(prev =>
        prev.map(p =>
          p.userId === userId
            ? { ...p, selectedNodeId: nodeId || undefined, lastActive: new Date().toISOString() }
            : p
        )
      );
    };

    // Node edit start
    const handleNodeEditStart = (userId: string, nodeId: string) => {
      setActiveUsers(prev =>
        prev.map(p =>
          p.userId === userId
            ? { ...p, editingNodeId: nodeId, lastActive: new Date().toISOString() }
            : p
        )
      );
    };

    // Node edit end
    const handleNodeEditEnd = (userId: string, nodeId: string) => {
      setActiveUsers(prev =>
        prev.map(p =>
          p.userId === userId
            ? { ...p, editingNodeId: undefined, lastActive: new Date().toISOString() }
            : p
        )
      );
    };

    // Operation received
    const handleOperation = (operation: Operation) => {
      // Add activity for the operation
      let description = '';
      switch (operation.type) {
        case OperationType.INSERT_NODE:
          description = 'added a node';
          break;
        case OperationType.DELETE_NODE:
          description = 'deleted a node';
          break;
        case OperationType.UPDATE_NODE:
          description = 'updated a node';
          break;
        case OperationType.MOVE_NODE:
          description = 'moved a node';
          break;
        case OperationType.INSERT_EDGE:
          description = 'added a connection';
          break;
        case OperationType.DELETE_EDGE:
          description = 'removed a connection';
          break;
        default:
          description = 'made a change';
      }

      const user = activeUsers.find(u => u.userId === operation.userId)?.user || currentUser;

      addActivity({
        id: operation.id,
        type: operation.type as any,
        workflowId,
        user,
        timestamp: operation.timestamp,
        metadata: { operation },
        description,
      });
    };

    // Register event listeners
    service.on('connection_state_change', handleConnectionStateChange);
    service.on('user_joined', handleUserJoined);
    service.on('user_left', handleUserLeft);
    service.on('cursor_move', handleCursorMove);
    service.on('node_select', handleNodeSelect);
    service.on('node_edit_start', handleNodeEditStart);
    service.on('node_edit_end', handleNodeEditEnd);
    service.on('operation', handleOperation);

    return () => {
      service.off('connection_state_change', handleConnectionStateChange);
      service.off('user_joined', handleUserJoined);
      service.off('user_left', handleUserLeft);
      service.off('cursor_move', handleCursorMove);
      service.off('node_select', handleNodeSelect);
      service.off('node_edit_start', handleNodeEditStart);
      service.off('node_edit_end', handleNodeEditEnd);
      service.off('operation', handleOperation);
    };
  }, [workflowId, currentUser, activeUsers]);

  // Helper to add activity
  const addActivity = useCallback((activity: Activity) => {
    setActivities(prev => [activity, ...prev.slice(0, 99)]); // Keep last 100 activities
  }, []);

  // Connect to collaboration server
  const connect = useCallback(async () => {
    if (serviceRef.current) {
      await serviceRef.current.connect();
    }
  }, []);

  // Disconnect from collaboration server
  const disconnect = useCallback(() => {
    if (serviceRef.current) {
      serviceRef.current.disconnect();
    }
  }, []);

  // Update cursor position
  const updateCursorPosition = useCallback((position: { x: number; y: number }) => {
    if (serviceRef.current) {
      serviceRef.current.updateCursor(position);
    }
  }, []);

  // Select node
  const selectNode = useCallback((nodeId: string | null) => {
    if (serviceRef.current) {
      serviceRef.current.selectNode(nodeId);
    }
  }, []);

  // Start editing node
  const startEditingNode = useCallback((nodeId: string) => {
    if (serviceRef.current) {
      serviceRef.current.startEditingNode(nodeId);
    }
  }, []);

  // Stop editing node
  const stopEditingNode = useCallback((nodeId: string) => {
    if (serviceRef.current) {
      serviceRef.current.stopEditingNode(nodeId);
    }
  }, []);

  // Send operation
  const sendOperation = useCallback(async (operation: Operation) => {
    if (serviceRef.current) {
      await serviceRef.current.sendOperation(operation);
    }
  }, []);

  // Add comment
  const addComment = useCallback(async (nodeId: string, content: string) => {
    const comment: Comment = {
      id: `comment-${Date.now()}-${Math.random()}`,
      workflowId,
      nodeId,
      author: currentUser,
      content,
      createdAt: new Date().toISOString(),
      resolved: false,
      mentions: extractMentions(content),
      reactions: [],
    };

    setComments(prev => [...prev, comment]);

    // In production, send to server
    // await fetch(`/api/v1/workflows/${workflowId}/comments`, {
    //   method: 'POST',
    //   body: JSON.stringify(comment),
    // });

    addActivity({
      id: `${Date.now()}-${Math.random()}`,
      type: 'comment_added' as any,
      workflowId,
      user: currentUser,
      timestamp: new Date().toISOString(),
      metadata: { commentId: comment.id, nodeId },
      description: `commented on ${nodeId}`,
    });
  }, [workflowId, currentUser, addActivity]);

  // Reply to comment
  const replyToComment = useCallback(async (commentId: string, content: string) => {
    const parentComment = comments.find(c => c.id === commentId);
    if (!parentComment) return;

    const reply: Comment = {
      id: `comment-${Date.now()}-${Math.random()}`,
      workflowId,
      nodeId: parentComment.nodeId,
      author: currentUser,
      content,
      createdAt: new Date().toISOString(),
      resolved: false,
      parentId: commentId,
      mentions: extractMentions(content),
      reactions: [],
    };

    setComments(prev => [...prev, reply]);

    // In production, send to server
  }, [workflowId, currentUser, comments]);

  // Resolve comment
  const resolveComment = useCallback(async (commentId: string) => {
    setComments(prev =>
      prev.map(c =>
        c.id === commentId
          ? {
              ...c,
              resolved: true,
              resolvedBy: currentUser,
              resolvedAt: new Date().toISOString(),
            }
          : c
      )
    );

    addActivity({
      id: `${Date.now()}-${Math.random()}`,
      type: 'comment_resolved' as any,
      workflowId,
      user: currentUser,
      timestamp: new Date().toISOString(),
      metadata: { commentId },
      description: 'resolved a comment',
    });

    // In production, send to server
  }, [workflowId, currentUser, addActivity]);

  // Share workflow
  const shareWorkflow = useCallback(async (email: string, permission: PermissionLevel) => {
    // In production, call API
    const newPermission = {
      userId: email, // In production, this would be a user ID
      level: permission,
      grantedBy: currentUser.id,
      grantedAt: new Date().toISOString(),
    };

    setPermissions(prev => [...prev, newPermission]);

    addActivity({
      id: `${Date.now()}-${Math.random()}`,
      type: 'permission_changed' as any,
      workflowId,
      user: currentUser,
      timestamp: new Date().toISOString(),
      metadata: { email, permission },
      description: `shared workflow with ${email} as ${permission}`,
    });
  }, [workflowId, currentUser, addActivity]);

  // Update permission
  const updatePermission = useCallback(async (userId: string, level: PermissionLevel) => {
    setPermissions(prev =>
      prev.map(p => (p.userId === userId ? { ...p, level } : p))
    );

    addActivity({
      id: `${Date.now()}-${Math.random()}`,
      type: 'permission_changed' as any,
      workflowId,
      user: currentUser,
      timestamp: new Date().toISOString(),
      metadata: { userId, level },
      description: `changed permissions for ${userId} to ${level}`,
    });

    // In production, send to server
  }, [workflowId, currentUser, addActivity]);

  // Add reaction to comment
  const addReaction = useCallback(async (commentId: string, emoji: string) => {
    setComments(prev =>
      prev.map(c => {
        if (c.id === commentId) {
          const existingReaction = c.reactions.find(
            r => r.userId === currentUser.id && r.emoji === emoji
          );

          if (existingReaction) {
            // Remove reaction
            return {
              ...c,
              reactions: c.reactions.filter(
                r => !(r.userId === currentUser.id && r.emoji === emoji)
              ),
            };
          } else {
            // Add reaction
            return {
              ...c,
              reactions: [
                ...c.reactions,
                {
                  emoji,
                  userId: currentUser.id,
                  timestamp: new Date().toISOString(),
                },
              ],
            };
          }
        }
        return c;
      })
    );

    // In production, send to server
  }, [currentUser]);

  // Extract @mentions from text
  function extractMentions(text: string): string[] {
    const mentionRegex = /@(\w+)/g;
    const mentions: string[] = [];
    let match;

    while ((match = mentionRegex.exec(text)) !== null) {
      mentions.push(match[1]);
    }

    return mentions;
  }

  return {
    connectionState,
    activeUsers,
    comments,
    activities,
    permissions,
    connect,
    disconnect,
    updateCursorPosition,
    selectNode,
    startEditingNode,
    stopEditingNode,
    sendOperation,
    addComment,
    replyToComment,
    resolveComment,
    shareWorkflow,
    updatePermission,
    addReaction,
  };
}

export default useCollaboration;
