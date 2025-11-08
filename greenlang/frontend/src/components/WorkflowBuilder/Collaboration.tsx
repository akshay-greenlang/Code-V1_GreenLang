/**
 * Collaboration Component
 *
 * Real-time collaborative editing for workflows with:
 * - WebSocket-based communication
 * - Live user presence
 * - Cursor sharing
 * - Collaborative commenting
 * - Conflict resolution
 * - User permissions
 * - Activity feed
 *
 * @module Collaboration
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useCollaboration } from '../../hooks/useCollaboration';
import { CommentThread } from './Comments/CommentThread';
import {
  User,
  UserPresence,
  PermissionLevel,
  Activity,
  Comment,
  CommentThread as CommentThreadType,
  WorkflowNode,
  Operation,
  OperationType,
} from '../../services/collaboration/types';

// Component props
interface CollaborationProps {
  workflowId: string;
  currentUser: User;
  nodes: WorkflowNode[];
  onNodesChange: (nodes: WorkflowNode[]) => void;
  onPermissionChange?: (userId: string, level: PermissionLevel) => void;
}

/**
 * Main Collaboration component
 */
export const Collaboration: React.FC<CollaborationProps> = ({
  workflowId,
  currentUser,
  nodes,
  onNodesChange,
  onPermissionChange,
}) => {
  const [showUserPanel, setShowUserPanel] = useState(true);
  const [showCommentsPanel, setShowCommentsPanel] = useState(false);
  const [showActivityFeed, setShowActivityFeed] = useState(false);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [selectedNodeForComment, setSelectedNodeForComment] = useState<string | null>(null);
  const [shareEmail, setShareEmail] = useState('');
  const [sharePermission, setSharePermission] = useState<PermissionLevel>('viewer');

  const {
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
  } = useCollaboration(workflowId, currentUser);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  // Track cursor position
  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    updateCursorPosition({ x: event.clientX, y: event.clientY });
  }, [updateCursorPosition]);

  // Get user color for avatar
  const getUserColor = (user: User): string => {
    return user.color || '#3b82f6';
  };

  // Get initials from name
  const getInitials = (name: string): string => {
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  // Handle share workflow
  const handleShare = useCallback(async () => {
    if (!shareEmail.trim()) return;

    try {
      await shareWorkflow(shareEmail, sharePermission);
      setShareEmail('');
      setShowShareDialog(false);
    } catch (error) {
      console.error('Failed to share workflow:', error);
    }
  }, [shareEmail, sharePermission, shareWorkflow]);

  // Filter comments by node
  const getCommentsForNode = useCallback((nodeId: string): CommentThreadType[] => {
    const nodeComments = comments.filter(c => c.nodeId === nodeId && !c.parentId);
    return nodeComments.map(rootComment => ({
      id: rootComment.id,
      nodeId: rootComment.nodeId,
      rootComment,
      replies: comments.filter(c => c.parentId === rootComment.id),
      resolved: rootComment.resolved,
      participants: [rootComment.author],
    }));
  }, [comments]);

  // Get unresolved comment count
  const unresolvedCommentCount = useMemo(() => {
    return comments.filter(c => !c.resolved && !c.parentId).length;
  }, [comments]);

  return (
    <div className="collaboration-container" onMouseMove={handleMouseMove}>
      {/* Connection Status */}
      <div className={`connection-status ${connectionState}`}>
        <div className="status-indicator" />
        <span>
          {connectionState === 'connected' && 'Connected'}
          {connectionState === 'connecting' && 'Connecting...'}
          {connectionState === 'reconnecting' && 'Reconnecting...'}
          {connectionState === 'disconnected' && 'Disconnected'}
          {connectionState === 'error' && 'Connection Error'}
        </span>
      </div>

      {/* Active Users Panel */}
      {showUserPanel && (
        <div className="active-users-panel">
          <div className="panel-header">
            <h3>Active Users ({activeUsers.length})</h3>
            <button onClick={() => setShowUserPanel(false)}>×</button>
          </div>
          <div className="users-list">
            {activeUsers.map(presence => (
              <UserAvatar
                key={presence.userId}
                presence={presence}
                getUserColor={getUserColor}
                getInitials={getInitials}
              />
            ))}
          </div>
          <button
            className="share-button"
            onClick={() => setShowShareDialog(true)}
          >
            + Share Workflow
          </button>
        </div>
      )}

      {/* User Cursors Overlay */}
      <div className="cursors-overlay">
        {activeUsers
          .filter(p => p.userId !== currentUser.id && p.cursorPosition)
          .map(presence => (
            <UserCursor
              key={presence.userId}
              presence={presence}
              getUserColor={getUserColor}
            />
          ))}
      </div>

      {/* Node Editing Indicators */}
      <div className="editing-indicators">
        {activeUsers
          .filter(p => p.userId !== currentUser.id && p.editingNodeId)
          .map(presence => (
            <NodeEditingIndicator
              key={presence.userId}
              presence={presence}
              nodes={nodes}
              getUserColor={getUserColor}
            />
          ))}
      </div>

      {/* Comments Panel */}
      {showCommentsPanel && (
        <div className="comments-panel">
          <div className="panel-header">
            <h3>
              Comments
              {unresolvedCommentCount > 0 && (
                <span className="unresolved-badge">{unresolvedCommentCount}</span>
              )}
            </h3>
            <button onClick={() => setShowCommentsPanel(false)}>×</button>
          </div>
          <div className="comments-list">
            {nodes.map(node => {
              const nodeComments = getCommentsForNode(node.id);
              if (nodeComments.length === 0) return null;

              return (
                <div key={node.id} className="node-comments">
                  <h4>{node.id}</h4>
                  {nodeComments.map(thread => (
                    <CommentThread
                      key={thread.id}
                      thread={thread}
                      currentUser={currentUser}
                      onReply={(content) => replyToComment(thread.rootComment.id, content)}
                      onResolve={() => resolveComment(thread.id)}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Activity Feed */}
      {showActivityFeed && (
        <div className="activity-feed">
          <div className="panel-header">
            <h3>Recent Activity</h3>
            <button onClick={() => setShowActivityFeed(false)}>×</button>
          </div>
          <div className="activities-list">
            {activities.map(activity => (
              <ActivityItem key={activity.id} activity={activity} />
            ))}
          </div>
        </div>
      )}

      {/* Share Dialog */}
      {showShareDialog && (
        <ShareDialog
          workflowId={workflowId}
          permissions={permissions}
          shareEmail={shareEmail}
          sharePermission={sharePermission}
          onEmailChange={setShareEmail}
          onPermissionChange={setSharePermission}
          onShare={handleShare}
          onClose={() => setShowShareDialog(false)}
          onUpdatePermission={updatePermission}
        />
      )}

      {/* Toolbar */}
      <div className="collaboration-toolbar">
        <button
          className={`toolbar-button ${showUserPanel ? 'active' : ''}`}
          onClick={() => setShowUserPanel(!showUserPanel)}
          title="Active Users"
        >
          👥 {activeUsers.length}
        </button>
        <button
          className={`toolbar-button ${showCommentsPanel ? 'active' : ''}`}
          onClick={() => setShowCommentsPanel(!showCommentsPanel)}
          title="Comments"
        >
          💬 {unresolvedCommentCount > 0 && <span className="badge">{unresolvedCommentCount}</span>}
        </button>
        <button
          className={`toolbar-button ${showActivityFeed ? 'active' : ''}`}
          onClick={() => setShowActivityFeed(!showActivityFeed)}
          title="Activity Feed"
        >
          📋
        </button>
      </div>
    </div>
  );
};

/**
 * User Avatar Component
 */
interface UserAvatarProps {
  presence: UserPresence;
  getUserColor: (user: User) => string;
  getInitials: (name: string) => string;
}

const UserAvatar: React.FC<UserAvatarProps> = ({
  presence,
  getUserColor,
  getInitials,
}) => {
  return (
    <div className="user-avatar">
      <div
        className="avatar-circle"
        style={{ backgroundColor: getUserColor(presence.user) }}
        title={presence.user.name}
      >
        {presence.user.avatar ? (
          <img src={presence.user.avatar} alt={presence.user.name} />
        ) : (
          getInitials(presence.user.name)
        )}
      </div>
      <div className="user-info">
        <div className="user-name">{presence.user.name}</div>
        <div className={`user-status ${presence.status}`}>
          {presence.status}
        </div>
        {presence.editingNodeId && (
          <div className="user-activity">
            ✏️ Editing {presence.editingNodeId}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * User Cursor Component
 */
interface UserCursorProps {
  presence: UserPresence;
  getUserColor: (user: User) => string;
}

const UserCursor: React.FC<UserCursorProps> = ({ presence, getUserColor }) => {
  if (!presence.cursorPosition) return null;

  return (
    <div
      className="user-cursor"
      style={{
        left: presence.cursorPosition.x,
        top: presence.cursorPosition.y,
        color: getUserColor(presence.user),
      }}
    >
      <svg width="24" height="24" viewBox="0 0 24 24">
        <path
          d="M5 3l14 9-6 1-3 6-5-16z"
          fill="currentColor"
          stroke="white"
          strokeWidth="1"
        />
      </svg>
      <div className="cursor-label" style={{ backgroundColor: getUserColor(presence.user) }}>
        {presence.user.name}
      </div>
    </div>
  );
};

/**
 * Node Editing Indicator Component
 */
interface NodeEditingIndicatorProps {
  presence: UserPresence;
  nodes: WorkflowNode[];
  getUserColor: (user: User) => string;
}

const NodeEditingIndicator: React.FC<NodeEditingIndicatorProps> = ({
  presence,
  nodes,
  getUserColor,
}) => {
  if (!presence.editingNodeId) return null;

  const node = nodes.find(n => n.id === presence.editingNodeId);
  if (!node) return null;

  return (
    <div
      className="editing-indicator"
      style={{
        left: node.position.x,
        top: node.position.y,
        borderColor: getUserColor(presence.user),
      }}
    >
      <div
        className="editing-label"
        style={{ backgroundColor: getUserColor(presence.user) }}
      >
        {presence.user.name} is editing
      </div>
    </div>
  );
};

/**
 * Activity Item Component
 */
interface ActivityItemProps {
  activity: Activity;
}

const ActivityItem: React.FC<ActivityItemProps> = ({ activity }) => {
  const getActivityIcon = (): string => {
    const icons: Record<string, string> = {
      node_added: '➕',
      node_deleted: '➖',
      node_updated: '✏️',
      node_moved: '↔️',
      edge_added: '🔗',
      edge_deleted: '🔓',
      comment_added: '💬',
      comment_resolved: '✅',
      user_joined: '👋',
      user_left: '👋',
      permission_changed: '🔐',
    };
    return icons[activity.type] || '📝';
  };

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="activity-item">
      <div className="activity-icon">{getActivityIcon()}</div>
      <div className="activity-content">
        <div className="activity-description">
          <strong>{activity.user.name}</strong> {activity.description}
        </div>
        <div className="activity-time">{formatTimestamp(activity.timestamp)}</div>
      </div>
    </div>
  );
};

/**
 * Share Dialog Component
 */
interface ShareDialogProps {
  workflowId: string;
  permissions: any[];
  shareEmail: string;
  sharePermission: PermissionLevel;
  onEmailChange: (email: string) => void;
  onPermissionChange: (permission: PermissionLevel) => void;
  onShare: () => void;
  onClose: () => void;
  onUpdatePermission: (userId: string, level: PermissionLevel) => void;
}

const ShareDialog: React.FC<ShareDialogProps> = ({
  workflowId,
  permissions,
  shareEmail,
  sharePermission,
  onEmailChange,
  onPermissionChange,
  onShare,
  onClose,
  onUpdatePermission,
}) => {
  const [showAccessToken, setShowAccessToken] = useState(false);
  const [accessToken, setAccessToken] = useState('');

  const generatePublicLink = async () => {
    // In production, this would call an API
    const token = `${workflowId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setAccessToken(token);
    setShowAccessToken(true);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="share-dialog">
      <div className="dialog-overlay" onClick={onClose} />
      <div className="dialog-content">
        <div className="dialog-header">
          <h3>Share Workflow</h3>
          <button onClick={onClose}>×</button>
        </div>

        <div className="dialog-body">
          {/* Share with user */}
          <div className="share-section">
            <h4>Share with user</h4>
            <div className="share-form">
              <input
                type="email"
                placeholder="Enter email address"
                value={shareEmail}
                onChange={(e) => onEmailChange(e.target.value)}
              />
              <select
                value={sharePermission}
                onChange={(e) => onPermissionChange(e.target.value as PermissionLevel)}
              >
                <option value="viewer">Viewer</option>
                <option value="editor">Editor</option>
                <option value="owner">Owner</option>
              </select>
              <button onClick={onShare}>Share</button>
            </div>
          </div>

          {/* Current permissions */}
          <div className="permissions-section">
            <h4>People with access</h4>
            <div className="permissions-list">
              {permissions.map(perm => (
                <div key={perm.userId} className="permission-item">
                  <div className="permission-user">{perm.userId}</div>
                  <select
                    value={perm.level}
                    onChange={(e) => onUpdatePermission(perm.userId, e.target.value as PermissionLevel)}
                  >
                    <option value="viewer">Viewer</option>
                    <option value="editor">Editor</option>
                    <option value="owner">Owner</option>
                  </select>
                </div>
              ))}
            </div>
          </div>

          {/* Public link */}
          <div className="public-link-section">
            <h4>Public link</h4>
            <button onClick={generatePublicLink}>Generate public link</button>
            {showAccessToken && (
              <div className="access-token">
                <input
                  type="text"
                  value={`${window.location.origin}/workflows/${workflowId}?token=${accessToken}`}
                  readOnly
                />
                <button
                  onClick={() =>
                    copyToClipboard(
                      `${window.location.origin}/workflows/${workflowId}?token=${accessToken}`
                    )
                  }
                >
                  Copy
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Collaboration;
