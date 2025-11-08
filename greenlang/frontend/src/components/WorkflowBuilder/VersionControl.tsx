/**
 * VersionControl Component
 *
 * Provides comprehensive version control functionality for workflows including:
 * - Version history panel
 * - Visual diff viewer
 * - Rollback functionality
 * - Version branching
 * - Version tagging
 * - Auto-save drafts
 *
 * @module VersionControl
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useVersionControl } from '../../hooks/useVersionControl';

// Type definitions
interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: Record<string, any>;
}

interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  type?: string;
}

interface WorkflowVersion {
  id: string;
  version: number;
  workflowId: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  commitMessage: string;
  author: {
    id: string;
    name: string;
    email: string;
  };
  timestamp: string;
  tags: VersionTag[];
  parentVersionId?: string;
  isBranch: boolean;
  branchName?: string;
}

interface VersionTag {
  name: string;
  type: 'production' | 'staging' | 'development' | 'custom';
  color: string;
}

interface DiffNode extends WorkflowNode {
  status?: 'added' | 'removed' | 'modified' | 'unchanged';
  changes?: Record<string, { old: any; new: any }>;
}

interface DiffEdge extends WorkflowEdge {
  status?: 'added' | 'removed' | 'unchanged';
}

interface VersionDiff {
  nodes: DiffNode[];
  edges: DiffEdge[];
  summary: {
    nodesAdded: number;
    nodesRemoved: number;
    nodesModified: number;
    edgesAdded: number;
    edgesRemoved: number;
  };
}

interface VersionControlProps {
  workflowId: string;
  currentVersion?: WorkflowVersion;
  onVersionRestore: (version: WorkflowVersion) => void;
  onVersionBranch: (version: WorkflowVersion, branchName: string) => void;
  autoSaveEnabled?: boolean;
  autoSaveInterval?: number;
}

/**
 * Main VersionControl component
 */
export const VersionControl: React.FC<VersionControlProps> = ({
  workflowId,
  currentVersion,
  onVersionRestore,
  onVersionBranch,
  autoSaveEnabled = true,
  autoSaveInterval = 30000, // 30 seconds
}) => {
  const [selectedVersions, setSelectedVersions] = useState<[string, string] | null>(null);
  const [showDiffViewer, setShowDiffViewer] = useState(false);
  const [showRollbackDialog, setShowRollbackDialog] = useState(false);
  const [rollbackVersion, setRollbackVersion] = useState<WorkflowVersion | null>(null);
  const [showBranchDialog, setShowBranchDialog] = useState(false);
  const [branchVersion, setBranchVersion] = useState<WorkflowVersion | null>(null);
  const [branchName, setBranchName] = useState('');
  const [commitMessage, setCommitMessage] = useState('');
  const [filterTag, setFilterTag] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [draftSaved, setDraftSaved] = useState(false);

  const queryClient = useQueryClient();

  const {
    versions,
    loading,
    error,
    fetchVersions,
    compareVersions,
    restoreVersion,
    createVersion,
    tagVersion,
    deleteVersion,
    saveDraft,
    conflictDetected,
    resolveConflict,
  } = useVersionControl(workflowId);

  // Auto-save functionality
  useEffect(() => {
    if (!autoSaveEnabled || !currentVersion) return;

    const autoSaveTimer = setInterval(async () => {
      try {
        await saveDraft(currentVersion);
        setDraftSaved(true);
        setTimeout(() => setDraftSaved(false), 2000);
      } catch (error) {
        console.error('Auto-save failed:', error);
      }
    }, autoSaveInterval);

    return () => clearInterval(autoSaveTimer);
  }, [autoSaveEnabled, autoSaveInterval, currentVersion, saveDraft]);

  // Filtered and searched versions
  const filteredVersions = useMemo(() => {
    let result = versions;

    // Filter by tag
    if (filterTag !== 'all') {
      result = result.filter(v =>
        v.tags.some(tag => tag.type === filterTag)
      );
    }

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(v =>
        v.commitMessage.toLowerCase().includes(query) ||
        v.author.name.toLowerCase().includes(query) ||
        v.branchName?.toLowerCase().includes(query)
      );
    }

    return result;
  }, [versions, filterTag, searchQuery]);

  // Handle version comparison
  const handleCompareVersions = useCallback(async (versionId1: string, versionId2: string) => {
    setSelectedVersions([versionId1, versionId2]);
    setShowDiffViewer(true);
  }, []);

  // Handle rollback request
  const handleRollbackRequest = useCallback((version: WorkflowVersion) => {
    setRollbackVersion(version);
    setShowRollbackDialog(true);
  }, []);

  // Confirm rollback
  const handleConfirmRollback = useCallback(async () => {
    if (!rollbackVersion) return;

    try {
      await restoreVersion(rollbackVersion.id);
      onVersionRestore(rollbackVersion);
      setShowRollbackDialog(false);
      setRollbackVersion(null);
      queryClient.invalidateQueries(['workflow-versions', workflowId]);
    } catch (error) {
      console.error('Rollback failed:', error);
    }
  }, [rollbackVersion, restoreVersion, onVersionRestore, queryClient, workflowId]);

  // Handle branch request
  const handleBranchRequest = useCallback((version: WorkflowVersion) => {
    setBranchVersion(version);
    setShowBranchDialog(true);
  }, []);

  // Confirm branch creation
  const handleConfirmBranch = useCallback(async () => {
    if (!branchVersion || !branchName.trim()) return;

    try {
      onVersionBranch(branchVersion, branchName);
      setShowBranchDialog(false);
      setBranchVersion(null);
      setBranchName('');
    } catch (error) {
      console.error('Branch creation failed:', error);
    }
  }, [branchVersion, branchName, onVersionBranch]);

  // Handle tag version
  const handleTagVersion = useCallback(async (
    versionId: string,
    tagName: string,
    tagType: VersionTag['type']
  ) => {
    try {
      const tag: VersionTag = {
        name: tagName,
        type: tagType,
        color: getTagColor(tagType),
      };
      await tagVersion(versionId, tag);
      queryClient.invalidateQueries(['workflow-versions', workflowId]);
    } catch (error) {
      console.error('Tag version failed:', error);
    }
  }, [tagVersion, queryClient, workflowId]);

  // Helper function to get tag color
  const getTagColor = (type: VersionTag['type']): string => {
    const colors = {
      production: '#10b981',
      staging: '#f59e0b',
      development: '#3b82f6',
      custom: '#8b5cf6',
    };
    return colors[type];
  };

  // Format date
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
    });
  };

  if (loading) {
    return (
      <div className="version-control-loading">
        <div className="spinner" />
        <p>Loading version history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="version-control-error">
        <div className="error-icon">⚠️</div>
        <p>Failed to load version history</p>
        <button onClick={() => fetchVersions()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="version-control-container">
      {/* Header */}
      <div className="version-control-header">
        <h2>Version History</h2>
        {autoSaveEnabled && (
          <div className={`auto-save-indicator ${draftSaved ? 'saved' : ''}`}>
            {draftSaved ? '✓ Draft saved' : '○ Auto-save enabled'}
          </div>
        )}
      </div>

      {/* Filters and Search */}
      <div className="version-control-filters">
        <input
          type="text"
          placeholder="Search versions..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="version-search"
        />
        <select
          value={filterTag}
          onChange={(e) => setFilterTag(e.target.value)}
          className="version-tag-filter"
        >
          <option value="all">All versions</option>
          <option value="production">Production</option>
          <option value="staging">Staging</option>
          <option value="development">Development</option>
        </select>
      </div>

      {/* Conflict Warning */}
      {conflictDetected && (
        <div className="conflict-warning">
          <div className="warning-icon">⚠️</div>
          <div className="warning-content">
            <strong>Version Conflict Detected</strong>
            <p>The workflow has been modified by another user. Please review and resolve conflicts.</p>
            <button onClick={resolveConflict}>Resolve Conflicts</button>
          </div>
        </div>
      )}

      {/* Version List */}
      <div className="version-list">
        {filteredVersions.length === 0 ? (
          <div className="no-versions">
            <p>No versions found</p>
          </div>
        ) : (
          filteredVersions.map((version) => (
            <VersionItem
              key={version.id}
              version={version}
              isCurrentVersion={currentVersion?.id === version.id}
              onCompare={handleCompareVersions}
              onRollback={handleRollbackRequest}
              onBranch={handleBranchRequest}
              onTag={handleTagVersion}
              formatDate={formatDate}
            />
          ))
        )}
      </div>

      {/* Diff Viewer Modal */}
      {showDiffViewer && selectedVersions && (
        <DiffViewerModal
          workflowId={workflowId}
          versionIds={selectedVersions}
          onClose={() => {
            setShowDiffViewer(false);
            setSelectedVersions(null);
          }}
          compareVersions={compareVersions}
        />
      )}

      {/* Rollback Confirmation Dialog */}
      {showRollbackDialog && rollbackVersion && (
        <RollbackDialog
          version={rollbackVersion}
          onConfirm={handleConfirmRollback}
          onCancel={() => {
            setShowRollbackDialog(false);
            setRollbackVersion(null);
          }}
        />
      )}

      {/* Branch Creation Dialog */}
      {showBranchDialog && branchVersion && (
        <BranchDialog
          version={branchVersion}
          branchName={branchName}
          onBranchNameChange={setBranchName}
          onConfirm={handleConfirmBranch}
          onCancel={() => {
            setShowBranchDialog(false);
            setBranchVersion(null);
            setBranchName('');
          }}
        />
      )}
    </div>
  );
};

/**
 * Individual version item component
 */
interface VersionItemProps {
  version: WorkflowVersion;
  isCurrentVersion: boolean;
  onCompare: (versionId1: string, versionId2: string) => void;
  onRollback: (version: WorkflowVersion) => void;
  onBranch: (version: WorkflowVersion) => void;
  onTag: (versionId: string, tagName: string, tagType: VersionTag['type']) => void;
  formatDate: (date: string) => string;
}

const VersionItem: React.FC<VersionItemProps> = ({
  version,
  isCurrentVersion,
  onCompare,
  onRollback,
  onBranch,
  onTag,
  formatDate,
}) => {
  const [showActions, setShowActions] = useState(false);
  const [showTagDialog, setShowTagDialog] = useState(false);

  return (
    <div className={`version-item ${isCurrentVersion ? 'current' : ''}`}>
      <div className="version-item-header">
        <div className="version-info">
          <span className="version-number">v{version.version}</span>
          {version.isBranch && (
            <span className="branch-badge">{version.branchName}</span>
          )}
          {version.tags.map((tag) => (
            <span
              key={tag.name}
              className="version-tag"
              style={{ backgroundColor: tag.color }}
            >
              {tag.name}
            </span>
          ))}
        </div>
        <button
          className="version-actions-toggle"
          onClick={() => setShowActions(!showActions)}
        >
          ⋮
        </button>
      </div>

      <div className="version-message">{version.commitMessage}</div>

      <div className="version-meta">
        <span className="version-author">{version.author.name}</span>
        <span className="version-time">{formatDate(version.timestamp)}</span>
      </div>

      {showActions && (
        <div className="version-actions">
          {!isCurrentVersion && (
            <>
              <button onClick={() => onRollback(version)}>
                Restore this version
              </button>
              <button onClick={() => onBranch(version)}>
                Create branch
              </button>
            </>
          )}
          <button onClick={() => setShowTagDialog(true)}>
            Add tag
          </button>
          <button onClick={() => onCompare(version.id, version.parentVersionId || '')}>
            View changes
          </button>
        </div>
      )}

      {showTagDialog && (
        <TagDialog
          versionId={version.id}
          onTag={onTag}
          onClose={() => setShowTagDialog(false)}
        />
      )}
    </div>
  );
};

/**
 * Diff Viewer Modal Component
 */
interface DiffViewerModalProps {
  workflowId: string;
  versionIds: [string, string];
  onClose: () => void;
  compareVersions: (v1: string, v2: string) => Promise<VersionDiff>;
}

const DiffViewerModal: React.FC<DiffViewerModalProps> = ({
  workflowId,
  versionIds,
  onClose,
  compareVersions,
}) => {
  const [diff, setDiff] = useState<VersionDiff | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'split' | 'unified'>('split');

  useEffect(() => {
    const loadDiff = async () => {
      try {
        setLoading(true);
        const result = await compareVersions(versionIds[0], versionIds[1]);
        setDiff(result);
      } catch (error) {
        console.error('Failed to load diff:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDiff();
  }, [versionIds, compareVersions]);

  return (
    <div className="diff-viewer-modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h3>Version Comparison</h3>
          <div className="view-mode-toggle">
            <button
              className={viewMode === 'split' ? 'active' : ''}
              onClick={() => setViewMode('split')}
            >
              Split View
            </button>
            <button
              className={viewMode === 'unified' ? 'active' : ''}
              onClick={() => setViewMode('unified')}
            >
              Unified View
            </button>
          </div>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        {loading ? (
          <div className="diff-loading">
            <div className="spinner" />
            <p>Comparing versions...</p>
          </div>
        ) : diff ? (
          <>
            <div className="diff-summary">
              <div className="summary-item added">
                +{diff.summary.nodesAdded} nodes added
              </div>
              <div className="summary-item removed">
                -{diff.summary.nodesRemoved} nodes removed
              </div>
              <div className="summary-item modified">
                ~{diff.summary.nodesModified} nodes modified
              </div>
              <div className="summary-item">
                {diff.summary.edgesAdded > 0 && `+${diff.summary.edgesAdded} connections added`}
                {diff.summary.edgesRemoved > 0 && ` -${diff.summary.edgesRemoved} connections removed`}
              </div>
            </div>

            <div className={`diff-content ${viewMode}`}>
              {viewMode === 'split' ? (
                <SplitDiffView diff={diff} />
              ) : (
                <UnifiedDiffView diff={diff} />
              )}
            </div>
          </>
        ) : (
          <div className="diff-error">
            <p>Failed to load version comparison</p>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Split Diff View Component
 */
const SplitDiffView: React.FC<{ diff: VersionDiff }> = ({ diff }) => {
  return (
    <div className="split-diff-view">
      <div className="diff-pane left">
        <h4>Before</h4>
        <div className="diff-nodes">
          {diff.nodes
            .filter(n => n.status === 'removed' || n.status === 'modified')
            .map(node => (
              <DiffNodeCard key={node.id} node={node} side="before" />
            ))}
        </div>
      </div>
      <div className="diff-pane right">
        <h4>After</h4>
        <div className="diff-nodes">
          {diff.nodes
            .filter(n => n.status === 'added' || n.status === 'modified')
            .map(node => (
              <DiffNodeCard key={node.id} node={node} side="after" />
            ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Unified Diff View Component
 */
const UnifiedDiffView: React.FC<{ diff: VersionDiff }> = ({ diff }) => {
  return (
    <div className="unified-diff-view">
      {diff.nodes.map(node => (
        <DiffNodeCard key={node.id} node={node} side="unified" />
      ))}
    </div>
  );
};

/**
 * Diff Node Card Component
 */
const DiffNodeCard: React.FC<{ node: DiffNode; side: 'before' | 'after' | 'unified' }> = ({
  node,
  side,
}) => {
  const getStatusColor = () => {
    switch (node.status) {
      case 'added': return '#10b981';
      case 'removed': return '#ef4444';
      case 'modified': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  return (
    <div
      className={`diff-node-card ${node.status}`}
      style={{ borderLeftColor: getStatusColor() }}
    >
      <div className="node-header">
        <span className="node-type">{node.type}</span>
        <span className="node-status">{node.status}</span>
      </div>
      <div className="node-id">{node.id}</div>
      {node.changes && (
        <div className="node-changes">
          {Object.entries(node.changes).map(([key, change]) => (
            <div key={key} className="change-item">
              <strong>{key}:</strong>
              {side === 'before' && <span className="old-value">{JSON.stringify(change.old)}</span>}
              {side === 'after' && <span className="new-value">{JSON.stringify(change.new)}</span>}
              {side === 'unified' && (
                <>
                  <span className="old-value">{JSON.stringify(change.old)}</span>
                  →
                  <span className="new-value">{JSON.stringify(change.new)}</span>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Rollback Confirmation Dialog
 */
const RollbackDialog: React.FC<{
  version: WorkflowVersion;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ version, onConfirm, onCancel }) => {
  return (
    <div className="rollback-dialog">
      <div className="dialog-overlay" onClick={onCancel} />
      <div className="dialog-content">
        <h3>Confirm Rollback</h3>
        <p>
          Are you sure you want to restore version {version.version}?
        </p>
        <p className="warning">
          This will create a new version with the content from v{version.version}.
          Your current work will not be lost.
        </p>
        <div className="dialog-actions">
          <button className="cancel-button" onClick={onCancel}>Cancel</button>
          <button className="confirm-button" onClick={onConfirm}>Restore Version</button>
        </div>
      </div>
    </div>
  );
};

/**
 * Branch Creation Dialog
 */
const BranchDialog: React.FC<{
  version: WorkflowVersion;
  branchName: string;
  onBranchNameChange: (name: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
}> = ({ version, branchName, onBranchNameChange, onConfirm, onCancel }) => {
  return (
    <div className="branch-dialog">
      <div className="dialog-overlay" onClick={onCancel} />
      <div className="dialog-content">
        <h3>Create Branch from v{version.version}</h3>
        <input
          type="text"
          placeholder="Enter branch name..."
          value={branchName}
          onChange={(e) => onBranchNameChange(e.target.value)}
          className="branch-name-input"
        />
        <div className="dialog-actions">
          <button className="cancel-button" onClick={onCancel}>Cancel</button>
          <button
            className="confirm-button"
            onClick={onConfirm}
            disabled={!branchName.trim()}
          >
            Create Branch
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Tag Dialog Component
 */
const TagDialog: React.FC<{
  versionId: string;
  onTag: (versionId: string, tagName: string, tagType: VersionTag['type']) => void;
  onClose: () => void;
}> = ({ versionId, onTag, onClose }) => {
  const [tagName, setTagName] = useState('');
  const [tagType, setTagType] = useState<VersionTag['type']>('development');

  const handleSubmit = () => {
    if (tagName.trim()) {
      onTag(versionId, tagName, tagType);
      onClose();
    }
  };

  return (
    <div className="tag-dialog">
      <div className="dialog-overlay" onClick={onClose} />
      <div className="dialog-content">
        <h3>Add Tag</h3>
        <input
          type="text"
          placeholder="Tag name..."
          value={tagName}
          onChange={(e) => setTagName(e.target.value)}
          className="tag-name-input"
        />
        <select
          value={tagType}
          onChange={(e) => setTagType(e.target.value as VersionTag['type'])}
          className="tag-type-select"
        >
          <option value="production">Production</option>
          <option value="staging">Staging</option>
          <option value="development">Development</option>
          <option value="custom">Custom</option>
        </select>
        <div className="dialog-actions">
          <button className="cancel-button" onClick={onClose}>Cancel</button>
          <button
            className="confirm-button"
            onClick={handleSubmit}
            disabled={!tagName.trim()}
          >
            Add Tag
          </button>
        </div>
      </div>
    </div>
  );
};

export default VersionControl;
