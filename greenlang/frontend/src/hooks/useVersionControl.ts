/**
 * useVersionControl Hook
 *
 * Custom hook for managing workflow version control including:
 * - Version fetching and caching
 * - Version comparison
 * - Rollback functionality
 * - Conflict detection and resolution
 * - Optimistic updates
 * - IndexedDB caching
 *
 * @module useVersionControl
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { openDB, IDBPDatabase } from 'idb';

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

interface ConflictInfo {
  versionId: string;
  conflictingFields: string[];
  localChanges: Record<string, any>;
  remoteChanges: Record<string, any>;
}

interface UseVersionControlResult {
  versions: WorkflowVersion[];
  loading: boolean;
  error: Error | null;
  fetchVersions: () => Promise<void>;
  compareVersions: (versionId1: string, versionId2: string) => Promise<VersionDiff>;
  restoreVersion: (versionId: string) => Promise<void>;
  createVersion: (data: Partial<WorkflowVersion>) => Promise<WorkflowVersion>;
  tagVersion: (versionId: string, tag: VersionTag) => Promise<void>;
  deleteVersion: (versionId: string) => Promise<void>;
  saveDraft: (version: WorkflowVersion) => Promise<void>;
  conflictDetected: boolean;
  conflictInfo: ConflictInfo | null;
  resolveConflict: () => Promise<void>;
}

// IndexedDB database name and version
const DB_NAME = 'greenlang-version-control';
const DB_VERSION = 1;
const STORE_NAME = 'versions';

/**
 * Initialize IndexedDB database
 */
async function initDB(): Promise<IDBPDatabase> {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('workflowId', 'workflowId');
        store.createIndex('timestamp', 'timestamp');
      }
    },
  });
}

/**
 * API client for version control endpoints
 */
class VersionControlAPI {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
    };
  }

  async fetchVersions(workflowId: string): Promise<WorkflowVersion[]> {
    const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/versions`, {
      headers: this.headers,
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch versions: ${response.statusText}`);
    }

    return response.json();
  }

  async getVersion(workflowId: string, versionId: string): Promise<WorkflowVersion> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}`,
      { headers: this.headers }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch version: ${response.statusText}`);
    }

    return response.json();
  }

  async compareVersions(
    workflowId: string,
    versionId1: string,
    versionId2: string
  ): Promise<VersionDiff> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/compare?v1=${versionId1}&v2=${versionId2}`,
      { headers: this.headers }
    );

    if (!response.ok) {
      throw new Error(`Failed to compare versions: ${response.statusText}`);
    }

    return response.json();
  }

  async createVersion(
    workflowId: string,
    data: Partial<WorkflowVersion>
  ): Promise<WorkflowVersion> {
    const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/versions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Failed to create version: ${response.statusText}`);
    }

    return response.json();
  }

  async restoreVersion(workflowId: string, versionId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}/restore`,
      {
        method: 'POST',
        headers: this.headers,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to restore version: ${response.statusText}`);
    }
  }

  async tagVersion(
    workflowId: string,
    versionId: string,
    tag: VersionTag
  ): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}/tags`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(tag),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to tag version: ${response.statusText}`);
    }
  }

  async deleteVersion(workflowId: string, versionId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}`,
      {
        method: 'DELETE',
        headers: this.headers,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to delete version: ${response.statusText}`);
    }
  }

  async saveDraft(workflowId: string, version: WorkflowVersion): Promise<void> {
    const response = await fetch(`${this.baseUrl}/workflows/${workflowId}/drafts`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(version),
    });

    if (!response.ok) {
      throw new Error(`Failed to save draft: ${response.statusText}`);
    }
  }

  async checkConflicts(workflowId: string, versionId: string): Promise<ConflictInfo | null> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}/conflicts`,
      { headers: this.headers }
    );

    if (!response.ok) {
      throw new Error(`Failed to check conflicts: ${response.statusText}`);
    }

    const data = await response.json();
    return data.hasConflict ? data.conflict : null;
  }

  async resolveConflict(
    workflowId: string,
    versionId: string,
    resolution: Record<string, any>
  ): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/workflows/${workflowId}/versions/${versionId}/conflicts/resolve`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(resolution),
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to resolve conflict: ${response.statusText}`);
    }
  }
}

/**
 * Version Control Hook
 */
export function useVersionControl(workflowId: string): UseVersionControlResult {
  const [versions, setVersions] = useState<WorkflowVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [conflictDetected, setConflictDetected] = useState(false);
  const [conflictInfo, setConflictInfo] = useState<ConflictInfo | null>(null);

  const queryClient = useQueryClient();
  const apiClient = useRef(new VersionControlAPI());
  const dbRef = useRef<IDBPDatabase | null>(null);

  // Initialize IndexedDB
  useEffect(() => {
    initDB().then(db => {
      dbRef.current = db;
    }).catch(err => {
      console.error('Failed to initialize IndexedDB:', err);
    });

    return () => {
      if (dbRef.current) {
        dbRef.current.close();
      }
    };
  }, []);

  // Cache versions in IndexedDB
  const cacheVersions = useCallback(async (versions: WorkflowVersion[]) => {
    if (!dbRef.current) return;

    const tx = dbRef.current.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    for (const version of versions) {
      await store.put(version);
    }

    await tx.done;
  }, []);

  // Get versions from cache
  const getCachedVersions = useCallback(async (): Promise<WorkflowVersion[]> => {
    if (!dbRef.current) return [];

    const tx = dbRef.current.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('workflowId');

    const versions = await index.getAll(workflowId);
    await tx.done;

    return versions.sort((a, b) =>
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [workflowId]);

  // Fetch versions from API
  const fetchVersions = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Try to load from cache first
      const cachedVersions = await getCachedVersions();
      if (cachedVersions.length > 0) {
        setVersions(cachedVersions);
      }

      // Fetch fresh data from API
      const freshVersions = await apiClient.current.fetchVersions(workflowId);
      setVersions(freshVersions);

      // Update cache
      await cacheVersions(freshVersions);
    } catch (err) {
      setError(err as Error);
      console.error('Failed to fetch versions:', err);
    } finally {
      setLoading(false);
    }
  }, [workflowId, getCachedVersions, cacheVersions]);

  // Initial fetch
  useEffect(() => {
    fetchVersions();
  }, [fetchVersions]);

  // Compare two versions
  const compareVersions = useCallback(async (
    versionId1: string,
    versionId2: string
  ): Promise<VersionDiff> => {
    try {
      const diff = await apiClient.current.compareVersions(workflowId, versionId1, versionId2);
      return diff;
    } catch (err) {
      console.error('Failed to compare versions:', err);
      throw err;
    }
  }, [workflowId]);

  // Restore a version
  const restoreVersion = useCallback(async (versionId: string) => {
    try {
      // Optimistic update
      queryClient.setQueryData(['workflow-versions', workflowId], (old: any) => {
        const version = versions.find(v => v.id === versionId);
        if (!version) return old;

        return [
          {
            ...version,
            id: `${Date.now()}`,
            version: (versions[0]?.version || 0) + 1,
            timestamp: new Date().toISOString(),
            commitMessage: `Restored from version ${version.version}`,
          },
          ...old,
        ];
      });

      await apiClient.current.restoreVersion(workflowId, versionId);
      await fetchVersions();
    } catch (err) {
      // Rollback optimistic update
      queryClient.invalidateQueries(['workflow-versions', workflowId]);
      console.error('Failed to restore version:', err);
      throw err;
    }
  }, [workflowId, versions, queryClient, fetchVersions]);

  // Create a new version
  const createVersion = useCallback(async (
    data: Partial<WorkflowVersion>
  ): Promise<WorkflowVersion> => {
    try {
      const newVersion = await apiClient.current.createVersion(workflowId, data);

      // Optimistic update
      setVersions(prev => [newVersion, ...prev]);

      // Update cache
      if (dbRef.current) {
        const tx = dbRef.current.transaction(STORE_NAME, 'readwrite');
        await tx.objectStore(STORE_NAME).put(newVersion);
        await tx.done;
      }

      return newVersion;
    } catch (err) {
      console.error('Failed to create version:', err);
      throw err;
    }
  }, [workflowId]);

  // Tag a version
  const tagVersion = useCallback(async (versionId: string, tag: VersionTag) => {
    try {
      // Optimistic update
      setVersions(prev => prev.map(v =>
        v.id === versionId
          ? { ...v, tags: [...v.tags, tag] }
          : v
      ));

      await apiClient.current.tagVersion(workflowId, versionId, tag);
    } catch (err) {
      // Rollback on error
      await fetchVersions();
      console.error('Failed to tag version:', err);
      throw err;
    }
  }, [workflowId, fetchVersions]);

  // Delete a version
  const deleteVersion = useCallback(async (versionId: string) => {
    try {
      // Optimistic update
      setVersions(prev => prev.filter(v => v.id !== versionId));

      await apiClient.current.deleteVersion(workflowId, versionId);

      // Update cache
      if (dbRef.current) {
        const tx = dbRef.current.transaction(STORE_NAME, 'readwrite');
        await tx.objectStore(STORE_NAME).delete(versionId);
        await tx.done;
      }
    } catch (err) {
      // Rollback on error
      await fetchVersions();
      console.error('Failed to delete version:', err);
      throw err;
    }
  }, [workflowId, fetchVersions]);

  // Save draft
  const saveDraft = useCallback(async (version: WorkflowVersion) => {
    try {
      await apiClient.current.saveDraft(workflowId, version);

      // Update local cache
      if (dbRef.current) {
        const tx = dbRef.current.transaction(STORE_NAME, 'readwrite');
        await tx.objectStore(STORE_NAME).put({
          ...version,
          id: `draft-${version.id}`,
          isDraft: true,
        });
        await tx.done;
      }
    } catch (err) {
      console.error('Failed to save draft:', err);
      throw err;
    }
  }, [workflowId]);

  // Check for conflicts
  useEffect(() => {
    const checkConflicts = async () => {
      if (versions.length === 0) return;

      try {
        const latestVersion = versions[0];
        const conflict = await apiClient.current.checkConflicts(
          workflowId,
          latestVersion.id
        );

        if (conflict) {
          setConflictDetected(true);
          setConflictInfo(conflict);
        } else {
          setConflictDetected(false);
          setConflictInfo(null);
        }
      } catch (err) {
        console.error('Failed to check conflicts:', err);
      }
    };

    // Check for conflicts every 30 seconds
    const interval = setInterval(checkConflicts, 30000);
    checkConflicts();

    return () => clearInterval(interval);
  }, [workflowId, versions]);

  // Resolve conflict
  const resolveConflict = useCallback(async () => {
    if (!conflictInfo) return;

    try {
      // For now, use a simple last-write-wins strategy
      // In a real implementation, this would show a UI for manual resolution
      await apiClient.current.resolveConflict(
        workflowId,
        conflictInfo.versionId,
        conflictInfo.remoteChanges
      );

      setConflictDetected(false);
      setConflictInfo(null);
      await fetchVersions();
    } catch (err) {
      console.error('Failed to resolve conflict:', err);
      throw err;
    }
  }, [workflowId, conflictInfo, fetchVersions]);

  return {
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
    conflictInfo,
    resolveConflict,
  };
}

export default useVersionControl;
