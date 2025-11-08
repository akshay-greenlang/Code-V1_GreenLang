/**
 * VersionControl Component Tests
 *
 * Comprehensive tests for version control functionality including:
 * - Version history display
 * - Version comparison
 * - Rollback functionality
 * - Branching
 * - Tagging
 * - Auto-save
 *
 * @module VersionControlTests
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { VersionControl } from '../VersionControl';
import { useVersionControl } from '../../../hooks/useVersionControl';

// Mock the useVersionControl hook
jest.mock('../../../hooks/useVersionControl');

const mockUseVersionControl = useVersionControl as jest.MockedFunction<typeof useVersionControl>;

describe('VersionControl', () => {
  let queryClient: QueryClient;

  const mockVersions = [
    {
      id: 'v1',
      version: 1,
      workflowId: 'workflow-1',
      nodes: [],
      edges: [],
      commitMessage: 'Initial version',
      author: { id: 'user-1', name: 'John Doe', email: 'john@example.com' },
      timestamp: new Date('2024-01-01T10:00:00Z').toISOString(),
      tags: [],
      isBranch: false,
    },
    {
      id: 'v2',
      version: 2,
      workflowId: 'workflow-1',
      nodes: [],
      edges: [],
      commitMessage: 'Added data processing nodes',
      author: { id: 'user-1', name: 'John Doe', email: 'john@example.com' },
      timestamp: new Date('2024-01-01T11:00:00Z').toISOString(),
      tags: [{ name: 'production', type: 'production', color: '#10b981' }],
      isBranch: false,
    },
    {
      id: 'v3',
      version: 3,
      workflowId: 'workflow-1',
      nodes: [],
      edges: [],
      commitMessage: 'Bug fixes',
      author: { id: 'user-2', name: 'Jane Smith', email: 'jane@example.com' },
      timestamp: new Date('2024-01-01T12:00:00Z').toISOString(),
      tags: [],
      isBranch: false,
    },
  ];

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    mockUseVersionControl.mockReturnValue({
      versions: mockVersions,
      loading: false,
      error: null,
      fetchVersions: jest.fn(),
      compareVersions: jest.fn(),
      restoreVersion: jest.fn(),
      createVersion: jest.fn(),
      tagVersion: jest.fn(),
      deleteVersion: jest.fn(),
      saveDraft: jest.fn(),
      conflictDetected: false,
      conflictInfo: null,
      resolveConflict: jest.fn(),
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <VersionControl
          workflowId="workflow-1"
          currentVersion={mockVersions[2]}
          onVersionRestore={jest.fn()}
          onVersionBranch={jest.fn()}
          {...props}
        />
      </QueryClientProvider>
    );
  };

  describe('Rendering', () => {
    it('should render version history panel', () => {
      renderComponent();
      expect(screen.getByText('Version History')).toBeInTheDocument();
    });

    it('should display all versions', () => {
      renderComponent();
      expect(screen.getByText('Initial version')).toBeInTheDocument();
      expect(screen.getByText('Added data processing nodes')).toBeInTheDocument();
      expect(screen.getByText('Bug fixes')).toBeInTheDocument();
    });

    it('should show version numbers', () => {
      renderComponent();
      expect(screen.getByText('v1')).toBeInTheDocument();
      expect(screen.getByText('v2')).toBeInTheDocument();
      expect(screen.getByText('v3')).toBeInTheDocument();
    });

    it('should display author names', () => {
      renderComponent();
      expect(screen.getAllByText('John Doe')).toHaveLength(2);
      expect(screen.getByText('Jane Smith')).toBeInTheDocument();
    });

    it('should show tags', () => {
      renderComponent();
      expect(screen.getByText('production')).toBeInTheDocument();
    });

    it('should highlight current version', () => {
      renderComponent();
      const currentVersionItem = screen.getByText('Bug fixes').closest('.version-item');
      expect(currentVersionItem).toHaveClass('current');
    });
  });

  describe('Loading State', () => {
    it('should show loading indicator when loading', () => {
      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        loading: true,
        versions: [],
      });

      renderComponent();
      expect(screen.getByText('Loading version history...')).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('should show error message on error', () => {
      const error = new Error('Failed to load versions');
      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        loading: false,
        error,
        versions: [],
      });

      renderComponent();
      expect(screen.getByText('Failed to load version history')).toBeInTheDocument();
    });

    it('should allow retry on error', () => {
      const fetchVersions = jest.fn();
      const error = new Error('Failed to load versions');

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        loading: false,
        error,
        versions: [],
        fetchVersions,
      });

      renderComponent();

      const retryButton = screen.getByText('Retry');
      fireEvent.click(retryButton);

      expect(fetchVersions).toHaveBeenCalled();
    });
  });

  describe('Search and Filter', () => {
    it('should filter versions by search query', () => {
      renderComponent();

      const searchInput = screen.getByPlaceholderText('Search versions...');
      fireEvent.change(searchInput, { target: { value: 'Bug' } });

      expect(screen.getByText('Bug fixes')).toBeInTheDocument();
      expect(screen.queryByText('Initial version')).not.toBeInTheDocument();
    });

    it('should filter by tag type', () => {
      renderComponent();

      const filterSelect = screen.getByRole('combobox');
      fireEvent.change(filterSelect, { target: { value: 'production' } });

      expect(screen.getByText('Added data processing nodes')).toBeInTheDocument();
      expect(screen.queryByText('Initial version')).not.toBeInTheDocument();
      expect(screen.queryByText('Bug fixes')).not.toBeInTheDocument();
    });
  });

  describe('Version Comparison', () => {
    it('should open diff viewer when comparing versions', async () => {
      const compareVersions = jest.fn().mockResolvedValue({
        nodes: [],
        edges: [],
        summary: {
          nodesAdded: 2,
          nodesRemoved: 1,
          nodesModified: 3,
          edgesAdded: 1,
          edgesRemoved: 0,
        },
      });

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        compareVersions,
      });

      renderComponent();

      // Click on version actions
      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      // Click "View changes"
      const viewChangesButton = screen.getByText('View changes');
      fireEvent.click(viewChangesButton);

      await waitFor(() => {
        expect(screen.getByText('Version Comparison')).toBeInTheDocument();
      });

      expect(compareVersions).toHaveBeenCalled();
    });

    it('should display diff summary', async () => {
      const compareVersions = jest.fn().mockResolvedValue({
        nodes: [],
        edges: [],
        summary: {
          nodesAdded: 2,
          nodesRemoved: 1,
          nodesModified: 3,
          edgesAdded: 1,
          edgesRemoved: 0,
        },
      });

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        compareVersions,
      });

      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const viewChangesButton = screen.getByText('View changes');
      fireEvent.click(viewChangesButton);

      await waitFor(() => {
        expect(screen.getByText('+2 nodes added')).toBeInTheDocument();
        expect(screen.getByText('-1 nodes removed')).toBeInTheDocument();
        expect(screen.getByText('~3 nodes modified')).toBeInTheDocument();
      });
    });

    it('should switch between split and unified view', async () => {
      const compareVersions = jest.fn().mockResolvedValue({
        nodes: [],
        edges: [],
        summary: {
          nodesAdded: 0,
          nodesRemoved: 0,
          nodesModified: 0,
          edgesAdded: 0,
          edgesRemoved: 0,
        },
      });

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        compareVersions,
      });

      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const viewChangesButton = screen.getByText('View changes');
      fireEvent.click(viewChangesButton);

      await waitFor(() => {
        expect(screen.getByText('Split View')).toBeInTheDocument();
        expect(screen.getByText('Unified View')).toBeInTheDocument();
      });

      const unifiedButton = screen.getByText('Unified View');
      fireEvent.click(unifiedButton);

      expect(unifiedButton).toHaveClass('active');
    });
  });

  describe('Version Rollback', () => {
    it('should show rollback confirmation dialog', () => {
      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const restoreButton = screen.getByText('Restore this version');
      fireEvent.click(restoreButton);

      expect(screen.getByText('Confirm Rollback')).toBeInTheDocument();
    });

    it('should restore version on confirmation', async () => {
      const restoreVersion = jest.fn();
      const onVersionRestore = jest.fn();

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        restoreVersion,
      });

      renderComponent({ onVersionRestore });

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const restoreButton = screen.getByText('Restore this version');
      fireEvent.click(restoreButton);

      const confirmButton = screen.getByText('Restore Version');
      fireEvent.click(confirmButton);

      await waitFor(() => {
        expect(restoreVersion).toHaveBeenCalledWith('v1');
        expect(onVersionRestore).toHaveBeenCalled();
      });
    });

    it('should cancel rollback', () => {
      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const restoreButton = screen.getByText('Restore this version');
      fireEvent.click(restoreButton);

      const cancelButton = screen.getByText('Cancel');
      fireEvent.click(cancelButton);

      expect(screen.queryByText('Confirm Rollback')).not.toBeInTheDocument();
    });
  });

  describe('Version Branching', () => {
    it('should show branch creation dialog', () => {
      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const branchButton = screen.getByText('Create branch');
      fireEvent.click(branchButton);

      expect(screen.getByText(/Create Branch from/)).toBeInTheDocument();
    });

    it('should create branch with name', async () => {
      const onVersionBranch = jest.fn();

      renderComponent({ onVersionBranch });

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const branchButton = screen.getByText('Create branch');
      fireEvent.click(branchButton);

      const branchNameInput = screen.getByPlaceholderText('Enter branch name...');
      fireEvent.change(branchNameInput, { target: { value: 'feature-x' } });

      const createButton = screen.getByText('Create Branch');
      fireEvent.click(createButton);

      await waitFor(() => {
        expect(onVersionBranch).toHaveBeenCalledWith(mockVersions[0], 'feature-x');
      });
    });

    it('should not create branch without name', () => {
      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const branchButton = screen.getByText('Create branch');
      fireEvent.click(branchButton);

      const createButton = screen.getByText('Create Branch');
      expect(createButton).toBeDisabled();
    });
  });

  describe('Version Tagging', () => {
    it('should show tag dialog', () => {
      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const addTagButton = screen.getByText('Add tag');
      fireEvent.click(addTagButton);

      expect(screen.getByText('Add Tag')).toBeInTheDocument();
    });

    it('should add tag to version', async () => {
      const tagVersion = jest.fn();

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        tagVersion,
      });

      renderComponent();

      const actionsButtons = screen.getAllByText('⋮');
      fireEvent.click(actionsButtons[0]);

      const addTagButton = screen.getByText('Add tag');
      fireEvent.click(addTagButton);

      const tagNameInput = screen.getByPlaceholderText('Tag name...');
      fireEvent.change(tagNameInput, { target: { value: 'v1.0.0' } });

      const tagTypeSelect = screen.getByRole('combobox', { name: '' });
      fireEvent.change(tagTypeSelect, { target: { value: 'production' } });

      const addButton = screen.getByText('Add Tag');
      fireEvent.click(addButton);

      await waitFor(() => {
        expect(tagVersion).toHaveBeenCalled();
      });
    });
  });

  describe('Auto-save', () => {
    jest.useFakeTimers();

    it('should show auto-save indicator when enabled', () => {
      renderComponent({ autoSaveEnabled: true });
      expect(screen.getByText(/Auto-save enabled/)).toBeInTheDocument();
    });

    it('should auto-save at specified interval', async () => {
      const saveDraft = jest.fn();

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        saveDraft,
      });

      renderComponent({
        autoSaveEnabled: true,
        autoSaveInterval: 1000,
      });

      jest.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(saveDraft).toHaveBeenCalled();
      });
    });

    it('should show draft saved message', async () => {
      const saveDraft = jest.fn().mockResolvedValue(undefined);

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        saveDraft,
      });

      renderComponent({
        autoSaveEnabled: true,
        autoSaveInterval: 1000,
      });

      jest.advanceTimersByTime(1000);

      await waitFor(() => {
        expect(screen.getByText('✓ Draft saved')).toBeInTheDocument();
      });
    });

    jest.useRealTimers();
  });

  describe('Conflict Detection', () => {
    it('should show conflict warning', () => {
      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        conflictDetected: true,
        conflictInfo: {
          versionId: 'v3',
          conflictingFields: ['nodes[0].data.name'],
          localChanges: { name: 'Local Name' },
          remoteChanges: { name: 'Remote Name' },
        },
      });

      renderComponent();

      expect(screen.getByText('Version Conflict Detected')).toBeInTheDocument();
    });

    it('should allow conflict resolution', async () => {
      const resolveConflict = jest.fn();

      mockUseVersionControl.mockReturnValue({
        ...mockUseVersionControl(),
        conflictDetected: true,
        conflictInfo: {
          versionId: 'v3',
          conflictingFields: ['nodes[0].data.name'],
          localChanges: { name: 'Local Name' },
          remoteChanges: { name: 'Remote Name' },
        },
        resolveConflict,
      });

      renderComponent();

      const resolveButton = screen.getByText('Resolve Conflicts');
      fireEvent.click(resolveButton);

      await waitFor(() => {
        expect(resolveConflict).toHaveBeenCalled();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have accessible labels', () => {
      renderComponent();

      expect(screen.getByPlaceholderText('Search versions...')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('should be keyboard navigable', () => {
      renderComponent();

      const searchInput = screen.getByPlaceholderText('Search versions...');
      searchInput.focus();

      expect(document.activeElement).toBe(searchInput);
    });
  });
});
