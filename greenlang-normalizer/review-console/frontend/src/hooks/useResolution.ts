/**
 * useResolution Hook
 *
 * React Query hook for submitting and managing resolutions.
 */

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useCallback } from 'react';
import { queueAPI } from '../api/client';
import { queueKeys } from './useQueue';
import type {
  ResolutionSubmission,
  Resolution,
  ResolutionDecision,
  CandidateMatch,
} from '../api/types';

/**
 * Hook for managing resolution form state and submission
 */
export function useResolution(queueItemId: string) {
  const queryClient = useQueryClient();

  // Form state
  const [decision, setDecision] = useState<ResolutionDecision | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [reviewerNotes, setReviewerNotes] = useState('');

  // Submit mutation
  const submitMutation = useMutation({
    mutationFn: (data: ResolutionSubmission) =>
      queueAPI.submitResolution(queueItemId, data),
    onSuccess: () => {
      // Invalidate queue caches
      queryClient.invalidateQueries({ queryKey: queueKeys.all });
    },
  });

  // Reset form
  const reset = useCallback(() => {
    setDecision(null);
    setSelectedCandidateId(null);
    setReviewerNotes('');
  }, []);

  // Accept top candidate
  const acceptTopCandidate = useCallback((candidate: CandidateMatch) => {
    setDecision('accept');
    setSelectedCandidateId(candidate.id);
  }, []);

  // Select a specific candidate
  const selectCandidate = useCallback((candidateId: string) => {
    setDecision('select');
    setSelectedCandidateId(candidateId);
  }, []);

  // Reject all candidates
  const rejectAll = useCallback(() => {
    setDecision('reject');
    setSelectedCandidateId(null);
  }, []);

  // Defer for later
  const defer = useCallback(() => {
    setDecision('defer');
    setSelectedCandidateId(null);
  }, []);

  // Escalate to senior reviewer
  const escalate = useCallback(() => {
    setDecision('escalate');
    setSelectedCandidateId(null);
  }, []);

  // Submit resolution
  const submit = useCallback(async (): Promise<Resolution> => {
    if (!decision) {
      throw new Error('No decision selected');
    }

    const submission: ResolutionSubmission = {
      decision,
      selectedCandidateId: selectedCandidateId ?? undefined,
      reviewerNotes,
    };

    const result = await submitMutation.mutateAsync(submission);
    reset();
    return result;
  }, [decision, selectedCandidateId, reviewerNotes, submitMutation, reset]);

  // Check if form is valid
  const isValid = useCallback(() => {
    if (!decision) return false;

    // Accept and select require a candidate
    if ((decision === 'accept' || decision === 'select') && !selectedCandidateId) {
      return false;
    }

    // Reject and escalate should have notes
    if ((decision === 'reject' || decision === 'escalate') && !reviewerNotes.trim()) {
      return false;
    }

    return true;
  }, [decision, selectedCandidateId, reviewerNotes]);

  return {
    // State
    decision,
    selectedCandidateId,
    reviewerNotes,

    // Setters
    setDecision,
    setSelectedCandidateId,
    setReviewerNotes,

    // Quick actions
    acceptTopCandidate,
    selectCandidate,
    rejectAll,
    defer,
    escalate,

    // Submission
    submit,
    reset,
    isValid: isValid(),

    // Mutation state
    isSubmitting: submitMutation.isPending,
    error: submitMutation.error,
    isSuccess: submitMutation.isSuccess,
  };
}

/**
 * Hook for batch resolution operations
 */
export function useBatchResolution() {
  const queryClient = useQueryClient();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Toggle selection
  const toggleSelection = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  // Select all
  const selectAll = useCallback((ids: string[]) => {
    setSelectedIds(new Set(ids));
  }, []);

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  // Batch accept mutation
  const batchAcceptMutation = useMutation({
    mutationFn: async (items: Array<{ id: string; candidateId: string }>) => {
      const results = await Promise.allSettled(
        items.map(({ id, candidateId }) =>
          queueAPI.submitResolution(id, {
            decision: 'accept',
            selectedCandidateId: candidateId,
            reviewerNotes: 'Batch accepted',
          })
        )
      );
      return results;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queueKeys.all });
      clearSelection();
    },
  });

  // Batch defer mutation
  const batchDeferMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const results = await Promise.allSettled(
        ids.map((id) =>
          queueAPI.submitResolution(id, {
            decision: 'defer',
            reviewerNotes: 'Batch deferred',
          })
        )
      );
      return results;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queueKeys.all });
      clearSelection();
    },
  });

  return {
    selectedIds,
    selectedCount: selectedIds.size,
    toggleSelection,
    selectAll,
    clearSelection,
    isSelected: (id: string) => selectedIds.has(id),

    // Batch operations
    batchAccept: batchAcceptMutation.mutate,
    batchDefer: batchDeferMutation.mutate,
    isBatchProcessing: batchAcceptMutation.isPending || batchDeferMutation.isPending,
  };
}

/**
 * Hook for tracking resolution history
 */
export function useResolutionHistory() {
  const [history, setHistory] = useState<Resolution[]>([]);

  const addToHistory = useCallback((resolution: Resolution) => {
    setHistory((prev) => [resolution, ...prev].slice(0, 50)); // Keep last 50
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return {
    history,
    addToHistory,
    clearHistory,
    recentCount: history.length,
  };
}
