/**
 * ResolutionForm Component
 *
 * Form for submitting resolution decisions with notes.
 * Includes quick action buttons and keyboard shortcuts.
 */

import React, { useEffect, useCallback } from 'react';
import {
  CheckIcon,
  XMarkIcon,
  ClockIcon,
  ArrowUpCircleIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import type { ResolutionDecision, CandidateMatch } from '../api/types';
import { useResolution } from '../hooks/useResolution';

interface ResolutionFormProps {
  queueItemId: string;
  candidates: CandidateMatch[];
  selectedCandidateId: string | null;
  onResolved: () => void;
  onSkip: () => void;
}

export const ResolutionForm: React.FC<ResolutionFormProps> = ({
  queueItemId,
  candidates,
  selectedCandidateId,
  onResolved,
  onSkip,
}) => {
  const {
    decision,
    reviewerNotes,
    setReviewerNotes,
    acceptTopCandidate,
    selectCandidate,
    rejectAll,
    defer,
    escalate,
    submit,
    reset,
    isValid,
    isSubmitting,
    error,
  } = useResolution(queueItemId);

  // Sync selected candidate from parent
  useEffect(() => {
    if (selectedCandidateId && candidates.length > 0) {
      const isTop = candidates[0].id === selectedCandidateId;
      if (isTop) {
        acceptTopCandidate(candidates[0]);
      } else {
        selectCandidate(selectedCandidateId);
      }
    }
  }, [selectedCandidateId, candidates, acceptTopCandidate, selectCandidate]);

  // Handle submit
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      try {
        await submit();
        toast.success('Resolution submitted successfully');
        onResolved();
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to submit resolution');
      }
    },
    [submit, onResolved]
  );

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in textarea
      if (e.target instanceof HTMLTextAreaElement) return;

      switch (e.key.toLowerCase()) {
        case 'a':
          if (candidates.length > 0) {
            acceptTopCandidate(candidates[0]);
          }
          break;
        case 'r':
          rejectAll();
          break;
        case 's':
          onSkip();
          break;
        case 'd':
          defer();
          break;
        case 'e':
          escalate();
          break;
        case 'enter':
          if (e.ctrlKey || e.metaKey) {
            if (isValid) {
              handleSubmit(e as unknown as React.FormEvent);
            }
          }
          break;
        case 'escape':
          reset();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    candidates,
    acceptTopCandidate,
    rejectAll,
    defer,
    escalate,
    isValid,
    handleSubmit,
    reset,
    onSkip,
  ]);

  // Decision button component
  interface DecisionButtonProps {
    value: ResolutionDecision;
    label: string;
    shortcut: string;
    icon: React.ComponentType<{ className?: string }>;
    variant: 'success' | 'danger' | 'warning' | 'info';
    onClick: () => void;
  }

  const DecisionButton: React.FC<DecisionButtonProps> = ({
    value,
    label,
    shortcut,
    icon: Icon,
    variant,
    onClick,
  }) => {
    const isActive = decision === value;

    const variantClasses = {
      success: isActive
        ? 'bg-green-600 text-white border-green-600'
        : 'border-green-300 text-green-700 hover:bg-green-50',
      danger: isActive
        ? 'bg-red-600 text-white border-red-600'
        : 'border-red-300 text-red-700 hover:bg-red-50',
      warning: isActive
        ? 'bg-yellow-600 text-white border-yellow-600'
        : 'border-yellow-300 text-yellow-700 hover:bg-yellow-50',
      info: isActive
        ? 'bg-blue-600 text-white border-blue-600'
        : 'border-blue-300 text-blue-700 hover:bg-blue-50',
    };

    return (
      <button
        type="button"
        onClick={onClick}
        className={clsx(
          'flex items-center justify-center gap-2 px-4 py-3',
          'border-2 rounded-lg font-medium transition-all',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
          variantClasses[variant]
        )}
      >
        <Icon className="w-5 h-5" />
        <span>{label}</span>
        <kbd
          className={clsx(
            'ml-2 px-1.5 py-0.5 text-xs rounded',
            isActive
              ? 'bg-white/20 text-white'
              : 'bg-gl-neutral-100 text-gl-neutral-600'
          )}
        >
          {shortcut}
        </kbd>
      </button>
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Quick actions */}
      <div>
        <h3 className="text-sm font-medium text-gl-neutral-700 mb-3">
          Quick Actions
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <DecisionButton
            value="accept"
            label="Accept Top"
            shortcut="A"
            icon={CheckIcon}
            variant="success"
            onClick={() => candidates.length > 0 && acceptTopCandidate(candidates[0])}
          />
          <DecisionButton
            value="reject"
            label="Reject All"
            shortcut="R"
            icon={XMarkIcon}
            variant="danger"
            onClick={rejectAll}
          />
          <DecisionButton
            value="defer"
            label="Defer"
            shortcut="D"
            icon={ClockIcon}
            variant="warning"
            onClick={defer}
          />
          <DecisionButton
            value="escalate"
            label="Escalate"
            shortcut="E"
            icon={ArrowUpCircleIcon}
            variant="info"
            onClick={escalate}
          />
        </div>
      </div>

      {/* Selected decision indicator */}
      {decision && (
        <div
          className={clsx(
            'p-4 rounded-lg border-l-4 animate-in',
            {
              'bg-green-50 border-green-500':
                decision === 'accept' || decision === 'select',
              'bg-red-50 border-red-500': decision === 'reject',
              'bg-yellow-50 border-yellow-500': decision === 'defer',
              'bg-blue-50 border-blue-500': decision === 'escalate',
            }
          )}
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">
              Decision:{' '}
              <span className="capitalize">
                {decision === 'select' ? 'Select Candidate' : decision}
              </span>
            </span>
            {selectedCandidateId && (decision === 'accept' || decision === 'select') && (
              <span className="text-sm text-gl-neutral-600">
                - {candidates.find((c) => c.id === selectedCandidateId)?.canonicalName}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Reviewer notes */}
      <div>
        <label htmlFor="reviewer-notes" className="label">
          Reviewer Notes
          {(decision === 'reject' || decision === 'escalate') && (
            <span className="text-red-500 ml-1">*</span>
          )}
        </label>
        <textarea
          id="reviewer-notes"
          rows={4}
          className={clsx(
            'input resize-none',
            (decision === 'reject' || decision === 'escalate') &&
              !reviewerNotes.trim() &&
              'input-error'
          )}
          placeholder={
            decision === 'reject'
              ? 'Please explain why no candidates match...'
              : decision === 'escalate'
              ? 'Please explain why this needs escalation...'
              : 'Optional notes about this resolution...'
          }
          value={reviewerNotes}
          onChange={(e) => setReviewerNotes(e.target.value)}
        />
        <p className="mt-1 text-xs text-gl-neutral-500">
          {decision === 'reject' || decision === 'escalate'
            ? 'Notes are required for this action'
            : 'Add any relevant context or observations'}
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-700">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-between pt-4 border-t border-gl-neutral-200">
        <button
          type="button"
          onClick={onSkip}
          className="btn-ghost"
        >
          Skip for now
          <kbd className="ml-2 kbd">S</kbd>
        </button>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={reset}
            className="btn-secondary"
            disabled={!decision}
          >
            Cancel
            <kbd className="ml-2 kbd">Esc</kbd>
          </button>

          <button
            type="submit"
            disabled={!isValid || isSubmitting}
            className={clsx(
              'btn-primary min-w-[140px]',
              isSubmitting && 'opacity-75 cursor-wait'
            )}
          >
            {isSubmitting ? (
              <>
                <svg
                  className="animate-spin h-4 w-4"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Submitting...
              </>
            ) : (
              <>
                Submit
                <ArrowRightIcon className="w-4 h-4" />
                <kbd className="ml-1 kbd">Ctrl+Enter</kbd>
              </>
            )}
          </button>
        </div>
      </div>
    </form>
  );
};

export default ResolutionForm;
