/**
 * ItemDetailView Component
 *
 * Full detail view for a queue item showing original input,
 * candidate matches, and resolution form.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ClockIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { formatDistanceToNow, format } from 'date-fns';
import toast from 'react-hot-toast';
import { useQueueItem, useSkipItem } from '../hooks/useQueue';
import { CandidateCard } from './CandidateCard';
import { ResolutionForm } from './ResolutionForm';
import type { EntityType } from '../api/types';

// Entity type labels
const entityTypeLabels: Record<EntityType, string> = {
  company: 'Company',
  product: 'Product',
  facility: 'Facility',
  material: 'Material',
  country: 'Country',
  emission_factor: 'Emission Factor',
  regulation: 'Regulation',
};

/**
 * Loading skeleton for item detail
 */
const ItemDetailSkeleton: React.FC = () => (
  <div className="p-6 space-y-6 animate-pulse">
    <div className="flex items-center gap-4">
      <div className="skeleton w-8 h-8 rounded" />
      <div className="skeleton w-48 h-6" />
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <div className="card p-6">
          <div className="skeleton w-32 h-5 mb-4" />
          <div className="space-y-3">
            <div className="skeleton w-full h-4" />
            <div className="skeleton w-3/4 h-4" />
            <div className="skeleton w-1/2 h-4" />
          </div>
        </div>
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card p-6">
              <div className="skeleton w-full h-24" />
            </div>
          ))}
        </div>
      </div>
      <div className="card p-6">
        <div className="skeleton w-32 h-5 mb-4" />
        <div className="space-y-4">
          <div className="skeleton w-full h-12" />
          <div className="skeleton w-full h-12" />
          <div className="skeleton w-full h-24" />
        </div>
      </div>
    </div>
  </div>
);

/**
 * Context info panel
 */
interface ContextPanelProps {
  context: Record<string, string>;
}

const ContextPanel: React.FC<ContextPanelProps> = ({ context }) => {
  if (Object.keys(context).length === 0) return null;

  return (
    <div className="mt-4 pt-4 border-t border-gl-neutral-200">
      <h4 className="text-sm font-medium text-gl-neutral-700 mb-3 flex items-center gap-2">
        <InformationCircleIcon className="w-4 h-4" />
        Context Information
      </h4>
      <dl className="grid grid-cols-2 gap-3">
        {Object.entries(context).map(([key, value]) => (
          <div key={key}>
            <dt className="text-xs text-gl-neutral-500 uppercase tracking-wide">
              {key.replace(/_/g, ' ')}
            </dt>
            <dd className="text-sm text-gl-neutral-900 font-medium mt-0.5">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
};

export const ItemDetailView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);

  const { item, isLoading, error, refetch } = useQueueItem(id);
  const skipMutation = useSkipItem();

  // Auto-select top candidate if exists
  useEffect(() => {
    if (item?.candidates.length && !selectedCandidateId) {
      setSelectedCandidateId(item.candidates[0].id);
    }
  }, [item, selectedCandidateId]);

  // Keyboard shortcuts for candidate selection
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Number keys 1-9 to select candidates
      const num = parseInt(e.key);
      if (num >= 1 && num <= 9 && item?.candidates) {
        const candidate = item.candidates[num - 1];
        if (candidate) {
          setSelectedCandidateId(candidate.id);
        }
      }

      // N for next item (after resolution)
      if (e.key.toLowerCase() === 'n') {
        // This will be handled elsewhere
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [item]);

  // Handle resolution complete
  const handleResolved = useCallback(() => {
    navigate('/queue');
    toast.success('Item resolved. Moving to next item...');
  }, [navigate]);

  // Handle skip
  const handleSkip = useCallback(async () => {
    if (!id) return;
    try {
      await skipMutation.mutateAsync({ id });
      navigate('/queue');
      toast('Item skipped', { icon: '⏭️' });
    } catch (err) {
      toast.error('Failed to skip item');
    }
  }, [id, skipMutation, navigate]);

  // Loading state
  if (isLoading) {
    return <ItemDetailSkeleton />;
  }

  // Error state
  if (error || !item) {
    return (
      <div className="p-6">
        <div className="card p-8 text-center">
          <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-gl-neutral-900 mb-2">
            {error ? 'Failed to load item' : 'Item not found'}
          </h2>
          <p className="text-sm text-gl-neutral-500 mb-4">
            {error instanceof Error ? error.message : 'The requested item could not be found'}
          </p>
          <Link to="/queue" className="btn-primary">
            Back to Queue
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            to="/queue"
            className="p-2 text-gl-neutral-400 hover:text-gl-neutral-600 rounded-lg hover:bg-gl-neutral-100"
            aria-label="Back to queue"
          >
            <ArrowLeftIcon className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-xl font-bold text-gl-neutral-900">
              Review Item
            </h1>
            <p className="text-sm text-gl-neutral-500 flex items-center gap-2">
              <span>{entityTypeLabels[item.entityType]}</span>
              <span className="text-gl-neutral-300">|</span>
              <span className="flex items-center gap-1">
                <ClockIcon className="w-3.5 h-3.5" />
                {formatDistanceToNow(new Date(item.createdAt), { addSuffix: true })}
              </span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span
            className={clsx('badge', {
              'badge-warning': item.status === 'pending',
              'badge-primary': item.status === 'in_review',
              'badge-gray': item.status === 'deferred',
              'badge-danger': item.status === 'escalated',
            })}
          >
            {item.status.replace('_', ' ')}
          </span>
          {item.priority >= 4 && (
            <span className="badge badge-danger flex items-center gap-1">
              <ExclamationTriangleIcon className="w-3 h-3" />
              High Priority
            </span>
          )}
        </div>
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Original input and candidates */}
        <div className="lg:col-span-2 space-y-6">
          {/* Original input card */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gl-neutral-900 flex items-center gap-2">
                <DocumentTextIcon className="w-5 h-5 text-gl-neutral-400" />
                Original Input
              </h2>
            </div>
            <div className="card-body">
              {/* Raw value */}
              <div className="mb-4">
                <label className="text-xs text-gl-neutral-500 uppercase tracking-wide">
                  Raw Value
                </label>
                <p className="text-lg font-semibold text-gl-neutral-900 mt-1 p-3 bg-gl-neutral-50 rounded-lg border border-gl-neutral-200">
                  {item.originalInput.rawValue}
                </p>
              </div>

              {/* Normalized value */}
              {item.originalInput.normalizedValue !== item.originalInput.rawValue && (
                <div className="mb-4">
                  <label className="text-xs text-gl-neutral-500 uppercase tracking-wide">
                    Normalized Value
                  </label>
                  <p className="text-base text-gl-neutral-700 mt-1 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    {item.originalInput.normalizedValue}
                  </p>
                </div>
              )}

              {/* Source info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-gl-neutral-500 uppercase tracking-wide">
                    Source
                  </label>
                  <p className="text-sm text-gl-neutral-900 mt-1">
                    {item.originalInput.source}
                  </p>
                </div>
                {item.originalInput.sourceRowId && (
                  <div>
                    <label className="text-xs text-gl-neutral-500 uppercase tracking-wide">
                      Source Row ID
                    </label>
                    <p className="text-sm font-mono text-gl-neutral-900 mt-1">
                      {item.originalInput.sourceRowId}
                    </p>
                  </div>
                )}
                <div>
                  <label className="text-xs text-gl-neutral-500 uppercase tracking-wide">
                    Timestamp
                  </label>
                  <p className="text-sm text-gl-neutral-900 mt-1">
                    {format(new Date(item.originalInput.timestamp), 'PPpp')}
                  </p>
                </div>
              </div>

              {/* Context info */}
              <ContextPanel context={item.originalInput.context} />

              {/* Review reason */}
              <div className="mt-4 pt-4 border-t border-gl-neutral-200">
                <div className="flex items-start gap-2 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-yellow-800">
                      Flagged for Review
                    </p>
                    <p className="text-sm text-yellow-700 mt-0.5">
                      {item.reason}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Candidate matches */}
          <div>
            <h2 className="text-lg font-semibold text-gl-neutral-900 mb-4">
              Candidate Matches ({item.candidates.length})
            </h2>

            {item.candidates.length === 0 ? (
              <div className="card p-8 text-center">
                <ExclamationTriangleIcon className="w-12 h-12 text-gl-neutral-300 mx-auto mb-4" />
                <h3 className="text-base font-medium text-gl-neutral-900 mb-2">
                  No candidates found
                </h3>
                <p className="text-sm text-gl-neutral-500">
                  The normalizer could not find any matching entities.
                  Consider rejecting or escalating this item.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {item.candidates.map((candidate, index) => (
                  <CandidateCard
                    key={candidate.id}
                    candidate={candidate}
                    rank={index + 1}
                    isSelected={selectedCandidateId === candidate.id}
                    isTopCandidate={index === 0}
                    onSelect={() => setSelectedCandidateId(candidate.id)}
                    keyboardShortcut={index < 9 ? String(index + 1) : undefined}
                  />
                ))}
              </div>
            )}

            {/* Confidence margin warning */}
            {item.confidenceMargin < 10 && item.candidates.length > 1 && (
              <div className="mt-4 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="flex items-start gap-2">
                  <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-yellow-800">
                      Close Confidence Scores
                    </p>
                    <p className="text-sm text-yellow-700 mt-0.5">
                      The margin between top candidates is only{' '}
                      <strong>{item.confidenceMargin.toFixed(1)}%</strong>.
                      Please review carefully before making a decision.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right column - Resolution form */}
        <div className="lg:col-span-1">
          <div className="card sticky top-6">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gl-neutral-900">
                Resolution
              </h2>
            </div>
            <div className="card-body">
              <ResolutionForm
                queueItemId={item.id}
                candidates={item.candidates}
                selectedCandidateId={selectedCandidateId}
                onResolved={handleResolved}
                onSkip={handleSkip}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ItemDetailView;
