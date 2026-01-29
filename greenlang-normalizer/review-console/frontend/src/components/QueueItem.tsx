/**
 * QueueItem Component
 *
 * Displays a single queue item in the list view with key information
 * including entity type, confidence scores, and status.
 */

import React from 'react';
import {
  BuildingOfficeIcon,
  CubeIcon,
  MapPinIcon,
  BeakerIcon,
  GlobeAltIcon,
  CalculatorIcon,
  DocumentTextIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { formatDistanceToNow } from 'date-fns';
import type { QueueItem as QueueItemType, EntityType } from '../api/types';

// Entity type icons
const entityTypeIcons: Record<EntityType, React.ComponentType<{ className?: string }>> = {
  company: BuildingOfficeIcon,
  product: CubeIcon,
  facility: MapPinIcon,
  material: BeakerIcon,
  country: GlobeAltIcon,
  emission_factor: CalculatorIcon,
  regulation: DocumentTextIcon,
};

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
 * Get confidence level class
 */
function getConfidenceClass(confidence: number): string {
  if (confidence >= 90) return 'confidence-high';
  if (confidence >= 70) return 'confidence-medium';
  if (confidence >= 50) return 'confidence-low';
  return 'confidence-very-low';
}

/**
 * Get status badge props
 */
function getStatusBadge(status: QueueItemType['status']): {
  label: string;
  className: string;
} {
  switch (status) {
    case 'pending':
      return { label: 'Pending', className: 'badge-warning' };
    case 'in_review':
      return { label: 'In Review', className: 'badge-primary' };
    case 'deferred':
      return { label: 'Deferred', className: 'badge-gray' };
    case 'escalated':
      return { label: 'Escalated', className: 'badge-danger' };
    case 'resolved':
      return { label: 'Resolved', className: 'badge-success' };
    default:
      return { label: status, className: 'badge-gray' };
  }
}

/**
 * Priority indicator
 */
const PriorityIndicator: React.FC<{ priority: number }> = ({ priority }) => {
  if (priority <= 2) return null;

  return (
    <div
      className={clsx(
        'flex items-center gap-1 text-xs font-medium',
        priority >= 4 ? 'text-red-600' : 'text-yellow-600'
      )}
      title={`Priority: ${priority}/5`}
    >
      <ExclamationTriangleIcon className="w-4 h-4" />
      {priority >= 4 && <span>High Priority</span>}
    </div>
  );
};

interface QueueItemProps {
  item: QueueItemType;
  onClick: () => void;
  selected?: boolean;
}

export const QueueItem: React.FC<QueueItemProps> = ({
  item,
  onClick,
  selected = false,
}) => {
  const Icon = entityTypeIcons[item.entityType] || DocumentTextIcon;
  const statusBadge = getStatusBadge(item.status);
  const confidenceClass = getConfidenceClass(item.topConfidence);

  // Format time ago
  const timeAgo = formatDistanceToNow(new Date(item.createdAt), {
    addSuffix: true,
  });

  return (
    <article
      className={clsx(
        'card p-4 cursor-pointer transition-all',
        'hover:border-gl-primary-300 hover:shadow-md',
        selected && 'border-gl-primary-500 ring-2 ring-gl-primary-200'
      )}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      aria-label={`Review ${item.originalInput.rawValue}`}
    >
      <div className="flex gap-4">
        {/* Entity type icon */}
        <div
          className={clsx(
            'flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center',
            'bg-gl-neutral-100 text-gl-neutral-600'
          )}
        >
          <Icon className="w-6 h-6" />
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {/* Header row */}
          <div className="flex items-start justify-between gap-4 mb-2">
            <div className="min-w-0">
              <h3 className="text-base font-medium text-gl-neutral-900 truncate">
                {item.originalInput.rawValue}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-gl-neutral-500">
                  {entityTypeLabels[item.entityType]}
                </span>
                <span className="text-gl-neutral-300">|</span>
                <span className="text-xs text-gl-neutral-500">
                  {item.originalInput.source}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <PriorityIndicator priority={item.priority} />
              <span className={statusBadge.className}>{statusBadge.label}</span>
            </div>
          </div>

          {/* Confidence and candidates */}
          <div className="flex items-center gap-4 mb-2">
            <div className="flex-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gl-neutral-500">
                  Top confidence: {item.topConfidence.toFixed(1)}%
                </span>
                {item.confidenceMargin < 10 && item.candidates.length > 1 && (
                  <span className="text-xs text-yellow-600 font-medium">
                    Close match (margin: {item.confidenceMargin.toFixed(1)}%)
                  </span>
                )}
              </div>
              <div className="confidence-bar">
                <div
                  className={clsx('confidence-bar-fill', confidenceClass)}
                  style={{ width: `${item.topConfidence}%` }}
                />
              </div>
            </div>
          </div>

          {/* Footer row */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-xs text-gl-neutral-500">
                {item.candidates.length} candidate{item.candidates.length !== 1 && 's'}
              </span>
              <span className="text-xs text-gl-neutral-400">
                {item.reason}
              </span>
            </div>

            <div className="flex items-center gap-1 text-xs text-gl-neutral-400">
              <ClockIcon className="w-3.5 h-3.5" />
              <span>{timeAgo}</span>
            </div>
          </div>

          {/* Top candidate preview */}
          {item.candidates.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gl-neutral-100">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gl-neutral-500">
                  Best match:
                </span>
                <span className="text-sm text-gl-neutral-900 font-medium">
                  {item.candidates[0].canonicalName}
                </span>
                <span className="text-xs text-gl-neutral-400">
                  via {item.candidates[0].matchMethod}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </article>
  );
};

export default QueueItem;
