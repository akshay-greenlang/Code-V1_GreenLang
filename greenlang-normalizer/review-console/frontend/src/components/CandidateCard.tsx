/**
 * CandidateCard Component
 *
 * Displays a candidate match with confidence score, match details,
 * and selection controls for the resolution workflow.
 */

import React from 'react';
import {
  CheckCircleIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import { CheckCircleIcon as CheckCircleSolidIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';
import type { CandidateMatch } from '../api/types';

/**
 * Get confidence level info
 */
function getConfidenceLevel(confidence: number): {
  level: string;
  color: string;
  bgColor: string;
  barColor: string;
} {
  if (confidence >= 90) {
    return {
      level: 'High',
      color: 'text-green-700',
      bgColor: 'bg-green-50',
      barColor: 'bg-green-500',
    };
  }
  if (confidence >= 70) {
    return {
      level: 'Medium',
      color: 'text-yellow-700',
      bgColor: 'bg-yellow-50',
      barColor: 'bg-yellow-500',
    };
  }
  if (confidence >= 50) {
    return {
      level: 'Low',
      color: 'text-orange-700',
      bgColor: 'bg-orange-50',
      barColor: 'bg-orange-500',
    };
  }
  return {
    level: 'Very Low',
    color: 'text-red-700',
    bgColor: 'bg-red-50',
    barColor: 'bg-red-500',
  };
}

/**
 * Match method badge
 */
const MatchMethodBadge: React.FC<{ method: string }> = ({ method }) => {
  const methodLabels: Record<string, { label: string; color: string }> = {
    exact: { label: 'Exact', color: 'badge-success' },
    fuzzy: { label: 'Fuzzy', color: 'badge-primary' },
    semantic: { label: 'Semantic', color: 'badge-warning' },
    alias: { label: 'Alias', color: 'badge-gray' },
    phonetic: { label: 'Phonetic', color: 'badge-primary' },
  };

  const config = methodLabels[method] || { label: method, color: 'badge-gray' };

  return <span className={config.color}>{config.label}</span>;
};

/**
 * Match factor row
 */
interface MatchFactorRowProps {
  name: string;
  score: number;
  weight: number;
  description: string;
}

const MatchFactorRow: React.FC<MatchFactorRowProps> = ({
  name,
  score,
  weight,
  description,
}) => (
  <div className="flex items-center gap-3 py-2">
    <div className="flex-1 min-w-0">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-gl-neutral-700">{name}</span>
        <span className="text-sm text-gl-neutral-500">
          {(score * 100).toFixed(0)}% (weight: {(weight * 100).toFixed(0)}%)
        </span>
      </div>
      <div className="h-1.5 bg-gl-neutral-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gl-primary-500 rounded-full transition-all"
          style={{ width: `${score * 100}%` }}
        />
      </div>
      <p className="text-xs text-gl-neutral-500 mt-1">{description}</p>
    </div>
  </div>
);

interface CandidateCardProps {
  candidate: CandidateMatch;
  rank: number;
  isSelected: boolean;
  isTopCandidate: boolean;
  onSelect: () => void;
  keyboardShortcut?: string;
}

export const CandidateCard: React.FC<CandidateCardProps> = ({
  candidate,
  rank,
  isSelected,
  isTopCandidate,
  onSelect,
  keyboardShortcut,
}) => {
  const [showDetails, setShowDetails] = React.useState(false);
  const confidenceInfo = getConfidenceLevel(candidate.confidence);

  return (
    <article
      className={clsx(
        'card overflow-hidden transition-all',
        isSelected
          ? 'ring-2 ring-gl-primary-500 border-gl-primary-500'
          : 'hover:border-gl-neutral-300',
        isTopCandidate && !isSelected && 'border-green-200 bg-green-50/30'
      )}
    >
      {/* Header */}
      <div
        className={clsx(
          'p-4 cursor-pointer',
          isSelected && 'bg-gl-primary-50'
        )}
        onClick={onSelect}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect();
          }
        }}
        aria-label={`Select ${candidate.canonicalName}`}
        aria-pressed={isSelected}
      >
        <div className="flex items-start gap-4">
          {/* Rank and selection indicator */}
          <div className="flex-shrink-0">
            <div
              className={clsx(
                'w-10 h-10 rounded-full flex items-center justify-center',
                'text-sm font-semibold transition-colors',
                isSelected
                  ? 'bg-gl-primary-500 text-white'
                  : 'bg-gl-neutral-100 text-gl-neutral-600'
              )}
            >
              {isSelected ? (
                <CheckCircleSolidIcon className="w-6 h-6" />
              ) : (
                <span>#{rank}</span>
              )}
            </div>
          </div>

          {/* Main content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-gl-neutral-900">
                  {candidate.canonicalName}
                </h3>
                <p className="text-sm text-gl-neutral-500 mt-0.5">
                  ID: {candidate.canonicalId}
                </p>
              </div>

              <div className="flex items-center gap-2 flex-shrink-0">
                {keyboardShortcut && (
                  <kbd className="kbd">{keyboardShortcut}</kbd>
                )}
                <MatchMethodBadge method={candidate.matchMethod} />
              </div>
            </div>

            {/* Confidence bar */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gl-neutral-700">
                  Confidence Score
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={clsx(
                      'px-2 py-0.5 rounded text-xs font-medium',
                      confidenceInfo.bgColor,
                      confidenceInfo.color
                    )}
                  >
                    {confidenceInfo.level}
                  </span>
                  <span className="text-lg font-bold text-gl-neutral-900">
                    {candidate.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="h-3 bg-gl-neutral-100 rounded-full overflow-hidden">
                <div
                  className={clsx(
                    'h-full rounded-full transition-all duration-500',
                    confidenceInfo.barColor
                  )}
                  style={{ width: `${candidate.confidence}%` }}
                />
              </div>
            </div>

            {/* Quick metadata */}
            {candidate.metadata && Object.keys(candidate.metadata).length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {Object.entries(candidate.metadata)
                  .slice(0, 3)
                  .map(([key, value]) => (
                    <span
                      key={key}
                      className="inline-flex items-center px-2 py-1 bg-gl-neutral-100 rounded text-xs text-gl-neutral-600"
                    >
                      <span className="font-medium">{key}:</span>
                      <span className="ml-1">{String(value)}</span>
                    </span>
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* Expand/collapse button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowDetails(!showDetails);
          }}
          className="mt-4 flex items-center gap-1 text-sm text-gl-neutral-500 hover:text-gl-neutral-700"
        >
          <InformationCircleIcon className="w-4 h-4" />
          <span>{showDetails ? 'Hide' : 'Show'} match details</span>
          {showDetails ? (
            <ChevronUpIcon className="w-4 h-4" />
          ) : (
            <ChevronDownIcon className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Match details (expandable) */}
      {showDetails && (
        <div className="border-t border-gl-neutral-200 p-4 bg-gl-neutral-50 animate-in">
          <h4 className="text-sm font-semibold text-gl-neutral-900 mb-3">
            Match Factors
          </h4>
          <div className="space-y-1 divide-y divide-gl-neutral-200">
            {candidate.matchDetails.factors.map((factor) => (
              <MatchFactorRow
                key={factor.name}
                name={factor.name}
                score={factor.score}
                weight={factor.weight}
                description={factor.description}
              />
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-gl-neutral-200">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gl-neutral-700">
                Overall Score
              </span>
              <span className="text-lg font-bold text-gl-neutral-900">
                {(candidate.matchDetails.score * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Selection indicator for top candidate */}
      {isTopCandidate && !isSelected && (
        <div className="px-4 py-2 bg-green-50 border-t border-green-100 flex items-center gap-2">
          <CheckCircleIcon className="w-4 h-4 text-green-600" />
          <span className="text-xs font-medium text-green-700">
            Top recommended match
          </span>
        </div>
      )}
    </article>
  );
};

export default CandidateCard;
