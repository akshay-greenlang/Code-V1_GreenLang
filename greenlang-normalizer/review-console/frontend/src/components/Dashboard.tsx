/**
 * Dashboard Component
 *
 * Overview dashboard showing review queue statistics,
 * performance metrics, and trend charts.
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  QueueListIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { dashboardAPI } from '../api/client';
import type { DashboardStats, EntityType, ConfidenceLevel } from '../api/types';

// Entity type display configuration
const entityTypeConfig: Record<EntityType, { label: string; color: string }> = {
  company: { label: 'Companies', color: 'bg-blue-500' },
  product: { label: 'Products', color: 'bg-green-500' },
  facility: { label: 'Facilities', color: 'bg-purple-500' },
  material: { label: 'Materials', color: 'bg-orange-500' },
  country: { label: 'Countries', color: 'bg-teal-500' },
  emission_factor: { label: 'Emission Factors', color: 'bg-red-500' },
  regulation: { label: 'Regulations', color: 'bg-indigo-500' },
};

// Confidence level display configuration
const confidenceLevelConfig: Record<ConfidenceLevel, { label: string; color: string }> = {
  high: { label: 'High (>90%)', color: 'bg-green-500' },
  medium: { label: 'Medium (70-90%)', color: 'bg-yellow-500' },
  low: { label: 'Low (50-70%)', color: 'bg-orange-500' },
  very_low: { label: 'Very Low (<50%)', color: 'bg-red-500' },
};

/**
 * Stat card component
 */
interface StatCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ComponentType<{ className?: string }>;
  color: 'green' | 'blue' | 'yellow' | 'red' | 'gray';
}

const StatCard: React.FC<StatCardProps> = ({ title, value, change, icon: Icon, color }) => {
  const colorClasses = {
    green: 'bg-green-50 text-green-600',
    blue: 'bg-blue-50 text-blue-600',
    yellow: 'bg-yellow-50 text-yellow-600',
    red: 'bg-red-50 text-red-600',
    gray: 'bg-gl-neutral-50 text-gl-neutral-600',
  };

  return (
    <div className="card p-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-gl-neutral-500">{title}</p>
          <p className="mt-2 text-3xl font-semibold text-gl-neutral-900">{value}</p>
          {change !== undefined && (
            <div className="mt-2 flex items-center gap-1">
              {change >= 0 ? (
                <ArrowTrendingUpIcon className="w-4 h-4 text-green-500" />
              ) : (
                <ArrowTrendingDownIcon className="w-4 h-4 text-red-500" />
              )}
              <span
                className={clsx(
                  'text-sm font-medium',
                  change >= 0 ? 'text-green-600' : 'text-red-600'
                )}
              >
                {Math.abs(change)}%
              </span>
              <span className="text-sm text-gl-neutral-500">vs last week</span>
            </div>
          )}
        </div>
        <div className={clsx('p-3 rounded-lg', colorClasses[color])}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  );
};

/**
 * Distribution bar component
 */
interface DistributionBarProps {
  data: Array<{ key: string; value: number; label: string; color: string }>;
  total: number;
}

const DistributionBar: React.FC<DistributionBarProps> = ({ data, total }) => {
  if (total === 0) return null;

  return (
    <div className="space-y-3">
      <div className="flex h-4 rounded-full overflow-hidden bg-gl-neutral-100">
        {data.map((item) => {
          const percentage = (item.value / total) * 100;
          if (percentage === 0) return null;
          return (
            <div
              key={item.key}
              className={clsx('transition-all', item.color)}
              style={{ width: `${percentage}%` }}
              title={`${item.label}: ${item.value} (${percentage.toFixed(1)}%)`}
            />
          );
        })}
      </div>
      <div className="flex flex-wrap gap-4">
        {data.map((item) => (
          <div key={item.key} className="flex items-center gap-2">
            <div className={clsx('w-3 h-3 rounded-full', item.color)} />
            <span className="text-xs text-gl-neutral-600">
              {item.label}: {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Loading skeleton for dashboard
 */
const DashboardSkeleton: React.FC = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="card p-6">
          <div className="skeleton h-4 w-24 mb-4" />
          <div className="skeleton h-8 w-16 mb-2" />
          <div className="skeleton h-4 w-32" />
        </div>
      ))}
    </div>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {[...Array(2)].map((_, i) => (
        <div key={i} className="card p-6">
          <div className="skeleton h-6 w-32 mb-4" />
          <div className="skeleton h-40 w-full" />
        </div>
      ))}
    </div>
  </div>
);

/**
 * Format seconds to human readable time
 */
function formatTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

/**
 * Dashboard component
 */
export const Dashboard: React.FC = () => {
  const { data: stats, isLoading, error } = useQuery<DashboardStats>({
    queryKey: ['dashboard', 'stats'],
    queryFn: () => dashboardAPI.getStats(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (isLoading) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold text-gl-neutral-900 mb-6">Dashboard</h1>
        <DashboardSkeleton />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="card p-6 text-center">
          <ExclamationTriangleIcon className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-gl-neutral-900 mb-2">
            Failed to load dashboard
          </h2>
          <p className="text-sm text-gl-neutral-500">
            {error instanceof Error ? error.message : 'An error occurred'}
          </p>
        </div>
      </div>
    );
  }

  if (!stats) return null;

  // Prepare entity type distribution data
  const entityTypeData = Object.entries(stats.byEntityType).map(([key, value]) => ({
    key,
    value,
    label: entityTypeConfig[key as EntityType]?.label || key,
    color: entityTypeConfig[key as EntityType]?.color || 'bg-gray-500',
  }));

  const entityTypeTotal = entityTypeData.reduce((sum, item) => sum + item.value, 0);

  // Prepare confidence level distribution data
  const confidenceData = Object.entries(stats.byConfidenceLevel).map(([key, value]) => ({
    key,
    value,
    label: confidenceLevelConfig[key as ConfidenceLevel]?.label || key,
    color: confidenceLevelConfig[key as ConfidenceLevel]?.color || 'bg-gray-500',
  }));

  const confidenceTotal = confidenceData.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gl-neutral-900">Dashboard</h1>
        <Link to="/queue" className="btn-primary">
          <QueueListIcon className="w-5 h-5" />
          Start Reviewing
        </Link>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Pending Review"
          value={stats.pending}
          icon={QueueListIcon}
          color={stats.pending > 100 ? 'red' : stats.pending > 50 ? 'yellow' : 'green'}
        />
        <StatCard
          title="In Review"
          value={stats.inReview}
          icon={ClockIcon}
          color="blue"
        />
        <StatCard
          title="Resolved Today"
          value={stats.resolvedToday}
          change={12}
          icon={CheckCircleIcon}
          color="green"
        />
        <StatCard
          title="Avg. Time to Resolve"
          value={formatTime(stats.averageTimeToResolve)}
          change={-8}
          icon={ClockIcon}
          color="gray"
        />
      </div>

      {/* Distribution charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* By Entity Type */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gl-neutral-900 mb-4">
            By Entity Type
          </h2>
          <DistributionBar data={entityTypeData} total={entityTypeTotal} />
        </div>

        {/* By Confidence Level */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gl-neutral-900 mb-4">
            By Confidence Level
          </h2>
          <DistributionBar data={confidenceData} total={confidenceTotal} />
        </div>
      </div>

      {/* Recent resolutions */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-lg font-semibold text-gl-neutral-900">
            Recent Resolutions
          </h2>
        </div>
        <div className="table-container">
          <table className="table">
            <thead className="table-header">
              <tr>
                <th>Reviewer</th>
                <th>Decision</th>
                <th>Notes</th>
                <th>Time</th>
                <th>Resolved</th>
              </tr>
            </thead>
            <tbody className="table-body">
              {stats.recentResolutions.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-8 text-gl-neutral-500">
                    No recent resolutions
                  </td>
                </tr>
              ) : (
                stats.recentResolutions.map((resolution) => (
                  <tr key={resolution.id}>
                    <td className="font-medium">{resolution.reviewerName}</td>
                    <td>
                      <span
                        className={clsx('badge', {
                          'badge-success': resolution.decision === 'accept',
                          'badge-primary': resolution.decision === 'select',
                          'badge-danger': resolution.decision === 'reject',
                          'badge-warning': resolution.decision === 'defer',
                          'badge-gray': resolution.decision === 'escalate',
                        })}
                      >
                        {resolution.decision}
                      </span>
                    </td>
                    <td className="max-w-xs truncate text-gl-neutral-600">
                      {resolution.reviewerNotes || '-'}
                    </td>
                    <td className="text-gl-neutral-600">
                      {formatTime(resolution.timeToResolve)}
                    </td>
                    <td className="text-gl-neutral-600">
                      {new Date(resolution.resolvedAt).toLocaleString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
