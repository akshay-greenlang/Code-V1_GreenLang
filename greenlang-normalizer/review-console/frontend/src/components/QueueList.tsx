/**
 * QueueList Component
 *
 * Paginated list of queue items with filtering, sorting, and search.
 */

import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MagnifyingGlassIcon,
  FunnelIcon,
  ArrowPathIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { useQueueList } from '../hooks/useQueue';
import { QueueItem } from './QueueItem';
import type { EntityType, QueueItemStatus } from '../api/types';

// Filter options
const entityTypeOptions: Array<{ value: EntityType | ''; label: string }> = [
  { value: '', label: 'All Types' },
  { value: 'company', label: 'Companies' },
  { value: 'product', label: 'Products' },
  { value: 'facility', label: 'Facilities' },
  { value: 'material', label: 'Materials' },
  { value: 'country', label: 'Countries' },
  { value: 'emission_factor', label: 'Emission Factors' },
  { value: 'regulation', label: 'Regulations' },
];

const statusOptions: Array<{ value: QueueItemStatus | ''; label: string }> = [
  { value: '', label: 'All Statuses' },
  { value: 'pending', label: 'Pending' },
  { value: 'in_review', label: 'In Review' },
  { value: 'deferred', label: 'Deferred' },
  { value: 'escalated', label: 'Escalated' },
];

const perPageOptions = [10, 25, 50, 100];

/**
 * Filter panel component
 */
interface FilterPanelProps {
  filters: {
    entityType?: EntityType;
    status?: QueueItemStatus;
    dateFrom?: string;
    dateTo?: string;
    search?: string;
  };
  onFilterChange: (filters: Partial<typeof filters>) => void;
  onClear: () => void;
  isOpen: boolean;
  onClose: () => void;
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  onFilterChange,
  onClear,
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  const hasActiveFilters = Object.values(filters).some((v) => v && v !== '');

  return (
    <div className="card p-4 mb-4 animate-in">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gl-neutral-900">Filters</h3>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <button
              onClick={onClear}
              className="text-sm text-gl-primary-600 hover:text-gl-primary-700"
            >
              Clear all
            </button>
          )}
          <button
            onClick={onClose}
            className="p-1 text-gl-neutral-400 hover:text-gl-neutral-600"
            aria-label="Close filters"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Entity Type */}
        <div>
          <label htmlFor="filter-entity-type" className="label">
            Entity Type
          </label>
          <select
            id="filter-entity-type"
            className="select"
            value={filters.entityType || ''}
            onChange={(e) =>
              onFilterChange({
                entityType: e.target.value as EntityType | undefined,
              })
            }
          >
            {entityTypeOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Status */}
        <div>
          <label htmlFor="filter-status" className="label">
            Status
          </label>
          <select
            id="filter-status"
            className="select"
            value={filters.status || ''}
            onChange={(e) =>
              onFilterChange({
                status: e.target.value as QueueItemStatus | undefined,
              })
            }
          >
            {statusOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Date From */}
        <div>
          <label htmlFor="filter-date-from" className="label">
            From Date
          </label>
          <input
            type="date"
            id="filter-date-from"
            className="input"
            value={filters.dateFrom || ''}
            onChange={(e) => onFilterChange({ dateFrom: e.target.value || undefined })}
          />
        </div>

        {/* Date To */}
        <div>
          <label htmlFor="filter-date-to" className="label">
            To Date
          </label>
          <input
            type="date"
            id="filter-date-to"
            className="input"
            value={filters.dateTo || ''}
            onChange={(e) => onFilterChange({ dateTo: e.target.value || undefined })}
          />
        </div>
      </div>
    </div>
  );
};

/**
 * Empty state component
 */
const EmptyState: React.FC<{ hasFilters: boolean }> = ({ hasFilters }) => (
  <div className="empty-state py-16">
    <MagnifyingGlassIcon className="empty-state-icon" />
    <h3 className="empty-state-title">
      {hasFilters ? 'No items match your filters' : 'Queue is empty'}
    </h3>
    <p className="empty-state-description">
      {hasFilters
        ? 'Try adjusting your filters or search query'
        : 'All items have been reviewed. Great work!'}
    </p>
  </div>
);

/**
 * QueueList component
 */
export const QueueList: React.FC = () => {
  const navigate = useNavigate();
  const [showFilters, setShowFilters] = useState(false);
  const [searchValue, setSearchValue] = useState('');

  const {
    items,
    pagination,
    filters,
    isLoading,
    isFetching,
    updateFilters,
    clearFilters,
    setPage,
    setPerPage,
    refresh,
  } = useQueueList();

  // Handle search submit
  const handleSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      updateFilters({ search: searchValue || undefined });
    },
    [searchValue, updateFilters]
  );

  // Handle item click
  const handleItemClick = useCallback(
    (id: string) => {
      navigate(`/queue/${id}`);
    },
    [navigate]
  );

  // Check if filters are active
  const hasActiveFilters = Object.values(filters).some((v) => v && v !== '');

  return (
    <div className="p-6 space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gl-neutral-900">Review Queue</h1>
          <p className="text-sm text-gl-neutral-500 mt-1">
            {pagination.totalItems} items pending review
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={refresh}
            disabled={isFetching}
            className="btn-secondary"
            aria-label="Refresh list"
          >
            <ArrowPathIcon
              className={clsx('w-5 h-5', isFetching && 'animate-spin')}
            />
          </button>
        </div>
      </div>

      {/* Search and filter bar */}
      <div className="flex flex-col sm:flex-row gap-4">
        <form onSubmit={handleSearch} className="flex-1 flex gap-2">
          <div className="relative flex-1">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gl-neutral-400" />
            <input
              type="text"
              placeholder="Search by name, ID, or source..."
              className="input pl-10"
              value={searchValue}
              onChange={(e) => setSearchValue(e.target.value)}
            />
          </div>
          <button type="submit" className="btn-primary">
            Search
          </button>
        </form>

        <button
          onClick={() => setShowFilters(!showFilters)}
          className={clsx(
            'btn-secondary',
            hasActiveFilters && 'border-gl-primary-500 text-gl-primary-600'
          )}
        >
          <FunnelIcon className="w-5 h-5" />
          Filters
          {hasActiveFilters && (
            <span className="ml-1 px-1.5 py-0.5 text-xs bg-gl-primary-500 text-white rounded-full">
              {Object.values(filters).filter((v) => v && v !== '').length}
            </span>
          )}
        </button>
      </div>

      {/* Filter panel */}
      <FilterPanel
        filters={filters}
        onFilterChange={updateFilters}
        onClear={clearFilters}
        isOpen={showFilters}
        onClose={() => setShowFilters(false)}
      />

      {/* Items list */}
      {isLoading ? (
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="card p-4">
              <div className="flex gap-4">
                <div className="skeleton w-16 h-16 rounded" />
                <div className="flex-1 space-y-2">
                  <div className="skeleton h-5 w-48" />
                  <div className="skeleton h-4 w-32" />
                  <div className="skeleton h-4 w-64" />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        <EmptyState hasFilters={hasActiveFilters} />
      ) : (
        <div className="space-y-3">
          {items.map((item) => (
            <QueueItem
              key={item.id}
              item={item}
              onClick={() => handleItemClick(item.id)}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {pagination.totalPages > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-gl-neutral-200">
          <div className="flex items-center gap-2">
            <span className="text-sm text-gl-neutral-600">Show</span>
            <select
              className="select w-20"
              value={pagination.perPage}
              onChange={(e) => setPerPage(Number(e.target.value))}
            >
              {perPageOptions.map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
            <span className="text-sm text-gl-neutral-600">per page</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gl-neutral-600">
              Page {pagination.page} of {pagination.totalPages}
            </span>

            <div className="flex gap-1">
              <button
                onClick={() => setPage(pagination.page - 1)}
                disabled={pagination.page <= 1}
                className="btn-secondary btn-sm"
                aria-label="Previous page"
              >
                <ChevronLeftIcon className="w-4 h-4" />
              </button>

              {/* Page numbers */}
              {Array.from({ length: Math.min(5, pagination.totalPages) }, (_, i) => {
                let pageNum: number;
                if (pagination.totalPages <= 5) {
                  pageNum = i + 1;
                } else if (pagination.page <= 3) {
                  pageNum = i + 1;
                } else if (pagination.page >= pagination.totalPages - 2) {
                  pageNum = pagination.totalPages - 4 + i;
                } else {
                  pageNum = pagination.page - 2 + i;
                }

                return (
                  <button
                    key={pageNum}
                    onClick={() => setPage(pageNum)}
                    className={clsx(
                      'btn-sm min-w-[32px]',
                      pageNum === pagination.page ? 'btn-primary' : 'btn-secondary'
                    )}
                  >
                    {pageNum}
                  </button>
                );
              })}

              <button
                onClick={() => setPage(pagination.page + 1)}
                disabled={!pagination.hasMore}
                className="btn-secondary btn-sm"
                aria-label="Next page"
              >
                <ChevronRightIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QueueList;
