/**
 * API Types for GreenLang Review Console
 *
 * Type definitions for entity resolution review queue data structures.
 */

// Entity types supported by the normalizer
export type EntityType =
  | 'company'
  | 'product'
  | 'facility'
  | 'material'
  | 'country'
  | 'emission_factor'
  | 'regulation';

// Resolution decision types
export type ResolutionDecision =
  | 'accept'      // Accept the top candidate match
  | 'select'      // Select a different candidate
  | 'reject'      // Reject all candidates, mark as no match
  | 'defer'       // Defer for later review
  | 'escalate';   // Escalate to senior reviewer

// Queue item status
export type QueueItemStatus =
  | 'pending'
  | 'in_review'
  | 'resolved'
  | 'deferred'
  | 'escalated';

// Confidence level thresholds
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'very_low';

/**
 * Candidate match from the normalizer
 */
export interface CandidateMatch {
  id: string;
  canonicalId: string;
  canonicalName: string;
  confidence: number;           // 0-100
  matchMethod: string;          // e.g., 'exact', 'fuzzy', 'semantic', 'alias'
  matchDetails: {
    score: number;
    factors: MatchFactor[];
  };
  metadata: Record<string, unknown>;
}

export interface MatchFactor {
  name: string;
  weight: number;
  score: number;
  description: string;
}

/**
 * Original input that needs resolution
 */
export interface OriginalInput {
  rawValue: string;
  normalizedValue: string;
  source: string;
  sourceRowId?: string;
  context: Record<string, string>;
  timestamp: string;
}

/**
 * Queue item representing an entity needing human review
 */
export interface QueueItem {
  id: string;
  entityType: EntityType;
  status: QueueItemStatus;
  priority: number;             // 1-5, higher is more urgent
  originalInput: OriginalInput;
  candidates: CandidateMatch[];
  topConfidence: number;
  confidenceMargin: number;     // Difference between top two candidates
  reason: string;               // Why flagged for review
  createdAt: string;
  updatedAt: string;
  assignedTo?: string;
  resolution?: Resolution;
}

/**
 * Resolution decision made by a reviewer
 */
export interface Resolution {
  id: string;
  queueItemId: string;
  decision: ResolutionDecision;
  selectedCandidateId?: string;
  reviewerNotes: string;
  reviewerId: string;
  reviewerName: string;
  resolvedAt: string;
  timeToResolve: number;        // Seconds from assignment to resolution
}

/**
 * Dashboard statistics
 */
export interface DashboardStats {
  pending: number;
  inReview: number;
  resolvedToday: number;
  resolvedThisWeek: number;
  averageTimeToResolve: number; // Seconds
  byEntityType: Record<EntityType, number>;
  byConfidenceLevel: Record<ConfidenceLevel, number>;
  recentResolutions: Resolution[];
  trendData: TrendDataPoint[];
}

export interface TrendDataPoint {
  date: string;
  pending: number;
  resolved: number;
}

/**
 * Filter parameters for queue list
 */
export interface QueueFilters {
  entityType?: EntityType;
  status?: QueueItemStatus;
  minConfidence?: number;
  maxConfidence?: number;
  dateFrom?: string;
  dateTo?: string;
  search?: string;
  assignedTo?: string;
  priority?: number;
}

/**
 * Pagination parameters
 */
export interface PaginationParams {
  page: number;
  perPage: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    page: number;
    perPage: number;
    totalItems: number;
    totalPages: number;
    hasMore: boolean;
  };
}

/**
 * Resolution submission payload
 */
export interface ResolutionSubmission {
  decision: ResolutionDecision;
  selectedCandidateId?: string;
  reviewerNotes: string;
}

/**
 * API error response
 */
export interface APIError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * User/Reviewer info
 */
export interface User {
  id: string;
  name: string;
  email: string;
  role: 'reviewer' | 'senior_reviewer' | 'admin';
  avatarUrl?: string;
}

/**
 * Keyboard shortcut definition
 */
export interface KeyboardShortcut {
  key: string;
  modifiers?: ('ctrl' | 'shift' | 'alt' | 'meta')[];
  description: string;
  action: string;
}
