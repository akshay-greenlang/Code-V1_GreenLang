/**
 * GL-CDP-APP v1.0 - Formatting Utilities
 *
 * Provides consistent number, date, score, and color formatting
 * used across all dashboard, table, chart, and detail components.
 */

import {
  ScoringLevel,
  ScoringBand,
  SCORING_LEVEL_COLORS,
  SCORING_BAND_COLORS,
  GapSeverity,
  ResponseStatus,
  SupplierStatus,
  TransitionMilestoneStatus,
} from '../types';

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null || isNaN(value)) return '--';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatTCO2e(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '-- tCO2e';
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_000_000) {
    return `${sign}${(abs / 1_000_000).toFixed(1)}M tCO2e`;
  }
  if (abs >= 1_000) {
    return `${sign}${(abs / 1_000).toFixed(1)}K tCO2e`;
  }
  return `${sign}${abs.toFixed(2)} tCO2e`;
}

export function formatPercentage(value: number | null | undefined, decimals = 1): string {
  if (value == null || isNaN(value)) return '--%';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

export function formatCompactNumber(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '--';
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(0);
}

export function formatCurrency(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '--';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

// ---------------------------------------------------------------------------
// Date formatting
// ---------------------------------------------------------------------------

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return '--';
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

export function formatDateTime(dateStr: string | null | undefined): string {
  if (!dateStr) return '--';
  try {
    return new Date(dateStr).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

export function formatDaysRemaining(deadline: string | null | undefined): string {
  if (!deadline) return '--';
  const now = new Date();
  const target = new Date(deadline);
  const diff = Math.ceil((target.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
  if (diff < 0) return `${Math.abs(diff)} days overdue`;
  if (diff === 0) return 'Due today';
  if (diff === 1) return '1 day remaining';
  return `${diff} days remaining`;
}

// ---------------------------------------------------------------------------
// Score formatting
// ---------------------------------------------------------------------------

export function formatScore(score: number | null | undefined): string {
  if (score == null || isNaN(score)) return '--%';
  return `${score.toFixed(1)}%`;
}

export function getLevelFromScore(score: number): ScoringLevel {
  if (score >= 80) return ScoringLevel.A;
  if (score >= 70) return ScoringLevel.A_MINUS;
  if (score >= 60) return ScoringLevel.B;
  if (score >= 50) return ScoringLevel.B_MINUS;
  if (score >= 40) return ScoringLevel.C;
  if (score >= 30) return ScoringLevel.C_MINUS;
  if (score >= 20) return ScoringLevel.D;
  return ScoringLevel.D_MINUS;
}

export function getBandFromLevel(level: ScoringLevel): ScoringBand {
  switch (level) {
    case ScoringLevel.A:
    case ScoringLevel.A_MINUS:
      return ScoringBand.LEADERSHIP;
    case ScoringLevel.B:
    case ScoringLevel.B_MINUS:
      return ScoringBand.MANAGEMENT;
    case ScoringLevel.C:
    case ScoringLevel.C_MINUS:
      return ScoringBand.AWARENESS;
    case ScoringLevel.D:
    case ScoringLevel.D_MINUS:
      return ScoringBand.DISCLOSURE;
  }
}

export function getScoringLevelColor(level: ScoringLevel): string {
  return SCORING_LEVEL_COLORS[level] || '#9e9e9e';
}

export function getScoringBandColor(band: ScoringBand): string {
  return SCORING_BAND_COLORS[band] || '#9e9e9e';
}

// ---------------------------------------------------------------------------
// Status colors
// ---------------------------------------------------------------------------

export function getStatusColor(
  status: string,
): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' {
  switch (status) {
    case ResponseStatus.NOT_STARTED:
    case 'not_started':
      return 'default';
    case ResponseStatus.DRAFT:
    case 'draft':
      return 'warning';
    case ResponseStatus.IN_REVIEW:
    case 'in_review':
      return 'info';
    case ResponseStatus.APPROVED:
    case 'approved':
      return 'success';
    case ResponseStatus.SUBMITTED:
    case 'submitted':
      return 'primary';
    case SupplierStatus.INVITED:
    case 'invited':
      return 'info';
    case SupplierStatus.IN_PROGRESS:
      return 'warning';
    case SupplierStatus.SUBMITTED:
      return 'primary';
    case SupplierStatus.SCORED:
    case 'scored':
      return 'success';
    case SupplierStatus.DECLINED:
    case 'declined':
      return 'error';
    case TransitionMilestoneStatus.COMPLETED:
    case 'completed':
      return 'success';
    case TransitionMilestoneStatus.IN_PROGRESS:
      return 'info';
    case TransitionMilestoneStatus.DELAYED:
    case 'delayed':
      return 'error';
    default:
      return 'default';
  }
}

export function getSeverityColor(
  severity: GapSeverity | string,
): 'error' | 'warning' | 'info' | 'success' {
  switch (severity) {
    case GapSeverity.CRITICAL:
    case 'critical':
      return 'error';
    case GapSeverity.HIGH:
    case 'high':
      return 'error';
    case GapSeverity.MEDIUM:
    case 'medium':
      return 'warning';
    case GapSeverity.LOW:
    case 'low':
      return 'info';
    default:
      return 'info';
  }
}
