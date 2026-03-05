/**
 * GL-ISO14064-APP v1.0 - Formatting Utilities
 *
 * Provides consistent number, date, and color formatting used across
 * all dashboard, table, chart, and detail components.
 */

import {
  GHGGas,
  GAS_COLORS,
  ISOCategory,
  CATEGORY_COLORS,
  FindingSeverity,
  InventoryStatus,
  VerificationStage,
  ActionStatus,
  FindingStatus,
  SignificanceLevel,
  DataQualityTier,
} from '../types';

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/**
 * Format a number with locale-aware thousands separators.
 * Defaults to 2 decimal places.
 */
export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value == null || isNaN(value)) return '--';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format a value as tonnes CO2e with appropriate scale suffix.
 * < 1,000  => "123.45 tCO2e"
 * >= 1,000 and < 1,000,000 => "12.3K tCO2e"
 * >= 1,000,000 => "1.2M tCO2e"
 */
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

/**
 * Format a percentage value. Handles null/undefined gracefully.
 */
export function formatPercentage(value: number | null | undefined, decimals = 1): string {
  if (value == null || isNaN(value)) return '--%';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

// ---------------------------------------------------------------------------
// Date formatting
// ---------------------------------------------------------------------------

/**
 * Format an ISO 8601 date string as a locale-aware short date.
 */
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

/**
 * Format an ISO 8601 date string as a locale-aware date + time.
 */
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

// ---------------------------------------------------------------------------
// Color utilities
// ---------------------------------------------------------------------------

/**
 * Get the theme color for an ISO 14064-1 category.
 */
export function getCategoryColor(category: ISOCategory): string {
  return CATEGORY_COLORS[category] || '#9e9e9e';
}

/**
 * Get the theme color for a GHG gas type.
 */
export function getGasColor(gas: GHGGas): string {
  return GAS_COLORS[gas] || '#9e9e9e';
}

/**
 * Get the MUI color token for a finding severity level.
 */
export function getSeverityColor(
  severity: FindingSeverity,
): 'error' | 'warning' | 'info' | 'success' {
  switch (severity) {
    case FindingSeverity.CRITICAL:
      return 'error';
    case FindingSeverity.HIGH:
      return 'error';
    case FindingSeverity.MEDIUM:
      return 'warning';
    case FindingSeverity.LOW:
      return 'info';
    default:
      return 'info';
  }
}

/**
 * Get the MUI color token for a generic status string.
 */
export function getStatusColor(
  status: string,
): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' {
  switch (status) {
    // Inventory / report statuses
    case InventoryStatus.DRAFT:
    case 'draft':
      return 'default';
    case InventoryStatus.IN_REVIEW:
    case 'in_review':
    case 'review':
      return 'info';
    case InventoryStatus.APPROVED:
    case 'approved':
      return 'primary';
    case InventoryStatus.VERIFIED:
    case 'verified':
      return 'success';
    case InventoryStatus.PUBLISHED:
    case 'published':
    case 'final':
      return 'success';

    // Verification stages
    case VerificationStage.INTERNAL_REVIEW:
    case 'internal_review':
      return 'info';
    case VerificationStage.EXTERNAL_VERIFICATION:
    case 'external_verification':
      return 'warning';

    // Action statuses
    case ActionStatus.PLANNED:
    case 'planned':
      return 'default';
    case ActionStatus.IN_PROGRESS:
    case 'in_progress':
      return 'info';
    case ActionStatus.COMPLETED:
    case 'completed':
      return 'success';
    case ActionStatus.DEFERRED:
    case 'deferred':
      return 'warning';
    case ActionStatus.CANCELLED:
    case 'cancelled':
      return 'error';

    // Finding statuses
    case FindingStatus.OPEN:
    case 'open':
      return 'error';
    case FindingStatus.IN_PROGRESS:
      return 'warning';
    case FindingStatus.RESOLVED:
    case 'resolved':
      return 'success';
    case FindingStatus.ACCEPTED:
    case 'accepted':
      return 'info';

    // Significance
    case SignificanceLevel.SIGNIFICANT:
    case 'significant':
      return 'error';
    case SignificanceLevel.NOT_SIGNIFICANT:
    case 'not_significant':
      return 'success';
    case SignificanceLevel.UNDER_REVIEW:
    case 'under_review':
      return 'warning';

    default:
      return 'default';
  }
}

/**
 * Get a human-readable label for a data quality tier.
 */
export function getDataQualityLabel(tier: DataQualityTier): string {
  switch (tier) {
    case DataQualityTier.TIER_1:
      return 'Tier 1 - Highest';
    case DataQualityTier.TIER_2:
      return 'Tier 2 - High';
    case DataQualityTier.TIER_3:
      return 'Tier 3 - Medium';
    case DataQualityTier.TIER_4:
      return 'Tier 4 - Low';
    default:
      return String(tier);
  }
}

/**
 * Get the MUI color for a data quality tier.
 */
export function getDataQualityColor(
  tier: DataQualityTier,
): 'success' | 'info' | 'warning' | 'error' {
  switch (tier) {
    case DataQualityTier.TIER_1:
      return 'success';
    case DataQualityTier.TIER_2:
      return 'info';
    case DataQualityTier.TIER_3:
      return 'warning';
    case DataQualityTier.TIER_4:
      return 'error';
    default:
      return 'warning';
  }
}
