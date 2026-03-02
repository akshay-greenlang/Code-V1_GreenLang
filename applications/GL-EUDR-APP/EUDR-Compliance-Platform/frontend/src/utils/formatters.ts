/**
 * GL-EUDR-APP Formatting Utilities
 *
 * Number, date, coordinate, and display formatting functions
 * plus color/label maps for statuses and commodities.
 */

import type { RiskLevel, ComplianceStatus, EUDRCommodity, DDSStatus } from '../types';

// ---------------------------------------------------------------------------
// Number Formatting
// ---------------------------------------------------------------------------

export function formatNumber(value: number, decimals = 0): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatCurrency(value: number, currency = 'EUR'): string {
  return value.toLocaleString('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function formatPercentage(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

export function formatArea(hectares: number): string {
  if (hectares >= 1000) {
    return `${(hectares / 1000).toFixed(2)} km²`;
  }
  return `${hectares.toFixed(2)} ha`;
}

// ---------------------------------------------------------------------------
// Date Formatting
// ---------------------------------------------------------------------------

export function formatDate(isoString: string | null | undefined): string {
  if (!isoString) return '-';
  const date = new Date(isoString);
  return date.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

export function formatDateTime(isoString: string | null | undefined): string {
  if (!isoString) return '-';
  const date = new Date(isoString);
  return date.toLocaleString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatRelativeTime(isoString: string): string {
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diffMs = now - then;
  const diffMinutes = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 30) return `${diffDays}d ago`;
  return formatDate(isoString);
}

// ---------------------------------------------------------------------------
// Coordinate Formatting
// ---------------------------------------------------------------------------

export function formatCoordinate(
  latitude: number,
  longitude: number
): string {
  const latDir = latitude >= 0 ? 'N' : 'S';
  const lonDir = longitude >= 0 ? 'E' : 'W';
  return `${Math.abs(latitude).toFixed(6)}° ${latDir}, ${Math.abs(longitude).toFixed(6)}° ${lonDir}`;
}

// ---------------------------------------------------------------------------
// Color Maps
// ---------------------------------------------------------------------------

export const riskColorMap: Record<RiskLevel, string> = {
  low: '#2e7d32',
  standard: '#1565c0',
  high: '#ed6c02',
  critical: '#d32f2f',
};

export const statusColorMap: Record<ComplianceStatus, string> = {
  compliant: '#2e7d32',
  non_compliant: '#d32f2f',
  pending: '#ed6c02',
  under_review: '#1565c0',
  expired: '#757575',
};

export const ddsStatusColorMap: Record<DDSStatus, string> = {
  draft: '#757575',
  pending_review: '#ed6c02',
  validated: '#1565c0',
  submitted: '#0288d1',
  accepted: '#2e7d32',
  rejected: '#d32f2f',
  amended: '#f57c00',
};

// ---------------------------------------------------------------------------
// Label Maps
// ---------------------------------------------------------------------------

export const commodityLabelMap: Record<EUDRCommodity, string> = {
  cattle: 'Cattle',
  cocoa: 'Cocoa',
  coffee: 'Coffee',
  oil_palm: 'Oil Palm',
  rubber: 'Rubber',
  soya: 'Soya',
  wood: 'Wood',
};

export const riskLabelMap: Record<RiskLevel, string> = {
  low: 'Low',
  standard: 'Standard',
  high: 'High',
  critical: 'Critical',
};

export const complianceLabelMap: Record<ComplianceStatus, string> = {
  compliant: 'Compliant',
  non_compliant: 'Non-Compliant',
  pending: 'Pending',
  under_review: 'Under Review',
  expired: 'Expired',
};
