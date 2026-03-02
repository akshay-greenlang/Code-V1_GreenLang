/**
 * GL-GHG Corporate Platform - Formatting Utilities
 *
 * Provides consistent formatting for emissions values, percentages,
 * intensity metrics, dates, and other display values throughout
 * the application.
 *
 * Also exports color and label maps for scopes, gases, and categories.
 */

// ---------------------------------------------------------------------------
// Emissions formatting
// ---------------------------------------------------------------------------

/**
 * Format a number as tCO2e with automatic scale (K/M/B).
 * Examples:
 *   0.42       -> "0.4 tCO2e"
 *   1,234      -> "1.2K tCO2e"
 *   1,234,567  -> "1.2M tCO2e"
 */
export function formatEmissions(value: number, decimals = 1): string {
  if (Math.abs(value) >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(decimals)}B tCO2e`;
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(decimals)}M tCO2e`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(decimals)}K tCO2e`;
  }
  return `${value.toFixed(decimals)} tCO2e`;
}

/**
 * Format raw emissions value without unit suffix (for chart labels).
 */
export function formatEmissionsRaw(value: number, decimals = 1): string {
  if (Math.abs(value) >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(decimals)}B`;
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(decimals)}M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(decimals)}K`;
  }
  return value.toFixed(decimals);
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/**
 * Format a number with locale-aware thousands separators.
 */
export function formatNumber(value: number, decimals = 0): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

// ---------------------------------------------------------------------------
// Percentage formatting
// ---------------------------------------------------------------------------

/**
 * Format a percentage value (e.g. 85.3 -> "85.3%").
 */
export function formatPercentage(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format a signed percentage change with + or - prefix.
 */
export function formatChange(value: number, decimals = 1): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(decimals)}%`;
}

// ---------------------------------------------------------------------------
// Intensity formatting
// ---------------------------------------------------------------------------

/**
 * Format an intensity metric value with its unit.
 * Example: formatIntensity(0.42, "tCO2e/M$ revenue") -> "0.42 tCO2e/M$ revenue"
 */
export function formatIntensity(value: number, unit: string, decimals = 2): string {
  return `${value.toFixed(decimals)} ${unit}`;
}

// ---------------------------------------------------------------------------
// Date formatting
// ---------------------------------------------------------------------------

/**
 * Format a date string to a localized short date.
 */
export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Format a date string to ISO format (YYYY-MM-DD).
 */
export function formatDateISO(dateStr: string): string {
  return new Date(dateStr).toISOString().split('T')[0];
}

/**
 * Format a reporting period as "Jan 2024 - Dec 2024".
 */
export function formatReportingPeriod(startStr: string, endStr: string): string {
  const start = new Date(startStr);
  const end = new Date(endStr);
  const startLabel = start.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  const endLabel = end.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  return `${startLabel} - ${endLabel}`;
}

// ---------------------------------------------------------------------------
// File size formatting
// ---------------------------------------------------------------------------

/**
 * Format file size in human-readable units.
 */
export function formatFileSize(bytes: number): string {
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(1)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

/**
 * Truncate a string to a maximum length with ellipsis.
 */
export function truncate(str: string, maxLen = 50): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen - 3) + '...';
}

// ---------------------------------------------------------------------------
// Quality helpers
// ---------------------------------------------------------------------------

/**
 * Get a MUI color for a data quality score (0-100).
 */
export function getQualityColor(score: number): 'success' | 'warning' | 'error' {
  if (score >= 80) return 'success';
  if (score >= 60) return 'warning';
  return 'error';
}

/**
 * Get a human-readable label for a data quality score.
 */
export function getQualityLabel(score: number): string {
  if (score >= 90) return 'Excellent';
  if (score >= 80) return 'Good';
  if (score >= 60) return 'Fair';
  if (score >= 40) return 'Poor';
  return 'Very Poor';
}

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------

/** Scope color palette used across charts and badges. */
export const scopeColorMap = {
  scope_1: '#e53935',
  scope_2: '#1e88e5',
  scope_3: '#43a047',
} as const;

/** Extended scope colors including lighter variants for hover/fill. */
export const scopeColorMapExtended = {
  scope_1: { main: '#e53935', light: '#ef5350', dark: '#c62828', bg: 'rgba(229, 57, 53, 0.08)' },
  scope_2: { main: '#1e88e5', light: '#42a5f5', dark: '#1565c0', bg: 'rgba(30, 136, 229, 0.08)' },
  scope_3: { main: '#43a047', light: '#66bb6a', dark: '#2e7d32', bg: 'rgba(67, 160, 71, 0.08)' },
} as const;

// ---------------------------------------------------------------------------
// Label maps
// ---------------------------------------------------------------------------

/** Human-readable labels for the 7 GHG Protocol gases. */
export const gasLabelMap: Record<string, string> = {
  CO2: 'Carbon Dioxide (CO2)',
  CH4: 'Methane (CH4)',
  N2O: 'Nitrous Oxide (N2O)',
  HFCs: 'Hydrofluorocarbons (HFCs)',
  PFCs: 'Perfluorocarbons (PFCs)',
  SF6: 'Sulfur Hexafluoride (SF6)',
  NF3: 'Nitrogen Trifluoride (NF3)',
};

/** Short gas labels for chart axes and table headers. */
export const gasShortLabelMap: Record<string, string> = {
  CO2: 'CO2',
  CH4: 'CH4',
  N2O: 'N2O',
  HFCs: 'HFCs',
  PFCs: 'PFCs',
  SF6: 'SF6',
  NF3: 'NF3',
};

/** Gas color palette for charts. */
export const gasColorMap: Record<string, string> = {
  CO2: '#757575',
  CH4: '#1e88e5',
  N2O: '#43a047',
  HFCs: '#ef6c00',
  PFCs: '#8e24aa',
  SF6: '#e53935',
  NF3: '#00897b',
};

/** Human-readable labels for Scope 1 source categories. */
export const scope1CategoryLabelMap: Record<string, string> = {
  stationary_combustion: 'Stationary Combustion',
  mobile_combustion: 'Mobile Combustion',
  process_emissions: 'Process Emissions',
  fugitive_emissions: 'Fugitive Emissions',
};

/** Human-readable labels for Scope 3 categories (1-15). */
export const categoryLabelMap: Record<string, string> = {
  cat_1: 'Cat 1: Purchased Goods & Services',
  cat_2: 'Cat 2: Capital Goods',
  cat_3: 'Cat 3: Fuel & Energy Activities',
  cat_4: 'Cat 4: Upstream Transportation',
  cat_5: 'Cat 5: Waste Generated in Operations',
  cat_6: 'Cat 6: Business Travel',
  cat_7: 'Cat 7: Employee Commuting',
  cat_8: 'Cat 8: Upstream Leased Assets',
  cat_9: 'Cat 9: Downstream Transportation',
  cat_10: 'Cat 10: Processing of Sold Products',
  cat_11: 'Cat 11: Use of Sold Products',
  cat_12: 'Cat 12: End-of-Life Treatment',
  cat_13: 'Cat 13: Downstream Leased Assets',
  cat_14: 'Cat 14: Franchises',
  cat_15: 'Cat 15: Investments',
};

/** Short category labels for charts. */
export const categoryShortLabelMap: Record<string, string> = {
  cat_1: 'Purchased G&S',
  cat_2: 'Capital Goods',
  cat_3: 'Fuel & Energy',
  cat_4: 'Upstream Transport',
  cat_5: 'Waste Generated',
  cat_6: 'Business Travel',
  cat_7: 'Employee Commuting',
  cat_8: 'Upstream Leased',
  cat_9: 'Downstream Transport',
  cat_10: 'Processing Sold',
  cat_11: 'Use of Sold',
  cat_12: 'End-of-Life',
  cat_13: 'Downstream Leased',
  cat_14: 'Franchises',
  cat_15: 'Investments',
};

/** Data quality tier labels. */
export const tierLabelMap: Record<string, string> = {
  tier_1: 'Tier 1 - Supplier Specific',
  tier_2: 'Tier 2 - Average Data',
  tier_3: 'Tier 3 - Spend-Based',
  tier_4: 'Tier 4 - Estimated',
};

/** Verification level labels. */
export const verificationLevelLabelMap: Record<string, string> = {
  limited: 'Limited Assurance',
  reasonable: 'Reasonable Assurance',
  not_verified: 'Not Verified',
};
