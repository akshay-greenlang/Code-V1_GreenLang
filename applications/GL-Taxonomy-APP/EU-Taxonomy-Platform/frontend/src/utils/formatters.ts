/**
 * Formatting utilities for the EU Taxonomy Platform.
 */

export const percentFormat = (value: number, decimals = 1): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

export const percentFromValue = (value: number, decimals = 1): string => {
  return `${value.toFixed(decimals)}%`;
};

export const currencyFormat = (
  value: number,
  currency = 'EUR',
  locale = 'en-EU'
): string => {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(value);
};

export const currencyFormatDetailed = (
  value: number,
  currency = 'EUR',
  locale = 'en-EU'
): string => {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    maximumFractionDigits: 2,
  }).format(value);
};

export const numberFormat = (value: number, decimals = 0): string => {
  return new Intl.NumberFormat('en-EU', {
    maximumFractionDigits: decimals,
  }).format(value);
};

export const compactNumber = (value: number): string => {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toFixed(0);
};

export const dateFormat = (dateStr: string): string => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
};

export const dateTimeFormat = (dateStr: string): string => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const kpiFormat = (eligible: number, aligned: number): string => {
  return `${percentFromValue(eligible)} eligible / ${percentFromValue(aligned)} aligned`;
};

export const garFormat = (ratio: number): string => {
  return `GAR: ${percentFromValue(ratio)}`;
};

export const gradeColor = (grade: string): string => {
  switch (grade) {
    case 'A': return '#2E7D32';
    case 'B': return '#558B2F';
    case 'C': return '#EF6C00';
    case 'D': return '#E65100';
    case 'F': return '#C62828';
    default: return '#757575';
  }
};

export const severityColor = (severity: string): string => {
  switch (severity) {
    case 'critical': return '#C62828';
    case 'high': return '#E65100';
    case 'medium': return '#EF6C00';
    case 'low': return '#2E7D32';
    default: return '#757575';
  }
};

export const alignmentStatusColor = (status: string): string => {
  switch (status) {
    case 'aligned': return '#2E7D32';
    case 'ms_pass': return '#558B2F';
    case 'dnsh_pass': return '#689F38';
    case 'sc_pass': return '#7CB342';
    case 'eligible': return '#0277BD';
    case 'not_eligible': return '#9E9E9E';
    case 'not_aligned': return '#C62828';
    case 'not_started': return '#BDBDBD';
    default: return '#757575';
  }
};

export const truncateText = (text: string, maxLength = 50): string => {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength)}...`;
};
