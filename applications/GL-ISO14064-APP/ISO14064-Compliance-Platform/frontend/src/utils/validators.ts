/**
 * GL-ISO14064-APP v1.0 - Validation Utilities
 *
 * Validation functions for ISO 14064-1:2018 compliance.
 * Each returns { valid: boolean, errors: string[] }.
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Reporting period must be at least 12 months per Clause 5.3. */
export function validateReportingPeriod(start: Date, end: Date): ValidationResult {
  const errors: string[] = [];
  if (!(start instanceof Date) || isNaN(start.getTime())) {
    errors.push('Start date is invalid.');
  }
  if (!(end instanceof Date) || isNaN(end.getTime())) {
    errors.push('End date is invalid.');
  }
  if (errors.length > 0) return { valid: false, errors };
  if (end <= start) {
    errors.push('End date must be after start date.');
  }
  const monthsDiff =
    (end.getFullYear() - start.getFullYear()) * 12 +
    (end.getMonth() - start.getMonth());
  if (monthsDiff < 12) {
    errors.push(
      `Reporting period must be at least 12 months per ISO 14064-1 Clause 5.3 (currently ${monthsDiff} months).`,
    );
  }
  return { valid: errors.length === 0, errors };
}

/** Emission source must have required fields. */
export function validateEmissionSource(source: {
  source_name?: string;
  category?: string;
  gas?: string;
  activity_data?: number;
  emission_factor?: number;
}): ValidationResult {
  const errors: string[] = [];
  if (!source.source_name?.trim()) errors.push('Source name is required.');
  if (!source.category?.trim()) errors.push('ISO category is required.');
  if (!source.gas?.trim()) errors.push('GHG gas type is required.');
  if (source.activity_data == null || source.activity_data < 0)
    errors.push('Activity data must be a non-negative number.');
  if (source.emission_factor == null || source.emission_factor < 0)
    errors.push('Emission factor must be a non-negative number.');
  return { valid: errors.length === 0, errors };
}

/** Removal source must have permanence assessment. */
export function validateRemovalSource(removal: {
  removal_type?: string;
  quantity_tco2e?: number;
  permanence_level?: string;
}): ValidationResult {
  const errors: string[] = [];
  if (!removal.removal_type?.trim()) errors.push('Removal type is required.');
  if (removal.quantity_tco2e == null || removal.quantity_tco2e < 0)
    errors.push('Removal quantity must be a non-negative number.');
  if (!removal.permanence_level?.trim())
    errors.push('Permanence assessment is required for all removal sources.');
  return { valid: errors.length === 0, errors };
}

/** Significance threshold must be positive. */
export function validateSignificanceThreshold(threshold: number): ValidationResult {
  const errors: string[] = [];
  if (threshold == null || threshold <= 0)
    errors.push('Significance threshold must be greater than 0.');
  if (threshold > 100)
    errors.push('Significance threshold cannot exceed 100%.');
  return { valid: errors.length === 0, errors };
}

/** Boundary must have all required fields. */
export function validateBoundaryCompleteness(boundary: {
  consolidation_approach?: string;
  included_entities?: string[];
  included_categories?: string[];
  reporting_period_start?: string;
  reporting_period_end?: string;
}): ValidationResult {
  const errors: string[] = [];
  if (!boundary.consolidation_approach?.trim())
    errors.push('Consolidation approach is required.');
  if (!boundary.included_entities?.length)
    errors.push('At least one entity must be included.');
  if (!boundary.included_categories?.length)
    errors.push('At least one ISO category must be included.');
  if (!boundary.reporting_period_start)
    errors.push('Reporting period start date is required.');
  if (!boundary.reporting_period_end)
    errors.push('Reporting period end date is required.');
  return { valid: errors.length === 0, errors };
}

/** Uncertainty range lower bound must be less than upper. */
export function validateUncertaintyRange(lower: number, upper: number): ValidationResult {
  const errors: string[] = [];
  if (lower == null || upper == null) {
    errors.push('Both lower and upper bounds are required.');
  } else if (lower >= upper) {
    errors.push('Lower bound must be less than upper bound.');
  }
  return { valid: errors.length === 0, errors };
}

/** GWP value must be positive. */
export function validateGWPValue(value: number): ValidationResult {
  const errors: string[] = [];
  if (value == null || value <= 0)
    errors.push('GWP value must be a positive number.');
  return { valid: errors.length === 0, errors };
}

/** Data quality score must be 0-100. */
export function validateDataQualityScore(score: number): ValidationResult {
  const errors: string[] = [];
  if (score == null || score < 0 || score > 100)
    errors.push('Data quality score must be between 0 and 100.');
  return { valid: errors.length === 0, errors };
}

/** Base year must be <= current year. */
export function validateBaseYear(year: number): ValidationResult {
  const errors: string[] = [];
  const currentYear = new Date().getFullYear();
  if (!Number.isInteger(year)) errors.push('Base year must be an integer.');
  else if (year < 1990) errors.push('Base year cannot be before 1990.');
  else if (year > currentYear)
    errors.push(`Base year cannot be in the future (current year: ${currentYear}).`);
  return { valid: errors.length === 0, errors };
}

/** Email must match standard format. */
export function validateEmail(email: string): ValidationResult {
  const errors: string[] = [];
  if (!email?.trim()) errors.push('Email is required.');
  else if (!EMAIL_REGEX.test(email)) errors.push('Email format is invalid.');
  return { valid: errors.length === 0, errors };
}

/** Organization name must be non-empty and within length limits. */
export function validateOrganizationName(name: string): ValidationResult {
  const errors: string[] = [];
  if (!name?.trim()) errors.push('Organization name is required.');
  else if (name.length > 500)
    errors.push('Organization name cannot exceed 500 characters.');
  return { valid: errors.length === 0, errors };
}

/** Check if string is a valid UUID. */
export function isValidUUID(id: string): ValidationResult {
  const errors: string[] = [];
  if (!id?.trim()) errors.push('ID is required.');
  else if (!UUID_REGEX.test(id)) errors.push('ID is not a valid UUID format.');
  return { valid: errors.length === 0, errors };
}
