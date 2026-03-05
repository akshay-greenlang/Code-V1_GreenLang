/**
 * GL-CDP-APP v1.0 - Validation Utilities
 *
 * Validation functions for CDP disclosure compliance.
 * Each returns { valid: boolean, errors: string[] }.
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/** Response text must be non-empty for required questions. */
export function validateResponseText(text: string, isRequired: boolean): ValidationResult {
  const errors: string[] = [];
  if (isRequired && !text?.trim()) {
    errors.push('Response is required for this question.');
  }
  if (text && text.length > 50000) {
    errors.push('Response text exceeds maximum length of 50,000 characters.');
  }
  return { valid: errors.length === 0, errors };
}

/** Numeric response must be a valid number. */
export function validateNumericResponse(value: number | null | undefined, isRequired: boolean): ValidationResult {
  const errors: string[] = [];
  if (isRequired && (value == null || isNaN(value))) {
    errors.push('A numeric value is required.');
  }
  if (value != null && value < 0) {
    errors.push('Value must be non-negative.');
  }
  return { valid: errors.length === 0, errors };
}

/** Percentage must be between 0 and 100. */
export function validatePercentage(value: number | null | undefined, isRequired: boolean): ValidationResult {
  const errors: string[] = [];
  if (isRequired && (value == null || isNaN(value))) {
    errors.push('A percentage value is required.');
  }
  if (value != null && (value < 0 || value > 100)) {
    errors.push('Percentage must be between 0 and 100.');
  }
  return { valid: errors.length === 0, errors };
}

/** Table data must have at least one row for required questions. */
export function validateTableData(
  rows: Record<string, unknown>[] | null | undefined,
  isRequired: boolean,
): ValidationResult {
  const errors: string[] = [];
  if (isRequired && (!rows || rows.length === 0)) {
    errors.push('At least one table row is required.');
  }
  return { valid: errors.length === 0, errors };
}

/** Email must match standard format. */
export function validateEmail(email: string): ValidationResult {
  const errors: string[] = [];
  if (!email?.trim()) errors.push('Email is required.');
  else if (!EMAIL_REGEX.test(email)) errors.push('Email format is invalid.');
  return { valid: errors.length === 0, errors };
}

/** Organization name must be non-empty. */
export function validateOrganizationName(name: string): ValidationResult {
  const errors: string[] = [];
  if (!name?.trim()) errors.push('Organization name is required.');
  else if (name.length > 500) errors.push('Organization name cannot exceed 500 characters.');
  return { valid: errors.length === 0, errors };
}

/** Reporting year must be valid. */
export function validateReportingYear(year: number): ValidationResult {
  const errors: string[] = [];
  const currentYear = new Date().getFullYear();
  if (!Number.isInteger(year)) errors.push('Reporting year must be an integer.');
  else if (year < 2015) errors.push('Reporting year cannot be before 2015.');
  else if (year > currentYear + 1) errors.push('Reporting year cannot be more than 1 year in the future.');
  return { valid: errors.length === 0, errors };
}

/** Transition plan target year must be in the future. */
export function validateTargetYear(year: number): ValidationResult {
  const errors: string[] = [];
  const currentYear = new Date().getFullYear();
  if (!Number.isInteger(year)) errors.push('Target year must be an integer.');
  else if (year <= currentYear) errors.push('Target year must be in the future.');
  else if (year > 2100) errors.push('Target year cannot exceed 2100.');
  return { valid: errors.length === 0, errors };
}

/** Emissions reduction target must be 0-100%. */
export function validateReductionTarget(pct: number): ValidationResult {
  const errors: string[] = [];
  if (pct == null || isNaN(pct)) errors.push('Reduction target is required.');
  else if (pct < 0 || pct > 100) errors.push('Reduction target must be between 0 and 100%.');
  return { valid: errors.length === 0, errors };
}

/** Supplier invitation must have required fields. */
export function validateSupplierInvitation(supplier: {
  supplier_name?: string;
  supplier_email?: string;
}): ValidationResult {
  const errors: string[] = [];
  if (!supplier.supplier_name?.trim()) errors.push('Supplier name is required.');
  if (!supplier.supplier_email?.trim()) errors.push('Supplier email is required.');
  else if (!EMAIL_REGEX.test(supplier.supplier_email)) errors.push('Supplier email is invalid.');
  return { valid: errors.length === 0, errors };
}

/** Module completeness check: all required questions must be answered. */
export function validateModuleCompleteness(
  totalRequired: number,
  answeredRequired: number,
): ValidationResult {
  const errors: string[] = [];
  if (answeredRequired < totalRequired) {
    errors.push(
      `${totalRequired - answeredRequired} required questions remain unanswered.`,
    );
  }
  return { valid: errors.length === 0, errors };
}
