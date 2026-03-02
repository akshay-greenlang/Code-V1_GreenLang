/**
 * GL-GHG Corporate Platform - Validation Utilities
 *
 * Form validation helpers for the GHG inventory setup wizard,
 * target configuration, emissions data entry, and other input forms.
 *
 * Each validator returns null on success or an error message string on failure.
 */

// ---------------------------------------------------------------------------
// General validators
// ---------------------------------------------------------------------------

/**
 * Validate that a string is not empty after trimming whitespace.
 */
export function isRequired(value: string): string | null {
  return value.trim().length > 0 ? null : 'This field is required';
}

/**
 * Validate minimum string length.
 */
export function minLength(value: string, min: number): string | null {
  return value.trim().length >= min ? null : `Must be at least ${min} characters`;
}

/**
 * Validate maximum string length.
 */
export function maxLength(value: string, max: number): string | null {
  return value.trim().length <= max ? null : `Must be at most ${max} characters`;
}

// ---------------------------------------------------------------------------
// Year validators
// ---------------------------------------------------------------------------

/**
 * Validate a year is within a reasonable range for GHG reporting (1990 - current+1).
 */
export function validateYear(year: number): string | null {
  const currentYear = new Date().getFullYear();
  if (!Number.isFinite(year) || !Number.isInteger(year)) {
    return 'Year must be a whole number';
  }
  if (year < 1990) return 'Year must be 1990 or later';
  if (year > currentYear + 1) return `Year cannot be after ${currentYear + 1}`;
  return null;
}

/**
 * Validate that target year is after base year.
 */
export function isTargetAfterBase(targetYear: number, baseYear: number): string | null {
  if (targetYear <= baseYear) return 'Target year must be after base year';
  return null;
}

// ---------------------------------------------------------------------------
// Percentage validators
// ---------------------------------------------------------------------------

/**
 * Validate a percentage is between 0 and 100.
 */
export function validatePercentage(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  if (value < 0 || value > 100) return 'Must be between 0% and 100%';
  return null;
}

/**
 * Validate an ownership percentage (> 0 and <= 100).
 */
export function isValidOwnership(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  if (value <= 0) return 'Ownership must be greater than 0%';
  if (value > 100) return 'Ownership cannot exceed 100%';
  return null;
}

/**
 * Validate a reduction target percentage (1-100).
 */
export function isValidReduction(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  if (value < 1) return 'Reduction must be at least 1%';
  if (value > 100) return 'Reduction cannot exceed 100%';
  return null;
}

// ---------------------------------------------------------------------------
// Emissions validators
// ---------------------------------------------------------------------------

/**
 * Validate an emissions value (must be a finite non-negative number).
 */
export function validateEmissions(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  if (value < 0) return 'Emissions value cannot be negative';
  if (value > 1_000_000_000_000) return 'Value exceeds maximum allowed (1 trillion tCO2e)';
  return null;
}

/**
 * Validate a positive number.
 */
export function isPositive(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  return value > 0 ? null : 'Value must be positive';
}

/**
 * Validate a non-negative number.
 */
export function isNonNegative(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  return value >= 0 ? null : 'Value cannot be negative';
}

// ---------------------------------------------------------------------------
// Threshold validators
// ---------------------------------------------------------------------------

/**
 * Validate a significance threshold for base year recalculation (1-10%).
 */
export function isValidThreshold(value: number): string | null {
  if (!Number.isFinite(value)) return 'Must be a valid number';
  if (value < 1 || value > 10) return 'Threshold must be between 1% and 10%';
  return null;
}

// ---------------------------------------------------------------------------
// Collection validators
// ---------------------------------------------------------------------------

/**
 * Validate that at least one scope is selected.
 */
export function hasSelectedScopes(scopes: string[]): string | null {
  return scopes.length > 0 ? null : 'At least one scope must be selected';
}

/**
 * Validate that at least one report section is selected.
 */
export function hasSelectedSections(sections: string[]): string | null {
  return sections.length > 0 ? null : 'At least one section must be selected';
}

/**
 * Validate that at least one entity is included in the boundary.
 */
export function hasIncludedEntities(entityCount: number): string | null {
  return entityCount > 0 ? null : 'At least one entity must be included in the boundary';
}

// ---------------------------------------------------------------------------
// Compound validators
// ---------------------------------------------------------------------------

/**
 * Run multiple validators on a single value and return the first error.
 */
export function composeValidators(
  ...validators: Array<(value: unknown) => string | null>
): (value: unknown) => string | null {
  return (value: unknown) => {
    for (const validator of validators) {
      const error = validator(value);
      if (error) return error;
    }
    return null;
  };
}
