/**
 * CDP Integration UI Components
 *
 * Exports all CDP-related components for questionnaire management,
 * progress tracking, data mapping, and compliance scorecard.
 */

export { default as CDPQuestionnaireEditor } from './CDPQuestionnaireEditor';
export { default as CDPProgressTracker } from './CDPProgressTracker';
export { default as CDPDataMapping } from './CDPDataMapping';
export { default as ComplianceScorecard } from './ComplianceScorecard';

// Re-export types from slice for component consumers
export type {
  CDPQuestion,
  CDPSection,
  CDPQuestionnaire,
  CDPProgress,
  CDPDataMappingItem,
  CDPValidation,
} from '../../store/slices/cdpSlice';

export type {
  ComplianceStandard,
  ComplianceRequirement,
  ComplianceScorecard as ComplianceScorecardType,
  ComplianceGap,
  ActionItem,
} from '../../store/slices/complianceSlice';
