/**
 * APP-003 GL-VCCI-APP v1.1 -- CDP & Compliance Component Tests
 *
 * Comprehensive tests for the 4 CDP components (CDPQuestionnaireEditor,
 * CDPProgressTracker, CDPDataMapping, ComplianceScorecard) and 2 pages
 * (CDPManagement, ComplianceDashboard).
 *
 * Target: 30+ tests, ~400 lines
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { BrowserRouter } from 'react-router-dom';

// Components under test
import CDPQuestionnaireEditor from '../../components/cdp/CDPQuestionnaireEditor';
import CDPProgressTracker from '../../components/cdp/CDPProgressTracker';
import CDPDataMapping from '../../components/cdp/CDPDataMapping';
import ComplianceScorecard from '../../components/cdp/ComplianceScorecard';
import CDPManagement from '../../pages/CDPManagement';
import ComplianceDashboard from '../../pages/ComplianceDashboard';

// Slices
import cdpReducer from '../../store/slices/cdpSlice';
import complianceReducer from '../../store/slices/complianceSlice';

import type { CDPQuestionnaire, CDPProgress, CDPDataMappingItem } from '../../store/slices/cdpSlice';
import type { ComplianceStandard } from '../../store/slices/complianceSlice';

// ============================================================================
// Mocks
// ============================================================================

vi.mock('recharts', async () => {
  const OriginalModule = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: any) => (
      <div data-testid="responsive-container" style={{ width: 800, height: 400 }}>
        {children}
      </div>
    ),
  };
});

vi.mock('../../services/api', () => ({
  default: {
    getCDPQuestionnaire: vi.fn().mockResolvedValue(null),
    autoPopulateCDP: vi.fn().mockResolvedValue(null),
    saveCDPDraft: vi.fn().mockResolvedValue({}),
    validateCDP: vi.fn().mockResolvedValue({ isValid: true, totalErrors: 0, totalWarnings: 0, sectionResults: [] }),
    exportCDP: vi.fn().mockResolvedValue(new Blob()),
    getCDPProgress: vi.fn().mockResolvedValue(null),
    getCDPScorePrediction: vi.fn().mockResolvedValue({ score: 'B' }),
    getComplianceScorecard: vi.fn().mockResolvedValue(null),
    getComplianceGaps: vi.fn().mockResolvedValue([]),
  },
}));

// ============================================================================
// Mock Data
// ============================================================================

const buildQuestion = (id: string, overrides: Record<string, any> = {}) => ({
  id,
  sectionId: overrides.sectionId || 'sec-c0',
  questionNumber: overrides.questionNumber || `C0.${id}`,
  questionText: overrides.questionText || `Question ${id}`,
  helpText: overrides.helpText || 'Help text for this question',
  fieldType: overrides.fieldType || 'text',
  options: overrides.options,
  required: overrides.required ?? true,
  value: overrides.value ?? '',
  autoPopulated: overrides.autoPopulated ?? false,
  dataSource: overrides.dataSource,
  confidence: overrides.confidence,
  validationError: overrides.validationError,
});

const mockQuestionnaire: CDPQuestionnaire = {
  id: 'q-2025',
  year: 2025,
  status: 'in_progress',
  overallCompletion: 62,
  lastSavedAt: '2026-02-28T10:00:00Z',
  submissionDeadline: '2026-07-31T23:59:59Z',
  sections: [
    {
      id: 'sec-c0', code: 'C0', name: 'Introduction',
      description: 'General introduction questions',
      completionPercentage: 100,
      isValid: true,
      questions: [
        buildQuestion('q1', { value: 'GreenLang Corp', autoPopulated: true, dataSource: 'ERP', confidence: 'high' }),
        buildQuestion('q2', { value: 'Energy', autoPopulated: true }),
      ],
    },
    {
      id: 'sec-c1', code: 'C1', name: 'Governance',
      description: 'Governance questions',
      completionPercentage: 50,
      isValid: false,
      questions: [
        buildQuestion('q3', { sectionId: 'sec-c1', questionNumber: 'C1.1', value: 'Board oversight', autoPopulated: false }),
        buildQuestion('q4', { sectionId: 'sec-c1', questionNumber: 'C1.2', value: '', validationError: 'Required field' }),
      ],
    },
    {
      id: 'sec-c6', code: 'C6', name: 'Emissions Data',
      description: 'Scope 1/2/3 emissions',
      completionPercentage: 80,
      isValid: false,
      questions: [
        buildQuestion('q5', { sectionId: 'sec-c6', questionNumber: 'C6.1', fieldType: 'number', value: 12500, autoPopulated: true }),
        buildQuestion('q6', { sectionId: 'sec-c6', questionNumber: 'C6.2', fieldType: 'select', options: ['Location-based', 'Market-based'], value: 'Location-based' }),
      ],
    },
  ],
};

const mockProgress: CDPProgress = {
  overallCompletion: 78.5,
  totalQuestions: 42,
  answeredQuestions: 33,
  autoFilledQuestions: 24,
  manualQuestions: 9,
  sectionProgress: [
    { sectionId: 'sec-c0', sectionName: 'Introduction', sectionCode: 'C0', totalQuestions: 5, answeredQuestions: 5, autoFilledQuestions: 4, manualQuestions: 1, completionPercentage: 100, isValid: true },
    { sectionId: 'sec-c1', sectionName: 'Governance', sectionCode: 'C1', totalQuestions: 12, answeredQuestions: 8, autoFilledQuestions: 6, manualQuestions: 2, completionPercentage: 66.7, isValid: false },
    { sectionId: 'sec-c6', sectionName: 'Emissions Data', sectionCode: 'C6', totalQuestions: 25, answeredQuestions: 20, autoFilledQuestions: 14, manualQuestions: 6, completionPercentage: 80, isValid: false },
  ],
  dataGaps: [
    { id: 'gap-1', sectionId: 'sec-c1', questionId: 'q4', description: 'Missing governance oversight details', severity: 'critical', suggestedAction: 'Provide board-level climate oversight' },
    { id: 'gap-2', sectionId: 'sec-c6', questionId: 'q7', description: 'Scope 3 Cat 4 data incomplete', severity: 'warning', suggestedAction: 'Import transport data' },
    { id: 'gap-3', sectionId: 'sec-c6', questionId: 'q8', description: 'Verification status not set', severity: 'info' },
  ],
};

const mockMappings: CDPDataMappingItem[] = [
  { id: 'm1', questionId: 'q1', questionNumber: 'C0.1', questionText: 'Organization name', sectionId: 'sec-c0', sectionName: 'C0 - Introduction', dataSource: 'erp', value: 'GreenLang', displayValue: 'GreenLang Corp', confidence: 'high' },
  { id: 'm2', questionId: 'q5', questionNumber: 'C6.1', questionText: 'Scope 1 emissions', sectionId: 'sec-c6', sectionName: 'C6 - Emissions Data', dataSource: 'calculated', value: 12500, displayValue: '12,500 tCO2e', confidence: 'high' },
  { id: 'm3', questionId: 'q3', questionNumber: 'C1.1', questionText: 'Governance description', sectionId: 'sec-c1', sectionName: 'C1 - Governance', dataSource: 'manual', value: 'Board oversight', displayValue: 'Board oversight', confidence: 'medium' },
  { id: 'm4', questionId: 'q4', questionNumber: 'C1.2', questionText: 'Climate risk officer', sectionId: 'sec-c1', sectionName: 'C1 - Governance', dataSource: 'unmapped', value: null, displayValue: '', confidence: 'low' },
];

const mockComplianceStandards: ComplianceStandard[] = [
  {
    id: 'std-1', name: 'GHG Protocol Scope 3', shortName: 'GHG Protocol',
    coveragePercentage: 92, requirementsMet: 23, requirementsTotal: 25, completionPercentage: 88,
    predictedScore: 'A-', lastUpdated: '2026-02-28T10:00:00Z',
    requirements: [
      { id: 'r1', code: 'GHG-01', name: 'Scope 3 Category Screening', description: 'Screen all 15 categories', met: true, dataPointsFilled: 15, dataPointsRequired: 15, priority: 'critical' },
      { id: 'r2', code: 'GHG-02', name: 'Activity Data Collection', description: 'Collect activity data', met: true, dataPointsFilled: 10, dataPointsRequired: 10, priority: 'high' },
      { id: 'r3', code: 'GHG-03', name: 'Uncertainty Assessment', description: 'Assess uncertainty', met: false, dataPointsFilled: 5, dataPointsRequired: 8, priority: 'medium' },
    ],
  },
  {
    id: 'std-2', name: 'ESRS E1 Climate', shortName: 'ESRS E1',
    coveragePercentage: 85, requirementsMet: 17, requirementsTotal: 20, completionPercentage: 82,
    lastUpdated: '2026-02-28T10:00:00Z',
    requirements: [
      { id: 'r4', code: 'E1-01', name: 'Transition Plan', description: 'Disclose transition plan', met: true, dataPointsFilled: 8, dataPointsRequired: 8, priority: 'critical' },
      { id: 'r5', code: 'E1-02', name: 'GHG Emissions', description: 'Report Scope 1/2/3', met: false, dataPointsFilled: 6, dataPointsRequired: 10, priority: 'critical' },
    ],
  },
  {
    id: 'std-3', name: 'CDP Climate Change', shortName: 'CDP',
    coveragePercentage: 88, requirementsMet: 13, requirementsTotal: 15, completionPercentage: 85,
    lastUpdated: '2026-02-28T10:00:00Z',
    requirements: [
      { id: 'r6', code: 'CDP-01', name: 'Governance', description: 'Board oversight', met: true, dataPointsFilled: 5, dataPointsRequired: 5, priority: 'high' },
    ],
  },
  {
    id: 'std-4', name: 'IFRS S2 Climate', shortName: 'IFRS S2',
    coveragePercentage: 76, requirementsMet: 11, requirementsTotal: 15, completionPercentage: 70,
    lastUpdated: '2026-02-28T10:00:00Z',
    requirements: [
      { id: 'r7', code: 'S2-01', name: 'Climate Risk Disclosure', description: 'Disclose climate risks', met: false, dataPointsFilled: 3, dataPointsRequired: 8, priority: 'critical' },
    ],
  },
  {
    id: 'std-5', name: 'ISO 14083 Transport', shortName: 'ISO 14083',
    coveragePercentage: 70, requirementsMet: 7, requirementsTotal: 10, completionPercentage: 65,
    lastUpdated: '2026-02-28T10:00:00Z',
    requirements: [
      { id: 'r8', code: 'ISO-01', name: 'WTW Methodology', description: 'Apply WTW methodology', met: true, dataPointsFilled: 4, dataPointsRequired: 4, priority: 'high' },
    ],
  },
];

// ============================================================================
// Helpers
// ============================================================================

function createMockStore(overrides: { cdp?: Record<string, any>; compliance?: Record<string, any> } = {}) {
  return configureStore({
    reducer: {
      cdp: cdpReducer,
      compliance: complianceReducer,
      dashboard: () => ({}),
      transactions: () => ({}),
      suppliers: () => ({}),
      reports: () => ({}),
      settings: () => ({}),
      uncertainty: () => ({}),
    },
    preloadedState: {
      cdp: {
        questionnaire: null,
        progress: null,
        mappings: [],
        scorePrediction: null,
        validation: null,
        selectedYear: 2025,
        loading: false,
        saving: false,
        error: null,
        ...(overrides.cdp || {}),
      },
      compliance: {
        scorecard: null,
        gaps: [],
        actionItems: [],
        loading: false,
        error: null,
        ...(overrides.compliance || {}),
      },
    } as any,
  });
}

function renderWithProviders(ui: React.ReactElement, overrides: { cdp?: Record<string, any>; compliance?: Record<string, any> } = {}) {
  const store = createMockStore(overrides);
  return {
    ...render(
      <Provider store={store}>
        <BrowserRouter>{ui}</BrowserRouter>
      </Provider>
    ),
    store,
  };
}

// ============================================================================
// 1. CDPQuestionnaireEditor
// ============================================================================

describe('CDPQuestionnaireEditor', () => {
  const onSave = vi.fn();
  const onQuestionChange = vi.fn();

  beforeEach(() => {
    onSave.mockReset();
    onQuestionChange.mockReset();
  });

  it('renders the year and overall progress', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    expect(screen.getByText('CDP Climate Change 2025')).toBeTruthy();
    expect(screen.getByText(/questions answered/)).toBeTruthy();
  });

  it('renders stepper with section codes', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    expect(screen.getByText('C0')).toBeTruthy();
    expect(screen.getByText('C1')).toBeTruthy();
    expect(screen.getByText('C6')).toBeTruthy();
  });

  it('shows auto-filled badge on auto-populated questions', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    const autoFilledChips = screen.getAllByText('Auto-filled');
    expect(autoFilledChips.length).toBeGreaterThan(0);
  });

  it('renders the Save Draft button', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    expect(screen.getByText('Save Draft')).toBeTruthy();
  });

  it('calls onSave when Save Draft is clicked', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    fireEvent.click(screen.getByText('Save Draft'));
    expect(onSave).toHaveBeenCalledWith(mockQuestionnaire);
  });

  it('navigates between sections with Next/Previous buttons', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    // Initially on section 1 of 3
    expect(screen.getByText('Section 1 of 3')).toBeTruthy();

    // Click Next
    fireEvent.click(screen.getByText('Next'));
    expect(screen.getByText('Section 2 of 3')).toBeTruthy();

    // Click Previous
    fireEvent.click(screen.getByText('Previous'));
    expect(screen.getByText('Section 1 of 3')).toBeTruthy();
  });

  it('disables Previous on first section and Next on last section', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    // Previous should be disabled on first section
    const prevBtn = screen.getByText('Previous').closest('button');
    expect(prevBtn?.disabled).toBe(true);

    // Navigate to last section
    fireEvent.click(screen.getByText('Next'));
    fireEvent.click(screen.getByText('Next'));
    const nextBtn = screen.getByText('Next').closest('button');
    expect(nextBtn?.disabled).toBe(true);
  });

  it('shows validation status with answered/required counts', () => {
    render(<CDPQuestionnaireEditor year={2025} questionnaire={mockQuestionnaire} onSave={onSave} />);
    // First section: 2/2 answered
    expect(screen.getByText('2/2 answered')).toBeTruthy();
  });
});

// ============================================================================
// 2. CDPProgressTracker
// ============================================================================

describe('CDPProgressTracker', () => {
  it('renders the overall completion donut', () => {
    render(<CDPProgressTracker progress={mockProgress} />);
    expect(screen.getByText('Overall Completion')).toBeTruthy();
    expect(screen.getByText('79%')).toBeTruthy();
    expect(screen.getByText('Complete')).toBeTruthy();
  });

  it('shows auto-filled and manual summary counts', () => {
    render(<CDPProgressTracker progress={mockProgress} />);
    expect(screen.getByText('24')).toBeTruthy(); // auto-filled
    expect(screen.getByText('9')).toBeTruthy();  // manual
    expect(screen.getByText('Auto-filled')).toBeTruthy();
    expect(screen.getByText('Manual')).toBeTruthy();
  });

  it('renders section progress bars', () => {
    render(<CDPProgressTracker progress={mockProgress} />);
    expect(screen.getByText('Section Progress')).toBeTruthy();
    expect(screen.getByText('C0')).toBeTruthy();
    expect(screen.getByText('C1')).toBeTruthy();
    expect(screen.getByText('C6')).toBeTruthy();
  });

  it('shows answered/total counts per section', () => {
    render(<CDPProgressTracker progress={mockProgress} />);
    expect(screen.getByText('5/5')).toBeTruthy();
    expect(screen.getByText('8/12')).toBeTruthy();
    expect(screen.getByText('20/25')).toBeTruthy();
  });

  it('renders data gap alerts with severity badges', () => {
    render(<CDPProgressTracker progress={mockProgress} />);
    expect(screen.getByText('Data Gaps (3)')).toBeTruthy();
    expect(screen.getByText('critical')).toBeTruthy();
    expect(screen.getByText('warning')).toBeTruthy();
    expect(screen.getByText('info')).toBeTruthy();
  });

  it('shows deadline countdown when deadline is provided', () => {
    // Future deadline: 500 days from now
    const futureDate = new Date(Date.now() + 500 * 24 * 60 * 60 * 1000).toISOString();
    render(<CDPProgressTracker progress={mockProgress} deadline={futureDate} />);
    expect(screen.getByText(/days until deadline/)).toBeTruthy();
  });
});

// ============================================================================
// 3. CDPDataMapping
// ============================================================================

describe('CDPDataMapping', () => {
  it('renders the Data Source Mapping title', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByText('Data Source Mapping')).toBeTruthy();
  });

  it('shows statistics chips', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByText('Total: 4')).toBeTruthy();
    expect(screen.getByText('Mapped: 3')).toBeTruthy();
    expect(screen.getByText('Unmapped: 1')).toBeTruthy();
  });

  it('renders table headers', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByText('Question ID')).toBeTruthy();
    expect(screen.getByText('Question Text')).toBeTruthy();
    expect(screen.getByText('Data Source')).toBeTruthy();
    expect(screen.getByText('Confidence')).toBeTruthy();
  });

  it('renders data source badges (ERP, Calculated, Manual, Unmapped)', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByText('ERP')).toBeTruthy();
    expect(screen.getByText('Calculated')).toBeTruthy();
    expect(screen.getByText('Manual')).toBeTruthy();
    expect(screen.getByText('Unmapped')).toBeTruthy();
  });

  it('renders the Section filter dropdown', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByLabelText('Section')).toBeTruthy();
  });

  it('renders the Source filter dropdown', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByLabelText('Source')).toBeTruthy();
  });

  it('shows the results count', () => {
    render(<CDPDataMapping mappings={mockMappings} />);
    expect(screen.getByText('Showing 4 of 4 mappings')).toBeTruthy();
  });
});

// ============================================================================
// 4. ComplianceScorecard
// ============================================================================

describe('ComplianceScorecard', () => {
  it('renders the Compliance Overview heading', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    expect(screen.getByText('Compliance Overview')).toBeTruthy();
  });

  it('renders 5 standard cards', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    expect(screen.getByText('GHG Protocol')).toBeTruthy();
    expect(screen.getByText('ESRS E1')).toBeTruthy();
    expect(screen.getByText('CDP')).toBeTruthy();
    expect(screen.getByText('IFRS S2')).toBeTruthy();
    expect(screen.getByText('ISO 14083')).toBeTruthy();
  });

  it('shows coverage percentages on each card', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    // GHG Protocol predicted score
    expect(screen.getByText('A-')).toBeTruthy();
    // Other cards show percentage like "85%"
    expect(screen.getByText('85%')).toBeTruthy();
    expect(screen.getByText('88%')).toBeTruthy();
  });

  it('shows requirement checklist items', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    expect(screen.getByText(/GHG-01: Scope 3 Category Screening/)).toBeTruthy();
  });

  it('shows critical gap alert when there are critical unmet requirements', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    // There are 2 critical unmet requirements (r5 E1-02 and r7 S2-01)
    expect(screen.getByText('Critical Compliance Gaps')).toBeTruthy();
  });

  it('renders the export button when handler is provided', () => {
    const onExport = vi.fn();
    render(<ComplianceScorecard standards={mockComplianceStandards} onExportReport={onExport} />);
    expect(screen.getByText('Export Report')).toBeTruthy();
  });

  it('renders gap summary chips', () => {
    render(<ComplianceScorecard standards={mockComplianceStandards} />);
    // 2 Critical gaps (E1-02 + S2-01), 1 Medium gap (GHG-03)
    expect(screen.getByText('2 Critical')).toBeTruthy();
    expect(screen.getByText('1 Medium')).toBeTruthy();
  });
});

// ============================================================================
// 5. CDPManagement Page
// ============================================================================

describe('CDPManagement Page', () => {
  it('renders the page heading', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    expect(screen.getByText('CDP Management')).toBeTruthy();
  });

  it('renders the year selector', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    expect(screen.getByLabelText('Year')).toBeTruthy();
  });

  it('renders the Auto-populate button', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    expect(screen.getByText('Auto-populate')).toBeTruthy();
  });

  it('renders the Validate button', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    expect(screen.getByText('Validate')).toBeTruthy();
  });

  it('renders the Export button', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    expect(screen.getByText('Export')).toBeTruthy();
  });

  it('shows editor and progress tracker when questionnaire is loaded', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: { questionnaire: mockQuestionnaire, progress: mockProgress, mappings: mockMappings },
    });
    // Editor renders the year header
    expect(screen.getByText('CDP Climate Change 2025')).toBeTruthy();
    // Progress tracker renders
    expect(screen.getByText('Overall Completion')).toBeTruthy();
  });

  it('shows no-data alert when questionnaire is null', () => {
    renderWithProviders(<CDPManagement />);
    expect(screen.getByText(/No CDP questionnaire data available/)).toBeTruthy();
  });

  it('shows score prediction chip when available', () => {
    renderWithProviders(<CDPManagement />, {
      cdp: {
        questionnaire: mockQuestionnaire,
        progress: mockProgress,
        mappings: mockMappings,
        scorePrediction: 'B',
      },
    });
    expect(screen.getByText('Predicted Score: B')).toBeTruthy();
  });
});

// ============================================================================
// 6. ComplianceDashboard Page
// ============================================================================

describe('ComplianceDashboard Page', () => {
  it('renders the page heading', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText('Compliance Dashboard')).toBeTruthy();
  });

  it('renders the Refresh and Export Report buttons', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText('Refresh')).toBeTruthy();
    // Two Export Report buttons: one in the page header, one in the scorecard
    const exportBtns = screen.getAllByText('Export Report');
    expect(exportBtns.length).toBeGreaterThanOrEqual(1);
  });

  it('renders the scorecard component when data is present', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText('Compliance Overview')).toBeTruthy();
  });

  it('renders the radar chart section', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText('Coverage by Standard')).toBeTruthy();
  });

  it('renders the trend chart section', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText('Compliance Trend (Last 8 Months)')).toBeTruthy();
  });

  it('shows "no compliance gaps" alert when actionItems is empty', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [],
      },
    });
    expect(screen.getByText(/No compliance gaps found/)).toBeTruthy();
  });

  it('renders action items table when items exist', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        scorecard: { overallScore: 84.7, standards: mockComplianceStandards, lastCalculated: '2026-02-28T10:00:00Z' },
        actionItems: [
          { id: 'ai-1', gap: 'Missing Scope 3 data', standard: 'GHG Protocol', severity: 'critical', action: 'Collect Cat 1 data', status: 'open', dueDate: '2026-06-30T00:00:00Z' },
          { id: 'ai-2', gap: 'Transition plan', standard: 'ESRS E1', severity: 'high', action: 'Draft transition plan', status: 'in_progress' },
        ],
      },
    });
    expect(screen.getByText('Action Items')).toBeTruthy();
    expect(screen.getByText('Missing Scope 3 data')).toBeTruthy();
    expect(screen.getByText('Transition plan')).toBeTruthy();
  });

  it('shows error alert when error is present', () => {
    renderWithProviders(<ComplianceDashboard />, {
      compliance: {
        error: 'Failed to load compliance data',
      },
    });
    expect(screen.getByText('Failed to load compliance data')).toBeTruthy();
  });
});
