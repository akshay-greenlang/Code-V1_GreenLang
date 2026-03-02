/**
 * APP-003 GL-VCCI-APP v1.1 -- CDP & Compliance Redux Slice Tests
 *
 * Tests the cdpSlice and complianceSlice initial states, all async thunks
 * (pending / fulfilled / rejected), synchronous reducers, and state
 * mutations.
 *
 * Target: 15+ tests, ~200 lines
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { configureStore } from '@reduxjs/toolkit';

import cdpReducer, {
  fetchCDPQuestionnaire,
  autoPopulateCDP,
  saveCDPDraft,
  validateCDP,
  fetchCDPProgress,
  fetchScorePrediction,
  setSelectedYear,
  updateQuestionValue,
  clearError as clearCdpError,
  clearValidation,
} from '../../store/slices/cdpSlice';

import complianceReducer, {
  fetchComplianceScorecard,
  fetchComplianceGaps,
  clearError as clearComplianceError,
  updateActionItemStatus,
} from '../../store/slices/complianceSlice';

// ============================================================================
// Mock API
// ============================================================================

const mockApi = {
  getCDPQuestionnaire: vi.fn(),
  autoPopulateCDP: vi.fn(),
  saveCDPDraft: vi.fn(),
  validateCDP: vi.fn(),
  exportCDP: vi.fn(),
  getCDPProgress: vi.fn(),
  getCDPScorePrediction: vi.fn(),
  getComplianceScorecard: vi.fn(),
  getComplianceGaps: vi.fn(),
};

vi.mock('../../services/api', () => ({
  default: {
    getCDPQuestionnaire: (...args: any[]) => mockApi.getCDPQuestionnaire(...args),
    autoPopulateCDP: (...args: any[]) => mockApi.autoPopulateCDP(...args),
    saveCDPDraft: (...args: any[]) => mockApi.saveCDPDraft(...args),
    validateCDP: (...args: any[]) => mockApi.validateCDP(...args),
    exportCDP: (...args: any[]) => mockApi.exportCDP(...args),
    getCDPProgress: (...args: any[]) => mockApi.getCDPProgress(...args),
    getCDPScorePrediction: (...args: any[]) => mockApi.getCDPScorePrediction(...args),
    getComplianceScorecard: (...args: any[]) => mockApi.getComplianceScorecard(...args),
    getComplianceGaps: (...args: any[]) => mockApi.getComplianceGaps(...args),
  },
}));

// ============================================================================
// Helpers
// ============================================================================

function createCdpStore(preloadedState?: any) {
  return configureStore({
    reducer: { cdp: cdpReducer },
    preloadedState: preloadedState ? { cdp: preloadedState } : undefined,
  });
}

function createComplianceStore(preloadedState?: any) {
  return configureStore({
    reducer: { compliance: complianceReducer },
    preloadedState: preloadedState ? { compliance: preloadedState } : undefined,
  });
}

const mockQuestionnaire = {
  id: 'q-2025',
  year: 2025,
  status: 'draft',
  sections: [
    {
      id: 'sec-c0',
      code: 'C0',
      name: 'Introduction',
      description: '',
      completionPercentage: 50,
      isValid: false,
      questions: [
        { id: 'q1', sectionId: 'sec-c0', questionNumber: 'C0.1', questionText: 'Org name', fieldType: 'text', required: true, value: 'GreenLang', autoPopulated: true },
        { id: 'q2', sectionId: 'sec-c0', questionNumber: 'C0.2', questionText: 'Sector', fieldType: 'text', required: true, value: '', autoPopulated: false },
      ],
    },
  ],
  overallCompletion: 50,
};

const mockValidation = {
  isValid: false,
  totalErrors: 2,
  totalWarnings: 1,
  sectionResults: [
    {
      sectionId: 'sec-c0',
      sectionName: 'Introduction',
      isValid: false,
      errors: [{ questionId: 'q2', questionNumber: 'C0.2', message: 'Required', severity: 'error' }],
    },
  ],
};

const mockProgress = {
  overallCompletion: 78,
  totalQuestions: 42,
  answeredQuestions: 33,
  autoFilledQuestions: 24,
  manualQuestions: 9,
  sectionProgress: [],
  dataGaps: [],
};

const mockScorecard = {
  overallScore: 84.7,
  standards: [
    { id: 's1', name: 'GHG Protocol', shortName: 'GHG', coveragePercentage: 92, requirementsMet: 23, requirementsTotal: 25, completionPercentage: 88, requirements: [], lastUpdated: '' },
  ],
  lastCalculated: '2026-02-28T10:00:00Z',
};

const mockGaps = [
  { id: 'g1', standardId: 's1', standardName: 'GHG Protocol', requirementId: 'r1', requirementName: 'Cat 1 data', description: 'Missing', severity: 'critical', action: 'Collect data', status: 'open' },
];

// ============================================================================
// CDP Slice Tests
// ============================================================================

describe('cdpSlice', () => {
  beforeEach(() => vi.clearAllMocks());

  // ---------- Initial state ----------

  it('has the correct initial state', () => {
    const store = createCdpStore();
    const state = store.getState().cdp;

    expect(state.questionnaire).toBeNull();
    expect(state.progress).toBeNull();
    expect(state.mappings).toEqual([]);
    expect(state.scorePrediction).toBeNull();
    expect(state.validation).toBeNull();
    expect(state.selectedYear).toBe(new Date().getFullYear());
    expect(state.loading).toBe(false);
    expect(state.saving).toBe(false);
    expect(state.error).toBeNull();
  });

  // ---------- Synchronous reducers ----------

  it('setSelectedYear updates the year', () => {
    const store = createCdpStore();
    store.dispatch(setSelectedYear(2024));
    expect(store.getState().cdp.selectedYear).toBe(2024);
  });

  it('updateQuestionValue modifies the question value and clears autoPopulated', () => {
    const store = createCdpStore({
      questionnaire: mockQuestionnaire,
      progress: null,
      mappings: [],
      scorePrediction: null,
      validation: null,
      selectedYear: 2025,
      loading: false,
      saving: false,
      error: null,
    });

    store.dispatch(updateQuestionValue({ sectionId: 'sec-c0', questionId: 'q1', value: 'New Corp' }));
    const q = store.getState().cdp.questionnaire!.sections[0].questions[0];
    expect(q.value).toBe('New Corp');
    expect(q.autoPopulated).toBe(false);
  });

  it('clearError resets the error', () => {
    const store = createCdpStore({
      questionnaire: null, progress: null, mappings: [], scorePrediction: null,
      validation: null, selectedYear: 2025, loading: false, saving: false,
      error: 'Some error',
    });
    store.dispatch(clearCdpError());
    expect(store.getState().cdp.error).toBeNull();
  });

  it('clearValidation resets the validation', () => {
    const store = createCdpStore({
      questionnaire: null, progress: null, mappings: [], scorePrediction: null,
      validation: mockValidation, selectedYear: 2025, loading: false, saving: false, error: null,
    });
    store.dispatch(clearValidation());
    expect(store.getState().cdp.validation).toBeNull();
  });

  // ---------- fetchCDPQuestionnaire ----------

  it('fetchCDPQuestionnaire.pending sets loading', async () => {
    mockApi.getCDPQuestionnaire.mockResolvedValueOnce(mockQuestionnaire);
    const store = createCdpStore();

    const promise = store.dispatch(fetchCDPQuestionnaire(2025));
    expect(store.getState().cdp.loading).toBe(true);
    await promise;
  });

  it('fetchCDPQuestionnaire.fulfilled stores the questionnaire', async () => {
    mockApi.getCDPQuestionnaire.mockResolvedValueOnce(mockQuestionnaire);
    const store = createCdpStore();

    await store.dispatch(fetchCDPQuestionnaire(2025));
    expect(store.getState().cdp.questionnaire).toEqual(mockQuestionnaire);
    expect(store.getState().cdp.loading).toBe(false);
  });

  it('fetchCDPQuestionnaire.rejected sets error', async () => {
    mockApi.getCDPQuestionnaire.mockRejectedValueOnce(new Error('Not found'));
    const store = createCdpStore();

    await store.dispatch(fetchCDPQuestionnaire(2025));
    expect(store.getState().cdp.error).toBe('Not found');
    expect(store.getState().cdp.loading).toBe(false);
  });

  // ---------- autoPopulateCDP ----------

  it('autoPopulateCDP.fulfilled replaces the questionnaire', async () => {
    const populated = { ...mockQuestionnaire, overallCompletion: 92 };
    mockApi.autoPopulateCDP.mockResolvedValueOnce(populated);
    const store = createCdpStore();

    await store.dispatch(autoPopulateCDP(2025));
    expect(store.getState().cdp.questionnaire!.overallCompletion).toBe(92);
  });

  it('autoPopulateCDP.rejected sets error', async () => {
    mockApi.autoPopulateCDP.mockRejectedValueOnce(new Error('Service unavailable'));
    const store = createCdpStore();

    await store.dispatch(autoPopulateCDP(2025));
    expect(store.getState().cdp.error).toBe('Service unavailable');
  });

  // ---------- saveCDPDraft ----------

  it('saveCDPDraft.pending sets saving flag', async () => {
    mockApi.saveCDPDraft.mockResolvedValueOnce({});
    const store = createCdpStore({
      questionnaire: mockQuestionnaire, progress: null, mappings: [], scorePrediction: null,
      validation: null, selectedYear: 2025, loading: false, saving: false, error: null,
    });

    const promise = store.dispatch(saveCDPDraft({ year: 2025, data: mockQuestionnaire }));
    expect(store.getState().cdp.saving).toBe(true);
    await promise;
    expect(store.getState().cdp.saving).toBe(false);
  });

  it('saveCDPDraft.fulfilled updates lastSavedAt', async () => {
    mockApi.saveCDPDraft.mockResolvedValueOnce({});
    const store = createCdpStore({
      questionnaire: { ...mockQuestionnaire, lastSavedAt: undefined },
      progress: null, mappings: [], scorePrediction: null,
      validation: null, selectedYear: 2025, loading: false, saving: false, error: null,
    });

    await store.dispatch(saveCDPDraft({ year: 2025, data: mockQuestionnaire }));
    expect(store.getState().cdp.questionnaire!.lastSavedAt).toBeDefined();
  });

  // ---------- validateCDP ----------

  it('validateCDP.fulfilled stores validation result', async () => {
    mockApi.validateCDP.mockResolvedValueOnce(mockValidation);
    const store = createCdpStore();

    await store.dispatch(validateCDP(2025));
    expect(store.getState().cdp.validation).toEqual(mockValidation);
  });

  it('validateCDP.rejected sets error', async () => {
    mockApi.validateCDP.mockRejectedValueOnce(new Error('Validation service error'));
    const store = createCdpStore();

    await store.dispatch(validateCDP(2025));
    expect(store.getState().cdp.error).toBe('Validation service error');
  });

  // ---------- fetchCDPProgress ----------

  it('fetchCDPProgress.fulfilled stores progress', async () => {
    mockApi.getCDPProgress.mockResolvedValueOnce(mockProgress);
    const store = createCdpStore();

    await store.dispatch(fetchCDPProgress(2025));
    expect(store.getState().cdp.progress).toEqual(mockProgress);
  });

  // ---------- fetchScorePrediction ----------

  it('fetchScorePrediction.fulfilled stores the score', async () => {
    mockApi.getCDPScorePrediction.mockResolvedValueOnce({ score: 'A-' });
    const store = createCdpStore();

    await store.dispatch(fetchScorePrediction(2025));
    expect(store.getState().cdp.scorePrediction).toBe('A-');
  });
});

// ============================================================================
// Compliance Slice Tests
// ============================================================================

describe('complianceSlice', () => {
  beforeEach(() => vi.clearAllMocks());

  // ---------- Initial state ----------

  it('has the correct initial state', () => {
    const store = createComplianceStore();
    const state = store.getState().compliance;

    expect(state.scorecard).toBeNull();
    expect(state.gaps).toEqual([]);
    expect(state.actionItems).toEqual([]);
    expect(state.loading).toBe(false);
    expect(state.error).toBeNull();
  });

  // ---------- Synchronous reducers ----------

  it('clearError resets error to null', () => {
    const store = createComplianceStore({
      scorecard: null, gaps: [], actionItems: [], loading: false,
      error: 'Previous error',
    });
    store.dispatch(clearComplianceError());
    expect(store.getState().compliance.error).toBeNull();
  });

  it('updateActionItemStatus changes the status of an action item', () => {
    const store = createComplianceStore({
      scorecard: null, gaps: [], loading: false, error: null,
      actionItems: [
        { id: 'ai-1', gap: 'Missing data', standard: 'GHG', severity: 'critical', action: 'Collect', status: 'open' },
      ],
    });

    store.dispatch(updateActionItemStatus({ id: 'ai-1', status: 'in_progress' }));
    expect(store.getState().compliance.actionItems[0].status).toBe('in_progress');
  });

  // ---------- fetchComplianceScorecard ----------

  it('fetchComplianceScorecard.pending sets loading', async () => {
    mockApi.getComplianceScorecard.mockResolvedValueOnce(mockScorecard);
    const store = createComplianceStore();

    const promise = store.dispatch(fetchComplianceScorecard());
    expect(store.getState().compliance.loading).toBe(true);
    await promise;
  });

  it('fetchComplianceScorecard.fulfilled stores the scorecard', async () => {
    mockApi.getComplianceScorecard.mockResolvedValueOnce(mockScorecard);
    const store = createComplianceStore();

    await store.dispatch(fetchComplianceScorecard());
    expect(store.getState().compliance.scorecard).toEqual(mockScorecard);
    expect(store.getState().compliance.loading).toBe(false);
  });

  it('fetchComplianceScorecard.rejected sets error', async () => {
    mockApi.getComplianceScorecard.mockRejectedValueOnce(new Error('DB connection lost'));
    const store = createComplianceStore();

    await store.dispatch(fetchComplianceScorecard());
    expect(store.getState().compliance.error).toBe('DB connection lost');
  });

  // ---------- fetchComplianceGaps ----------

  it('fetchComplianceGaps.fulfilled stores gaps and derives actionItems', async () => {
    mockApi.getComplianceGaps.mockResolvedValueOnce(mockGaps);
    const store = createComplianceStore();

    await store.dispatch(fetchComplianceGaps());
    const state = store.getState().compliance;

    expect(state.gaps).toEqual(mockGaps);
    expect(state.actionItems).toHaveLength(1);
    expect(state.actionItems[0].gap).toBe('Cat 1 data');
    expect(state.actionItems[0].standard).toBe('GHG Protocol');
    expect(state.actionItems[0].severity).toBe('critical');
    expect(state.actionItems[0].status).toBe('open');
  });

  it('fetchComplianceGaps.rejected sets error', async () => {
    mockApi.getComplianceGaps.mockRejectedValueOnce(new Error('Unauthorized'));
    const store = createComplianceStore();

    await store.dispatch(fetchComplianceGaps());
    expect(store.getState().compliance.error).toBe('Unauthorized');
  });
});
