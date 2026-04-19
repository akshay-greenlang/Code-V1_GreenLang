/**
 * CDP Questionnaire Redux Slice
 *
 * Manages state for CDP Climate Change questionnaire editing,
 * progress tracking, data mapping, and score prediction.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';

// =============================================================================
// Type Definitions
// =============================================================================

export interface CDPQuestion {
  id: string;
  sectionId: string;
  questionNumber: string;
  questionText: string;
  helpText?: string;
  fieldType: 'text' | 'textarea' | 'select' | 'multiselect' | 'checkbox' | 'number' | 'date' | 'table';
  options?: string[];
  required: boolean;
  value: any;
  autoPopulated: boolean;
  dataSource?: string;
  confidence?: 'high' | 'medium' | 'low';
  validationError?: string;
}

export interface CDPSection {
  id: string;
  code: string;
  name: string;
  description: string;
  questions: CDPQuestion[];
  completionPercentage: number;
  isValid: boolean;
}

export interface CDPQuestionnaire {
  id: string;
  year: number;
  status: 'draft' | 'in_progress' | 'validated' | 'submitted';
  sections: CDPSection[];
  lastSavedAt?: string;
  submissionDeadline?: string;
  overallCompletion: number;
}

export interface CDPProgress {
  overallCompletion: number;
  totalQuestions: number;
  answeredQuestions: number;
  autoFilledQuestions: number;
  manualQuestions: number;
  sectionProgress: Array<{
    sectionId: string;
    sectionName: string;
    sectionCode: string;
    totalQuestions: number;
    answeredQuestions: number;
    autoFilledQuestions: number;
    manualQuestions: number;
    completionPercentage: number;
    isValid: boolean;
  }>;
  dataGaps: Array<{
    id: string;
    sectionId: string;
    questionId: string;
    description: string;
    severity: 'critical' | 'warning' | 'info';
    suggestedAction?: string;
  }>;
}

export interface CDPDataMappingItem {
  id: string;
  questionId: string;
  questionNumber: string;
  questionText: string;
  sectionId: string;
  sectionName: string;
  dataSource: 'erp' | 'calculated' | 'manual' | 'unmapped';
  value: any;
  displayValue: string;
  confidence: 'high' | 'medium' | 'low';
  lastUpdated?: string;
}

export interface CDPValidation {
  isValid: boolean;
  totalErrors: number;
  totalWarnings: number;
  sectionResults: Array<{
    sectionId: string;
    sectionName: string;
    isValid: boolean;
    errors: Array<{
      questionId: string;
      questionNumber: string;
      message: string;
      severity: 'error' | 'warning';
    }>;
  }>;
}

// =============================================================================
// State Interface
// =============================================================================

interface CDPState {
  questionnaire: CDPQuestionnaire | null;
  progress: CDPProgress | null;
  mappings: CDPDataMappingItem[];
  scorePrediction: string | null;
  validation: CDPValidation | null;
  selectedYear: number;
  loading: boolean;
  saving: boolean;
  error: string | null;
}

const initialState: CDPState = {
  questionnaire: null,
  progress: null,
  mappings: [],
  scorePrediction: null,
  validation: null,
  selectedYear: new Date().getFullYear(),
  loading: false,
  saving: false,
  error: null,
};

// =============================================================================
// Async Thunks
// =============================================================================

export const fetchCDPQuestionnaire = createAsyncThunk(
  'cdp/fetchQuestionnaire',
  async (year: number) => {
    const response = await api.getCDPQuestionnaire(year);
    return response;
  }
);

export const autoPopulateCDP = createAsyncThunk(
  'cdp/autoPopulate',
  async (year: number) => {
    const response = await api.autoPopulateCDP(year);
    return response;
  }
);

export const saveCDPDraft = createAsyncThunk(
  'cdp/saveDraft',
  async ({ year, data }: { year: number; data: any }) => {
    await api.saveCDPDraft(year, data);
    return { year, savedAt: new Date().toISOString() };
  }
);

export const validateCDP = createAsyncThunk(
  'cdp/validate',
  async (year: number) => {
    const response = await api.validateCDP(year);
    return response;
  }
);

export const exportCDP = createAsyncThunk(
  'cdp/export',
  async ({ year, format }: { year: number; format: string }) => {
    const blob = await api.exportCDP(year, format);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cdp_questionnaire_${year}.${format === 'excel' ? 'xlsx' : format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    return { year, format };
  }
);

export const fetchCDPProgress = createAsyncThunk(
  'cdp/fetchProgress',
  async (year: number) => {
    const response = await api.getCDPProgress(year);
    return response;
  }
);

export const fetchScorePrediction = createAsyncThunk(
  'cdp/fetchScorePrediction',
  async (year: number) => {
    const response = await api.getCDPScorePrediction(year);
    return response.score;
  }
);

// =============================================================================
// Slice
// =============================================================================

const cdpSlice = createSlice({
  name: 'cdp',
  initialState,
  reducers: {
    setSelectedYear: (state, action: PayloadAction<number>) => {
      state.selectedYear = action.payload;
    },
    updateQuestionValue: (
      state,
      action: PayloadAction<{ sectionId: string; questionId: string; value: any }>
    ) => {
      if (state.questionnaire) {
        const section = state.questionnaire.sections.find(
          (s) => s.id === action.payload.sectionId
        );
        if (section) {
          const question = section.questions.find(
            (q) => q.id === action.payload.questionId
          );
          if (question) {
            question.value = action.payload.value;
            question.autoPopulated = false;
          }
        }
      }
    },
    clearError: (state) => {
      state.error = null;
    },
    clearValidation: (state) => {
      state.validation = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch questionnaire
    builder
      .addCase(fetchCDPQuestionnaire.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCDPQuestionnaire.fulfilled, (state, action) => {
        state.loading = false;
        state.questionnaire = action.payload;
      })
      .addCase(fetchCDPQuestionnaire.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch CDP questionnaire';
      });

    // Auto-populate
    builder
      .addCase(autoPopulateCDP.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(autoPopulateCDP.fulfilled, (state, action) => {
        state.loading = false;
        state.questionnaire = action.payload;
      })
      .addCase(autoPopulateCDP.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to auto-populate CDP data';
      });

    // Save draft
    builder
      .addCase(saveCDPDraft.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(saveCDPDraft.fulfilled, (state, action) => {
        state.saving = false;
        if (state.questionnaire) {
          state.questionnaire.lastSavedAt = action.payload.savedAt;
        }
      })
      .addCase(saveCDPDraft.rejected, (state, action) => {
        state.saving = false;
        state.error = action.error.message || 'Failed to save CDP draft';
      });

    // Validate
    builder
      .addCase(validateCDP.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(validateCDP.fulfilled, (state, action) => {
        state.loading = false;
        state.validation = action.payload;
      })
      .addCase(validateCDP.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to validate CDP questionnaire';
      });

    // Export
    builder
      .addCase(exportCDP.pending, (state) => {
        state.loading = true;
      })
      .addCase(exportCDP.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(exportCDP.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to export CDP questionnaire';
      });

    // Fetch progress
    builder
      .addCase(fetchCDPProgress.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchCDPProgress.fulfilled, (state, action) => {
        state.loading = false;
        state.progress = action.payload;
      })
      .addCase(fetchCDPProgress.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch CDP progress';
      });

    // Score prediction
    builder
      .addCase(fetchScorePrediction.fulfilled, (state, action) => {
        state.scorePrediction = action.payload;
      });
  },
});

export const { setSelectedYear, updateQuestionValue, clearError, clearValidation } =
  cdpSlice.actions;
export default cdpSlice.reducer;
