import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ValidationResult } from '../../types';
import { validationApi } from '../../services/api';
import type { RootState } from '../index';

interface ValidationState {
  result: ValidationResult | null;
  checklist: { criterion_code: string; criterion_name: string; category: string; status: string }[];
  readiness: { readiness_score: number; category_scores: { category: string; score: number }[]; blockers: string[] } | null;
  loading: boolean;
  validating: boolean;
  error: string | null;
}

const initialState: ValidationState = {
  result: null,
  checklist: [],
  readiness: null,
  loading: false,
  validating: false,
  error: null,
};

export const runValidation = createAsyncThunk(
  'validation/runValidation',
  async (targetId: string) => validationApi.validate(targetId)
);

export const fetchValidationResult = createAsyncThunk(
  'validation/fetchResult',
  async (targetId: string) => validationApi.getResult(targetId)
);

export const fetchChecklist = createAsyncThunk(
  'validation/fetchChecklist',
  async (orgId: string) => validationApi.getChecklist(orgId)
);

export const fetchReadiness = createAsyncThunk(
  'validation/fetchReadiness',
  async (orgId: string) => validationApi.getReadiness(orgId)
);

const validationSlice = createSlice({
  name: 'validation',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearResult(state) { state.result = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(runValidation.pending, (state) => { state.validating = true; state.error = null; })
      .addCase(runValidation.fulfilled, (state, action: PayloadAction<ValidationResult>) => {
        state.validating = false;
        state.result = action.payload;
      })
      .addCase(runValidation.rejected, (state, action) => {
        state.validating = false;
        state.error = action.error.message || 'Validation failed';
      })
      .addCase(fetchValidationResult.pending, (state) => { state.loading = true; })
      .addCase(fetchValidationResult.fulfilled, (state, action: PayloadAction<ValidationResult>) => {
        state.loading = false;
        state.result = action.payload;
      })
      .addCase(fetchValidationResult.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch validation result';
      })
      .addCase(fetchChecklist.fulfilled, (state, action) => {
        state.checklist = action.payload;
      })
      .addCase(fetchReadiness.fulfilled, (state, action) => {
        state.readiness = action.payload;
      });
  },
});

export const { clearError, clearResult } = validationSlice.actions;
export const selectValidationResult = (state: RootState) => state.validation.result;
export const selectChecklist = (state: RootState) => state.validation.checklist;
export const selectReadiness = (state: RootState) => state.validation.readiness;
export const selectValidationLoading = (state: RootState) => state.validation.loading;
export const selectValidating = (state: RootState) => state.validation.validating;
export default validationSlice.reducer;
