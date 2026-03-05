/**
 * Reports Redux Slice
 *
 * Manages report generation state: report list, generation progress,
 * submission checklist, and export operations.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { ReportsState, CDPReport, SubmissionChecklist, ReportFormat } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: ReportsState = {
  reports: [],
  checklist: null,
  generating: false,
  loading: false,
  error: null,
};

export const generateReport = createAsyncThunk<
  CDPReport,
  { questionnaireId: string; format: ReportFormat; title?: string }
>(
  'reports/generate',
  async ({ questionnaireId, format, title }) =>
    cdpApi.generateReport({ questionnaire_id: questionnaireId, format, title }),
);

export const fetchReports = createAsyncThunk<CDPReport[], string>(
  'reports/fetchAll',
  async (questionnaireId) => cdpApi.getReports(questionnaireId),
);

export const fetchChecklist = createAsyncThunk<SubmissionChecklist, string>(
  'reports/fetchChecklist',
  async (questionnaireId) => cdpApi.getSubmissionChecklist(questionnaireId),
);

export const submitToORS = createAsyncThunk<void, string>(
  'reports/submitToORS',
  async (questionnaireId) => cdpApi.submitToORS(questionnaireId),
);

const reportsSlice = createSlice({
  name: 'reports',
  initialState,
  reducers: {
    clearReports: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(generateReport.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(generateReport.fulfilled, (state, action) => {
        state.generating = false;
        state.reports.push(action.payload);
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.generating = false;
        state.error = action.error.message ?? 'Failed to generate report';
      })
      .addCase(fetchReports.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchReports.fulfilled, (state, action) => {
        state.loading = false;
        state.reports = action.payload;
      })
      .addCase(fetchReports.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load reports';
      })
      .addCase(fetchChecklist.fulfilled, (state, action) => {
        state.checklist = action.payload;
      });
  },
});

export const { clearReports } = reportsSlice.actions;
export default reportsSlice.reducer;
