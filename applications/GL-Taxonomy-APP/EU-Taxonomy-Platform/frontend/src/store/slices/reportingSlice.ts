import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DisclosureReport, Article8Data, ReportHistory, ReportComparison } from '../../types';
import { reportingApi } from '../../services/api';
import type { RootState } from '../index';

interface ReportingState {
  currentReport: DisclosureReport | null;
  article8Data: Article8Data | null;
  history: ReportHistory[];
  comparison: ReportComparison | null;
  loading: boolean;
  error: string | null;
}

const initialState: ReportingState = {
  currentReport: null,
  article8Data: null,
  history: [],
  comparison: null,
  loading: false,
  error: null,
};

export const fetchArticle8 = createAsyncThunk(
  'reporting/article8',
  async ({ orgId, period }: { orgId: string; period: string }) =>
    reportingApi.article8(orgId, period)
);

export const fetchReportHistory = createAsyncThunk(
  'reporting/history',
  async (orgId: string) => reportingApi.history(orgId)
);

export const compareReports = createAsyncThunk(
  'reporting/compare',
  async ({ orgId, p1, p2 }: { orgId: string; p1: string; p2: string }) =>
    reportingApi.compare(orgId, p1, p2)
);

export const createReport = createAsyncThunk(
  'reporting/create',
  async ({ orgId, payload }: { orgId: string; payload: Partial<DisclosureReport> }) =>
    reportingApi.create(orgId, payload)
);

const reportingSlice = createSlice({
  name: 'reporting',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchArticle8.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchArticle8.fulfilled, (state, action: PayloadAction<Article8Data>) => {
        state.loading = false;
        state.article8Data = action.payload;
      })
      .addCase(fetchArticle8.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch report data';
      })
      .addCase(fetchReportHistory.fulfilled, (state, action) => { state.history = action.payload; })
      .addCase(compareReports.fulfilled, (state, action) => { state.comparison = action.payload; })
      .addCase(createReport.fulfilled, (state, action) => { state.currentReport = action.payload; });
  },
});

export const { clearError } = reportingSlice.actions;
export const selectCurrentReport = (state: RootState) => state.reporting.currentReport;
export const selectArticle8Data = (state: RootState) => state.reporting.article8Data;
export const selectReportHistory = (state: RootState) => state.reporting.history;
export const selectReportComparison = (state: RootState) => state.reporting.comparison;
export const selectReportingLoading = (state: RootState) => state.reporting.loading;
export default reportingSlice.reducer;
