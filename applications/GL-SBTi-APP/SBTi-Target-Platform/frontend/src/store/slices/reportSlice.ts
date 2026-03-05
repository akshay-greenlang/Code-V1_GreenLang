import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Report } from '../../types';
import { reportApi } from '../../services/api';
import type { RootState } from '../index';

interface ReportState {
  reports: Report[];
  selectedReport: Report | null;
  generating: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: ReportState = {
  reports: [],
  selectedReport: null,
  generating: false,
  loading: false,
  error: null,
};

export const fetchReports = createAsyncThunk(
  'report/fetchReports',
  async (orgId: string) => reportApi.getReports(orgId)
);

export const generateReport = createAsyncThunk(
  'report/generate',
  async (data: { org_id: string; report_type: string; target_ids: string[]; year: number }) =>
    reportApi.generateReport(data)
);

export const fetchReport = createAsyncThunk(
  'report/fetchReport',
  async (id: string) => reportApi.getReport(id)
);

export const deleteReport = createAsyncThunk(
  'report/delete',
  async (id: string) => { await reportApi.deleteReport(id); return id; }
);

const reportSlice = createSlice({
  name: 'report',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearSelectedReport(state) { state.selectedReport = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchReports.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchReports.fulfilled, (state, action: PayloadAction<Report[]>) => {
        state.loading = false;
        state.reports = action.payload;
      })
      .addCase(fetchReports.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch reports';
      })
      .addCase(generateReport.pending, (state) => { state.generating = true; })
      .addCase(generateReport.fulfilled, (state, action: PayloadAction<Report>) => {
        state.generating = false;
        state.reports.push(action.payload);
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.generating = false;
        state.error = action.error.message || 'Failed to generate report';
      })
      .addCase(fetchReport.fulfilled, (state, action: PayloadAction<Report>) => {
        state.selectedReport = action.payload;
      })
      .addCase(deleteReport.fulfilled, (state, action: PayloadAction<string>) => {
        state.reports = state.reports.filter((r) => r.id !== action.payload);
      });
  },
});

export const { clearError, clearSelectedReport } = reportSlice.actions;
export const selectReports = (state: RootState) => state.report.reports;
export const selectSelectedReport = (state: RootState) => state.report.selectedReport;
export const selectReportGenerating = (state: RootState) => state.report.generating;
export const selectReportLoading = (state: RootState) => state.report.loading;
export default reportSlice.reducer;
