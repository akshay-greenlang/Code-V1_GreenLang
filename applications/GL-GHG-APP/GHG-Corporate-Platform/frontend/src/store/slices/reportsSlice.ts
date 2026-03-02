/**
 * Reports Redux Slice
 *
 * Manages report generation, disclosure tracking, completeness checks,
 * and data export functionality.
 *
 * Async thunks:
 *   - generateReport: Generate a GHG Protocol compliant report
 *   - fetchReports: Load report history for an inventory
 *   - fetchDisclosures: Load disclosure checklist
 *   - fetchCompleteness: Load completeness assessment
 *   - exportData: Export inventory data in specified format
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  ReportsState,
  Report,
  Disclosure,
  CompletenessResult,
  ExportResult,
  GenerateReportRequest,
  ExportDataRequest,
  ReportFormat,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: ReportsState = {
  reports: [],
  disclosures: [],
  completeness: null,
  generating: false,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const generateReport = createAsyncThunk<
  Report,
  GenerateReportRequest
>(
  'reports/generateReport',
  async (payload) => {
    return ghgApi.generateReport(payload);
  },
);

export const fetchReports = createAsyncThunk<
  Report[],
  string
>(
  'reports/fetchReports',
  async (inventoryId) => {
    return ghgApi.getReports(inventoryId);
  },
);

export const fetchReport = createAsyncThunk<
  Report,
  string
>(
  'reports/fetchReport',
  async (reportId) => {
    return ghgApi.getReport(reportId);
  },
);

export const fetchDisclosures = createAsyncThunk<
  Disclosure[],
  string
>(
  'reports/fetchDisclosures',
  async (inventoryId) => {
    return ghgApi.getDisclosures(inventoryId);
  },
);

export const fetchCompleteness = createAsyncThunk<
  CompletenessResult,
  string
>(
  'reports/fetchCompleteness',
  async (inventoryId) => {
    return ghgApi.getCompleteness(inventoryId);
  },
);

export const exportData = createAsyncThunk<
  ExportResult,
  ExportDataRequest
>(
  'reports/exportData',
  async (payload) => {
    return ghgApi.exportData(payload);
  },
);

export const downloadReport = createAsyncThunk<
  { reportId: string; blob: Blob },
  string
>(
  'reports/downloadReport',
  async (reportId) => {
    const blob = await ghgApi.downloadReport(reportId);
    return { reportId, blob };
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const reportsSlice = createSlice({
  name: 'reports',
  initialState,
  reducers: {
    clearReports: () => initialState,
    clearReportsError: (state) => {
      state.error = null;
    },
    setReportFormat: (state, action: PayloadAction<ReportFormat>) => {
      // Store last-selected format for convenience
      void action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- generateReport --
      .addCase(generateReport.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(generateReport.fulfilled, (state, action) => {
        state.generating = false;
        state.reports.unshift(action.payload);
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.generating = false;
        state.error = action.error.message ?? 'Report generation failed';
      })

      // -- fetchReports --
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

      // -- fetchReport --
      .addCase(fetchReport.fulfilled, (state, action) => {
        const idx = state.reports.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) {
          state.reports[idx] = action.payload;
        } else {
          state.reports.push(action.payload);
        }
      })

      // -- fetchDisclosures --
      .addCase(fetchDisclosures.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchDisclosures.fulfilled, (state, action) => {
        state.loading = false;
        state.disclosures = action.payload;
      })
      .addCase(fetchDisclosures.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load disclosures';
      })

      // -- fetchCompleteness --
      .addCase(fetchCompleteness.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchCompleteness.fulfilled, (state, action) => {
        state.loading = false;
        state.completeness = action.payload;
      })
      .addCase(fetchCompleteness.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load completeness assessment';
      })

      // -- exportData --
      .addCase(exportData.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(exportData.fulfilled, (state) => {
        state.generating = false;
      })
      .addCase(exportData.rejected, (state, action) => {
        state.generating = false;
        state.error = action.error.message ?? 'Data export failed';
      });
  },
});

export const { clearReports, clearReportsError, setReportFormat } = reportsSlice.actions;
export default reportsSlice.reducer;
