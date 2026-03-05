/**
 * Reports Redux Slice
 *
 * Manages ISO 14064-1 report generation, mandatory element tracking,
 * and data export functionality per Clause 9 requirements.
 *
 * Async thunks:
 *   - generateReport: Generate an ISO 14064-1 compliant report
 *   - fetchReports: Load report history for an inventory
 *   - fetchReport: Load a single report by ID
 *   - fetchMandatoryElements: Load mandatory element completion status
 *   - exportData: Export inventory data in specified format
 *   - downloadReport: Download a generated report file
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  ReportsState,
  ISOReport,
  MandatoryElement,
  ExportResult,
  GenerateReportRequest,
  ExportDataRequest,
  ReportFormat,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: ReportsState = {
  reports: [],
  mandatoryElements: [],
  completeness_pct: 0,
  generating: false,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const generateReport = createAsyncThunk<
  ISOReport,
  GenerateReportRequest
>(
  'reports/generateReport',
  async (payload) => {
    return iso14064Api.generateReport(payload);
  },
);

export const fetchReports = createAsyncThunk<
  ISOReport[],
  string
>(
  'reports/fetchReports',
  async (inventoryId) => {
    return iso14064Api.getReports(inventoryId);
  },
);

export const fetchReport = createAsyncThunk<
  ISOReport,
  string
>(
  'reports/fetchReport',
  async (reportId) => {
    return iso14064Api.getReport(reportId);
  },
);

export const fetchMandatoryElements = createAsyncThunk<
  MandatoryElement[],
  string
>(
  'reports/fetchMandatoryElements',
  async (inventoryId) => {
    return iso14064Api.getMandatoryElements(inventoryId);
  },
);

export const exportData = createAsyncThunk<
  ExportResult,
  ExportDataRequest
>(
  'reports/exportData',
  async (payload) => {
    return iso14064Api.exportData(payload);
  },
);

export const downloadReport = createAsyncThunk<
  { reportId: string; blob: Blob },
  string
>(
  'reports/downloadReport',
  async (reportId) => {
    const blob = await iso14064Api.downloadReport(reportId);
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

      // -- fetchMandatoryElements --
      .addCase(fetchMandatoryElements.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchMandatoryElements.fulfilled, (state, action) => {
        state.loading = false;
        state.mandatoryElements = action.payload;
        const completed = action.payload.filter((e) => e.complete).length;
        const total = action.payload.length;
        state.completeness_pct = total > 0 ? (completed / total) * 100 : 0;
      })
      .addCase(fetchMandatoryElements.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load mandatory elements';
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
