import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';
import type { Report, ReportRequest, PaginatedResponse } from '../../types';

interface ReportsState {
  reports: Report[];
  selectedReport: Report | null;
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
  filters: {
    type?: string;
    status?: string[];
  };
  loading: boolean;
  generating: boolean;
  error: string | null;
}

const initialState: ReportsState = {
  reports: [],
  selectedReport: null,
  pagination: {
    page: 1,
    pageSize: 25,
    total: 0,
    totalPages: 0,
  },
  filters: {},
  loading: false,
  generating: false,
  error: null,
};

// Async thunks
export const fetchReports = createAsyncThunk(
  'reports/fetchAll',
  async (params?: {
    page?: number;
    pageSize?: number;
    type?: string;
    status?: string[];
  }) => {
    const response = await api.getReports(params);
    return response;
  }
);

export const fetchReport = createAsyncThunk(
  'reports/fetchOne',
  async (id: string) => {
    const report = await api.getReport(id);
    return report;
  }
);

export const generateReport = createAsyncThunk(
  'reports/generate',
  async (request: ReportRequest) => {
    const report = await api.generateReport(request);
    return report;
  }
);

export const downloadReport = createAsyncThunk(
  'reports/download',
  async (reportId: string) => {
    const blob = await api.downloadReport(reportId);

    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `report_${reportId}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    return reportId;
  }
);

export const deleteReport = createAsyncThunk(
  'reports/delete',
  async (id: string) => {
    await api.deleteReport(id);
    return id;
  }
);

const reportsSlice = createSlice({
  name: 'reports',
  initialState,
  reducers: {
    setPage: (state, action: PayloadAction<number>) => {
      state.pagination.page = action.payload;
    },
    setPageSize: (state, action: PayloadAction<number>) => {
      state.pagination.pageSize = action.payload;
    },
    setFilters: (state, action: PayloadAction<ReportsState['filters']>) => {
      state.filters = action.payload;
      state.pagination.page = 1;
    },
  },
  extraReducers: (builder) => {
    // Fetch all reports
    builder
      .addCase(fetchReports.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchReports.fulfilled, (state, action: PayloadAction<PaginatedResponse<Report>>) => {
        state.loading = false;
        state.reports = action.payload.data;
        state.pagination = action.payload.pagination;
      })
      .addCase(fetchReports.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch reports';
      });

    // Fetch single report
    builder
      .addCase(fetchReport.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchReport.fulfilled, (state, action: PayloadAction<Report>) => {
        state.loading = false;
        state.selectedReport = action.payload;
      })
      .addCase(fetchReport.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch report';
      });

    // Generate report
    builder
      .addCase(generateReport.pending, (state) => {
        state.generating = true;
        state.error = null;
      })
      .addCase(generateReport.fulfilled, (state, action: PayloadAction<Report>) => {
        state.generating = false;
        state.reports = [action.payload, ...state.reports];
        state.selectedReport = action.payload;
      })
      .addCase(generateReport.rejected, (state, action) => {
        state.generating = false;
        state.error = action.error.message || 'Failed to generate report';
      });

    // Download report
    builder
      .addCase(downloadReport.pending, (state) => {
        state.loading = true;
      })
      .addCase(downloadReport.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(downloadReport.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to download report';
      });

    // Delete report
    builder
      .addCase(deleteReport.fulfilled, (state, action: PayloadAction<string>) => {
        state.reports = state.reports.filter(r => r.id !== action.payload);
        state.pagination.total -= 1;
      });
  },
});

export const { setPage, setPageSize, setFilters } = reportsSlice.actions;
export default reportsSlice.reducer;
