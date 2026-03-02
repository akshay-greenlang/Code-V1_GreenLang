/**
 * Risk Redux Slice
 *
 * Manages risk assessment state: per-supplier assessments, heatmap data,
 * risk alerts, and trend analysis for the EUDR compliance platform.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import apiClient from '../../services/api';
import type {
  RiskAssessment,
  RiskAlert,
  RiskTrendPoint,
  RiskHeatmapEntry,
  PaginatedResponse,
} from '../../types';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface RiskState {
  assessment: RiskAssessment | null;
  heatmap: RiskHeatmapEntry[];
  alerts: RiskAlert[];
  trends: RiskTrendPoint[];
  alertsPagination: {
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
  };
  loading: boolean;
  heatmapLoading: boolean;
  alertsLoading: boolean;
  trendsLoading: boolean;
  error: string | null;
}

const initialState: RiskState = {
  assessment: null,
  heatmap: [],
  alerts: [],
  trends: [],
  alertsPagination: { total: 0, page: 1, per_page: 25, total_pages: 0 },
  loading: false,
  heatmapLoading: false,
  alertsLoading: false,
  trendsLoading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async Thunks
// ---------------------------------------------------------------------------

export const fetchRiskAssessment = createAsyncThunk(
  'risk/fetchAssessment',
  async (supplierId: string, { rejectWithValue }) => {
    try {
      return await apiClient.getRiskAssessment(supplierId);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch risk assessment';
      return rejectWithValue(message);
    }
  }
);

export const fetchRiskHeatmap = createAsyncThunk(
  'risk/fetchHeatmap',
  async (
    params: { commodity?: string; country?: string } | undefined,
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getRiskHeatmap(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch risk heatmap';
      return rejectWithValue(message);
    }
  }
);

export const fetchRiskAlerts = createAsyncThunk(
  'risk/fetchAlerts',
  async (
    params:
      | { severity?: string; is_resolved?: boolean; page?: number; per_page?: number }
      | undefined,
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getRiskAlerts(params);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch risk alerts';
      return rejectWithValue(message);
    }
  }
);

export const resolveRiskAlert = createAsyncThunk(
  'risk/resolveAlert',
  async (id: string, { rejectWithValue }) => {
    try {
      return await apiClient.resolveRiskAlert(id);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to resolve alert';
      return rejectWithValue(message);
    }
  }
);

export const fetchRiskTrends = createAsyncThunk(
  'risk/fetchTrends',
  async (
    params: { supplierId?: string; period?: string } | undefined,
    { rejectWithValue }
  ) => {
    try {
      return await apiClient.getRiskTrends(params?.supplierId, params?.period);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : 'Failed to fetch risk trends';
      return rejectWithValue(message);
    }
  }
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const riskSlice = createSlice({
  name: 'risk',
  initialState,
  reducers: {
    clearRiskError(state) {
      state.error = null;
    },
    clearAssessment(state) {
      state.assessment = null;
    },
  },
  extraReducers: (builder) => {
    // Assessment
    builder
      .addCase(fetchRiskAssessment.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchRiskAssessment.fulfilled, (state, action) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(fetchRiskAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
      });

    // Heatmap
    builder
      .addCase(fetchRiskHeatmap.pending, (state) => {
        state.heatmapLoading = true;
        state.error = null;
      })
      .addCase(fetchRiskHeatmap.fulfilled, (state, action) => {
        state.heatmapLoading = false;
        state.heatmap = action.payload;
      })
      .addCase(fetchRiskHeatmap.rejected, (state, action) => {
        state.heatmapLoading = false;
        state.error = action.payload as string;
      });

    // Alerts
    builder
      .addCase(fetchRiskAlerts.pending, (state) => {
        state.alertsLoading = true;
        state.error = null;
      })
      .addCase(fetchRiskAlerts.fulfilled, (state, action) => {
        state.alertsLoading = false;
        const payload = action.payload as PaginatedResponse<RiskAlert>;
        state.alerts = payload.items;
        state.alertsPagination = {
          total: payload.total,
          page: payload.page,
          per_page: payload.per_page,
          total_pages: payload.total_pages,
        };
      })
      .addCase(fetchRiskAlerts.rejected, (state, action) => {
        state.alertsLoading = false;
        state.error = action.payload as string;
      });

    // Resolve alert
    builder
      .addCase(resolveRiskAlert.fulfilled, (state, action) => {
        const idx = state.alerts.findIndex(
          (a) => a.id === action.payload.id
        );
        if (idx !== -1) state.alerts[idx] = action.payload;
      })
      .addCase(resolveRiskAlert.rejected, (state, action) => {
        state.error = action.payload as string;
      });

    // Trends
    builder
      .addCase(fetchRiskTrends.pending, (state) => {
        state.trendsLoading = true;
        state.error = null;
      })
      .addCase(fetchRiskTrends.fulfilled, (state, action) => {
        state.trendsLoading = false;
        state.trends = action.payload;
      })
      .addCase(fetchRiskTrends.rejected, (state, action) => {
        state.trendsLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const { clearRiskError, clearAssessment } = riskSlice.actions;

export default riskSlice.reducer;
