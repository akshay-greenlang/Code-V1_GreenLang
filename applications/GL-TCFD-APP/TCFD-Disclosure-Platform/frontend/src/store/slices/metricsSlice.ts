import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ClimateMetric, ClimateTarget, TargetProgress, EmissionsSummary, PeerBenchmarkData, PaginatedResponse } from '../../types';
import { metricsApi } from '../../services/api';
import type { RootState } from '../index';

interface MetricsState {
  metrics: ClimateMetric[];
  targets: ClimateTarget[];
  targetProgress: Record<string, TargetProgress>;
  emissionsSummary: EmissionsSummary | null;
  intensityTrend: { year: number; revenue_intensity: number; employee_intensity: number }[];
  peerBenchmarks: PeerBenchmarkData[];
  industryMetrics: { metric: string; value: number; unit: string; industry_avg: number; percentile: number }[];
  loading: boolean;
  error: string | null;
}

const initialState: MetricsState = {
  metrics: [],
  targets: [],
  targetProgress: {},
  emissionsSummary: null,
  intensityTrend: [],
  peerBenchmarks: [],
  industryMetrics: [],
  loading: false,
  error: null,
};

export const fetchMetrics = createAsyncThunk(
  'metrics/fetchMetrics',
  async ({ orgId, params }: { orgId: string; params?: { category?: string; year?: number } }) =>
    metricsApi.getMetrics(orgId, params)
);

export const fetchTargets = createAsyncThunk(
  'metrics/fetchTargets',
  async (orgId: string) => metricsApi.getTargets(orgId)
);

export const fetchTargetProgress = createAsyncThunk(
  'metrics/fetchTargetProgress',
  async (targetId: string) => metricsApi.getTargetProgress(targetId)
);

export const fetchEmissionsSummary = createAsyncThunk(
  'metrics/fetchEmissionsSummary',
  async ({ orgId, year }: { orgId: string; year?: number }) =>
    metricsApi.getEmissionsSummary(orgId, year)
);

export const fetchIntensityTrend = createAsyncThunk(
  'metrics/fetchIntensityTrend',
  async (orgId: string) => metricsApi.getIntensityTrend(orgId)
);

export const fetchPeerBenchmarks = createAsyncThunk(
  'metrics/fetchPeerBenchmarks',
  async (orgId: string) => metricsApi.getPeerBenchmark(orgId)
);

export const fetchIndustryMetrics = createAsyncThunk(
  'metrics/fetchIndustryMetrics',
  async (orgId: string) => metricsApi.getIndustryMetrics(orgId)
);

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMetrics.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchMetrics.fulfilled, (state, action: PayloadAction<PaginatedResponse<ClimateMetric>>) => {
        state.loading = false;
        state.metrics = action.payload.items;
      })
      .addCase(fetchMetrics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch metrics';
      })
      .addCase(fetchTargets.fulfilled, (state, action: PayloadAction<ClimateTarget[]>) => {
        state.targets = action.payload;
      })
      .addCase(fetchTargetProgress.fulfilled, (state, action: PayloadAction<TargetProgress>) => {
        state.targetProgress[action.payload.target_id] = action.payload;
      })
      .addCase(fetchEmissionsSummary.fulfilled, (state, action: PayloadAction<EmissionsSummary>) => {
        state.emissionsSummary = action.payload;
      })
      .addCase(fetchIntensityTrend.fulfilled, (state, action) => {
        state.intensityTrend = action.payload;
      })
      .addCase(fetchPeerBenchmarks.fulfilled, (state, action: PayloadAction<PeerBenchmarkData[]>) => {
        state.peerBenchmarks = action.payload;
      })
      .addCase(fetchIndustryMetrics.fulfilled, (state, action) => {
        state.industryMetrics = action.payload;
      });
  },
});

export const { clearError } = metricsSlice.actions;
export const selectMetrics = (state: RootState) => state.metrics.metrics;
export const selectTargets = (state: RootState) => state.metrics.targets;
export const selectTargetProgress = (state: RootState) => state.metrics.targetProgress;
export const selectEmissionsSummary = (state: RootState) => state.metrics.emissionsSummary;
export const selectIntensityTrend = (state: RootState) => state.metrics.intensityTrend;
export const selectPeerBenchmarks = (state: RootState) => state.metrics.peerBenchmarks;
export const selectIndustryMetrics = (state: RootState) => state.metrics.industryMetrics;
export const selectMetricsLoading = (state: RootState) => state.metrics.loading;
export default metricsSlice.reducer;
