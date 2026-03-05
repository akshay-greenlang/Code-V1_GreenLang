import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DataQualityScore, DimensionScore, EvidenceTracker, ImprovementPlan } from '../../types';
import { dataQualityApi } from '../../services/api';
import type { RootState } from '../index';

interface DataQualityState {
  score: DataQualityScore | null;
  dimensions: DimensionScore[];
  evidence: EvidenceTracker | null;
  improvementPlan: ImprovementPlan | null;
  loading: boolean;
  error: string | null;
}

const initialState: DataQualityState = {
  score: null,
  dimensions: [],
  evidence: null,
  improvementPlan: null,
  loading: false,
  error: null,
};

export const assessDataQuality = createAsyncThunk(
  'dataQuality/assess',
  async (orgId: string) => dataQualityApi.assess(orgId)
);

export const fetchDataQualityDashboard = createAsyncThunk(
  'dataQuality/dashboard',
  async (orgId: string) => dataQualityApi.dashboard(orgId)
);

export const fetchDimensions = createAsyncThunk(
  'dataQuality/dimensions',
  async (orgId: string) => dataQualityApi.dimensions(orgId)
);

export const fetchEvidence = createAsyncThunk(
  'dataQuality/evidence',
  async (orgId: string) => dataQualityApi.evidence(orgId)
);

export const fetchImprovementPlan = createAsyncThunk(
  'dataQuality/improvement',
  async (orgId: string) => dataQualityApi.improvementPlan(orgId)
);

const dataQualitySlice = createSlice({
  name: 'dataQuality',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(assessDataQuality.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(assessDataQuality.fulfilled, (state, action: PayloadAction<DataQualityScore>) => {
        state.loading = false;
        state.score = action.payload;
      })
      .addCase(assessDataQuality.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to assess data quality';
      })
      .addCase(fetchDataQualityDashboard.fulfilled, (state, action) => { state.score = action.payload; })
      .addCase(fetchDimensions.fulfilled, (state, action) => { state.dimensions = action.payload; })
      .addCase(fetchEvidence.fulfilled, (state, action) => { state.evidence = action.payload; })
      .addCase(fetchImprovementPlan.fulfilled, (state, action) => { state.improvementPlan = action.payload; });
  },
});

export const { clearError } = dataQualitySlice.actions;
export const selectDataQualityScore = (state: RootState) => state.dataQuality.score;
export const selectDimensions = (state: RootState) => state.dataQuality.dimensions;
export const selectEvidenceTracker = (state: RootState) => state.dataQuality.evidence;
export const selectImprovementPlan = (state: RootState) => state.dataQuality.improvementPlan;
export const selectDataQualityLoading = (state: RootState) => state.dataQuality.loading;
export default dataQualitySlice.reducer;
