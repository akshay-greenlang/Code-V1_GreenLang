import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { FLAGAssessment } from '../../types';
import { flagApi } from '../../services/api';
import type { RootState } from '../index';

interface FLAGState {
  assessment: FLAGAssessment | null;
  emissionsSplit: { flag_emissions: number; non_flag_emissions: number; flag_pct: number; by_commodity: { commodity: string; emissions: number }[] } | null;
  loading: boolean;
  error: string | null;
}

const initialState: FLAGState = {
  assessment: null,
  emissionsSplit: null,
  loading: false,
  error: null,
};

export const fetchFLAGAssessment = createAsyncThunk(
  'flag/fetchAssessment',
  async (orgId: string) => flagApi.getTriggerAssessment(orgId)
);

export const runFLAGTrigger = createAsyncThunk(
  'flag/runTrigger',
  async (orgId: string) => flagApi.runTriggerAssessment(orgId)
);

export const fetchEmissionsSplit = createAsyncThunk(
  'flag/fetchSplit',
  async (orgId: string) => flagApi.getEmissionsSplit(orgId)
);

const flagSlice = createSlice({
  name: 'flag',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchFLAGAssessment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchFLAGAssessment.fulfilled, (state, action: PayloadAction<FLAGAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(fetchFLAGAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch FLAG assessment';
      })
      .addCase(runFLAGTrigger.fulfilled, (state, action: PayloadAction<FLAGAssessment>) => {
        state.assessment = action.payload;
      })
      .addCase(fetchEmissionsSplit.fulfilled, (state, action) => {
        state.emissionsSplit = action.payload;
      });
  },
});

export const { clearError } = flagSlice.actions;
export const selectFLAGAssessment = (state: RootState) => state.flag.assessment;
export const selectEmissionsSplit = (state: RootState) => state.flag.emissionsSplit;
export const selectFLAGLoading = (state: RootState) => state.flag.loading;
export default flagSlice.reducer;
