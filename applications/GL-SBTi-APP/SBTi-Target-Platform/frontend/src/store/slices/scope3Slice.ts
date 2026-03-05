import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Scope3Screening } from '../../types';
import { scope3Api } from '../../services/api';
import type { RootState } from '../index';

interface Scope3State {
  screening: Scope3Screening | null;
  coverage: { total_pct: number; included_categories: number[]; excluded_categories: number[]; two_thirds_met: boolean } | null;
  hotspots: { category_number: number; category_name: string; emissions: number; significance: string; hotspot_rank: number }[];
  loading: boolean;
  error: string | null;
}

const initialState: Scope3State = {
  screening: null,
  coverage: null,
  hotspots: [],
  loading: false,
  error: null,
};

export const fetchScope3Screening = createAsyncThunk(
  'scope3/fetchScreening',
  async (orgId: string) => scope3Api.getTriggerAssessment(orgId)
);

export const runScope3Trigger = createAsyncThunk(
  'scope3/runTrigger',
  async (orgId: string) => scope3Api.runTriggerAssessment(orgId)
);

export const fetchScope3Coverage = createAsyncThunk(
  'scope3/fetchCoverage',
  async (orgId: string) => scope3Api.getCoverage(orgId)
);

export const fetchScope3Hotspots = createAsyncThunk(
  'scope3/fetchHotspots',
  async (orgId: string) => scope3Api.getHotspots(orgId)
);

const scope3Slice = createSlice({
  name: 'scope3',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchScope3Screening.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchScope3Screening.fulfilled, (state, action: PayloadAction<Scope3Screening>) => {
        state.loading = false;
        state.screening = action.payload;
      })
      .addCase(fetchScope3Screening.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch Scope 3 screening';
      })
      .addCase(runScope3Trigger.fulfilled, (state, action: PayloadAction<Scope3Screening>) => {
        state.screening = action.payload;
      })
      .addCase(fetchScope3Coverage.fulfilled, (state, action) => {
        state.coverage = action.payload;
      })
      .addCase(fetchScope3Hotspots.fulfilled, (state, action) => {
        state.hotspots = action.payload;
      });
  },
});

export const { clearError } = scope3Slice.actions;
export const selectScope3Screening = (state: RootState) => state.scope3.screening;
export const selectScope3Coverage = (state: RootState) => state.scope3.coverage;
export const selectScope3Hotspots = (state: RootState) => state.scope3.hotspots;
export const selectScope3Loading = (state: RootState) => state.scope3.loading;
export default scope3Slice.reducer;
