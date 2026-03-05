import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Pathway, PathwayComparison } from '../../types';
import { pathwayApi } from '../../services/api';
import type { RootState } from '../index';

interface PathwayState {
  currentPathway: Pathway | null;
  comparisons: PathwayComparison[];
  loading: boolean;
  calculating: boolean;
  error: string | null;
}

const initialState: PathwayState = {
  currentPathway: null,
  comparisons: [],
  loading: false,
  calculating: false,
  error: null,
};

export const fetchPathway = createAsyncThunk(
  'pathway/fetchPathway',
  async (targetId: string) => pathwayApi.getPathway(targetId)
);

export const calculateACA = createAsyncThunk(
  'pathway/calculateACA',
  async (params: { base_year: number; target_year: number; base_emissions: number; alignment: string }) =>
    pathwayApi.calculateACA(params)
);

export const calculateSDA = createAsyncThunk(
  'pathway/calculateSDA',
  async (params: { sector: string; base_year: number; target_year: number; base_intensity: number; alignment: string }) =>
    pathwayApi.calculateSDA(params)
);

export const fetchComparisons = createAsyncThunk(
  'pathway/fetchComparisons',
  async (targetId: string) => pathwayApi.comparePathways(targetId)
);

const pathwaySlice = createSlice({
  name: 'pathway',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearPathway(state) { state.currentPathway = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchPathway.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchPathway.fulfilled, (state, action: PayloadAction<Pathway>) => {
        state.loading = false;
        state.currentPathway = action.payload;
      })
      .addCase(fetchPathway.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch pathway';
      })
      .addCase(calculateACA.pending, (state) => { state.calculating = true; })
      .addCase(calculateACA.fulfilled, (state, action: PayloadAction<Pathway>) => {
        state.calculating = false;
        state.currentPathway = action.payload;
      })
      .addCase(calculateACA.rejected, (state, action) => {
        state.calculating = false;
        state.error = action.error.message || 'Failed to calculate ACA pathway';
      })
      .addCase(calculateSDA.pending, (state) => { state.calculating = true; })
      .addCase(calculateSDA.fulfilled, (state, action: PayloadAction<Pathway>) => {
        state.calculating = false;
        state.currentPathway = action.payload;
      })
      .addCase(calculateSDA.rejected, (state, action) => {
        state.calculating = false;
        state.error = action.error.message || 'Failed to calculate SDA pathway';
      })
      .addCase(fetchComparisons.fulfilled, (state, action: PayloadAction<PathwayComparison[]>) => {
        state.comparisons = action.payload;
      });
  },
});

export const { clearError, clearPathway } = pathwaySlice.actions;
export const selectCurrentPathway = (state: RootState) => state.pathway.currentPathway;
export const selectPathwayComparisons = (state: RootState) => state.pathway.comparisons;
export const selectPathwayLoading = (state: RootState) => state.pathway.loading;
export const selectPathwayCalculating = (state: RootState) => state.pathway.calculating;
export default pathwaySlice.reducer;
