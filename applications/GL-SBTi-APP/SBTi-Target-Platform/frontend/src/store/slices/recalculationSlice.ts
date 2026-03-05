import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Recalculation, ThresholdCheck } from '../../types';
import { recalculationApi } from '../../services/api';
import type { RootState } from '../index';

interface RecalculationState {
  recalculations: Recalculation[];
  thresholdChecks: ThresholdCheck[];
  loading: boolean;
  error: string | null;
}

const initialState: RecalculationState = {
  recalculations: [],
  thresholdChecks: [],
  loading: false,
  error: null,
};

export const fetchThresholdChecks = createAsyncThunk(
  'recalculation/fetchThresholds',
  async (orgId: string) => recalculationApi.checkThresholds(orgId)
);

export const fetchRecalculations = createAsyncThunk(
  'recalculation/fetchAll',
  async (orgId: string) => recalculationApi.getRecalculations(orgId)
);

export const createRecalculation = createAsyncThunk(
  'recalculation/create',
  async (data: Partial<Recalculation>) => recalculationApi.createRecalculation(data)
);

export const approveRecalculation = createAsyncThunk(
  'recalculation/approve',
  async (id: string) => recalculationApi.approveRecalculation(id)
);

const recalculationSlice = createSlice({
  name: 'recalculation',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchThresholdChecks.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchThresholdChecks.fulfilled, (state, action: PayloadAction<ThresholdCheck[]>) => {
        state.loading = false;
        state.thresholdChecks = action.payload;
      })
      .addCase(fetchThresholdChecks.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to check thresholds';
      })
      .addCase(fetchRecalculations.fulfilled, (state, action: PayloadAction<Recalculation[]>) => {
        state.recalculations = action.payload;
      })
      .addCase(createRecalculation.fulfilled, (state, action: PayloadAction<Recalculation>) => {
        state.recalculations.push(action.payload);
      })
      .addCase(approveRecalculation.fulfilled, (state, action: PayloadAction<Recalculation>) => {
        const idx = state.recalculations.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) state.recalculations[idx] = action.payload;
      });
  },
});

export const { clearError } = recalculationSlice.actions;
export const selectRecalculations = (state: RootState) => state.recalculation.recalculations;
export const selectThresholdChecks = (state: RootState) => state.recalculation.thresholdChecks;
export const selectRecalculationLoading = (state: RootState) => state.recalculation.loading;
export default recalculationSlice.reducer;
