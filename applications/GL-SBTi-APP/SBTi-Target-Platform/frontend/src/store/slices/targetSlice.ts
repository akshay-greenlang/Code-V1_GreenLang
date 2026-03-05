import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { Target, PaginatedResponse } from '../../types';
import { targetApi } from '../../services/api';
import type { RootState } from '../index';

interface TargetState {
  targets: Target[];
  selectedTarget: Target | null;
  total: number;
  loading: boolean;
  saving: boolean;
  error: string | null;
}

const initialState: TargetState = {
  targets: [],
  selectedTarget: null,
  total: 0,
  loading: false,
  saving: false,
  error: null,
};

export const fetchTargets = createAsyncThunk(
  'target/fetchTargets',
  async (orgId: string) => targetApi.getTargets(orgId)
);

export const fetchTarget = createAsyncThunk(
  'target/fetchTarget',
  async (id: string) => targetApi.getTarget(id)
);

export const createTarget = createAsyncThunk(
  'target/createTarget',
  async (data: Partial<Target>) => targetApi.createTarget(data)
);

export const updateTarget = createAsyncThunk(
  'target/updateTarget',
  async ({ id, data }: { id: string; data: Partial<Target> }) => targetApi.updateTarget(id, data)
);

export const deleteTarget = createAsyncThunk(
  'target/deleteTarget',
  async (id: string) => { await targetApi.deleteTarget(id); return id; }
);

export const submitTarget = createAsyncThunk(
  'target/submitTarget',
  async (id: string) => targetApi.submitTarget(id)
);

const targetSlice = createSlice({
  name: 'target',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    clearSelectedTarget(state) { state.selectedTarget = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTargets.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchTargets.fulfilled, (state, action: PayloadAction<PaginatedResponse<Target>>) => {
        state.loading = false;
        state.targets = action.payload.items;
        state.total = action.payload.total;
      })
      .addCase(fetchTargets.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch targets';
      })
      .addCase(fetchTarget.fulfilled, (state, action: PayloadAction<Target>) => {
        state.selectedTarget = action.payload;
      })
      .addCase(createTarget.pending, (state) => { state.saving = true; })
      .addCase(createTarget.fulfilled, (state, action: PayloadAction<Target>) => {
        state.saving = false;
        state.targets.push(action.payload);
      })
      .addCase(createTarget.rejected, (state, action) => {
        state.saving = false;
        state.error = action.error.message || 'Failed to create target';
      })
      .addCase(updateTarget.fulfilled, (state, action: PayloadAction<Target>) => {
        state.saving = false;
        const idx = state.targets.findIndex((t) => t.id === action.payload.id);
        if (idx >= 0) state.targets[idx] = action.payload;
        if (state.selectedTarget?.id === action.payload.id) state.selectedTarget = action.payload;
      })
      .addCase(deleteTarget.fulfilled, (state, action: PayloadAction<string>) => {
        state.targets = state.targets.filter((t) => t.id !== action.payload);
      })
      .addCase(submitTarget.fulfilled, (state, action: PayloadAction<Target>) => {
        const idx = state.targets.findIndex((t) => t.id === action.payload.id);
        if (idx >= 0) state.targets[idx] = action.payload;
      });
  },
});

export const { clearError, clearSelectedTarget } = targetSlice.actions;
export const selectTargets = (state: RootState) => state.target.targets;
export const selectSelectedTarget = (state: RootState) => state.target.selectedTarget;
export const selectTargetLoading = (state: RootState) => state.target.loading;
export const selectTargetSaving = (state: RootState) => state.target.saving;
export default targetSlice.reducer;
