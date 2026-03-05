import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { ISSBMapping, DualComplianceScore, MigrationChecklistItem } from '../../types';
import { issbApi } from '../../services/api';
import type { RootState } from '../index';

interface ISSBState {
  mappings: ISSBMapping[];
  dualScorecard: DualComplianceScore[];
  migrationChecklist: MigrationChecklistItem[];
  loading: boolean;
  error: string | null;
}

const initialState: ISSBState = {
  mappings: [],
  dualScorecard: [],
  migrationChecklist: [],
  loading: false,
  error: null,
};

export const fetchISSBMappings = createAsyncThunk(
  'issb/fetchMappings',
  async (orgId: string) => issbApi.getMappings(orgId)
);

export const fetchDualScorecard = createAsyncThunk(
  'issb/fetchDualScorecard',
  async (orgId: string) => issbApi.getDualScorecard(orgId)
);

export const fetchMigrationChecklist = createAsyncThunk(
  'issb/fetchMigrationChecklist',
  async (orgId: string) => issbApi.getMigrationChecklist(orgId)
);

export const updateChecklistItem = createAsyncThunk(
  'issb/updateChecklistItem',
  async ({ id, data }: { id: string; data: Partial<MigrationChecklistItem> }) =>
    issbApi.updateChecklistItem(id, data)
);

const issbSlice = createSlice({
  name: 'issb',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchISSBMappings.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchISSBMappings.fulfilled, (state, action: PayloadAction<ISSBMapping[]>) => {
        state.loading = false;
        state.mappings = action.payload;
      })
      .addCase(fetchISSBMappings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch ISSB mappings';
      })
      .addCase(fetchDualScorecard.fulfilled, (state, action: PayloadAction<DualComplianceScore[]>) => {
        state.dualScorecard = action.payload;
      })
      .addCase(fetchMigrationChecklist.fulfilled, (state, action: PayloadAction<MigrationChecklistItem[]>) => {
        state.migrationChecklist = action.payload;
      })
      .addCase(updateChecklistItem.fulfilled, (state, action: PayloadAction<MigrationChecklistItem>) => {
        const idx = state.migrationChecklist.findIndex((i) => i.id === action.payload.id);
        if (idx >= 0) state.migrationChecklist[idx] = action.payload;
      });
  },
});

export const { clearError } = issbSlice.actions;
export const selectISSBMappings = (state: RootState) => state.issb.mappings;
export const selectDualScorecard = (state: RootState) => state.issb.dualScorecard;
export const selectMigrationChecklist = (state: RootState) => state.issb.migrationChecklist;
export const selectISSBLoading = (state: RootState) => state.issb.loading;
export default issbSlice.reducer;
