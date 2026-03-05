import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { DelegatedActVersion, RegulatoryUpdate, OmnibusImpact } from '../../types';
import { regulatoryApi } from '../../services/api';
import type { RootState } from '../index';

interface RegulatoryState {
  delegatedActs: DelegatedActVersion[];
  updates: RegulatoryUpdate[];
  omnibusImpacts: OmnibusImpact[];
  applicableVersion: DelegatedActVersion | null;
  loading: boolean;
  error: string | null;
}

const initialState: RegulatoryState = {
  delegatedActs: [],
  updates: [],
  omnibusImpacts: [],
  applicableVersion: null,
  loading: false,
  error: null,
};

export const fetchDelegatedActs = createAsyncThunk(
  'regulatory/delegatedActs',
  async () => regulatoryApi.delegatedActs()
);

export const fetchRegulatoryUpdates = createAsyncThunk(
  'regulatory/updates',
  async (limit?: number) => regulatoryApi.updates(limit)
);

export const fetchOmnibusImpact = createAsyncThunk(
  'regulatory/omnibus',
  async (orgId: string) => regulatoryApi.omnibusImpact(orgId)
);

export const fetchApplicableVersion = createAsyncThunk(
  'regulatory/applicable',
  async (orgId: string) => regulatoryApi.applicableVersion(orgId)
);

const regulatorySlice = createSlice({
  name: 'regulatory',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDelegatedActs.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchDelegatedActs.fulfilled, (state, action: PayloadAction<DelegatedActVersion[]>) => {
        state.loading = false;
        state.delegatedActs = action.payload;
      })
      .addCase(fetchDelegatedActs.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch delegated acts';
      })
      .addCase(fetchRegulatoryUpdates.fulfilled, (state, action) => { state.updates = action.payload; })
      .addCase(fetchOmnibusImpact.fulfilled, (state, action) => { state.omnibusImpacts = action.payload; })
      .addCase(fetchApplicableVersion.fulfilled, (state, action) => { state.applicableVersion = action.payload; });
  },
});

export const { clearError } = regulatorySlice.actions;
export const selectDelegatedActs = (state: RootState) => state.regulatory.delegatedActs;
export const selectRegulatoryUpdates = (state: RootState) => state.regulatory.updates;
export const selectOmnibusImpacts = (state: RootState) => state.regulatory.omnibusImpacts;
export const selectApplicableVersion = (state: RootState) => state.regulatory.applicableVersion;
export const selectRegulatoryLoading = (state: RootState) => state.regulatory.loading;
export default regulatorySlice.reducer;
