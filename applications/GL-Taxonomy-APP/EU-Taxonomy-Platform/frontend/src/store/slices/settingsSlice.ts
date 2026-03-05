import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { TaxonomySettings, ReportingPeriod, MRVMapping } from '../../types';
import { settingsApi } from '../../services/api';
import type { RootState } from '../index';

interface SettingsState {
  settings: TaxonomySettings | null;
  reportingPeriods: ReportingPeriod[];
  mrvMappings: MRVMapping[];
  loading: boolean;
  error: string | null;
}

const initialState: SettingsState = {
  settings: null,
  reportingPeriods: [],
  mrvMappings: [],
  loading: false,
  error: null,
};

export const fetchSettings = createAsyncThunk(
  'settings/fetch',
  async (orgId: string) => settingsApi.get(orgId)
);

export const updateSettings = createAsyncThunk(
  'settings/update',
  async ({ orgId, settings }: { orgId: string; settings: Partial<TaxonomySettings> }) =>
    settingsApi.update(orgId, settings)
);

export const fetchReportingPeriods = createAsyncThunk(
  'settings/periods',
  async (orgId: string) => settingsApi.reportingPeriods(orgId)
);

export const fetchMRVMappings = createAsyncThunk(
  'settings/mrvMapping',
  async (orgId: string) => settingsApi.mrvMapping(orgId)
);

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSettings.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchSettings.fulfilled, (state, action: PayloadAction<TaxonomySettings>) => {
        state.loading = false;
        state.settings = action.payload;
      })
      .addCase(fetchSettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch settings';
      })
      .addCase(updateSettings.fulfilled, (state, action) => { state.settings = action.payload; })
      .addCase(fetchReportingPeriods.fulfilled, (state, action) => { state.reportingPeriods = action.payload; })
      .addCase(fetchMRVMappings.fulfilled, (state, action) => { state.mrvMappings = action.payload; });
  },
});

export const { clearError } = settingsSlice.actions;
export const selectSettings = (state: RootState) => state.settings.settings;
export const selectReportingPeriods = (state: RootState) => state.settings.reportingPeriods;
export const selectMRVMappings = (state: RootState) => state.settings.mrvMappings;
export const selectSettingsLoading = (state: RootState) => state.settings.loading;
export default settingsSlice.reducer;
