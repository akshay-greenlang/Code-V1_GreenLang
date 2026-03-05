import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { OrganizationSettings } from '../../types';
import { settingsApi } from '../../services/api';
import type { RootState } from '../index';

interface SettingsState {
  settings: OrganizationSettings | null;
  organizations: { id: string; name: string }[];
  activeOrgId: string;
  loading: boolean;
  saving: boolean;
  error: string | null;
}

const initialState: SettingsState = {
  settings: null,
  organizations: [],
  activeOrgId: 'org_default',
  loading: false,
  saving: false,
  error: null,
};

export const fetchSettings = createAsyncThunk(
  'settings/fetchSettings',
  async (orgId: string) => settingsApi.getSettings(orgId)
);

export const updateSettings = createAsyncThunk(
  'settings/updateSettings',
  async ({ orgId, data }: { orgId: string; data: Partial<OrganizationSettings> }) =>
    settingsApi.updateSettings(orgId, data)
);

export const fetchOrganizations = createAsyncThunk(
  'settings/fetchOrganizations',
  async () => settingsApi.getOrganizations()
);

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    setActiveOrgId(state, action: PayloadAction<string>) {
      state.activeOrgId = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSettings.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchSettings.fulfilled, (state, action: PayloadAction<OrganizationSettings>) => {
        state.loading = false;
        state.settings = action.payload;
      })
      .addCase(fetchSettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch settings';
      })
      .addCase(updateSettings.pending, (state) => { state.saving = true; })
      .addCase(updateSettings.fulfilled, (state, action: PayloadAction<OrganizationSettings>) => {
        state.saving = false;
        state.settings = action.payload;
      })
      .addCase(updateSettings.rejected, (state, action) => {
        state.saving = false;
        state.error = action.error.message || 'Failed to save settings';
      })
      .addCase(fetchOrganizations.fulfilled, (state, action) => {
        state.organizations = action.payload;
      });
  },
});

export const { clearError, setActiveOrgId } = settingsSlice.actions;
export const selectSettings = (state: RootState) => state.settings.settings;
export const selectOrganizations = (state: RootState) => state.settings.organizations;
export const selectActiveOrgId = (state: RootState) => state.settings.activeOrgId;
export const selectSettingsLoading = (state: RootState) => state.settings.loading;
export const selectSettingsSaving = (state: RootState) => state.settings.saving;
export default settingsSlice.reducer;
