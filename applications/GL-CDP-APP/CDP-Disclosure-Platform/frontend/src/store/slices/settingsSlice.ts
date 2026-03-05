/**
 * Settings Redux Slice
 *
 * Manages application settings state: organization profile,
 * team members, MRV connections, and notification preferences.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { SettingsState, OrganizationSettings, UpdateSettingsRequest } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: SettingsState = {
  settings: null,
  loading: false,
  saving: false,
  error: null,
};

export const fetchSettings = createAsyncThunk<OrganizationSettings, string>(
  'settings/fetch',
  async (orgId) => cdpApi.getSettings(orgId),
);

export const updateSettings = createAsyncThunk<
  OrganizationSettings,
  { orgId: string; payload: UpdateSettingsRequest }
>(
  'settings/update',
  async ({ orgId, payload }) => cdpApi.updateSettings(orgId, payload),
);

export const addTeamMember = createAsyncThunk<
  void,
  { orgId: string; member: { name: string; email: string; role: string } }
>(
  'settings/addTeamMember',
  async ({ orgId, member }) => cdpApi.addTeamMember(orgId, member),
);

export const removeTeamMember = createAsyncThunk<
  string,
  { orgId: string; memberId: string }
>(
  'settings/removeTeamMember',
  async ({ orgId, memberId }) => {
    await cdpApi.removeTeamMember(orgId, memberId);
    return memberId;
  },
);

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    clearSettings: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSettings.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSettings.fulfilled, (state, action) => {
        state.loading = false;
        state.settings = action.payload;
      })
      .addCase(fetchSettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load settings';
      })
      .addCase(updateSettings.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(updateSettings.fulfilled, (state, action) => {
        state.saving = false;
        state.settings = action.payload;
      })
      .addCase(updateSettings.rejected, (state, action) => {
        state.saving = false;
        state.error = action.error.message ?? 'Failed to save settings';
      })
      .addCase(removeTeamMember.fulfilled, (state, action) => {
        if (state.settings) {
          state.settings.team_members = state.settings.team_members.filter(
            (m) => m.id !== action.payload,
          );
        }
      });
  },
});

export const { clearSettings } = settingsSlice.actions;
export default settingsSlice.reducer;
