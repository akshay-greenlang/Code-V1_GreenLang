/**
 * GL-VCCI Frontend - Settings Redux Slice
 *
 * Manages persistent user settings including profile, preferences,
 * notification toggles, and dashboard defaults. Syncs with the
 * backend settings API for cross-session persistence.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';

// ==============================================================================
// Type Definitions
// ==============================================================================

export interface UserProfile {
  name: string;
  email: string;
  company: string;
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  language: 'en' | 'es' | 'fr' | 'de';
  timezone: string;
  dateFormat: string;
  currency: string;
}

export interface NotificationSettings {
  email: boolean;
  inApp: boolean;
  weeklyReports: boolean;
}

export interface DashboardDefaults {
  defaultView: 'overview' | 'detailed';
  defaultPeriod: '30d' | '90d' | '1y' | 'ytd';
}

// ==============================================================================
// State Interface
// ==============================================================================

interface SettingsState {
  userProfile: UserProfile;
  preferences: UserPreferences;
  notifications: NotificationSettings;
  dashboard: DashboardDefaults;
  loading: boolean;
  error: string | null;
  lastSaved: string | null;
}

const initialState: SettingsState = {
  userProfile: {
    name: '',
    email: '',
    company: '',
  },
  preferences: {
    theme: 'light',
    language: 'en',
    timezone: 'UTC',
    dateFormat: 'YYYY-MM-DD',
    currency: 'USD',
  },
  notifications: {
    email: true,
    inApp: true,
    weeklyReports: true,
  },
  dashboard: {
    defaultView: 'overview',
    defaultPeriod: '90d',
  },
  loading: false,
  error: null,
  lastSaved: null,
};

// ==============================================================================
// Async Thunks
// ==============================================================================

export const fetchSettings = createAsyncThunk(
  'settings/fetch',
  async () => {
    const response = await api.getUserSettings();
    return response;
  }
);

export const updateSettings = createAsyncThunk(
  'settings/update',
  async (settings: {
    userProfile?: Partial<UserProfile>;
    preferences?: Partial<UserPreferences>;
    notifications?: Partial<NotificationSettings>;
    dashboard?: Partial<DashboardDefaults>;
  }) => {
    const response = await api.updateUserSettings(settings);
    return response;
  }
);

// ==============================================================================
// Slice
// ==============================================================================

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    setUserProfile: (state, action: PayloadAction<Partial<UserProfile>>) => {
      state.userProfile = { ...state.userProfile, ...action.payload };
    },
    setPreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      state.preferences = { ...state.preferences, ...action.payload };
    },
    setNotifications: (state, action: PayloadAction<Partial<NotificationSettings>>) => {
      state.notifications = { ...state.notifications, ...action.payload };
    },
    setDashboardDefaults: (state, action: PayloadAction<Partial<DashboardDefaults>>) => {
      state.dashboard = { ...state.dashboard, ...action.payload };
    },
    clearSettingsError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch settings
    builder
      .addCase(fetchSettings.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSettings.fulfilled, (state, action) => {
        state.loading = false;
        if (action.payload.userProfile) {
          state.userProfile = action.payload.userProfile;
        }
        if (action.payload.preferences) {
          state.preferences = { ...state.preferences, ...action.payload.preferences };
        }
        if (action.payload.notifications) {
          state.notifications = { ...state.notifications, ...action.payload.notifications };
        }
        if (action.payload.dashboard) {
          state.dashboard = { ...state.dashboard, ...action.payload.dashboard };
        }
      })
      .addCase(fetchSettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch settings';
      });

    // Update settings
    builder
      .addCase(updateSettings.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateSettings.fulfilled, (state, action) => {
        state.loading = false;
        state.lastSaved = new Date().toISOString();
        if (action.payload.userProfile) {
          state.userProfile = { ...state.userProfile, ...action.payload.userProfile };
        }
        if (action.payload.preferences) {
          state.preferences = { ...state.preferences, ...action.payload.preferences };
        }
        if (action.payload.notifications) {
          state.notifications = { ...state.notifications, ...action.payload.notifications };
        }
        if (action.payload.dashboard) {
          state.dashboard = { ...state.dashboard, ...action.payload.dashboard };
        }
      })
      .addCase(updateSettings.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to update settings';
      });
  },
});

export const {
  setUserProfile,
  setPreferences,
  setNotifications,
  setDashboardDefaults,
  clearSettingsError,
} = settingsSlice.actions;

export default settingsSlice.reducer;
