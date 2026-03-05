import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { EconomicActivity, NACEMapping, ActivityStatistics, PaginatedResponse, ActivitySearchParams } from '../../types';
import { activitiesApi } from '../../services/api';
import type { RootState } from '../index';

interface ActivitiesState {
  activities: EconomicActivity[];
  selectedActivity: EconomicActivity | null;
  naceMappings: NACEMapping[];
  statistics: ActivityStatistics | null;
  total: number;
  page: number;
  perPage: number;
  loading: boolean;
  error: string | null;
}

const initialState: ActivitiesState = {
  activities: [],
  selectedActivity: null,
  naceMappings: [],
  statistics: null,
  total: 0,
  page: 1,
  perPage: 25,
  loading: false,
  error: null,
};

export const fetchActivities = createAsyncThunk(
  'activities/fetch',
  async (params: ActivitySearchParams) => activitiesApi.list(params)
);

export const fetchActivity = createAsyncThunk(
  'activities/fetchOne',
  async (id: string) => activitiesApi.get(id)
);

export const searchNACE = createAsyncThunk(
  'activities/searchNACE',
  async (code: string) => activitiesApi.searchByNACE(code)
);

export const fetchStatistics = createAsyncThunk(
  'activities/statistics',
  async (orgId: string) => activitiesApi.statistics(orgId)
);

const activitiesSlice = createSlice({
  name: 'activities',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    setPage(state, action: PayloadAction<number>) { state.page = action.payload; },
    setSelectedActivity(state, action: PayloadAction<EconomicActivity | null>) { state.selectedActivity = action.payload; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchActivities.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchActivities.fulfilled, (state, action: PayloadAction<PaginatedResponse<EconomicActivity>>) => {
        state.loading = false;
        state.activities = action.payload.items;
        state.total = action.payload.total;
        state.page = action.payload.page;
      })
      .addCase(fetchActivities.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch activities';
      })
      .addCase(fetchActivity.fulfilled, (state, action) => { state.selectedActivity = action.payload; })
      .addCase(searchNACE.fulfilled, (state, action) => { state.naceMappings = action.payload; })
      .addCase(fetchStatistics.fulfilled, (state, action) => { state.statistics = action.payload; });
  },
});

export const { clearError, setPage, setSelectedActivity } = activitiesSlice.actions;
export const selectActivities = (state: RootState) => state.activities.activities;
export const selectSelectedActivity = (state: RootState) => state.activities.selectedActivity;
export const selectNACEMappings = (state: RootState) => state.activities.naceMappings;
export const selectActivityStatistics = (state: RootState) => state.activities.statistics;
export const selectActivitiesLoading = (state: RootState) => state.activities.loading;
export default activitiesSlice.reducer;
