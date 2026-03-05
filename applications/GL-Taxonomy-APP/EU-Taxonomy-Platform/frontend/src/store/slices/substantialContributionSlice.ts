import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { SCAssessment, SCCriteria, SCProfile, EnvironmentalObjective } from '../../types';
import { scApi } from '../../services/api';
import type { RootState } from '../index';

interface SCState {
  assessments: SCAssessment[];
  currentAssessment: SCAssessment | null;
  criteria: SCCriteria | null;
  profile: SCProfile | null;
  loading: boolean;
  error: string | null;
}

const initialState: SCState = {
  assessments: [],
  currentAssessment: null,
  criteria: null,
  profile: null,
  loading: false,
  error: null,
};

export const assessSC = createAsyncThunk(
  'sc/assess',
  async ({ activityId, objective }: { activityId: string; objective: EnvironmentalObjective }) =>
    scApi.assess(activityId, objective)
);

export const batchAssessSC = createAsyncThunk(
  'sc/batchAssess',
  async ({ orgId, objective }: { orgId: string; objective: EnvironmentalObjective }) =>
    scApi.batchAssess(orgId, objective)
);

export const fetchSCResults = createAsyncThunk(
  'sc/results',
  async (activityId: string) => scApi.getResults(activityId)
);

export const fetchSCCriteria = createAsyncThunk(
  'sc/criteria',
  async ({ activityId, objective }: { activityId: string; objective: EnvironmentalObjective }) =>
    scApi.getCriteria(activityId, objective)
);

export const fetchSCProfile = createAsyncThunk(
  'sc/profile',
  async (activityId: string) => scApi.getProfile(activityId)
);

const scSlice = createSlice({
  name: 'substantialContribution',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    setCurrentAssessment(state, action: PayloadAction<SCAssessment | null>) {
      state.currentAssessment = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(assessSC.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(assessSC.fulfilled, (state, action: PayloadAction<SCAssessment>) => {
        state.loading = false;
        state.currentAssessment = action.payload;
      })
      .addCase(assessSC.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to assess SC';
      })
      .addCase(batchAssessSC.fulfilled, (state, action) => { state.assessments = action.payload; state.loading = false; })
      .addCase(fetchSCResults.fulfilled, (state, action) => { state.assessments = action.payload; })
      .addCase(fetchSCCriteria.fulfilled, (state, action) => { state.criteria = action.payload; })
      .addCase(fetchSCProfile.fulfilled, (state, action) => { state.profile = action.payload; });
  },
});

export const { clearError, setCurrentAssessment } = scSlice.actions;
export const selectSCAssessments = (state: RootState) => state.substantialContribution.assessments;
export const selectCurrentSCAssessment = (state: RootState) => state.substantialContribution.currentAssessment;
export const selectSCCriteria = (state: RootState) => state.substantialContribution.criteria;
export const selectSCProfile = (state: RootState) => state.substantialContribution.profile;
export const selectSCLoading = (state: RootState) => state.substantialContribution.loading;
export default scSlice.reducer;
