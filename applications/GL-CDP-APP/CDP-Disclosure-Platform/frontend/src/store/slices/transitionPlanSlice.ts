/**
 * Transition Plan Redux Slice
 *
 * Manages 1.5C transition plan state: plan creation/update,
 * milestones, pathway points, and SBTi alignment.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  TransitionPlanState,
  TransitionPlan,
  TransitionMilestone,
  PathwayPoint,
  CreateTransitionPlanRequest,
  AddMilestoneRequest,
} from '../../types';
import { cdpApi } from '../../services/api';

const initialState: TransitionPlanState = {
  plan: null,
  pathway: [],
  loading: false,
  error: null,
};

export const fetchTransitionPlan = createAsyncThunk<TransitionPlan, string>(
  'transitionPlan/fetch',
  async (orgId) => cdpApi.getTransitionPlan(orgId),
);

export const createTransitionPlan = createAsyncThunk<
  TransitionPlan,
  { orgId: string; payload: CreateTransitionPlanRequest }
>(
  'transitionPlan/create',
  async ({ orgId, payload }) => cdpApi.createTransitionPlan(orgId, payload),
);

export const updateTransitionPlan = createAsyncThunk<
  TransitionPlan,
  { orgId: string; payload: Partial<CreateTransitionPlanRequest> }
>(
  'transitionPlan/update',
  async ({ orgId, payload }) => cdpApi.updateTransitionPlan(orgId, payload),
);

export const addMilestone = createAsyncThunk<
  TransitionMilestone,
  { orgId: string; payload: AddMilestoneRequest }
>(
  'transitionPlan/addMilestone',
  async ({ orgId, payload }) => cdpApi.addMilestone(orgId, payload),
);

export const fetchPathway = createAsyncThunk<PathwayPoint[], string>(
  'transitionPlan/fetchPathway',
  async (orgId) => cdpApi.getPathway(orgId),
);

export const checkSBTi = createAsyncThunk<
  { aligned: boolean; status: string; details: string },
  string
>(
  'transitionPlan/checkSBTi',
  async (orgId) => cdpApi.checkSBTiAlignment(orgId),
);

const transitionPlanSlice = createSlice({
  name: 'transitionPlan',
  initialState,
  reducers: {
    clearTransitionPlan: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchTransitionPlan.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTransitionPlan.fulfilled, (state, action) => {
        state.loading = false;
        state.plan = action.payload;
      })
      .addCase(fetchTransitionPlan.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load transition plan';
      })
      .addCase(createTransitionPlan.fulfilled, (state, action) => {
        state.plan = action.payload;
      })
      .addCase(updateTransitionPlan.fulfilled, (state, action) => {
        state.plan = action.payload;
      })
      .addCase(addMilestone.fulfilled, (state, action) => {
        if (state.plan) {
          state.plan.milestones.push(action.payload);
        }
      })
      .addCase(fetchPathway.fulfilled, (state, action) => {
        state.pathway = action.payload;
      })
      .addCase(checkSBTi.fulfilled, (state, action) => {
        if (state.plan) {
          state.plan.sbti_aligned = action.payload.aligned;
          state.plan.sbti_status = action.payload.status;
        }
      });
  },
});

export const { clearTransitionPlan } = transitionPlanSlice.actions;
export default transitionPlanSlice.reducer;
