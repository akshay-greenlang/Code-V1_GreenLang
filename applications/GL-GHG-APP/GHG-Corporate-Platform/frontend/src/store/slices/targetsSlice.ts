/**
 * Targets Redux Slice
 *
 * Manages emission reduction targets, progress tracking,
 * SBTi alignment checks, and forecast trajectories.
 *
 * Async thunks:
 *   - setTarget: Create a new reduction target
 *   - fetchTargets: Load all targets for an organization
 *   - fetchProgress: Load progress data for a specific target
 *   - checkSBTi: Check SBTi alignment for a target
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  TargetsState,
  Target,
  TargetProgress,
  SBTiAlignmentCheck,
  SetTargetRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: TargetsState = {
  targets: [],
  progress: {},
  sbtiCheck: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const setTarget = createAsyncThunk<
  Target,
  { orgId: string; payload: SetTargetRequest }
>(
  'targets/setTarget',
  async ({ orgId, payload }) => {
    return ghgApi.setTarget(orgId, payload);
  },
);

export const fetchTargets = createAsyncThunk<
  Target[],
  string
>(
  'targets/fetchTargets',
  async (orgId) => {
    return ghgApi.getTargets(orgId);
  },
);

export const fetchTargetProgress = createAsyncThunk<
  { targetId: string; progress: TargetProgress },
  { orgId: string; targetId: string }
>(
  'targets/fetchProgress',
  async ({ orgId, targetId }) => {
    const progress = await ghgApi.getTargetProgress(orgId, targetId);
    return { targetId, progress };
  },
);

export const checkSBTi = createAsyncThunk<
  SBTiAlignmentCheck,
  { orgId: string; targetId: string }
>(
  'targets/checkSBTi',
  async ({ orgId, targetId }) => {
    return ghgApi.checkSBTiAlignment(orgId, targetId);
  },
);

export const deleteTarget = createAsyncThunk<
  string,
  { orgId: string; targetId: string }
>(
  'targets/deleteTarget',
  async ({ orgId, targetId }) => {
    await ghgApi.deleteTarget(orgId, targetId);
    return targetId;
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const targetsSlice = createSlice({
  name: 'targets',
  initialState,
  reducers: {
    clearTargets: () => initialState,
    clearSBTiCheck: (state) => {
      state.sbtiCheck = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- setTarget --
      .addCase(setTarget.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(setTarget.fulfilled, (state, action) => {
        state.loading = false;
        state.targets.push(action.payload);
      })
      .addCase(setTarget.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to create target';
      })

      // -- fetchTargets --
      .addCase(fetchTargets.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTargets.fulfilled, (state, action) => {
        state.loading = false;
        state.targets = action.payload;
      })
      .addCase(fetchTargets.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load targets';
      })

      // -- fetchTargetProgress --
      .addCase(fetchTargetProgress.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTargetProgress.fulfilled, (state, action) => {
        state.loading = false;
        state.progress[action.payload.targetId] = action.payload.progress;
      })
      .addCase(fetchTargetProgress.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load target progress';
      })

      // -- checkSBTi --
      .addCase(checkSBTi.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(checkSBTi.fulfilled, (state, action) => {
        state.loading = false;
        state.sbtiCheck = action.payload;
      })
      .addCase(checkSBTi.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to check SBTi alignment';
      })

      // -- deleteTarget --
      .addCase(deleteTarget.fulfilled, (state, action) => {
        state.targets = state.targets.filter((t) => t.id !== action.payload);
        delete state.progress[action.payload];
      });
  },
});

export const { clearTargets, clearSBTiCheck } = targetsSlice.actions;
export default targetsSlice.reducer;
