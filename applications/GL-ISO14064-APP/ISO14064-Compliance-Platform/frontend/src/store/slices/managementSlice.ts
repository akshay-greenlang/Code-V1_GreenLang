/**
 * Management Redux Slice
 *
 * Manages GHG management plan and quality management state per
 * ISO 14064-1 Clause 9.  Covers improvement actions (emission
 * reduction, removal enhancement, data improvement, process
 * improvement) and quality procedures.
 *
 * Async thunks:
 *   - fetchPlan: Load management plan for an organization-year
 *   - upsertPlan: Create or update the management plan
 *   - addAction: Add a management action
 *   - fetchActions: Load all management actions
 *   - updateAction: Update a management action
 *   - deleteAction: Remove a management action
 *   - fetchQualityPlan: Load quality management plan
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  ManagementState,
  ManagementPlan,
  ManagementAction,
  QualityManagementPlan,
  CreateManagementActionRequest,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: ManagementState = {
  plan: null,
  actions: [],
  qualityPlan: null,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const fetchPlan = createAsyncThunk<
  ManagementPlan,
  { orgId: string; reportingYear: number }
>(
  'management/fetchPlan',
  async ({ orgId, reportingYear }) => {
    return iso14064Api.getManagementPlan(orgId, reportingYear);
  },
);

export const upsertPlan = createAsyncThunk<
  ManagementPlan,
  { orgId: string; payload: Partial<ManagementPlan> }
>(
  'management/upsertPlan',
  async ({ orgId, payload }) => {
    return iso14064Api.upsertManagementPlan(orgId, payload);
  },
);

export const addAction = createAsyncThunk<
  ManagementAction,
  { orgId: string; payload: CreateManagementActionRequest }
>(
  'management/addAction',
  async ({ orgId, payload }) => {
    return iso14064Api.addManagementAction(orgId, payload);
  },
);

export const fetchActions = createAsyncThunk<
  ManagementAction[],
  string
>(
  'management/fetchActions',
  async (orgId) => {
    return iso14064Api.getManagementActions(orgId);
  },
);

export const updateAction = createAsyncThunk<
  ManagementAction,
  { orgId: string; actionId: string; payload: Partial<CreateManagementActionRequest> }
>(
  'management/updateAction',
  async ({ orgId, actionId, payload }) => {
    return iso14064Api.updateManagementAction(orgId, actionId, payload);
  },
);

export const deleteAction = createAsyncThunk<
  string,
  { orgId: string; actionId: string }
>(
  'management/deleteAction',
  async ({ orgId, actionId }) => {
    await iso14064Api.deleteManagementAction(orgId, actionId);
    return actionId;
  },
);

export const fetchQualityPlan = createAsyncThunk<
  QualityManagementPlan,
  string
>(
  'management/fetchQualityPlan',
  async (orgId) => {
    return iso14064Api.getQualityManagementPlan(orgId);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const managementSlice = createSlice({
  name: 'management',
  initialState,
  reducers: {
    clearManagement: () => initialState,
    clearManagementError: (state) => {
      state.error = null;
    },
    updateActionLocal: (state, action: PayloadAction<ManagementAction>) => {
      const idx = state.actions.findIndex((a) => a.id === action.payload.id);
      if (idx >= 0) {
        state.actions[idx] = action.payload;
      }
    },
    removeActionLocal: (state, action: PayloadAction<string>) => {
      state.actions = state.actions.filter((a) => a.id !== action.payload);
    },
  },
  extraReducers: (builder) => {
    builder
      // -- fetchPlan --
      .addCase(fetchPlan.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchPlan.fulfilled, (state, action) => {
        state.loading = false;
        state.plan = action.payload;
        state.actions = action.payload.actions;
      })
      .addCase(fetchPlan.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load management plan';
      })

      // -- upsertPlan --
      .addCase(upsertPlan.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(upsertPlan.fulfilled, (state, action) => {
        state.loading = false;
        state.plan = action.payload;
        state.actions = action.payload.actions;
      })
      .addCase(upsertPlan.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update management plan';
      })

      // -- addAction --
      .addCase(addAction.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(addAction.fulfilled, (state, action) => {
        state.loading = false;
        state.actions.push(action.payload);
      })
      .addCase(addAction.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to add management action';
      })

      // -- fetchActions --
      .addCase(fetchActions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchActions.fulfilled, (state, action) => {
        state.loading = false;
        state.actions = action.payload;
      })
      .addCase(fetchActions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch management actions';
      })

      // -- updateAction --
      .addCase(updateAction.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateAction.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.actions.findIndex((a) => a.id === action.payload.id);
        if (idx >= 0) {
          state.actions[idx] = action.payload;
        }
      })
      .addCase(updateAction.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update management action';
      })

      // -- deleteAction --
      .addCase(deleteAction.fulfilled, (state, action) => {
        state.actions = state.actions.filter((a) => a.id !== action.payload);
      })

      // -- fetchQualityPlan --
      .addCase(fetchQualityPlan.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchQualityPlan.fulfilled, (state, action) => {
        state.loading = false;
        state.qualityPlan = action.payload;
      })
      .addCase(fetchQualityPlan.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch quality management plan';
      });
  },
});

export const { clearManagement, clearManagementError, updateActionLocal, removeActionLocal } =
  managementSlice.actions;
export default managementSlice.reducer;
