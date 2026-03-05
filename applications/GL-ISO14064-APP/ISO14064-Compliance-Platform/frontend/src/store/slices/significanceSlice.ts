/**
 * Significance Redux Slice
 *
 * Manages significance assessment state per ISO 14064-1 Clause 5.2.2.
 * Organizations must assess significance of indirect categories (3-6)
 * using defined criteria before inclusion/exclusion decisions.
 *
 * Async thunks:
 *   - assessCategory: Run significance assessment for a category
 *   - fetchAssessments: Load all assessments for an inventory
 *   - fetchAssessmentForCategory: Load assessment for a specific category
 *   - updateAssessment: Update an existing assessment
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type {
  SignificanceState,
  SignificanceAssessment,
  ISOCategory,
} from '../../types';
import { iso14064Api } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: SignificanceState = {
  assessments: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const assessCategory = createAsyncThunk<
  SignificanceAssessment,
  { inventoryId: string; category: ISOCategory }
>(
  'significance/assessCategory',
  async ({ inventoryId, category }) => {
    return iso14064Api.assessSignificance(inventoryId, category);
  },
);

export const fetchAssessments = createAsyncThunk<
  SignificanceAssessment[],
  string
>(
  'significance/fetchAssessments',
  async (inventoryId) => {
    return iso14064Api.getSignificanceAssessments(inventoryId);
  },
);

export const fetchAssessmentForCategory = createAsyncThunk<
  SignificanceAssessment,
  { inventoryId: string; category: ISOCategory }
>(
  'significance/fetchAssessmentForCategory',
  async ({ inventoryId, category }) => {
    return iso14064Api.getSignificanceForCategory(inventoryId, category);
  },
);

export const updateAssessment = createAsyncThunk<
  SignificanceAssessment,
  { inventoryId: string; assessmentId: string; payload: Partial<SignificanceAssessment> }
>(
  'significance/updateAssessment',
  async ({ inventoryId, assessmentId, payload }) => {
    return iso14064Api.updateSignificanceAssessment(inventoryId, assessmentId, payload);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const significanceSlice = createSlice({
  name: 'significance',
  initialState,
  reducers: {
    clearSignificance: () => initialState,
    clearSignificanceError: (state) => {
      state.error = null;
    },
    setAssessmentLocal: (state, action: PayloadAction<SignificanceAssessment>) => {
      const idx = state.assessments.findIndex((a) => a.id === action.payload.id);
      if (idx >= 0) {
        state.assessments[idx] = action.payload;
      } else {
        state.assessments.push(action.payload);
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // -- assessCategory --
      .addCase(assessCategory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(assessCategory.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.assessments.findIndex(
          (a) => a.category === action.payload.category,
        );
        if (idx >= 0) {
          state.assessments[idx] = action.payload;
        } else {
          state.assessments.push(action.payload);
        }
      })
      .addCase(assessCategory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to assess significance';
      })

      // -- fetchAssessments --
      .addCase(fetchAssessments.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchAssessments.fulfilled, (state, action) => {
        state.loading = false;
        state.assessments = action.payload;
      })
      .addCase(fetchAssessments.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch significance assessments';
      })

      // -- fetchAssessmentForCategory --
      .addCase(fetchAssessmentForCategory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchAssessmentForCategory.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.assessments.findIndex(
          (a) => a.category === action.payload.category,
        );
        if (idx >= 0) {
          state.assessments[idx] = action.payload;
        } else {
          state.assessments.push(action.payload);
        }
      })
      .addCase(fetchAssessmentForCategory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to fetch assessment for category';
      })

      // -- updateAssessment --
      .addCase(updateAssessment.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(updateAssessment.fulfilled, (state, action) => {
        state.loading = false;
        const idx = state.assessments.findIndex((a) => a.id === action.payload.id);
        if (idx >= 0) {
          state.assessments[idx] = action.payload;
        }
      })
      .addCase(updateAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to update significance assessment';
      });
  },
});

export const { clearSignificance, clearSignificanceError, setAssessmentLocal } =
  significanceSlice.actions;
export default significanceSlice.reducer;
