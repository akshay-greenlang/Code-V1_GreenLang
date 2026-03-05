/**
 * GL-ISO14064-APP v1.0 - Quality Management Redux Slice
 *
 * State management for ISO 14064-1 Clause 7 quality management:
 * quality plans, procedures, data quality matrix, and corrective
 * actions.  Connects to /api/v1/iso14064/quality endpoints.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '../../services/api';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface QualityPlan {
  plan_id: string;
  org_id: string;
  title: string;
  description: string;
  status: string;
  objectives: string[];
  scope: string;
  responsible_person: string;
  review_frequency: string;
  last_reviewed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface QualityProcedure {
  procedure_id: string;
  plan_id: string;
  title: string;
  procedure_type: string;
  description: string;
  responsible: string;
  frequency: string;
  status: string;
  last_executed_at: string | null;
  next_due_at: string | null;
  created_at: string;
}

export interface DataQualityEntry {
  category: string;
  dimension: string;
  score: number;
  tier: string;
  notes: string;
}

export interface CorrectiveAction {
  action_id: string;
  plan_id: string;
  title: string;
  description: string;
  priority: string;
  status: string;
  assigned_to: string;
  root_cause: string;
  due_date: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface QualityState {
  plans: QualityPlan[];
  currentPlan: QualityPlan | null;
  procedures: QualityProcedure[];
  dataQualityMatrix: DataQualityEntry[];
  correctiveActions: CorrectiveAction[];
  loading: boolean;
  error: string | null;
}

const initialState: QualityState = {
  plans: [],
  currentPlan: null,
  procedures: [],
  dataQualityMatrix: [],
  correctiveActions: [],
  loading: false,
  error: null,
};

/* ------------------------------------------------------------------ */
/*  Async Thunks                                                       */
/* ------------------------------------------------------------------ */

export const fetchQualityPlans = createAsyncThunk(
  'quality/fetchPlans',
  async (orgId: string) => {
    const resp = await api.get(`/api/v1/iso14064/quality/plans?org_id=${orgId}`);
    return resp as QualityPlan[];
  },
);

export const fetchQualityPlan = createAsyncThunk(
  'quality/fetchPlan',
  async (planId: string) => {
    const resp = await api.get(`/api/v1/iso14064/quality/plans/${planId}`);
    return resp as QualityPlan;
  },
);

export const createQualityPlan = createAsyncThunk(
  'quality/createPlan',
  async (data: Partial<QualityPlan>) => {
    const resp = await api.post('/api/v1/iso14064/quality/plans', data);
    return resp as QualityPlan;
  },
);

export const fetchProcedures = createAsyncThunk(
  'quality/fetchProcedures',
  async (planId: string) => {
    const resp = await api.get(`/api/v1/iso14064/quality/plans/${planId}/procedures`);
    return resp as QualityProcedure[];
  },
);

export const fetchDataQualityMatrix = createAsyncThunk(
  'quality/fetchMatrix',
  async (inventoryId: string) => {
    const resp = await api.get(`/api/v1/iso14064/quality/data-quality-matrix?inventory_id=${inventoryId}`);
    return resp as DataQualityEntry[];
  },
);

export const fetchCorrectiveActions = createAsyncThunk(
  'quality/fetchActions',
  async (planId: string) => {
    const resp = await api.get(`/api/v1/iso14064/quality/plans/${planId}/corrective-actions`);
    return resp as CorrectiveAction[];
  },
);

/* ------------------------------------------------------------------ */
/*  Slice                                                              */
/* ------------------------------------------------------------------ */

const qualitySlice = createSlice({
  name: 'quality',
  initialState,
  reducers: {
    clearQualityError(state) {
      state.error = null;
    },
    setCurrentPlan(state, action: PayloadAction<QualityPlan | null>) {
      state.currentPlan = action.payload;
    },
  },
  extraReducers: (builder) => {
    // Plans list
    builder
      .addCase(fetchQualityPlans.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchQualityPlans.fulfilled, (state, action) => { state.loading = false; state.plans = action.payload; })
      .addCase(fetchQualityPlans.rejected, (state, action) => { state.loading = false; state.error = action.error.message ?? 'Failed to fetch plans'; });

    // Single plan
    builder
      .addCase(fetchQualityPlan.fulfilled, (state, action) => { state.currentPlan = action.payload; });

    // Create plan
    builder
      .addCase(createQualityPlan.fulfilled, (state, action) => { state.plans.push(action.payload); state.currentPlan = action.payload; });

    // Procedures
    builder
      .addCase(fetchProcedures.pending, (state) => { state.loading = true; })
      .addCase(fetchProcedures.fulfilled, (state, action) => { state.loading = false; state.procedures = action.payload; })
      .addCase(fetchProcedures.rejected, (state, action) => { state.loading = false; state.error = action.error.message ?? 'Failed to fetch procedures'; });

    // Data quality matrix
    builder
      .addCase(fetchDataQualityMatrix.fulfilled, (state, action) => { state.dataQualityMatrix = action.payload; });

    // Corrective actions
    builder
      .addCase(fetchCorrectiveActions.fulfilled, (state, action) => { state.correctiveActions = action.payload; });
  },
});

export const { clearQualityError, setCurrentPlan } = qualitySlice.actions;
export default qualitySlice.reducer;
