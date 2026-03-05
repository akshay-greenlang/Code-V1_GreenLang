import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { KPICalculation, KPISummary, ObjectiveBreakdown, CapExPlan } from '../../types';
import { kpiApi } from '../../services/api';
import type { RootState } from '../index';

interface KPIState {
  calculation: KPICalculation | null;
  summary: KPISummary | null;
  objectiveBreakdown: ObjectiveBreakdown[];
  capexPlans: CapExPlan[];
  loading: boolean;
  error: string | null;
}

const initialState: KPIState = {
  calculation: null,
  summary: null,
  objectiveBreakdown: [],
  capexPlans: [],
  loading: false,
  error: null,
};

export const calculateKPI = createAsyncThunk(
  'kpi/calculate',
  async ({ orgId, period }: { orgId: string; period: string }) =>
    kpiApi.calculate(orgId, period)
);

export const fetchKPIDashboard = createAsyncThunk(
  'kpi/dashboard',
  async (params: { organization_id: string; reporting_period: string; include_comparison?: boolean }) =>
    kpiApi.dashboard(params)
);

export const fetchObjectiveBreakdown = createAsyncThunk(
  'kpi/objectiveBreakdown',
  async ({ orgId, period }: { orgId: string; period: string }) =>
    kpiApi.objectiveBreakdown(orgId, period)
);

export const fetchCapExPlans = createAsyncThunk(
  'kpi/capexPlans',
  async (activityId: string) => kpiApi.capexPlan(activityId)
);

const kpiSlice = createSlice({
  name: 'kpi',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
  },
  extraReducers: (builder) => {
    builder
      .addCase(calculateKPI.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(calculateKPI.fulfilled, (state, action: PayloadAction<KPICalculation>) => {
        state.loading = false;
        state.calculation = action.payload;
      })
      .addCase(calculateKPI.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to calculate KPIs';
      })
      .addCase(fetchKPIDashboard.fulfilled, (state, action) => { state.summary = action.payload; })
      .addCase(fetchObjectiveBreakdown.fulfilled, (state, action) => { state.objectiveBreakdown = action.payload; })
      .addCase(fetchCapExPlans.fulfilled, (state, action) => { state.capexPlans = action.payload; });
  },
});

export const { clearError } = kpiSlice.actions;
export const selectKPICalculation = (state: RootState) => state.kpi.calculation;
export const selectKPISummary = (state: RootState) => state.kpi.summary;
export const selectObjectiveBreakdown = (state: RootState) => state.kpi.objectiveBreakdown;
export const selectCapExPlans = (state: RootState) => state.kpi.capexPlans;
export const selectKPILoading = (state: RootState) => state.kpi.loading;
export default kpiSlice.reducer;
