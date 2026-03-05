import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { GovernanceAssessment, GovernanceRole, GovernanceCommittee, CompetencyEntry } from '../../types';
import { governanceApi } from '../../services/api';
import type { RootState } from '../index';

interface GovernanceState {
  assessment: GovernanceAssessment | null;
  roles: GovernanceRole[];
  committees: GovernanceCommittee[];
  competencies: CompetencyEntry[];
  loading: boolean;
  error: string | null;
}

const initialState: GovernanceState = {
  assessment: null,
  roles: [],
  committees: [],
  competencies: [],
  loading: false,
  error: null,
};

export const fetchAssessment = createAsyncThunk(
  'governance/fetchAssessment',
  async (orgId: string) => governanceApi.getAssessment(orgId)
);

export const fetchRoles = createAsyncThunk(
  'governance/fetchRoles',
  async (orgId: string) => governanceApi.getRoles(orgId)
);

export const fetchCommittees = createAsyncThunk(
  'governance/fetchCommittees',
  async (orgId: string) => governanceApi.getCommittees(orgId)
);

export const fetchCompetencies = createAsyncThunk(
  'governance/fetchCompetencies',
  async (orgId: string) => governanceApi.getCompetencies(orgId)
);

export const updateRole = createAsyncThunk(
  'governance/updateRole',
  async ({ id, data }: { id: string; data: Partial<GovernanceRole> }) => governanceApi.updateRole(id, data)
);

export const updateCompetency = createAsyncThunk(
  'governance/updateCompetency',
  async ({ id, data }: { id: string; data: Partial<CompetencyEntry> }) => governanceApi.updateCompetency(id, data)
);

const governanceSlice = createSlice({
  name: 'governance',
  initialState,
  reducers: {
    clearError(state) {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAssessment.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchAssessment.fulfilled, (state, action: PayloadAction<GovernanceAssessment>) => {
        state.loading = false;
        state.assessment = action.payload;
      })
      .addCase(fetchAssessment.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch governance assessment';
      })
      .addCase(fetchRoles.fulfilled, (state, action: PayloadAction<GovernanceRole[]>) => {
        state.roles = action.payload;
      })
      .addCase(fetchCommittees.fulfilled, (state, action: PayloadAction<GovernanceCommittee[]>) => {
        state.committees = action.payload;
      })
      .addCase(fetchCompetencies.fulfilled, (state, action: PayloadAction<CompetencyEntry[]>) => {
        state.competencies = action.payload;
      })
      .addCase(updateRole.fulfilled, (state, action: PayloadAction<GovernanceRole>) => {
        const idx = state.roles.findIndex((r) => r.id === action.payload.id);
        if (idx >= 0) state.roles[idx] = action.payload;
      })
      .addCase(updateCompetency.fulfilled, (state, action: PayloadAction<CompetencyEntry>) => {
        const idx = state.competencies.findIndex((c) => c.id === action.payload.id);
        if (idx >= 0) state.competencies[idx] = action.payload;
      });
  },
});

export const { clearError } = governanceSlice.actions;
export const selectGovernance = (state: RootState) => state.governance;
export const selectAssessment = (state: RootState) => state.governance.assessment;
export const selectRoles = (state: RootState) => state.governance.roles;
export const selectCommittees = (state: RootState) => state.governance.committees;
export const selectCompetencies = (state: RootState) => state.governance.competencies;
export default governanceSlice.reducer;
