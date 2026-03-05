import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { TCFDDisclosure, DisclosureSection, ComplianceCheck, PaginatedResponse } from '../../types';
import { disclosureApi } from '../../services/api';
import type { RootState } from '../index';

interface DisclosureState {
  disclosures: TCFDDisclosure[];
  activeDisclosure: TCFDDisclosure | null;
  sections: DisclosureSection[];
  complianceChecks: ComplianceCheck[];
  checklist: { code: string; title: string; pillar: string; status: string; completeness: number }[];
  activeSectionId: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: DisclosureState = {
  disclosures: [],
  activeDisclosure: null,
  sections: [],
  complianceChecks: [],
  checklist: [],
  activeSectionId: null,
  loading: false,
  error: null,
};

export const fetchDisclosures = createAsyncThunk(
  'disclosure/fetchDisclosures',
  async ({ orgId, params }: { orgId: string; params?: { year?: number; status?: string } }) =>
    disclosureApi.getDisclosures(orgId, params)
);

export const fetchDisclosure = createAsyncThunk(
  'disclosure/fetchDisclosure',
  async (id: string) => disclosureApi.getDisclosure(id)
);

export const fetchSections = createAsyncThunk(
  'disclosure/fetchSections',
  async (disclosureId: string) => disclosureApi.getSections(disclosureId)
);

export const updateSection = createAsyncThunk(
  'disclosure/updateSection',
  async ({ disclosureId, sectionId, data }: { disclosureId: string; sectionId: string; data: Partial<DisclosureSection> }) =>
    disclosureApi.updateSection(disclosureId, sectionId, data)
);

export const fetchComplianceChecks = createAsyncThunk(
  'disclosure/fetchComplianceChecks',
  async ({ disclosureId, framework }: { disclosureId: string; framework?: string }) =>
    disclosureApi.getComplianceChecks(disclosureId, framework)
);

export const runComplianceCheck = createAsyncThunk(
  'disclosure/runComplianceCheck',
  async (disclosureId: string) => disclosureApi.runComplianceCheck(disclosureId)
);

export const fetchChecklist = createAsyncThunk(
  'disclosure/fetchChecklist',
  async (orgId: string) => disclosureApi.getDisclosureChecklist(orgId)
);

const disclosureSlice = createSlice({
  name: 'disclosure',
  initialState,
  reducers: {
    clearError(state) { state.error = null; },
    setActiveSectionId(state, action: PayloadAction<string | null>) {
      state.activeSectionId = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchDisclosures.pending, (state) => { state.loading = true; state.error = null; })
      .addCase(fetchDisclosures.fulfilled, (state, action: PayloadAction<PaginatedResponse<TCFDDisclosure>>) => {
        state.loading = false;
        state.disclosures = action.payload.items;
      })
      .addCase(fetchDisclosures.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch disclosures';
      })
      .addCase(fetchDisclosure.fulfilled, (state, action: PayloadAction<TCFDDisclosure>) => {
        state.activeDisclosure = action.payload;
      })
      .addCase(fetchSections.fulfilled, (state, action: PayloadAction<DisclosureSection[]>) => {
        state.sections = action.payload;
      })
      .addCase(updateSection.fulfilled, (state, action: PayloadAction<DisclosureSection>) => {
        const idx = state.sections.findIndex((s) => s.id === action.payload.id);
        if (idx >= 0) state.sections[idx] = action.payload;
      })
      .addCase(fetchComplianceChecks.fulfilled, (state, action: PayloadAction<ComplianceCheck[]>) => {
        state.complianceChecks = action.payload;
      })
      .addCase(runComplianceCheck.fulfilled, (state, action: PayloadAction<ComplianceCheck[]>) => {
        state.complianceChecks = action.payload;
      })
      .addCase(fetchChecklist.fulfilled, (state, action) => {
        state.checklist = action.payload;
      });
  },
});

export const { clearError, setActiveSectionId } = disclosureSlice.actions;
export const selectDisclosures = (state: RootState) => state.disclosure.disclosures;
export const selectActiveDisclosure = (state: RootState) => state.disclosure.activeDisclosure;
export const selectSections = (state: RootState) => state.disclosure.sections;
export const selectComplianceChecks = (state: RootState) => state.disclosure.complianceChecks;
export const selectChecklist = (state: RootState) => state.disclosure.checklist;
export const selectActiveSectionId = (state: RootState) => state.disclosure.activeSectionId;
export const selectDisclosureLoading = (state: RootState) => state.disclosure.loading;
export default disclosureSlice.reducer;
