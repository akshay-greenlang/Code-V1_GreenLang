/**
 * Compliance Scorecard Redux Slice
 *
 * Manages state for multi-standard compliance tracking including
 * GHG Protocol, ESRS E1, CDP, IFRS S2, and ISO 14083.
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import api from '../../services/api';

// =============================================================================
// Type Definitions
// =============================================================================

export interface ComplianceRequirement {
  id: string;
  code: string;
  name: string;
  description: string;
  met: boolean;
  dataPointsFilled: number;
  dataPointsRequired: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
}

export interface ComplianceStandard {
  id: string;
  name: string;
  shortName: string;
  coveragePercentage: number;
  requirementsMet: number;
  requirementsTotal: number;
  predictedScore?: string;
  completionPercentage: number;
  requirements: ComplianceRequirement[];
  lastUpdated: string;
}

export interface ComplianceScorecard {
  overallScore: number;
  standards: ComplianceStandard[];
  lastCalculated: string;
}

export interface ComplianceGap {
  id: string;
  standardId: string;
  standardName: string;
  requirementId: string;
  requirementName: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  action: string;
  status: 'open' | 'in_progress' | 'resolved';
  dueDate?: string;
  assignee?: string;
}

export interface ActionItem {
  id: string;
  gap: string;
  standard: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  action: string;
  status: 'open' | 'in_progress' | 'resolved';
  dueDate?: string;
}

// =============================================================================
// State Interface
// =============================================================================

interface ComplianceState {
  scorecard: ComplianceScorecard | null;
  gaps: ComplianceGap[];
  actionItems: ActionItem[];
  loading: boolean;
  error: string | null;
}

const initialState: ComplianceState = {
  scorecard: null,
  gaps: [],
  actionItems: [],
  loading: false,
  error: null,
};

// =============================================================================
// Async Thunks
// =============================================================================

export const fetchComplianceScorecard = createAsyncThunk(
  'compliance/fetchScorecard',
  async () => {
    const response = await api.getComplianceScorecard();
    return response;
  }
);

export const fetchComplianceGaps = createAsyncThunk(
  'compliance/fetchGaps',
  async () => {
    const response = await api.getComplianceGaps();
    return response;
  }
);

// =============================================================================
// Slice
// =============================================================================

const complianceSlice = createSlice({
  name: 'compliance',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    updateActionItemStatus: (
      state,
      action: PayloadAction<{ id: string; status: ActionItem['status'] }>
    ) => {
      const item = state.actionItems.find((i) => i.id === action.payload.id);
      if (item) {
        item.status = action.payload.status;
      }
    },
  },
  extraReducers: (builder) => {
    // Fetch scorecard
    builder
      .addCase(fetchComplianceScorecard.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchComplianceScorecard.fulfilled, (state, action) => {
        state.loading = false;
        state.scorecard = action.payload;
      })
      .addCase(fetchComplianceScorecard.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch compliance scorecard';
      });

    // Fetch gaps
    builder
      .addCase(fetchComplianceGaps.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchComplianceGaps.fulfilled, (state, action) => {
        state.loading = false;
        state.gaps = action.payload;
        // Derive action items from gaps
        state.actionItems = action.payload.map((gap) => ({
          id: gap.id,
          gap: gap.requirementName,
          standard: gap.standardName,
          severity: gap.severity,
          action: gap.action,
          status: gap.status,
          dueDate: gap.dueDate,
        }));
      })
      .addCase(fetchComplianceGaps.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch compliance gaps';
      });
  },
});

export const { clearError, updateActionItemStatus } = complianceSlice.actions;
export default complianceSlice.reducer;
