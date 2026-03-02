/**
 * Verification Redux Slice
 *
 * Manages third-party verification workflow: engagement start,
 * approval/rejection, findings management, and resolution.
 *
 * Async thunks:
 *   - startVerification: Begin a new verification engagement
 *   - approve: Issue positive verification opinion
 *   - reject: Issue adverse opinion
 *   - addFinding: Add a verification finding
 *   - resolveFinding: Resolve a finding
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  VerificationState,
  VerificationRecord,
  Finding,
  StartVerificationRequest,
  AddFindingRequest,
} from '../../types';
import { ghgApi } from '../../services/api';

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

const initialState: VerificationState = {
  currentVerification: null,
  verifications: [],
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Async thunks
// ---------------------------------------------------------------------------

export const startVerification = createAsyncThunk<
  VerificationRecord,
  StartVerificationRequest
>(
  'verification/start',
  async (payload) => {
    return ghgApi.startVerification(payload);
  },
);

export const fetchVerification = createAsyncThunk<
  VerificationRecord,
  string
>(
  'verification/fetch',
  async (verificationId) => {
    return ghgApi.getVerification(verificationId);
  },
);

export const fetchVerifications = createAsyncThunk<
  VerificationRecord[],
  string
>(
  'verification/fetchAll',
  async (inventoryId) => {
    return ghgApi.getVerifications(inventoryId);
  },
);

export const approveVerification = createAsyncThunk<
  VerificationRecord,
  { verificationId: string; opinion: string }
>(
  'verification/approve',
  async ({ verificationId, opinion }) => {
    return ghgApi.approveVerification(verificationId, opinion);
  },
);

export const rejectVerification = createAsyncThunk<
  VerificationRecord,
  { verificationId: string; reason: string }
>(
  'verification/reject',
  async ({ verificationId, reason }) => {
    return ghgApi.rejectVerification(verificationId, reason);
  },
);

export const addFinding = createAsyncThunk<
  Finding,
  { verificationId: string; payload: AddFindingRequest }
>(
  'verification/addFinding',
  async ({ verificationId, payload }) => {
    return ghgApi.addFinding(verificationId, payload);
  },
);

export const resolveFinding = createAsyncThunk<
  Finding,
  { verificationId: string; findingId: string; resolution: string }
>(
  'verification/resolveFinding',
  async ({ verificationId, findingId, resolution }) => {
    return ghgApi.resolveFinding(verificationId, findingId, resolution);
  },
);

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

const verificationSlice = createSlice({
  name: 'verification',
  initialState,
  reducers: {
    clearVerification: () => initialState,
    setCurrentVerification: (state, action) => {
      state.currentVerification = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      // -- startVerification --
      .addCase(startVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(startVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.currentVerification = action.payload;
        state.verifications.push(action.payload);
      })
      .addCase(startVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to start verification';
      })

      // -- fetchVerification --
      .addCase(fetchVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.currentVerification = action.payload;
      })
      .addCase(fetchVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load verification';
      })

      // -- fetchVerifications --
      .addCase(fetchVerifications.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVerifications.fulfilled, (state, action) => {
        state.loading = false;
        state.verifications = action.payload;
      })
      .addCase(fetchVerifications.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load verifications';
      })

      // -- approveVerification --
      .addCase(approveVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(approveVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.currentVerification = action.payload;
        const idx = state.verifications.findIndex((v) => v.id === action.payload.id);
        if (idx >= 0) {
          state.verifications[idx] = action.payload;
        }
      })
      .addCase(approveVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to approve verification';
      })

      // -- rejectVerification --
      .addCase(rejectVerification.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(rejectVerification.fulfilled, (state, action) => {
        state.loading = false;
        state.currentVerification = action.payload;
        const idx = state.verifications.findIndex((v) => v.id === action.payload.id);
        if (idx >= 0) {
          state.verifications[idx] = action.payload;
        }
      })
      .addCase(rejectVerification.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to reject verification';
      })

      // -- addFinding --
      .addCase(addFinding.fulfilled, (state, action) => {
        if (state.currentVerification) {
          state.currentVerification.findings.push(action.payload);
          state.currentVerification.findings_summary.total_findings += 1;
          state.currentVerification.findings_summary.open_count += 1;
          const sev = action.payload.severity;
          if (sev === 'critical') state.currentVerification.findings_summary.critical_count += 1;
          else if (sev === 'high') state.currentVerification.findings_summary.high_count += 1;
          else if (sev === 'medium') state.currentVerification.findings_summary.medium_count += 1;
          else if (sev === 'low') state.currentVerification.findings_summary.low_count += 1;
        }
      })

      // -- resolveFinding --
      .addCase(resolveFinding.fulfilled, (state, action) => {
        if (state.currentVerification) {
          const idx = state.currentVerification.findings.findIndex(
            (f) => f.id === action.payload.id,
          );
          if (idx >= 0) {
            state.currentVerification.findings[idx] = action.payload;
            state.currentVerification.findings_summary.open_count -= 1;
            state.currentVerification.findings_summary.resolved_count += 1;
          }
        }
      });
  },
});

export const { clearVerification, setCurrentVerification } = verificationSlice.actions;
export default verificationSlice.reducer;
