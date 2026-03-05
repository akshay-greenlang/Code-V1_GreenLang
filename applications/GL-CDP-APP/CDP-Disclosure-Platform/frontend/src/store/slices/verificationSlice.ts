/**
 * Verification Redux Slice
 *
 * Manages verification state: records per scope, summary with
 * A-level requirements checker, and CRUD operations.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  CDPVerificationState,
  VerificationRecord,
  VerificationSummary,
  CreateVerificationRequest,
} from '../../types';
import { cdpApi } from '../../services/api';

const initialState: CDPVerificationState = {
  records: [],
  summary: null,
  loading: false,
  error: null,
};

export const fetchVerificationRecords = createAsyncThunk<VerificationRecord[], string>(
  'verification/fetchRecords',
  async (orgId) => cdpApi.getVerificationRecords(orgId),
);

export const createVerification = createAsyncThunk<
  VerificationRecord,
  { orgId: string; payload: CreateVerificationRequest }
>(
  'verification/create',
  async ({ orgId, payload }) => cdpApi.createVerification(orgId, payload),
);

export const fetchVerificationSummary = createAsyncThunk<VerificationSummary, string>(
  'verification/fetchSummary',
  async (orgId) => cdpApi.getVerificationSummary(orgId),
);

export const deleteVerification = createAsyncThunk<
  string,
  { orgId: string; recordId: string }
>(
  'verification/delete',
  async ({ orgId, recordId }) => {
    await cdpApi.deleteVerification(orgId, recordId);
    return recordId;
  },
);

const verificationSlice = createSlice({
  name: 'verification',
  initialState,
  reducers: {
    clearVerification: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchVerificationRecords.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchVerificationRecords.fulfilled, (state, action) => {
        state.loading = false;
        state.records = action.payload;
      })
      .addCase(fetchVerificationRecords.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load verification records';
      })
      .addCase(createVerification.fulfilled, (state, action) => {
        state.records.push(action.payload);
      })
      .addCase(fetchVerificationSummary.fulfilled, (state, action) => {
        state.summary = action.payload;
      })
      .addCase(deleteVerification.fulfilled, (state, action) => {
        state.records = state.records.filter((r) => r.id !== action.payload);
      });
  },
});

export const { clearVerification } = verificationSlice.actions;
export default verificationSlice.reducer;
