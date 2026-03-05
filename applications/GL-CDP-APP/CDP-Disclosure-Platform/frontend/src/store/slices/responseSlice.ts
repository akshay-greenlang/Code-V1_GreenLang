/**
 * Response Redux Slice
 *
 * Manages CDP response state: CRUD, workflow transitions,
 * evidence attachments, version history, and review comments.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type {
  ResponseState,
  Response as CDPResponse,
  ResponseVersion,
  Evidence,
  ReviewComment,
  SaveResponseRequest,
} from '../../types';
import { cdpApi } from '../../services/api';

const initialState: ResponseState = {
  responses: {},
  currentResponse: null,
  versions: [],
  evidence: [],
  comments: [],
  saving: false,
  loading: false,
  error: null,
};

export const fetchModuleResponses = createAsyncThunk<
  Record<string, CDPResponse>,
  { questionnaireId: string; moduleId: string }
>(
  'response/fetchModule',
  async ({ questionnaireId, moduleId }) =>
    cdpApi.getResponsesByModule(questionnaireId, moduleId),
);

export const fetchResponse = createAsyncThunk<CDPResponse, string>(
  'response/fetch',
  async (questionId) => cdpApi.getResponse(questionId),
);

export const saveResponse = createAsyncThunk<
  CDPResponse,
  { questionId: string; payload: SaveResponseRequest }
>(
  'response/save',
  async ({ questionId, payload }) => cdpApi.saveResponse(questionId, payload),
);

export const submitForReview = createAsyncThunk<
  void,
  { response_ids: string[]; reviewer: string }
>(
  'response/submitForReview',
  async (payload) => cdpApi.submitForReview(payload),
);

export const approveResponses = createAsyncThunk<
  void,
  { response_ids: string[]; comments?: string }
>(
  'response/approve',
  async (payload) => cdpApi.approveResponses(payload),
);

export const fetchVersions = createAsyncThunk<ResponseVersion[], string>(
  'response/fetchVersions',
  async (responseId) => cdpApi.getResponseVersions(responseId),
);

export const uploadEvidence = createAsyncThunk<
  Evidence,
  { responseId: string; formData: FormData }
>(
  'response/uploadEvidence',
  async ({ responseId, formData }) => cdpApi.uploadEvidence(responseId, formData),
);

export const deleteEvidence = createAsyncThunk<
  string,
  { responseId: string; evidenceId: string }
>(
  'response/deleteEvidence',
  async ({ responseId, evidenceId }) => {
    await cdpApi.deleteEvidence(responseId, evidenceId);
    return evidenceId;
  },
);

export const fetchComments = createAsyncThunk<ReviewComment[], string>(
  'response/fetchComments',
  async (responseId) => cdpApi.getComments(responseId),
);

const responseSlice = createSlice({
  name: 'response',
  initialState,
  reducers: {
    clearResponses: () => initialState,
    setCurrentResponse: (state, action) => {
      state.currentResponse = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchModuleResponses.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchModuleResponses.fulfilled, (state, action) => {
        state.loading = false;
        state.responses = { ...state.responses, ...action.payload };
      })
      .addCase(fetchModuleResponses.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load responses';
      })
      .addCase(fetchResponse.fulfilled, (state, action) => {
        state.currentResponse = action.payload;
        state.responses[action.payload.question_id] = action.payload;
      })
      .addCase(saveResponse.pending, (state) => {
        state.saving = true;
        state.error = null;
      })
      .addCase(saveResponse.fulfilled, (state, action) => {
        state.saving = false;
        state.responses[action.payload.question_id] = action.payload;
        state.currentResponse = action.payload;
      })
      .addCase(saveResponse.rejected, (state, action) => {
        state.saving = false;
        state.error = action.error.message ?? 'Failed to save response';
      })
      .addCase(fetchVersions.fulfilled, (state, action) => {
        state.versions = action.payload;
      })
      .addCase(uploadEvidence.fulfilled, (state, action) => {
        state.evidence.push(action.payload);
      })
      .addCase(deleteEvidence.fulfilled, (state, action) => {
        state.evidence = state.evidence.filter((e) => e.id !== action.payload);
      })
      .addCase(fetchComments.fulfilled, (state, action) => {
        state.comments = action.payload;
      });
  },
});

export const { clearResponses, setCurrentResponse } = responseSlice.actions;
export default responseSlice.reducer;
