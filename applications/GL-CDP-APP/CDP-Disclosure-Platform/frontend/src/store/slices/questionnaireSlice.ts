/**
 * Questionnaire Redux Slice
 *
 * Manages CDP questionnaire state: listing, current questionnaire,
 * module navigation, progress tracking, and auto-population.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { QuestionnaireState, Questionnaire, Module } from '../../types';
import { cdpApi } from '../../services/api';
import type { ModuleProgress } from '../../types';

const initialState: QuestionnaireState = {
  currentQuestionnaire: null,
  questionnaires: [],
  modules: [],
  currentModule: null,
  loading: false,
  error: null,
};

export const fetchQuestionnaires = createAsyncThunk<Questionnaire[], string>(
  'questionnaire/fetchAll',
  async (orgId) => cdpApi.listQuestionnaires(orgId),
);

export const fetchQuestionnaire = createAsyncThunk<Questionnaire, string>(
  'questionnaire/fetch',
  async (questionnaireId) => cdpApi.getQuestionnaire(questionnaireId),
);

export const createQuestionnaire = createAsyncThunk<
  Questionnaire,
  { org_id: string; reporting_year: number }
>(
  'questionnaire/create',
  async (payload) => cdpApi.createQuestionnaire(payload),
);

export const fetchModules = createAsyncThunk<Module[], string>(
  'questionnaire/fetchModules',
  async (questionnaireId) => cdpApi.getModules(questionnaireId),
);

export const fetchProgress = createAsyncThunk<ModuleProgress[], string>(
  'questionnaire/fetchProgress',
  async (questionnaireId) => cdpApi.getQuestionnaireProgress(questionnaireId),
);

export const autoPopulateModule = createAsyncThunk<
  void,
  { questionnaireId: string; moduleId: string }
>(
  'questionnaire/autoPopulate',
  async ({ questionnaireId, moduleId }) => {
    await cdpApi.autoPopulateModule(questionnaireId, moduleId);
  },
);

const questionnaireSlice = createSlice({
  name: 'questionnaire',
  initialState,
  reducers: {
    clearQuestionnaire: () => initialState,
    setCurrentModule: (state, action) => {
      state.currentModule = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchQuestionnaires.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchQuestionnaires.fulfilled, (state, action) => {
        state.loading = false;
        state.questionnaires = action.payload;
      })
      .addCase(fetchQuestionnaires.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load questionnaires';
      })
      .addCase(fetchQuestionnaire.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchQuestionnaire.fulfilled, (state, action) => {
        state.loading = false;
        state.currentQuestionnaire = action.payload;
      })
      .addCase(fetchQuestionnaire.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load questionnaire';
      })
      .addCase(createQuestionnaire.fulfilled, (state, action) => {
        state.currentQuestionnaire = action.payload;
        state.questionnaires.push(action.payload);
      })
      .addCase(fetchModules.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchModules.fulfilled, (state, action) => {
        state.loading = false;
        state.modules = action.payload;
      })
      .addCase(fetchModules.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message ?? 'Failed to load modules';
      });
  },
});

export const { clearQuestionnaire, setCurrentModule } = questionnaireSlice.actions;
export default questionnaireSlice.reducer;
