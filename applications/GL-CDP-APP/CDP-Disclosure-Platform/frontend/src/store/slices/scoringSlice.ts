/**
 * Scoring Redux Slice
 *
 * Manages CDP scoring simulation state: overall score prediction,
 * category scores, what-if scenarios, and A-level eligibility.
 */

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { ScoringState, ScoringResult, WhatIfScenario, ARequirement } from '../../types';
import { cdpApi } from '../../services/api';

const initialState: ScoringState = {
  result: null,
  whatIfScenarios: [],
  currentScenario: null,
  simulating: false,
  loading: false,
  error: null,
};

export const simulateScore = createAsyncThunk<ScoringResult, string>(
  'scoring/simulate',
  async (questionnaireId) =>
    cdpApi.simulateScore({ questionnaire_id: questionnaireId }),
);

export const runWhatIf = createAsyncThunk<
  WhatIfScenario,
  { questionnaireId: string; improvements: Array<{ question_id: string; improved_score: number }> }
>(
  'scoring/whatIf',
  async ({ questionnaireId, improvements }) =>
    cdpApi.runWhatIf({ questionnaire_id: questionnaireId, improvements }),
);

export const checkALevel = createAsyncThunk<ARequirement[], string>(
  'scoring/checkALevel',
  async (questionnaireId) => cdpApi.checkALevelEligibility(questionnaireId),
);

const scoringSlice = createSlice({
  name: 'scoring',
  initialState,
  reducers: {
    clearScoring: () => initialState,
    setCurrentScenario: (state, action) => {
      state.currentScenario = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(simulateScore.pending, (state) => {
        state.simulating = true;
        state.error = null;
      })
      .addCase(simulateScore.fulfilled, (state, action) => {
        state.simulating = false;
        state.result = action.payload;
      })
      .addCase(simulateScore.rejected, (state, action) => {
        state.simulating = false;
        state.error = action.error.message ?? 'Failed to simulate score';
      })
      .addCase(runWhatIf.pending, (state) => {
        state.simulating = true;
        state.error = null;
      })
      .addCase(runWhatIf.fulfilled, (state, action) => {
        state.simulating = false;
        state.currentScenario = action.payload;
        state.whatIfScenarios.push(action.payload);
      })
      .addCase(runWhatIf.rejected, (state, action) => {
        state.simulating = false;
        state.error = action.error.message ?? 'Failed to run what-if analysis';
      })
      .addCase(checkALevel.fulfilled, (state, action) => {
        if (state.result) {
          state.result.a_level_requirements = action.payload;
          state.result.a_level_eligible = action.payload.every((r) => r.met);
        }
      });
  },
});

export const { clearScoring, setCurrentScenario } = scoringSlice.actions;
export default scoringSlice.reducer;
