/**
 * GL-CDP-APP v1.0 - Redux Store Configuration
 *
 * Configures the Redux Toolkit store with 12 domain slices
 * covering questionnaire, response, scoring, gap analysis,
 * benchmarking, supply chain, transition plan, verification,
 * historical, reports, dashboard, and settings.
 */

import { configureStore } from '@reduxjs/toolkit';
import questionnaireReducer from './slices/questionnaireSlice';
import responseReducer from './slices/responseSlice';
import scoringReducer from './slices/scoringSlice';
import gapAnalysisReducer from './slices/gapAnalysisSlice';
import benchmarkingReducer from './slices/benchmarkingSlice';
import supplyChainReducer from './slices/supplyChainSlice';
import transitionPlanReducer from './slices/transitionPlanSlice';
import verificationReducer from './slices/verificationSlice';
import historicalReducer from './slices/historicalSlice';
import reportsReducer from './slices/reportsSlice';
import dashboardReducer from './slices/dashboardSlice';
import settingsReducer from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    questionnaire: questionnaireReducer,
    response: responseReducer,
    scoring: scoringReducer,
    gapAnalysis: gapAnalysisReducer,
    benchmarking: benchmarkingReducer,
    supplyChain: supplyChainReducer,
    transitionPlan: transitionPlanReducer,
    verification: verificationReducer,
    historical: historicalReducer,
    reports: reportsReducer,
    dashboard: dashboardReducer,
    settings: settingsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'reports/generateReport/fulfilled',
          'reports/downloadReport/fulfilled',
          'response/uploadEvidence/fulfilled',
        ],
      },
    }),
  devTools: import.meta.env.DEV,
});

export type AppDispatch = typeof store.dispatch;
export type AppRootState = ReturnType<typeof store.getState>;
