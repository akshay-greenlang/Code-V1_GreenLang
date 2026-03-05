/**
 * Redux Store Configuration
 *
 * Combines all 14 domain slices for the TCFD platform.
 */

import { configureStore } from '@reduxjs/toolkit';
import governanceReducer from './slices/governanceSlice';
import strategyReducer from './slices/strategySlice';
import scenarioReducer from './slices/scenarioSlice';
import physicalRiskReducer from './slices/physicalRiskSlice';
import transitionRiskReducer from './slices/transitionRiskSlice';
import opportunityReducer from './slices/opportunitySlice';
import financialReducer from './slices/financialSlice';
import riskMgmtReducer from './slices/riskMgmtSlice';
import metricsReducer from './slices/metricsSlice';
import disclosureReducer from './slices/disclosureSlice';
import gapReducer from './slices/gapSlice';
import issbReducer from './slices/issbSlice';
import dashboardReducer from './slices/dashboardSlice';
import settingsReducer from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    governance: governanceReducer,
    strategy: strategyReducer,
    scenario: scenarioReducer,
    physicalRisk: physicalRiskReducer,
    transitionRisk: transitionRiskReducer,
    opportunity: opportunityReducer,
    financial: financialReducer,
    riskMgmt: riskMgmtReducer,
    metrics: metricsReducer,
    disclosure: disclosureReducer,
    gap: gapReducer,
    issb: issbReducer,
    dashboard: dashboardReducer,
    settings: settingsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST'],
      },
    }),
  devTools: import.meta.env.DEV,
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
