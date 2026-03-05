/**
 * Redux Store Configuration
 *
 * Combines all 14 domain slices for the EU Taxonomy Alignment Platform.
 */

import { configureStore } from '@reduxjs/toolkit';
import dashboardReducer from './slices/dashboardSlice';
import activitiesReducer from './slices/activitiesSlice';
import screeningReducer from './slices/screeningSlice';
import substantialContributionReducer from './slices/substantialContributionSlice';
import dnshReducer from './slices/dnshSlice';
import safeguardsReducer from './slices/safeguardsSlice';
import kpiReducer from './slices/kpiSlice';
import garReducer from './slices/garSlice';
import alignmentReducer from './slices/alignmentSlice';
import reportingReducer from './slices/reportingSlice';
import portfolioReducer from './slices/portfolioSlice';
import dataQualityReducer from './slices/dataQualitySlice';
import regulatoryReducer from './slices/regulatorySlice';
import settingsReducer from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardReducer,
    activities: activitiesReducer,
    screening: screeningReducer,
    substantialContribution: substantialContributionReducer,
    dnsh: dnshReducer,
    safeguards: safeguardsReducer,
    kpi: kpiReducer,
    gar: garReducer,
    alignment: alignmentReducer,
    reporting: reportingReducer,
    portfolio: portfolioReducer,
    dataQuality: dataQualityReducer,
    regulatory: regulatoryReducer,
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
