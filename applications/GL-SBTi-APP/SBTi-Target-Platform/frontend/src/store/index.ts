/**
 * Redux Store Configuration
 *
 * Combines all 14 domain slices for the SBTi Target Validation Platform.
 */

import { configureStore } from '@reduxjs/toolkit';
import dashboardReducer from './slices/dashboardSlice';
import targetReducer from './slices/targetSlice';
import pathwayReducer from './slices/pathwaySlice';
import validationReducer from './slices/validationSlice';
import scope3Reducer from './slices/scope3Slice';
import flagReducer from './slices/flagSlice';
import sectorReducer from './slices/sectorSlice';
import progressReducer from './slices/progressSlice';
import temperatureReducer from './slices/temperatureSlice';
import recalculationReducer from './slices/recalculationSlice';
import reviewReducer from './slices/reviewSlice';
import fiReducer from './slices/fiSlice';
import reportReducer from './slices/reportSlice';
import settingsReducer from './slices/settingsSlice';

export const store = configureStore({
  reducer: {
    dashboard: dashboardReducer,
    target: targetReducer,
    pathway: pathwayReducer,
    validation: validationReducer,
    scope3: scope3Reducer,
    flag: flagReducer,
    sector: sectorReducer,
    progress: progressReducer,
    temperature: temperatureReducer,
    recalculation: recalculationReducer,
    review: reviewReducer,
    fi: fiReducer,
    report: reportReducer,
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
