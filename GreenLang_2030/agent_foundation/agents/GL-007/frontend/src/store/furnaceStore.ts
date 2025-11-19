/**
 * GL-007 Furnace Performance Monitor - Furnace State Store
 *
 * Global state management for furnace data using Zustand
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  FurnaceConfig,
  FurnacePerformance,
  Alert,
  MaintenanceSchedule,
  AnalyticsData,
  ThermalProfile,
  RefractoryCondition,
} from '../types';

// ============================================================================
// FURNACE STORE INTERFACE
// ============================================================================

interface FurnaceStore {
  // Current furnace selection
  selectedFurnaceId: string | null;
  furnaces: FurnaceConfig[];

  // Real-time performance data
  performance: Record<string, FurnacePerformance>;
  thermalProfiles: Record<string, ThermalProfile>;

  // Alerts
  activeAlerts: Alert[];
  alertHistory: Alert[];
  unacknowledgedCount: number;

  // Maintenance
  maintenanceSchedules: Record<string, MaintenanceSchedule>;
  refractoryConditions: Record<string, RefractoryCondition>;

  // Analytics
  analyticsData: Record<string, AnalyticsData>;

  // Loading states
  loading: {
    furnaces: boolean;
    performance: boolean;
    alerts: boolean;
    maintenance: boolean;
    analytics: boolean;
  };

  // Error states
  errors: {
    furnaces?: Error;
    performance?: Error;
    alerts?: Error;
    maintenance?: Error;
    analytics?: Error;
  };

  // Actions
  setSelectedFurnace: (furnaceId: string | null) => void;
  setFurnaces: (furnaces: FurnaceConfig[]) => void;
  addFurnace: (furnace: FurnaceConfig) => void;
  updateFurnace: (furnaceId: string, updates: Partial<FurnaceConfig>) => void;

  setPerformance: (furnaceId: string, performance: FurnacePerformance) => void;
  updatePerformance: (furnaceId: string, updates: Partial<FurnacePerformance>) => void;

  setThermalProfile: (furnaceId: string, profile: ThermalProfile) => void;

  addAlert: (alert: Alert) => void;
  updateAlert: (alertId: string, updates: Partial<Alert>) => void;
  acknowledgeAlert: (alertId: string, userId: string) => void;
  resolveAlert: (alertId: string) => void;
  clearAlerts: () => void;

  setMaintenanceSchedule: (furnaceId: string, schedule: MaintenanceSchedule) => void;
  setRefractoryCondition: (furnaceId: string, condition: RefractoryCondition) => void;

  setAnalytics: (furnaceId: string, analytics: AnalyticsData) => void;

  setLoading: (key: keyof FurnaceStore['loading'], value: boolean) => void;
  setError: (key: keyof FurnaceStore['errors'], error: Error | undefined) => void;

  reset: () => void;
}

// ============================================================================
// INITIAL STATE
// ============================================================================

const initialState = {
  selectedFurnaceId: null,
  furnaces: [],
  performance: {},
  thermalProfiles: {},
  activeAlerts: [],
  alertHistory: [],
  unacknowledgedCount: 0,
  maintenanceSchedules: {},
  refractoryConditions: {},
  analyticsData: {},
  loading: {
    furnaces: false,
    performance: false,
    alerts: false,
    maintenance: false,
    analytics: false,
  },
  errors: {},
};

// ============================================================================
// FURNACE STORE
// ============================================================================

export const useFurnaceStore = create<FurnaceStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Furnace selection
      setSelectedFurnace: (furnaceId) =>
        set({ selectedFurnaceId: furnaceId }),

      setFurnaces: (furnaces) =>
        set({ furnaces }),

      addFurnace: (furnace) =>
        set((state) => ({
          furnaces: [...state.furnaces, furnace],
        })),

      updateFurnace: (furnaceId, updates) =>
        set((state) => ({
          furnaces: state.furnaces.map((f) =>
            f.id === furnaceId ? { ...f, ...updates } : f
          ),
        })),

      // Performance data
      setPerformance: (furnaceId, performance) =>
        set((state) => ({
          performance: {
            ...state.performance,
            [furnaceId]: performance,
          },
        })),

      updatePerformance: (furnaceId, updates) =>
        set((state) => ({
          performance: {
            ...state.performance,
            [furnaceId]: {
              ...state.performance[furnaceId],
              ...updates,
              timestamp: new Date().toISOString(),
            },
          },
        })),

      // Thermal profile
      setThermalProfile: (furnaceId, profile) =>
        set((state) => ({
          thermalProfiles: {
            ...state.thermalProfiles,
            [furnaceId]: profile,
          },
        })),

      // Alerts
      addAlert: (alert) =>
        set((state) => {
          const activeAlerts = [alert, ...state.activeAlerts].slice(0, 100);
          const unacknowledgedCount = activeAlerts.filter(
            (a) => a.status === 'active'
          ).length;

          return {
            activeAlerts,
            unacknowledgedCount,
          };
        }),

      updateAlert: (alertId, updates) =>
        set((state) => ({
          activeAlerts: state.activeAlerts.map((a) =>
            a.id === alertId ? { ...a, ...updates } : a
          ),
        })),

      acknowledgeAlert: (alertId, userId) =>
        set((state) => ({
          activeAlerts: state.activeAlerts.map((a) =>
            a.id === alertId
              ? {
                  ...a,
                  status: 'acknowledged' as const,
                  acknowledgedBy: userId,
                  acknowledgedAt: new Date().toISOString(),
                }
              : a
          ),
          unacknowledgedCount: Math.max(0, state.unacknowledgedCount - 1),
        })),

      resolveAlert: (alertId) =>
        set((state) => {
          const alert = state.activeAlerts.find((a) => a.id === alertId);
          if (!alert) return state;

          const updatedAlert = {
            ...alert,
            status: 'resolved' as const,
            resolvedAt: new Date().toISOString(),
          };

          return {
            activeAlerts: state.activeAlerts.filter((a) => a.id !== alertId),
            alertHistory: [updatedAlert, ...state.alertHistory].slice(0, 500),
            unacknowledgedCount:
              alert.status === 'active'
                ? Math.max(0, state.unacknowledgedCount - 1)
                : state.unacknowledgedCount,
          };
        }),

      clearAlerts: () =>
        set({
          activeAlerts: [],
          alertHistory: [],
          unacknowledgedCount: 0,
        }),

      // Maintenance
      setMaintenanceSchedule: (furnaceId, schedule) =>
        set((state) => ({
          maintenanceSchedules: {
            ...state.maintenanceSchedules,
            [furnaceId]: schedule,
          },
        })),

      setRefractoryCondition: (furnaceId, condition) =>
        set((state) => ({
          refractoryConditions: {
            ...state.refractoryConditions,
            [furnaceId]: condition,
          },
        })),

      // Analytics
      setAnalytics: (furnaceId, analytics) =>
        set((state) => ({
          analyticsData: {
            ...state.analyticsData,
            [furnaceId]: analytics,
          },
        })),

      // Loading states
      setLoading: (key, value) =>
        set((state) => ({
          loading: {
            ...state.loading,
            [key]: value,
          },
        })),

      // Error states
      setError: (key, error) =>
        set((state) => ({
          errors: {
            ...state.errors,
            [key]: error,
          },
        })),

      // Reset
      reset: () => set(initialState),
    }),
    {
      name: 'furnace-store',
      partialize: (state) => ({
        selectedFurnaceId: state.selectedFurnaceId,
        furnaces: state.furnaces,
      }),
    }
  )
);

// ============================================================================
// SELECTORS
// ============================================================================

export const selectSelectedFurnace = (state: FurnaceStore) => {
  const { selectedFurnaceId, furnaces } = state;
  return furnaces.find((f) => f.id === selectedFurnaceId) || null;
};

export const selectSelectedPerformance = (state: FurnaceStore) => {
  const { selectedFurnaceId, performance } = state;
  return selectedFurnaceId ? performance[selectedFurnaceId] : null;
};

export const selectSelectedThermalProfile = (state: FurnaceStore) => {
  const { selectedFurnaceId, thermalProfiles } = state;
  return selectedFurnaceId ? thermalProfiles[selectedFurnaceId] : null;
};

export const selectActiveAlertsForFurnace = (furnaceId: string) => (state: FurnaceStore) => {
  return state.activeAlerts.filter((a) => a.furnaceId === furnaceId);
};

export const selectCriticalAlerts = (state: FurnaceStore) => {
  return state.activeAlerts.filter((a) => a.severity === 'critical' && a.status === 'active');
};

export const selectMaintenanceSchedule = (furnaceId: string) => (state: FurnaceStore) => {
  return state.maintenanceSchedules[furnaceId] || null;
};

export const selectAnalytics = (furnaceId: string) => (state: FurnaceStore) => {
  return state.analyticsData[furnaceId] || null;
};

export default useFurnaceStore;
