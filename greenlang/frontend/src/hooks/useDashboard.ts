/**
 * Custom React hook for dashboard management.
 *
 * Provides CRUD operations, layout management, and state
 * synchronization with the backend API.
 */

import { useState, useEffect, useCallback } from 'react';
import { Layout } from 'react-grid-layout';

interface WidgetConfig {
  id: string;
  type: string;
  title: string;
  config: Record<string, any>;
  dataSource: {
    channel: string;
    metricName?: string;
    tags?: Record<string, string>;
    aggregation?: string;
  };
  position?: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
}

interface DashboardTab {
  id: string;
  name: string;
  layout: Layout[];
  widgets: WidgetConfig[];
}

interface Dashboard {
  id: string;
  name: string;
  description?: string;
  layout: Record<string, any>;
  tabs?: DashboardTab[];
  widgets: WidgetConfig[];
  tags: string[];
  accessLevel: string;
  createdAt: string;
  updatedAt: string;
}

interface UseDashboardResult {
  dashboard: Dashboard | null;
  loading: boolean;
  error: string | null;
  saveDashboard: (dashboard: Partial<Dashboard>) => Promise<void>;
  updateLayout: (dashboardId: string, tabId: string, layout: Layout[]) => Promise<void>;
  addWidget: (dashboardId: string, tabId: string, widget: WidgetConfig) => Promise<void>;
  removeWidget: (dashboardId: string, widgetId: string) => Promise<void>;
  updateWidget: (dashboardId: string, widgetId: string, updates: Partial<WidgetConfig>) => Promise<void>;
  deleteDashboard: (dashboardId: string) => Promise<void>;
  shareDashboard: (dashboardId: string, options: ShareOptions) => Promise<ShareResponse>;
}

interface ShareOptions {
  expiresIn?: number;
  canView?: boolean;
  canEdit?: boolean;
}

interface ShareResponse {
  id: string;
  token: string;
  expiresAt?: string;
  canView: boolean;
  canEdit: boolean;
  createdAt: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export function useDashboard(dashboardId?: string): UseDashboardResult {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch dashboard on mount or when ID changes
  useEffect(() => {
    if (dashboardId) {
      fetchDashboard(dashboardId);
    }
  }, [dashboardId]);

  const fetchDashboard = async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/dashboards/${id}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch dashboard: ${response.statusText}`);
      }

      const data = await response.json();
      setDashboard(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch dashboard';
      setError(message);
      console.error('Error fetching dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  const saveDashboard = useCallback(async (dashboardData: Partial<Dashboard>) => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const method = dashboardData.id ? 'PUT' : 'POST';
      const url = dashboardData.id
        ? `${API_BASE_URL}/dashboards/${dashboardData.id}`
        : `${API_BASE_URL}/dashboards`;

      const response = await fetch(url, {
        method,
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(dashboardData)
      });

      if (!response.ok) {
        throw new Error(`Failed to save dashboard: ${response.statusText}`);
      }

      const data = await response.json();
      setDashboard(data);

      // Optimistic update
      if (dashboard && dashboardData.id === dashboard.id) {
        setDashboard({ ...dashboard, ...data });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save dashboard';
      setError(message);
      console.error('Error saving dashboard:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [dashboard]);

  const updateLayout = useCallback(async (
    dashboardId: string,
    tabId: string,
    layout: Layout[]
  ) => {
    if (!dashboard || dashboard.id !== dashboardId) {
      return;
    }

    // Optimistic update
    const updatedTabs = dashboard.tabs?.map(tab => {
      if (tab.id === tabId) {
        return { ...tab, layout };
      }
      return tab;
    });

    setDashboard({ ...dashboard, tabs: updatedTabs });

    // Debounced save
    try {
      await saveDashboard({
        id: dashboardId,
        tabs: updatedTabs
      });
    } catch (err) {
      console.error('Error updating layout:', err);
      // Revert on error
      setDashboard(dashboard);
    }
  }, [dashboard, saveDashboard]);

  const addWidget = useCallback(async (
    dashboardId: string,
    tabId: string,
    widget: WidgetConfig
  ) => {
    if (!dashboard || dashboard.id !== dashboardId) {
      return;
    }

    // Optimistic update
    const updatedTabs = dashboard.tabs?.map(tab => {
      if (tab.id === tabId) {
        return {
          ...tab,
          widgets: [...tab.widgets, widget]
        };
      }
      return tab;
    });

    setDashboard({ ...dashboard, tabs: updatedTabs });

    try {
      await saveDashboard({
        id: dashboardId,
        tabs: updatedTabs
      });
    } catch (err) {
      console.error('Error adding widget:', err);
      setDashboard(dashboard);
    }
  }, [dashboard, saveDashboard]);

  const removeWidget = useCallback(async (
    dashboardId: string,
    widgetId: string
  ) => {
    if (!dashboard || dashboard.id !== dashboardId) {
      return;
    }

    // Optimistic update
    const updatedTabs = dashboard.tabs?.map(tab => ({
      ...tab,
      widgets: tab.widgets.filter(w => w.id !== widgetId),
      layout: tab.layout.filter(l => l.i !== widgetId)
    }));

    setDashboard({ ...dashboard, tabs: updatedTabs });

    try {
      await saveDashboard({
        id: dashboardId,
        tabs: updatedTabs
      });
    } catch (err) {
      console.error('Error removing widget:', err);
      setDashboard(dashboard);
    }
  }, [dashboard, saveDashboard]);

  const updateWidget = useCallback(async (
    dashboardId: string,
    widgetId: string,
    updates: Partial<WidgetConfig>
  ) => {
    if (!dashboard || dashboard.id !== dashboardId) {
      return;
    }

    // Optimistic update
    const updatedTabs = dashboard.tabs?.map(tab => ({
      ...tab,
      widgets: tab.widgets.map(w => {
        if (w.id === widgetId) {
          return { ...w, ...updates };
        }
        return w;
      })
    }));

    setDashboard({ ...dashboard, tabs: updatedTabs });

    try {
      await saveDashboard({
        id: dashboardId,
        tabs: updatedTabs
      });
    } catch (err) {
      console.error('Error updating widget:', err);
      setDashboard(dashboard);
    }
  }, [dashboard, saveDashboard]);

  const deleteDashboard = useCallback(async (dashboardId: string) => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/dashboards/${dashboardId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to delete dashboard: ${response.statusText}`);
      }

      setDashboard(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete dashboard';
      setError(message);
      console.error('Error deleting dashboard:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const shareDashboard = useCallback(async (
    dashboardId: string,
    options: ShareOptions
  ): Promise<ShareResponse> => {
    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/dashboards/${dashboardId}/share`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(options)
      });

      if (!response.ok) {
        throw new Error(`Failed to share dashboard: ${response.statusText}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to share dashboard';
      setError(message);
      console.error('Error sharing dashboard:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    dashboard,
    loading,
    error,
    saveDashboard,
    updateLayout,
    addWidget,
    removeWidget,
    updateWidget,
    deleteDashboard,
    shareDashboard
  };
}

export default useDashboard;
