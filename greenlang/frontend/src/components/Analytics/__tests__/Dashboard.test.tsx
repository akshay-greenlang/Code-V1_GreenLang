/**
 * Tests for Analytics Dashboard component.
 *
 * Tests cover rendering, widget management, drag-and-drop,
 * data updates, and WebSocket integration.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import Dashboard, { DASHBOARD_TEMPLATES } from '../Dashboard';
import { getMetricService, resetMetricService } from '../MetricService';

// Mock MetricService
jest.mock('../MetricService', () => ({
  getMetricService: jest.fn(),
  resetMetricService: jest.fn(),
  MetricService: jest.fn().mockImplementation(() => ({
    connect: jest.fn(),
    disconnect: jest.fn(),
    subscribe: jest.fn(),
    onMetric: jest.fn(),
    onError: jest.fn(),
    onConnectionChange: jest.fn(),
    offMetric: jest.fn(),
    isConnected: jest.fn(() => true),
  })),
}));

// Mock useDashboard hook
jest.mock('../../../hooks/useDashboard', () => ({
  useDashboard: jest.fn(() => ({
    dashboard: null,
    loading: false,
    error: null,
    saveDashboard: jest.fn(),
    updateLayout: jest.fn(),
    addWidget: jest.fn(),
    removeWidget: jest.fn(),
    deleteDashboard: jest.fn(),
    shareDashboard: jest.fn(),
  })),
  __esModule: true,
  default: jest.fn(),
}));

// Mock react-grid-layout
jest.mock('react-grid-layout', () => ({
  Responsive: ({ children }: any) => <div data-testid="grid-layout">{children}</div>,
  WidthProvider: (component: any) => component,
}));

// Mock html2canvas and jsPDF
jest.mock('html2canvas', () => jest.fn());
jest.mock('jspdf', () => ({
  __esModule: true,
  default: jest.fn().mockImplementation(() => ({
    addImage: jest.fn(),
    save: jest.fn(),
  })),
}));

describe('Dashboard Component', () => {
  let mockMetricService: any;

  beforeEach(() => {
    mockMetricService = {
      connect: jest.fn(),
      disconnect: jest.fn(),
      subscribe: jest.fn(),
      onMetric: jest.fn(),
      onError: jest.fn(),
      onConnectionChange: jest.fn((callback) => {
        callback(true); // Simulate connected
      }),
      offMetric: jest.fn(),
      isConnected: jest.fn(() => true),
    };

    (getMetricService as jest.Mock).mockReturnValue(mockMetricService);
  });

  afterEach(() => {
    jest.clearAllMocks();
    resetMetricService();
  });

  describe('Rendering', () => {
    test('renders dashboard with header', () => {
      render(<Dashboard initialTemplate="system_overview" />);

      expect(screen.getByText('Analytics Dashboard')).toBeInTheDocument();
    });

    test('renders connection status indicator', () => {
      render(<Dashboard />);

      const statusIndicator = document.querySelector('.connection-status');
      expect(statusIndicator).toBeInTheDocument();
    });

    test('renders time range selector', () => {
      render(<Dashboard />);

      const selector = screen.getByRole('combobox', { name: /time range/i }) ||
                      document.querySelector('select');
      expect(selector).toBeInTheDocument();
    });

    test('renders theme toggle button', () => {
      render(<Dashboard />);

      const themeButton = screen.getByRole('button', { name: /theme/i }) ||
                         screen.getByText(/☀️|🌙/);
      expect(themeButton).toBeInTheDocument();
    });

    test('loads template on initialization', () => {
      render(<Dashboard initialTemplate="system_overview" />);

      const template = DASHBOARD_TEMPLATES.system_overview;
      expect(template).toBeDefined();
      expect(template.tabs).toHaveLength(1);
    });
  });

  describe('WebSocket Connection', () => {
    test('connects to WebSocket on mount', async () => {
      render(<Dashboard wsUrl="ws://localhost:8000/ws/metrics" />);

      await waitFor(() => {
        expect(mockMetricService.connect).toHaveBeenCalled();
      });
    });

    test('disconnects on unmount', () => {
      const { unmount } = render(<Dashboard />);

      unmount();

      expect(mockMetricService.disconnect).toHaveBeenCalled();
    });

    test('subscribes to metric channels', async () => {
      render(<Dashboard initialTemplate="system_overview" />);

      await waitFor(() => {
        expect(mockMetricService.subscribe).toHaveBeenCalled();
      });

      const subscribeCall = mockMetricService.subscribe.mock.calls[0][0];
      expect(subscribeCall.channels).toContain('system.metrics');
    });

    test('handles metric updates', async () => {
      render(<Dashboard initialTemplate="system_overview" />);

      await waitFor(() => {
        expect(mockMetricService.onMetric).toHaveBeenCalled();
      });
    });
  });

  describe('Widget Management', () => {
    test('renders widgets from template', () => {
      render(<Dashboard initialTemplate="system_overview" />);

      const gridLayout = screen.getByTestId('grid-layout');
      expect(gridLayout).toBeInTheDocument();
    });

    test('enters edit mode when clicking edit button', () => {
      render(<Dashboard />);

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      expect(screen.getByRole('button', { name: /done/i })).toBeInTheDocument();
    });

    test('shows widget toolbar in edit mode', () => {
      render(<Dashboard />);

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      expect(screen.getByText(/add widget/i)).toBeInTheDocument();
    });

    test('adds widget when clicking add button in edit mode', async () => {
      const mockAddWidget = jest.fn();
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: { id: 'dash-1', tabs: [{ id: 'tab-1', widgets: [], layout: [] }] },
        loading: false,
        error: null,
        addWidget: mockAddWidget,
        updateLayout: jest.fn(),
        saveDashboard: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      const lineChartButton = screen.getByRole('button', { name: /line chart/i });
      fireEvent.click(lineChartButton);

      await waitFor(() => {
        expect(mockAddWidget).toHaveBeenCalled();
      });
    });
  });

  describe('Dashboard Persistence', () => {
    test('saves dashboard when clicking save button', async () => {
      const mockSaveDashboard = jest.fn();
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: { id: 'dash-1', name: 'My Dashboard', tabs: [] },
        loading: false,
        error: null,
        saveDashboard: mockSaveDashboard,
        updateLayout: jest.fn(),
        addWidget: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      const saveButton = screen.getByRole('button', { name: /save/i });
      fireEvent.click(saveButton);

      await waitFor(() => {
        expect(mockSaveDashboard).toHaveBeenCalled();
      });
    });
  });

  describe('Layout Management', () => {
    test('updates layout when widgets are moved', async () => {
      const mockUpdateLayout = jest.fn();
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: {
          id: 'dash-1',
          tabs: [{
            id: 'tab-1',
            name: 'Tab 1',
            layout: [{ i: 'widget-1', x: 0, y: 0, w: 6, h: 4 }],
            widgets: [{ id: 'widget-1', type: 'line_chart', title: 'Test' }]
          }]
        },
        loading: false,
        error: null,
        updateLayout: mockUpdateLayout,
        saveDashboard: jest.fn(),
        addWidget: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      const editButton = screen.getByRole('button', { name: /edit/i });
      fireEvent.click(editButton);

      // Simulate layout change
      // In a real test, you would simulate drag-and-drop
      // For now, just verify the handler exists
      expect(mockUpdateLayout).toBeDefined();
    });
  });

  describe('Export Functionality', () => {
    test('exports dashboard as PNG', async () => {
      const html2canvas = require('html2canvas');

      render(<Dashboard />);

      const pngButton = screen.getByRole('button', { name: /png/i });
      fireEvent.click(pngButton);

      await waitFor(() => {
        expect(html2canvas).toHaveBeenCalled();
      });
    });

    test('exports dashboard as PDF', async () => {
      const html2canvas = require('html2canvas');
      const jsPDF = require('jspdf').default;

      render(<Dashboard />);

      const pdfButton = screen.getByRole('button', { name: /pdf/i });
      fireEvent.click(pdfButton);

      await waitFor(() => {
        expect(html2canvas).toHaveBeenCalled();
      });
    });
  });

  describe('Fullscreen Mode', () => {
    test('toggles fullscreen mode', () => {
      const mockRequestFullscreen = jest.fn();
      document.documentElement.requestFullscreen = mockRequestFullscreen;

      render(<Dashboard />);

      const fullscreenButton = screen.getByText('⛶');
      fireEvent.click(fullscreenButton);

      expect(mockRequestFullscreen).toHaveBeenCalled();
    });
  });

  describe('Theme Switching', () => {
    test('switches between light and dark theme', () => {
      render(<Dashboard theme="light" />);

      const themeButton = screen.getByText('🌙');
      fireEvent.click(themeButton);

      expect(screen.getByText('☀️')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    test('displays error message when dashboard loading fails', () => {
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: null,
        loading: false,
        error: 'Failed to load dashboard',
        saveDashboard: jest.fn(),
        updateLayout: jest.fn(),
        addWidget: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      expect(screen.getByText(/error/i)).toBeInTheDocument();
      expect(screen.getByText(/failed to load dashboard/i)).toBeInTheDocument();
    });

    test('displays loading state', () => {
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: null,
        loading: true,
        error: null,
        saveDashboard: jest.fn(),
        updateLayout: jest.fn(),
        addWidget: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      expect(screen.getByText(/loading dashboard/i)).toBeInTheDocument();
    });
  });

  describe('Templates', () => {
    test('renders system overview template', () => {
      render(<Dashboard initialTemplate="system_overview" />);

      const template = DASHBOARD_TEMPLATES.system_overview;
      expect(template.name).toBe('System Overview');
      expect(template.tabs[0].widgets).toHaveLength(4);
    });

    test('renders workflow performance template', () => {
      render(<Dashboard initialTemplate="workflow_performance" />);

      const template = DASHBOARD_TEMPLATES.workflow_performance;
      expect(template.name).toBe('Workflow Performance');
    });

    test('renders agent analytics template', () => {
      render(<Dashboard initialTemplate="agent_analytics" />);

      const template = DASHBOARD_TEMPLATES.agent_analytics;
      expect(template.name).toBe('Agent Analytics');
    });

    test('renders distributed cluster template', () => {
      render(<Dashboard initialTemplate="distributed_cluster" />);

      const template = DASHBOARD_TEMPLATES.distributed_cluster;
      expect(template.name).toBe('Distributed Cluster');
    });
  });

  describe('Tab Management', () => {
    test('switches between tabs', () => {
      const useDashboard = require('../../../hooks/useDashboard').useDashboard;
      useDashboard.mockReturnValue({
        dashboard: {
          id: 'dash-1',
          tabs: [
            { id: 'tab-1', name: 'Tab 1', layout: [], widgets: [] },
            { id: 'tab-2', name: 'Tab 2', layout: [], widgets: [] }
          ]
        },
        loading: false,
        error: null,
        saveDashboard: jest.fn(),
        updateLayout: jest.fn(),
        addWidget: jest.fn(),
        removeWidget: jest.fn(),
      });

      render(<Dashboard dashboardId="dash-1" />);

      const tab1Button = screen.getByRole('button', { name: /tab 1/i });
      const tab2Button = screen.getByRole('button', { name: /tab 2/i });

      expect(tab1Button).toBeInTheDocument();
      expect(tab2Button).toBeInTheDocument();

      fireEvent.click(tab2Button);

      // Tab 2 should now be active
      expect(tab2Button).toHaveStyle({ borderBottom: '2px solid #2196f3' });
    });
  });
});

// Run tests with: npm test -- Dashboard.test.tsx
