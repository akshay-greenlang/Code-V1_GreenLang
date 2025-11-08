/**
 * Analytics Dashboard with real-time metrics visualization.
 *
 * Provides a customizable grid-based dashboard with drag-and-drop widgets,
 * real-time metric updates, and multiple visualization options.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Responsive, WidthProvider, Layout } from 'react-grid-layout';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

import { getMetricService, Metric, MetricSubscription } from './MetricService';
import { useDashboard } from '../../hooks/useDashboard';
import LineChart from './widgets/LineChart';
import BarChart from './widgets/BarChart';
import GaugeChart from './widgets/GaugeChart';
import StatCard from './widgets/StatCard';
import TableWidget from './widgets/TableWidget';
import HeatmapChart from './widgets/HeatmapChart';
import PieChart from './widgets/PieChart';
import AlertWidget from './widgets/AlertWidget';

const ResponsiveGridLayout = WidthProvider(Responsive);

// Widget type definitions
export type WidgetType =
  | 'line_chart'
  | 'bar_chart'
  | 'gauge_chart'
  | 'stat_card'
  | 'table'
  | 'heatmap'
  | 'pie_chart'
  | 'alert';

export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  config: Record<string, any>;
  dataSource: {
    channel: string;
    metricName?: string;
    tags?: Record<string, string>;
    aggregation?: string;
  };
}

export interface DashboardTab {
  id: string;
  name: string;
  layout: Layout[];
  widgets: WidgetConfig[];
}

interface DashboardTemplateConfig {
  name: string;
  description: string;
  tabs: DashboardTab[];
}

// Default dashboard templates
const DASHBOARD_TEMPLATES: Record<string, DashboardTemplateConfig> = {
  system_overview: {
    name: 'System Overview',
    description: 'CPU, memory, disk, and network metrics',
    tabs: [
      {
        id: 'system',
        name: 'System',
        layout: [
          { i: 'cpu', x: 0, y: 0, w: 6, h: 4 },
          { i: 'memory', x: 6, y: 0, w: 6, h: 4 },
          { i: 'disk', x: 0, y: 4, w: 6, h: 4 },
          { i: 'network', x: 6, y: 4, w: 6, h: 4 },
        ],
        widgets: [
          {
            id: 'cpu',
            type: 'line_chart',
            title: 'CPU Usage',
            config: { yAxisLabel: 'Percentage', unit: '%' },
            dataSource: { channel: 'system.metrics', metricName: 'cpu.percent' }
          },
          {
            id: 'memory',
            type: 'line_chart',
            title: 'Memory Usage',
            config: { yAxisLabel: 'Percentage', unit: '%' },
            dataSource: { channel: 'system.metrics', metricName: 'memory.percent' }
          },
          {
            id: 'disk',
            type: 'gauge_chart',
            title: 'Disk Usage',
            config: { max: 100, unit: '%' },
            dataSource: { channel: 'system.metrics', metricName: 'disk.percent' }
          },
          {
            id: 'network',
            type: 'stat_card',
            title: 'Network I/O',
            config: { format: 'bytes' },
            dataSource: { channel: 'system.metrics', metricName: 'network.bytes_sent' }
          },
        ]
      }
    ]
  },
  workflow_performance: {
    name: 'Workflow Performance',
    description: 'Workflow executions, success rate, and duration',
    tabs: [
      {
        id: 'workflows',
        name: 'Workflows',
        layout: [
          { i: 'executions', x: 0, y: 0, w: 8, h: 4 },
          { i: 'success_rate', x: 8, y: 0, w: 4, h: 4 },
          { i: 'duration', x: 0, y: 4, w: 6, h: 4 },
          { i: 'status', x: 6, y: 4, w: 6, h: 4 },
        ],
        widgets: [
          {
            id: 'executions',
            type: 'line_chart',
            title: 'Workflow Executions',
            config: { yAxisLabel: 'Count' },
            dataSource: { channel: 'workflow.metrics', metricName: 'executions.total' }
          },
          {
            id: 'success_rate',
            type: 'gauge_chart',
            title: 'Success Rate',
            config: { max: 100, unit: '%' },
            dataSource: { channel: 'workflow.metrics', metricName: 'success_rate' }
          },
          {
            id: 'duration',
            type: 'bar_chart',
            title: 'Duration Percentiles',
            config: { orientation: 'vertical' },
            dataSource: { channel: 'workflow.metrics', metricName: 'duration' }
          },
          {
            id: 'status',
            type: 'pie_chart',
            title: 'Execution Status',
            config: {},
            dataSource: { channel: 'workflow.metrics', metricName: 'executions' }
          },
        ]
      }
    ]
  },
  agent_analytics: {
    name: 'Agent Analytics',
    description: 'Agent calls, latency, and error rates',
    tabs: [
      {
        id: 'agents',
        name: 'Agents',
        layout: [
          { i: 'calls', x: 0, y: 0, w: 6, h: 4 },
          { i: 'latency', x: 6, y: 0, w: 6, h: 4 },
          { i: 'errors', x: 0, y: 4, w: 6, h: 4 },
          { i: 'top_agents', x: 6, y: 4, w: 6, h: 4 },
        ],
        widgets: [
          {
            id: 'calls',
            type: 'line_chart',
            title: 'Agent Calls',
            config: { yAxisLabel: 'Count' },
            dataSource: { channel: 'agent.metrics', metricName: 'calls.total' }
          },
          {
            id: 'latency',
            type: 'line_chart',
            title: 'Agent Latency',
            config: { yAxisLabel: 'Milliseconds', unit: 'ms' },
            dataSource: { channel: 'agent.metrics', metricName: 'latency.avg' }
          },
          {
            id: 'errors',
            type: 'bar_chart',
            title: 'Errors by Type',
            config: { orientation: 'horizontal' },
            dataSource: { channel: 'agent.metrics', metricName: 'errors.by_type' }
          },
          {
            id: 'top_agents',
            type: 'table',
            title: 'Top Agents',
            config: { columns: ['agent', 'calls', 'avg_latency'] },
            dataSource: { channel: 'agent.metrics', metricName: 'top_agents' }
          },
        ]
      }
    ]
  },
  distributed_cluster: {
    name: 'Distributed Cluster',
    description: 'Node health, task distribution, and throughput',
    tabs: [
      {
        id: 'cluster',
        name: 'Cluster',
        layout: [
          { i: 'nodes', x: 0, y: 0, w: 4, h: 4 },
          { i: 'tasks', x: 4, y: 0, w: 8, h: 4 },
          { i: 'throughput', x: 0, y: 4, w: 6, h: 4 },
          { i: 'distribution', x: 6, y: 4, w: 6, h: 4 },
        ],
        widgets: [
          {
            id: 'nodes',
            type: 'stat_card',
            title: 'Active Nodes',
            config: { format: 'number' },
            dataSource: { channel: 'distributed.metrics', metricName: 'nodes.active' }
          },
          {
            id: 'tasks',
            type: 'line_chart',
            title: 'Task Queue',
            config: { yAxisLabel: 'Tasks' },
            dataSource: { channel: 'distributed.metrics', metricName: 'tasks' }
          },
          {
            id: 'throughput',
            type: 'line_chart',
            title: 'Throughput',
            config: { yAxisLabel: 'Tasks/sec' },
            dataSource: { channel: 'distributed.metrics', metricName: 'throughput' }
          },
          {
            id: 'distribution',
            type: 'pie_chart',
            title: 'Task Distribution',
            config: {},
            dataSource: { channel: 'distributed.metrics', metricName: 'tasks' }
          },
        ]
      }
    ]
  }
};

interface DashboardProps {
  dashboardId?: string;
  initialTemplate?: string;
  wsUrl?: string;
  wsToken?: string;
  theme?: 'light' | 'dark';
  autoRefresh?: number;
}

const Dashboard: React.FC<DashboardProps> = ({
  dashboardId,
  initialTemplate = 'system_overview',
  wsUrl = 'ws://localhost:8000/ws/metrics',
  wsToken,
  theme = 'light',
  autoRefresh = 5000,
}) => {
  const dashboardRef = useRef<HTMLDivElement>(null);
  const metricService = useRef(getMetricService(wsUrl, wsToken || undefined));

  const [tabs, setTabs] = useState<DashboardTab[]>([]);
  const [activeTab, setActiveTab] = useState<string>('');
  const [connected, setConnected] = useState<boolean>(false);
  const [fullscreen, setFullscreen] = useState<boolean>(false);
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [refreshInterval, setRefreshInterval] = useState<number>(autoRefresh);
  const [metricData, setMetricData] = useState<Map<string, Metric[]>>(new Map());
  const [editMode, setEditMode] = useState<boolean>(false);
  const [currentTheme, setCurrentTheme] = useState<'light' | 'dark'>(theme);

  const {
    dashboard,
    loading,
    error,
    saveDashboard,
    updateLayout,
    addWidget,
    removeWidget,
  } = useDashboard(dashboardId);

  // Initialize dashboard from template or saved state
  useEffect(() => {
    if (dashboard && dashboard.tabs) {
      setTabs(dashboard.tabs);
      if (dashboard.tabs.length > 0) {
        setActiveTab(dashboard.tabs[0].id);
      }
    } else if (initialTemplate && DASHBOARD_TEMPLATES[initialTemplate]) {
      const template = DASHBOARD_TEMPLATES[initialTemplate];
      setTabs(template.tabs);
      if (template.tabs.length > 0) {
        setActiveTab(template.tabs[0].id);
      }
    }
  }, [dashboard, initialTemplate]);

  // Connect to WebSocket
  useEffect(() => {
    const ms = metricService.current;

    ms.onConnectionChange((isConnected) => {
      setConnected(isConnected);
    });

    ms.onError((err) => {
      console.error('MetricService error:', err);
    });

    ms.connect();

    return () => {
      ms.disconnect();
    };
  }, []);

  // Subscribe to metric channels
  useEffect(() => {
    if (!connected || tabs.length === 0) {
      return;
    }

    const channels = new Set<string>();
    tabs.forEach(tab => {
      tab.widgets.forEach(widget => {
        channels.add(widget.dataSource.channel);
      });
    });

    const subscription: MetricSubscription = {
      channels: Array.from(channels),
      compression: true,
      aggregation_interval: '5s'
    };

    metricService.current.subscribe(subscription);

    // Set up metric callbacks
    channels.forEach(channel => {
      metricService.current.onMetric(channel, (metric: Metric) => {
        handleMetricUpdate(channel, metric);
      });
    });

    return () => {
      // Clean up callbacks
      channels.forEach(channel => {
        metricService.current.offMetric(channel, handleMetricUpdate);
      });
    };
  }, [connected, tabs]);

  // Handle metric updates
  const handleMetricUpdate = useCallback((channel: string, metric: Metric) => {
    setMetricData(prev => {
      const key = `${channel}:${metric.name}`;
      const existing = prev.get(key) || [];
      const updated = [...existing, metric];

      // Limit to last 1000 points
      if (updated.length > 1000) {
        updated.shift();
      }

      const newMap = new Map(prev);
      newMap.set(key, updated);
      return newMap;
    });
  }, []);

  // Handle layout change
  const handleLayoutChange = useCallback((layout: Layout[], layouts: any) => {
    if (!editMode) {
      return;
    }

    const currentTab = tabs.find(t => t.id === activeTab);
    if (!currentTab) {
      return;
    }

    const updatedTabs = tabs.map(tab => {
      if (tab.id === activeTab) {
        return { ...tab, layout };
      }
      return tab;
    });

    setTabs(updatedTabs);

    // Save to backend
    if (dashboardId) {
      updateLayout(dashboardId, activeTab, layout);
    }
  }, [activeTab, editMode, tabs, dashboardId, updateLayout]);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      dashboardRef.current?.requestFullscreen();
      setFullscreen(true);
    } else {
      document.exitFullscreen();
      setFullscreen(false);
    }
  }, []);

  // Export dashboard as PNG
  const exportAsPNG = useCallback(async () => {
    if (!dashboardRef.current) {
      return;
    }

    try {
      const canvas = await html2canvas(dashboardRef.current);
      const link = document.createElement('a');
      link.download = `dashboard-${Date.now()}.png`;
      link.href = canvas.toDataURL();
      link.click();
    } catch (error) {
      console.error('Error exporting PNG:', error);
    }
  }, []);

  // Export dashboard as PDF
  const exportAsPDF = useCallback(async () => {
    if (!dashboardRef.current) {
      return;
    }

    try {
      const canvas = await html2canvas(dashboardRef.current);
      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });
      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
      pdf.save(`dashboard-${Date.now()}.pdf`);
    } catch (error) {
      console.error('Error exporting PDF:', error);
    }
  }, []);

  // Add new widget
  const handleAddWidget = useCallback((widgetType: WidgetType) => {
    if (!activeTab) {
      return;
    }

    const newWidget: WidgetConfig = {
      id: `widget-${Date.now()}`,
      type: widgetType,
      title: `New ${widgetType}`,
      config: {},
      dataSource: {
        channel: 'system.metrics',
        metricName: 'cpu.percent'
      }
    };

    const currentTab = tabs.find(t => t.id === activeTab);
    if (!currentTab) {
      return;
    }

    const newLayout: Layout = {
      i: newWidget.id,
      x: 0,
      y: Infinity, // Add to bottom
      w: 6,
      h: 4
    };

    const updatedTabs = tabs.map(tab => {
      if (tab.id === activeTab) {
        return {
          ...tab,
          layout: [...tab.layout, newLayout],
          widgets: [...tab.widgets, newWidget]
        };
      }
      return tab;
    });

    setTabs(updatedTabs);

    if (dashboardId) {
      addWidget(dashboardId, activeTab, newWidget);
    }
  }, [activeTab, tabs, dashboardId, addWidget]);

  // Remove widget
  const handleRemoveWidget = useCallback((widgetId: string) => {
    const updatedTabs = tabs.map(tab => {
      if (tab.id === activeTab) {
        return {
          ...tab,
          layout: tab.layout.filter(l => l.i !== widgetId),
          widgets: tab.widgets.filter(w => w.id !== widgetId)
        };
      }
      return tab;
    });

    setTabs(updatedTabs);

    if (dashboardId) {
      removeWidget(dashboardId, widgetId);
    }
  }, [activeTab, tabs, dashboardId, removeWidget]);

  // Save dashboard
  const handleSaveDashboard = useCallback(() => {
    if (dashboardId) {
      saveDashboard({
        id: dashboardId,
        name: dashboard?.name || 'My Dashboard',
        tabs
      });
    }
  }, [dashboardId, dashboard, tabs, saveDashboard]);

  // Render widget based on type
  const renderWidget = useCallback((widget: WidgetConfig) => {
    const dataKey = `${widget.dataSource.channel}:${widget.dataSource.metricName}`;
    const data = metricData.get(dataKey) || [];

    const commonProps = {
      title: widget.title,
      data,
      config: widget.config,
      onRemove: editMode ? () => handleRemoveWidget(widget.id) : undefined
    };

    switch (widget.type) {
      case 'line_chart':
        return <LineChart {...commonProps} />;
      case 'bar_chart':
        return <BarChart {...commonProps} />;
      case 'gauge_chart':
        return <GaugeChart {...commonProps} />;
      case 'stat_card':
        return <StatCard {...commonProps} />;
      case 'table':
        return <TableWidget {...commonProps} />;
      case 'heatmap':
        return <HeatmapChart {...commonProps} />;
      case 'pie_chart':
        return <PieChart {...commonProps} />;
      case 'alert':
        return <AlertWidget {...commonProps} />;
      default:
        return <div>Unknown widget type: {widget.type}</div>;
    }
  }, [metricData, editMode, handleRemoveWidget]);

  // Get current tab
  const currentTab = tabs.find(t => t.id === activeTab);

  return (
    <div
      ref={dashboardRef}
      className={`analytics-dashboard theme-${currentTheme} ${fullscreen ? 'fullscreen' : ''}`}
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: currentTheme === 'dark' ? '#1a1a1a' : '#f5f5f5',
        color: currentTheme === 'dark' ? '#ffffff' : '#000000'
      }}
    >
      {/* Header */}
      <div
        className="dashboard-header"
        style={{
          padding: '16px',
          borderBottom: `1px solid ${currentTheme === 'dark' ? '#333' : '#ddd'}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <h1 style={{ margin: 0 }}>Analytics Dashboard</h1>
          <div
            className={`connection-status ${connected ? 'connected' : 'disconnected'}`}
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: connected ? '#4caf50' : '#f44336'
            }}
            title={connected ? 'Connected' : 'Disconnected'}
          />
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          {/* Time range selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            style={{
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #ccc'
            }}
          >
            <option value="1h">Last 1 hour</option>
            <option value="6h">Last 6 hours</option>
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="custom">Custom</option>
          </select>

          {/* Refresh interval selector */}
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            style={{
              padding: '8px',
              borderRadius: '4px',
              border: '1px solid #ccc'
            }}
          >
            <option value="5000">5s</option>
            <option value="10000">10s</option>
            <option value="30000">30s</option>
            <option value="60000">1m</option>
            <option value="300000">5m</option>
          </select>

          {/* Theme toggle */}
          <button
            onClick={() => setCurrentTheme(t => t === 'light' ? 'dark' : 'light')}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              cursor: 'pointer'
            }}
          >
            {currentTheme === 'light' ? '🌙' : '☀️'}
          </button>

          {/* Edit mode toggle */}
          <button
            onClick={() => setEditMode(!editMode)}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              cursor: 'pointer',
              backgroundColor: editMode ? '#2196f3' : 'transparent',
              color: editMode ? '#fff' : 'inherit'
            }}
          >
            {editMode ? 'Done' : 'Edit'}
          </button>

          {/* Fullscreen toggle */}
          <button
            onClick={toggleFullscreen}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              cursor: 'pointer'
            }}
          >
            ⛶
          </button>

          {/* Export buttons */}
          <button
            onClick={exportAsPNG}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              cursor: 'pointer'
            }}
          >
            PNG
          </button>
          <button
            onClick={exportAsPDF}
            style={{
              padding: '8px 16px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              cursor: 'pointer'
            }}
          >
            PDF
          </button>

          {/* Save button */}
          {editMode && (
            <button
              onClick={handleSaveDashboard}
              style={{
                padding: '8px 16px',
                borderRadius: '4px',
                border: '1px solid #ccc',
                cursor: 'pointer',
                backgroundColor: '#4caf50',
                color: '#fff'
              }}
            >
              Save
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div
        className="dashboard-tabs"
        style={{
          padding: '8px 16px',
          borderBottom: `1px solid ${currentTheme === 'dark' ? '#333' : '#ddd'}`,
          display: 'flex',
          gap: '8px'
        }}
      >
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '8px 16px',
              borderRadius: '4px 4px 0 0',
              border: 'none',
              cursor: 'pointer',
              backgroundColor: tab.id === activeTab
                ? (currentTheme === 'dark' ? '#333' : '#fff')
                : 'transparent',
              borderBottom: tab.id === activeTab ? '2px solid #2196f3' : 'none'
            }}
          >
            {tab.name}
          </button>
        ))}
      </div>

      {/* Widget toolbar (in edit mode) */}
      {editMode && (
        <div
          className="widget-toolbar"
          style={{
            padding: '8px 16px',
            borderBottom: `1px solid ${currentTheme === 'dark' ? '#333' : '#ddd'}`,
            display: 'flex',
            gap: '8px'
          }}
        >
          <span>Add Widget:</span>
          <button onClick={() => handleAddWidget('line_chart')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Line Chart</button>
          <button onClick={() => handleAddWidget('bar_chart')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Bar Chart</button>
          <button onClick={() => handleAddWidget('gauge_chart')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Gauge</button>
          <button onClick={() => handleAddWidget('stat_card')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Stat Card</button>
          <button onClick={() => handleAddWidget('table')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Table</button>
          <button onClick={() => handleAddWidget('heatmap')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Heatmap</button>
          <button onClick={() => handleAddWidget('pie_chart')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Pie Chart</button>
          <button onClick={() => handleAddWidget('alert')} style={{ padding: '4px 8px', cursor: 'pointer' }}>Alerts</button>
        </div>
      )}

      {/* Grid layout */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        {currentTab && (
          <ResponsiveGridLayout
            className="layout"
            layouts={{ lg: currentTab.layout }}
            breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
            cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
            rowHeight={100}
            isDraggable={editMode}
            isResizable={editMode}
            onLayoutChange={handleLayoutChange}
          >
            {currentTab.widgets.map(widget => (
              <div key={widget.id} style={{ backgroundColor: currentTheme === 'dark' ? '#2a2a2a' : '#fff', borderRadius: '8px', padding: '16px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                {renderWidget(widget)}
              </div>
            ))}
          </ResponsiveGridLayout>
        )}
      </div>

      {/* Loading/Error states */}
      {loading && (
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}>
          Loading dashboard...
        </div>
      )}
      {error && (
        <div style={{ position: 'absolute', top: '16px', right: '16px', padding: '16px', backgroundColor: '#f44336', color: '#fff', borderRadius: '4px' }}>
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
export { DASHBOARD_TEMPLATES };
