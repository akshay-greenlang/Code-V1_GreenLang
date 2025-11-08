/**
 * Alert widget showing active alerts with severity indicators.
 */

import React, { useState, useEffect } from 'react';
import { Metric } from '../MetricService';

interface Alert {
  id: string;
  name: string;
  severity: 'critical' | 'warning' | 'info';
  message: string;
  timestamp: string;
  state: 'firing' | 'pending' | 'resolved';
}

interface AlertWidgetProps {
  title: string;
  data: Metric[];
  config?: {
    maxAlerts?: number;
    showResolved?: boolean;
  };
  onRemove?: () => void;
}

const AlertWidget: React.FC<AlertWidgetProps> = ({ title, data, config, onRemove }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    // Convert metrics to alerts
    // In a real implementation, this would fetch from the alert API
    const convertedAlerts: Alert[] = data.slice(-10).map((metric, index) => ({
      id: `alert-${index}`,
      name: metric.name || 'Unknown Alert',
      severity: 'warning' as const,
      message: `Metric ${metric.name} exceeded threshold`,
      timestamp: metric.timestamp,
      state: 'firing' as const
    }));

    setAlerts(convertedAlerts);
  }, [data]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return '#f44336';
      case 'warning':
        return '#ff9800';
      case 'info':
        return '#2196f3';
      default:
        return '#999';
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'firing':
        return '🔥';
      case 'pending':
        return '⏳';
      case 'resolved':
        return '✅';
      default:
        return '❓';
    }
  };

  const handleAcknowledge = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, state: 'resolved' as const } : alert
    ));
  };

  const filteredAlerts = alerts.filter(alert =>
    config?.showResolved !== false || alert.state !== 'resolved'
  ).slice(0, config?.maxAlerts || 10);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontSize: '12px', color: '#999' }}>
            {filteredAlerts.filter(a => a.state === 'firing').length} active
          </span>
          {onRemove && (
            <button onClick={onRemove} style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '18px', color: '#f44336' }}>
              ×
            </button>
          )}
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        {filteredAlerts.map(alert => (
          <div
            key={alert.id}
            style={{
              padding: '12px',
              marginBottom: '8px',
              borderLeft: `4px solid ${getSeverityColor(alert.severity)}`,
              backgroundColor: '#f5f5f5',
              borderRadius: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start'
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <span>{getStateIcon(alert.state)}</span>
                <strong>{alert.name}</strong>
                <span style={{
                  fontSize: '10px',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  backgroundColor: getSeverityColor(alert.severity),
                  color: '#fff'
                }}>
                  {alert.severity.toUpperCase()}
                </span>
              </div>
              <div style={{ fontSize: '14px', color: '#666', marginBottom: '4px' }}>
                {alert.message}
              </div>
              <div style={{ fontSize: '12px', color: '#999' }}>
                {new Date(alert.timestamp).toLocaleString()}
              </div>
            </div>

            {alert.state === 'firing' && (
              <button
                onClick={() => handleAcknowledge(alert.id)}
                style={{
                  padding: '4px 8px',
                  borderRadius: '4px',
                  border: '1px solid #ccc',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Resolve
              </button>
            )}
          </div>
        ))}
      </div>

      {filteredAlerts.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>
          No active alerts
        </div>
      )}
    </div>
  );
};

export default AlertWidget;
