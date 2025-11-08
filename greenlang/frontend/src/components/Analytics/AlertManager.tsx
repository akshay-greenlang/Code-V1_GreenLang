/**
 * Alert Manager component for creating and managing alert rules.
 *
 * Provides UI for alert configuration, notification setup, and alert history.
 */

import React, { useState, useEffect, useCallback } from 'react';

interface AlertRule {
  id: string;
  name: string;
  description?: string;
  rule_type: 'threshold' | 'rate_of_change' | 'absence' | 'anomaly';
  condition: Record<string, any>;
  notifications: NotificationConfig[];
  severity: 'info' | 'warning' | 'critical';
  enabled: boolean;
  tags: Record<string, string>;
}

interface NotificationConfig {
  channel: 'email' | 'slack' | 'pagerduty' | 'webhook';
  config: Record<string, any>;
  enabled: boolean;
}

interface AlertHistory {
  rule_id: string;
  rule_name: string;
  state: string;
  value: number;
  timestamp: string;
  labels: Record<string, string>;
}

interface ActiveAlert {
  rule_id: string;
  state: string;
  value: number;
  labels: Record<string, string>;
  started_at: string;
  last_evaluated: string;
  fingerprint: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const AlertManager: React.FC = () => {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<ActiveAlert[]>([]);
  const [alertHistory, setAlertHistory] = useState<AlertHistory[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedRule, setSelectedRule] = useState<AlertRule | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch alert rules
  useEffect(() => {
    fetchRules();
    fetchActiveAlerts();
  }, []);

  const fetchRules = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setRules(data);
      }
    } catch (error) {
      console.error('Error fetching alert rules:', error);
    }
  };

  const fetchActiveAlerts = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/active`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setActiveAlerts(data);
      }
    } catch (error) {
      console.error('Error fetching active alerts:', error);
    }
  };

  const fetchRuleHistory = async (ruleId: string) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules/${ruleId}/history`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAlertHistory(data);
      }
    } catch (error) {
      console.error('Error fetching alert history:', error);
    }
  };

  const createRule = async (rule: Partial<AlertRule>) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(rule)
      });

      if (response.ok) {
        await fetchRules();
        setShowCreateModal(false);
      } else {
        console.error('Failed to create rule:', await response.text());
      }
    } catch (error) {
      console.error('Error creating rule:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateRule = async (ruleId: string, updates: Partial<AlertRule>) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules/${ruleId}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updates)
      });

      if (response.ok) {
        await fetchRules();
      }
    } catch (error) {
      console.error('Error updating rule:', error);
    } finally {
      setLoading(false);
    }
  };

  const deleteRule = async (ruleId: string) => {
    if (!confirm('Are you sure you want to delete this alert rule?')) {
      return;
    }

    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules/${ruleId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        await fetchRules();
      }
    } catch (error) {
      console.error('Error deleting rule:', error);
    } finally {
      setLoading(false);
    }
  };

  const testRule = async (ruleId: string) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules/${ruleId}/test`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Rule test result: ${JSON.stringify(result, null, 2)}`);
      }
    } catch (error) {
      console.error('Error testing rule:', error);
    } finally {
      setLoading(false);
    }
  };

  const silenceRule = async (ruleId: string, duration: number) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/rules/${ruleId}/silence`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ duration })
      });

      if (response.ok) {
        await fetchRules();
      }
    } catch (error) {
      console.error('Error silencing rule:', error);
    } finally {
      setLoading(false);
    }
  };

  const acknowledgeAlert = async (fingerprint: string) => {
    setLoading(true);
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${API_BASE_URL}/alerts/${fingerprint}/acknowledge`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        await fetchActiveAlerts();
      }
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    } finally {
      setLoading(false);
    }
  };

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

  const getStateColor = (state: string) => {
    switch (state) {
      case 'firing':
        return '#f44336';
      case 'pending':
        return '#ff9800';
      case 'resolved':
        return '#4caf50';
      default:
        return '#999';
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h1>Alert Manager</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          style={{
            padding: '10px 20px',
            backgroundColor: '#2196f3',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Create Alert Rule
        </button>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: '1px solid #ddd', marginBottom: '20px' }}>
        <button style={{ padding: '10px 20px', borderBottom: '2px solid #2196f3' }}>Alert Rules</button>
        <button style={{ padding: '10px 20px' }} onClick={() => fetchActiveAlerts()}>Active Alerts ({activeAlerts.length})</button>
        <button style={{ padding: '10px 20px' }}>Alert History</button>
      </div>

      {/* Active Alerts Panel */}
      {activeAlerts.length > 0 && (
        <div style={{ marginBottom: '30px', padding: '15px', backgroundColor: '#fff3cd', borderLeft: '4px solid #ff9800', borderRadius: '4px' }}>
          <h3 style={{ marginTop: 0 }}>Active Alerts ({activeAlerts.length})</h3>
          {activeAlerts.map(alert => {
            const rule = rules.find(r => r.id === alert.rule_id);
            return (
              <div
                key={alert.fingerprint}
                style={{
                  padding: '10px',
                  marginBottom: '10px',
                  backgroundColor: '#fff',
                  borderLeft: `4px solid ${getStateColor(alert.state)}`,
                  borderRadius: '4px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <strong>{rule?.name || alert.rule_id}</strong>
                    <div style={{ fontSize: '14px', color: '#666', marginTop: '4px' }}>
                      State: <span style={{ color: getStateColor(alert.state) }}>{alert.state}</span> |
                      Value: {alert.value?.toFixed(2)} |
                      Started: {new Date(alert.started_at).toLocaleString()}
                    </div>
                    {Object.entries(alert.labels).length > 0 && (
                      <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                        Labels: {JSON.stringify(alert.labels)}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => acknowledgeAlert(alert.fingerprint)}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#4caf50',
                      color: '#fff',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                  >
                    Acknowledge
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Alert Rules Table */}
      <div style={{ backgroundColor: '#fff', borderRadius: '4px', padding: '20px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #ddd' }}>
              <th style={{ padding: '12px', textAlign: 'left' }}>Name</th>
              <th style={{ padding: '12px', textAlign: 'left' }}>Type</th>
              <th style={{ padding: '12px', textAlign: 'left' }}>Severity</th>
              <th style={{ padding: '12px', textAlign: 'left' }}>Status</th>
              <th style={{ padding: '12px', textAlign: 'left' }}>Notifications</th>
              <th style={{ padding: '12px', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {rules.map(rule => (
              <tr key={rule.id} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '12px' }}>
                  <div>
                    <strong>{rule.name}</strong>
                    {rule.description && (
                      <div style={{ fontSize: '12px', color: '#999' }}>{rule.description}</div>
                    )}
                  </div>
                </td>
                <td style={{ padding: '12px' }}>{rule.rule_type}</td>
                <td style={{ padding: '12px' }}>
                  <span
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: getSeverityColor(rule.severity),
                      color: '#fff',
                      fontSize: '12px'
                    }}
                  >
                    {rule.severity.toUpperCase()}
                  </span>
                </td>
                <td style={{ padding: '12px' }}>
                  <span
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      backgroundColor: rule.enabled ? '#4caf50' : '#999',
                      color: '#fff',
                      fontSize: '12px'
                    }}
                  >
                    {rule.enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </td>
                <td style={{ padding: '12px' }}>
                  {rule.notifications.length} channel(s)
                </td>
                <td style={{ padding: '12px', textAlign: 'right' }}>
                  <button
                    onClick={() => testRule(rule.id)}
                    style={{ marginRight: '8px', padding: '4px 8px', cursor: 'pointer' }}
                  >
                    Test
                  </button>
                  <button
                    onClick={() => {
                      const duration = prompt('Silence duration (hours):');
                      if (duration) {
                        silenceRule(rule.id, parseInt(duration) * 3600);
                      }
                    }}
                    style={{ marginRight: '8px', padding: '4px 8px', cursor: 'pointer' }}
                  >
                    Silence
                  </button>
                  <button
                    onClick={() => {
                      setSelectedRule(rule);
                      setShowCreateModal(true);
                    }}
                    style={{ marginRight: '8px', padding: '4px 8px', cursor: 'pointer' }}
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => deleteRule(rule.id)}
                    style={{ padding: '4px 8px', cursor: 'pointer', color: '#f44336' }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {rules.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
            No alert rules configured. Click "Create Alert Rule" to get started.
          </div>
        )}
      </div>

      {/* Create/Edit Modal */}
      {showCreateModal && (
        <CreateAlertModal
          rule={selectedRule}
          onClose={() => {
            setShowCreateModal(false);
            setSelectedRule(null);
          }}
          onSave={(rule) => {
            if (rule.id) {
              updateRule(rule.id, rule);
            } else {
              createRule(rule);
            }
          }}
        />
      )}

      {loading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999
        }}>
          <div style={{ padding: '20px', backgroundColor: '#fff', borderRadius: '4px' }}>
            Loading...
          </div>
        </div>
      )}
    </div>
  );
};

// Create Alert Modal Component
interface CreateAlertModalProps {
  rule: AlertRule | null;
  onClose: () => void;
  onSave: (rule: Partial<AlertRule>) => void;
}

const CreateAlertModal: React.FC<CreateAlertModalProps> = ({ rule, onClose, onSave }) => {
  const [formData, setFormData] = useState<Partial<AlertRule>>(
    rule || {
      name: '',
      description: '',
      rule_type: 'threshold',
      condition: { metric: '', operator: '>', threshold: 0 },
      notifications: [],
      severity: 'warning',
      enabled: true,
      tags: {}
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999
    }}>
      <div style={{
        backgroundColor: '#fff',
        borderRadius: '8px',
        padding: '30px',
        minWidth: '600px',
        maxHeight: '90vh',
        overflow: 'auto'
      }}>
        <h2>{rule ? 'Edit Alert Rule' : 'Create Alert Rule'}</h2>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>

          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            />
          </div>

          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Rule Type</label>
            <select
              value={formData.rule_type}
              onChange={(e) => setFormData({ ...formData, rule_type: e.target.value as any })}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="threshold">Threshold</option>
              <option value="rate_of_change">Rate of Change</option>
              <option value="absence">Absence</option>
              <option value="anomaly">Anomaly</option>
            </select>
          </div>

          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Severity</label>
            <select
              value={formData.severity}
              onChange={(e) => setFormData({ ...formData, severity: e.target.value as any })}
              style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid #ddd' }}
            >
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="critical">Critical</option>
            </select>
          </div>

          <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: '10px 20px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                backgroundColor: '#fff',
                cursor: 'pointer'
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              style={{
                padding: '10px 20px',
                borderRadius: '4px',
                border: 'none',
                backgroundColor: '#2196f3',
                color: '#fff',
                cursor: 'pointer'
              }}
            >
              Save
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AlertManager;
