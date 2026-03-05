/**
 * ChangeMonitor - Base year change monitor.
 */
import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import type { Recalculation } from '../../types';
import { formatDate } from '../../utils/formatters';

interface ChangeMonitorProps { recalculations: Recalculation[]; }

const ChangeMonitor: React.FC<ChangeMonitorProps> = ({ recalculations }) => {
  const columns: Column<Recalculation>[] = [
    { id: 'trigger', label: 'Trigger', accessor: (r) => r.trigger.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) },
    { id: 'date', label: 'Date', accessor: (r) => formatDate(r.trigger_date), sortAccessor: (r) => r.trigger_date },
    { id: 'impact', label: 'Impact', accessor: (r) => `${r.impact_on_base_year_pct >= 0 ? '+' : ''}${r.impact_on_base_year_pct.toFixed(1)}%`, align: 'right' },
    { id: 'threshold', label: 'Exceeds 5%', accessor: (r) => r.threshold_exceeded ? 'Yes' : 'No', align: 'center' },
    { id: 'status', label: 'Status', accessor: (r) => <StatusBadge status={r.status} variant="target" /> },
  ];
  return <DataTable columns={columns} data={recalculations} keyAccessor={(r) => r.id} title="Base Year Changes" defaultSortColumn="date" defaultSortDirection="desc" />;
};

export default ChangeMonitor;
