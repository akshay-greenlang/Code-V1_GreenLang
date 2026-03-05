import React from 'react';
import { Card, CardContent, Typography, Chip, LinearProgress, Box } from '@mui/material';
import DataTable, { Column } from '../common/DataTable';
import type { RiskManagementRecord, RiskResponseAction } from '../../types';
import { formatDate, formatCurrency } from '../../utils/formatters';

interface ResponseTrackerProps { records: RiskManagementRecord[]; }

const ResponseTracker: React.FC<ResponseTrackerProps> = ({ records }) => {
  const allActions = records.flatMap((r) => r.response_actions.map((a) => ({ ...a, riskName: r.risk_name })));
  const columns: Column<typeof allActions[0]>[] = [
    { id: 'risk', label: 'Risk', accessor: (r) => r.riskName, sortAccessor: (r) => r.riskName },
    { id: 'action', label: 'Action', accessor: (r) => r.description, sortAccessor: (r) => r.description },
    { id: 'responsible', label: 'Responsible', accessor: (r) => r.responsible },
    { id: 'due', label: 'Due Date', accessor: (r) => formatDate(r.due_date), sortAccessor: (r) => r.due_date },
    { id: 'status', label: 'Status', accessor: (r) => <Chip label={r.status.replace(/_/g, ' ')} size="small" color={r.status === 'completed' ? 'success' : r.status === 'overdue' ? 'error' : r.status === 'in_progress' ? 'primary' : 'default'} />, sortAccessor: (r) => r.status },
    { id: 'effectiveness', label: 'Effectiveness', accessor: (r) => <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><LinearProgress variant="determinate" value={r.effectiveness} sx={{ width: 60, height: 6, borderRadius: 3 }} /><Typography variant="caption">{r.effectiveness}%</Typography></Box>, sortAccessor: (r) => r.effectiveness },
    { id: 'cost', label: 'Cost', accessor: (r) => formatCurrency(r.cost, 'USD', true), align: 'right', sortAccessor: (r) => r.cost },
  ];
  return <Card><CardContent sx={{ p: 0, '&:last-child': { pb: 0 } }}><DataTable title="Risk Response Actions" columns={columns} data={allActions} keyAccessor={(r) => r.id} /></CardContent></Card>;
};

export default ResponseTracker;
