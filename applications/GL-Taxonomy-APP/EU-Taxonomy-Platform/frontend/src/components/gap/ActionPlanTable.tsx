/**
 * ActionPlanTable - Table of gap remediation actions.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip, LinearProgress, Box } from '@mui/material';
import DataTable from '../common/DataTable';

const DEMO = [
  { id: '1', gap: 'Missing DNSH water assessment for 3 activities', category: 'DNSH', priority: 'High', responsible: 'ESG Team', status: 'In Progress', completion: 40, due: '2025-05-30' },
  { id: '2', gap: 'Incomplete EPC data for mortgage portfolio', category: 'Data', priority: 'Critical', responsible: 'Real Estate', status: 'In Progress', completion: 25, due: '2025-06-30' },
  { id: '3', gap: 'Anti-corruption policy not publicly available', category: 'Safeguards', priority: 'Medium', responsible: 'Compliance', status: 'Pending', completion: 0, due: '2025-04-15' },
  { id: '4', gap: 'Climate risk assessment not conducted for CCA', category: 'DNSH', priority: 'High', responsible: 'Risk Team', status: 'In Progress', completion: 60, due: '2025-05-15' },
  { id: '5', gap: 'SC evidence not third-party verified', category: 'SC', priority: 'Medium', responsible: 'External Audit', status: 'Pending', completion: 0, due: '2025-07-31' },
];

const columns = [
  { key: 'gap' as const, label: 'Gap Description' },
  { key: 'category' as const, label: 'Category', width: 90 },
  { key: 'priority' as const, label: 'Priority', width: 90, format: (v: unknown) => <Chip label={v as string} size="small" color={(v as string) === 'Critical' ? 'error' : (v as string) === 'High' ? 'warning' : 'default'} /> },
  { key: 'completion' as const, label: 'Progress', width: 120, format: (v: unknown) => (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <LinearProgress variant="determinate" value={v as number} sx={{ flexGrow: 1, height: 6, borderRadius: 3 }} />
      <Typography variant="caption">{v as number}%</Typography>
    </Box>
  )},
  { key: 'responsible' as const, label: 'Owner', width: 110 },
  { key: 'due' as const, label: 'Due', width: 100 },
];

const ActionPlanTable: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Action Plan</Typography>
      <DataTable columns={columns} data={DEMO} keyField="id" dense />
    </CardContent>
  </Card>
);

export default ActionPlanTable;
