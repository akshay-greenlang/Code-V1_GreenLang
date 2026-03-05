/**
 * DueDiligenceTracker - Track due diligence assessments across topics.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip, Box } from '@mui/material';
import DataTable from '../common/DataTable';

const DEMO_RECORDS = [
  { id: '1', topic: 'Human Rights', framework: 'UNGP', last_assessed: '2025-01-15', next_due: '2026-01-15', status: 'current', responsible: 'ESG Team' },
  { id: '2', topic: 'Anti-Corruption', framework: 'UNCAC', last_assessed: '2024-11-20', next_due: '2025-11-20', status: 'current', responsible: 'Compliance' },
  { id: '3', topic: 'Taxation', framework: 'OECD Tax', last_assessed: '2025-02-01', next_due: '2026-02-01', status: 'current', responsible: 'Tax Dept' },
  { id: '4', topic: 'Fair Competition', framework: 'EU Competition Law', last_assessed: '2024-06-10', next_due: '2025-06-10', status: 'due_soon', responsible: 'Legal' },
];

const statusChip = (status: string) => {
  const conf = status === 'current' ? { color: 'success' as const, label: 'Current' } :
    status === 'due_soon' ? { color: 'warning' as const, label: 'Due Soon' } :
    { color: 'error' as const, label: 'Overdue' };
  return <Chip label={conf.label} color={conf.color} size="small" />;
};

const columns = [
  { key: 'topic' as const, label: 'Topic' },
  { key: 'framework' as const, label: 'Framework' },
  { key: 'last_assessed' as const, label: 'Last Assessed' },
  { key: 'next_due' as const, label: 'Next Due' },
  { key: 'status' as const, label: 'Status', format: (val: unknown) => statusChip(val as string) },
  { key: 'responsible' as const, label: 'Responsible' },
];

const DueDiligenceTracker: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Due Diligence Tracker
      </Typography>
      <DataTable columns={columns} data={DEMO_RECORDS} keyField="id" searchable={false} dense />
    </CardContent>
  </Card>
);

export default DueDiligenceTracker;
