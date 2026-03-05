/**
 * ImprovementActions - Table of data quality improvement actions.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable from '../common/DataTable';

const DEMO = [
  { id: '1', action: 'Complete EPC ratings for mortgage portfolio', category: 'Completeness', priority: 'High', status: 'In Progress', responsible: 'Real Estate Team', due: '2025-06-30', impact: 12 },
  { id: '2', action: 'Obtain third-party verification for SC evidence', category: 'Verifiability', priority: 'High', status: 'Pending', responsible: 'External Audit', due: '2025-05-15', impact: 8 },
  { id: '3', action: 'Update climate risk assessment data', category: 'Timeliness', priority: 'Medium', status: 'In Progress', responsible: 'Risk Team', due: '2025-04-30', impact: 6 },
  { id: '4', action: 'Collect counterparty taxonomy disclosures', category: 'Completeness', priority: 'Medium', status: 'Pending', responsible: 'Relationship Mgrs', due: '2025-07-31', impact: 10 },
  { id: '5', action: 'Reconcile NACE code mapping discrepancies', category: 'Consistency', priority: 'Low', status: 'Completed', responsible: 'Data Team', due: '2025-03-15', impact: 3 },
];

const columns = [
  { key: 'action' as const, label: 'Action' },
  { key: 'category' as const, label: 'Category', width: 110 },
  { key: 'priority' as const, label: 'Priority', width: 90, format: (v: unknown) => <Chip label={v as string} size="small" color={(v as string) === 'High' ? 'error' : (v as string) === 'Medium' ? 'warning' : 'success'} /> },
  { key: 'status' as const, label: 'Status', width: 110, format: (v: unknown) => <Chip label={v as string} size="small" variant="outlined" /> },
  { key: 'responsible' as const, label: 'Responsible', width: 130 },
  { key: 'due' as const, label: 'Due Date', width: 100 },
  { key: 'impact' as const, label: 'Impact', align: 'right' as const, width: 70, format: (v: unknown) => `+${v}pts` },
];

const ImprovementActions: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Improvement Action Plan</Typography>
      <DataTable columns={columns} data={DEMO} keyField="id" dense />
    </CardContent>
  </Card>
);

export default ImprovementActions;
