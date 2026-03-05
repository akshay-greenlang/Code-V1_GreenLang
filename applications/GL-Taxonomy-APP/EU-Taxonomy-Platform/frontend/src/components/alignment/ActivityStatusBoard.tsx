/**
 * ActivityStatusBoard - Board showing all activities with their current alignment step.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip, Box } from '@mui/material';
import DataTable from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import { AlignmentStatus } from '../../types';

const DEMO_DATA = [
  { id: '1', name: 'Solar PV generation', step: 'Aligned', status: AlignmentStatus.ALIGNED, progress: '5/5' },
  { id: '2', name: 'Wind generation', step: 'Aligned', status: AlignmentStatus.ALIGNED, progress: '5/5' },
  { id: '3', name: 'Building renovation', step: 'DNSH', status: AlignmentStatus.DNSH_PASS, progress: '3/5' },
  { id: '4', name: 'Rail transport', step: 'SC Assessment', status: AlignmentStatus.SC_PASS, progress: '2/5' },
  { id: '5', name: 'Flat glass manufacture', step: 'Eligible', status: AlignmentStatus.ELIGIBLE, progress: '1/5' },
  { id: '6', name: 'Data-driven solutions', step: 'Aligned', status: AlignmentStatus.ALIGNED, progress: '5/5' },
  { id: '7', name: 'Water treatment', step: 'Not Started', status: AlignmentStatus.NOT_STARTED, progress: '0/5' },
];

const columns = [
  { key: 'name' as const, label: 'Activity' },
  { key: 'step' as const, label: 'Current Step' },
  { key: 'progress' as const, label: 'Progress', align: 'center' as const },
  { key: 'status' as const, label: 'Status', format: (val: unknown) => <StatusBadge status={val as AlignmentStatus} /> },
];

const ActivityStatusBoard: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Activity Status Board</Typography>
      <DataTable columns={columns} data={DEMO_DATA} keyField="id" dense />
    </CardContent>
  </Card>
);

export default ActivityStatusBoard;
