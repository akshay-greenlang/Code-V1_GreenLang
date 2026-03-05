/**
 * TSCChangeLog - Table of TSC changes across delegated act versions.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable from '../common/DataTable';

const DEMO = [
  { id: '1', activity: 'Electricity from solar PV', criterion: 'GHG emissions threshold', previous: '100 gCO2e/kWh', current: '100 gCO2e/kWh', change: 'Unchanged', impact: 'None' },
  { id: '2', activity: 'Building renovation', criterion: 'Primary energy demand reduction', previous: '30%', current: '25%', change: 'Relaxed', impact: 'Positive' },
  { id: '3', activity: 'Passenger cars', criterion: 'CO2 per km', previous: '50 g/km', current: '0 g/km (ZEV only)', change: 'Tightened', impact: 'High' },
  { id: '4', activity: 'Freight transport', criterion: 'Emissions per tonne-km', previous: 'N/A', current: '25 gCO2e/tkm', change: 'New', impact: 'Medium' },
  { id: '5', activity: 'Data processing', criterion: 'PUE threshold', previous: '1.5', current: '1.3', change: 'Tightened', impact: 'Medium' },
];

const columns = [
  { key: 'activity' as const, label: 'Activity' },
  { key: 'criterion' as const, label: 'Criterion' },
  { key: 'previous' as const, label: 'Previous', width: 100 },
  { key: 'current' as const, label: 'Current', width: 120 },
  { key: 'change' as const, label: 'Change', width: 100, format: (v: unknown) => {
    const c = v as string;
    return <Chip label={c} size="small" color={c === 'Tightened' ? 'error' : c === 'Relaxed' ? 'success' : c === 'New' ? 'info' : 'default'} />;
  }},
  { key: 'impact' as const, label: 'Impact', width: 80 },
];

const TSCChangeLog: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>TSC Change Log</Typography>
      <DataTable columns={columns} data={DEMO} keyField="id" dense />
    </CardContent>
  </Card>
);

export default TSCChangeLog;
