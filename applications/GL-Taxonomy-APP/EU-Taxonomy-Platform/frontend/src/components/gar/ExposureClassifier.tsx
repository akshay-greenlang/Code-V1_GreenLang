/**
 * ExposureClassifier - Classify exposures by type and taxonomy alignment.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable from '../common/DataTable';
import { currencyFormat } from '../../utils/formatters';

const DEMO_DATA = [
  { id: '1', type: 'General Lending', total: 1800000000, eligible: 720000000, aligned: 378000000, gar: 21.0, count: 245 },
  { id: '2', type: 'Mortgages', total: 1200000000, eligible: 960000000, aligned: 480000000, gar: 40.0, count: 3200 },
  { id: '3', type: 'Project Finance', total: 450000000, eligible: 360000000, aligned: 270000000, gar: 60.0, count: 18 },
  { id: '4', type: 'Equity Holdings', total: 350000000, eligible: 140000000, aligned: 52500000, gar: 15.0, count: 85 },
  { id: '5', type: 'Auto Loans', total: 250000000, eligible: 200000000, aligned: 62500000, gar: 25.0, count: 1500 },
  { id: '6', type: 'Sovereign', total: 800000000, eligible: 0, aligned: 0, gar: 0, count: 12 },
];

const columns = [
  { key: 'type' as const, label: 'Exposure Type' },
  { key: 'total' as const, label: 'Total', align: 'right' as const, format: (v: unknown) => currencyFormat(v as number) },
  { key: 'eligible' as const, label: 'Eligible', align: 'right' as const, format: (v: unknown) => currencyFormat(v as number) },
  { key: 'aligned' as const, label: 'Aligned', align: 'right' as const, format: (v: unknown) => currencyFormat(v as number) },
  { key: 'gar' as const, label: 'GAR %', align: 'right' as const, format: (v: unknown) => <Chip label={`${(v as number).toFixed(1)}%`} size="small" color={(v as number) > 30 ? 'success' : (v as number) > 0 ? 'warning' : 'default'} /> },
  { key: 'count' as const, label: 'Count', align: 'right' as const },
];

const ExposureClassifier: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Exposure Classification</Typography>
      <DataTable columns={columns} data={DEMO_DATA} keyField="id" searchable={false} dense />
    </CardContent>
  </Card>
);

export default ExposureClassifier;
