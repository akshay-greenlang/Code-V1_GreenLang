/**
 * PortfolioTable - CRUD table for portfolios.
 */

import React from 'react';
import { Card, CardContent, Typography, Button, Box, Chip } from '@mui/material';
import { Add } from '@mui/icons-material';
import DataTable from '../common/DataTable';
import { currencyFormat } from '../../utils/formatters';

const DEMO = [
  { id: '1', name: 'Green Bond Portfolio', value: 850000000, holdings: 42, aligned_pct: 68.5, updated: '2025-03-01' },
  { id: '2', name: 'Corporate Lending Book', value: 2100000000, holdings: 245, aligned_pct: 22.3, updated: '2025-03-01' },
  { id: '3', name: 'Mortgage Portfolio', value: 1200000000, holdings: 3200, aligned_pct: 40.0, updated: '2025-02-28' },
  { id: '4', name: 'SME Lending', value: 450000000, holdings: 180, aligned_pct: 15.2, updated: '2025-02-28' },
];

const columns = [
  { key: 'name' as const, label: 'Portfolio' },
  { key: 'value' as const, label: 'Value', align: 'right' as const, format: (v: unknown) => currencyFormat(v as number) },
  { key: 'holdings' as const, label: 'Holdings', align: 'right' as const },
  { key: 'aligned_pct' as const, label: 'Aligned %', align: 'right' as const, format: (v: unknown) => <Chip label={`${(v as number).toFixed(1)}%`} size="small" color={(v as number) > 30 ? 'success' : 'warning'} /> },
  { key: 'updated' as const, label: 'Updated' },
];

const PortfolioTable: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Portfolios</Typography>
        <Button variant="contained" startIcon={<Add />} size="small">New Portfolio</Button>
      </Box>
      <DataTable columns={columns} data={DEMO} keyField="id" searchable={false} dense />
    </CardContent>
  </Card>
);

export default PortfolioTable;
