/**
 * HoldingsEditor - Edit holdings within a portfolio.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip } from '@mui/material';
import DataTable from '../common/DataTable';
import { currencyFormat } from '../../utils/formatters';

const DEMO = [
  { id: '1', counterparty: 'SolarTech GmbH', lei: 'ABCD1234567890ABCDEF', type: 'General Lending', nominal: 25000000, nace: 'D35.11', eligible: true, aligned: true, quality: 'High' },
  { id: '2', counterparty: 'WindPower SA', lei: 'EFGH1234567890ABCDEF', type: 'Project Finance', nominal: 45000000, nace: 'D35.11', eligible: true, aligned: true, quality: 'High' },
  { id: '3', counterparty: 'GreenBuild Corp', lei: 'IJKL1234567890ABCDEF', type: 'General Lending', nominal: 18000000, nace: 'F41.2', eligible: true, aligned: false, quality: 'Medium' },
  { id: '4', counterparty: 'RetailCo Ltd', lei: 'MNOP1234567890ABCDEF', type: 'General Lending', nominal: 12000000, nace: 'G47.1', eligible: false, aligned: false, quality: 'Low' },
];

const columns = [
  { key: 'counterparty' as const, label: 'Counterparty' },
  { key: 'type' as const, label: 'Type', width: 120 },
  { key: 'nominal' as const, label: 'Nominal', align: 'right' as const, format: (v: unknown) => currencyFormat(v as number) },
  { key: 'nace' as const, label: 'NACE', width: 80 },
  { key: 'eligible' as const, label: 'Eligible', align: 'center' as const, format: (v: unknown) => <Chip label={v ? 'Yes' : 'No'} size="small" color={v ? 'success' : 'default'} /> },
  { key: 'aligned' as const, label: 'Aligned', align: 'center' as const, format: (v: unknown) => <Chip label={v ? 'Yes' : 'No'} size="small" color={v ? 'success' : 'default'} /> },
  { key: 'quality' as const, label: 'DQ', width: 80 },
];

const HoldingsEditor: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Holdings</Typography>
      <DataTable columns={columns} data={DEMO} keyField="id" dense />
    </CardContent>
  </Card>
);

export default HoldingsEditor;
