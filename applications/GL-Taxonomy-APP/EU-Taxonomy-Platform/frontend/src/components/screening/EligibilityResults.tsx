/**
 * EligibilityResults - Table displaying screening results with status.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip, Box } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';
import DataTable from '../common/DataTable';

const DEMO_RESULTS = [
  { id: '1', nace_code: 'D35.11', activity: 'Production of electricity', eligible: true, objectives: 'CCM, CCA', type: 'Own Performance', turnover_share: 32.1, rationale: 'Listed in Climate DA Annex I' },
  { id: '2', nace_code: 'C23.1', activity: 'Manufacture of glass', eligible: true, objectives: 'CCM, CE', type: 'Transitional', turnover_share: 12.8, rationale: 'Listed in Climate DA Annex I (transitional)' },
  { id: '3', nace_code: 'F41.2', activity: 'Building renovation', eligible: true, objectives: 'CCM', type: 'Own Performance', turnover_share: 15.7, rationale: 'Listed in Climate DA Annex I' },
  { id: '4', nace_code: 'H49.1', activity: 'Rail transport', eligible: true, objectives: 'CCM', type: 'Own Performance', turnover_share: 10.2, rationale: 'Listed in Climate DA Annex I' },
  { id: '5', nace_code: 'G47.1', activity: 'Retail trade', eligible: false, objectives: '-', type: '-', turnover_share: 18.5, rationale: 'Not listed in any Delegated Act annex' },
  { id: '6', nace_code: 'K64.1', activity: 'Monetary intermediation', eligible: false, objectives: '-', type: '-', turnover_share: 10.7, rationale: 'Financial intermediation - excluded' },
];

const columns = [
  { key: 'nace_code' as const, label: 'NACE', width: 80 },
  { key: 'activity' as const, label: 'Activity' },
  { key: 'eligible' as const, label: 'Eligible', width: 90, align: 'center' as const, format: (val: unknown) => val ? <CheckCircle fontSize="small" sx={{ color: '#2E7D32' }} /> : <Cancel fontSize="small" sx={{ color: '#C62828' }} /> },
  { key: 'objectives' as const, label: 'Objectives', width: 100 },
  { key: 'type' as const, label: 'Type', width: 120 },
  { key: 'turnover_share' as const, label: 'Turnover %', align: 'right' as const, width: 100, format: (val: unknown) => `${(val as number).toFixed(1)}%` },
  { key: 'rationale' as const, label: 'Rationale' },
];

const EligibilityResults: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Eligibility Results</Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Chip label="4 Eligible" color="success" size="small" />
          <Chip label="2 Not Eligible" color="default" size="small" />
        </Box>
      </Box>
      <DataTable columns={columns} data={DEMO_RESULTS} keyField="id" dense />
    </CardContent>
  </Card>
);

export default EligibilityResults;
