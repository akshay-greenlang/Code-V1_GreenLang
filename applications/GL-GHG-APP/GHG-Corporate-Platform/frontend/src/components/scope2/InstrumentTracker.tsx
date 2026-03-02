/**
 * InstrumentTracker - Contractual instrument management table
 *
 * Displays a table of contractual instruments (RECs, PPAs, green
 * tariffs, GECs, self-generation) with type, provider, MWh covered,
 * date range, status, and a summary of total green MWh impact.
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Grid,
} from '@mui/material';
import type { ContractualInstrument } from '../../types';
import DataTable, { Column } from '../common/DataTable';
import StatusBadge from '../common/StatusBadge';
import { formatNumber, formatDate } from '../../utils/formatters';

interface InstrumentTrackerProps {
  instruments: ContractualInstrument[];
  totalConsumptionMwh?: number;
}

const TYPE_LABELS: Record<string, string> = {
  REC: 'Renewable Energy Certificate',
  PPA: 'Power Purchase Agreement',
  green_tariff: 'Green Tariff',
  GEC: 'Green Electricity Certificate',
  self_generation: 'Self-Generation',
};

const InstrumentTracker: React.FC<InstrumentTrackerProps> = ({
  instruments,
  totalConsumptionMwh = 0,
}) => {
  const totalGreenMwh = instruments
    .filter((i) => i.status === 'active')
    .reduce((sum, i) => sum + i.mwh, 0);
  const greenPercent = totalConsumptionMwh > 0 ? (totalGreenMwh / totalConsumptionMwh) * 100 : 0;
  const activeCount = instruments.filter((i) => i.status === 'active').length;

  const columns: Column<ContractualInstrument>[] = [
    {
      id: 'type',
      label: 'Type',
      render: (row) => (
        <Chip label={TYPE_LABELS[row.type] || row.type} size="small" variant="outlined" />
      ),
    },
    { id: 'provider', label: 'Provider' },
    {
      id: 'mwh',
      label: 'MWh',
      align: 'right',
      render: (row) => formatNumber(row.mwh, 1),
      getValue: (row) => row.mwh,
    },
    {
      id: 'start_date',
      label: 'Start Date',
      render: (row) => formatDate(row.start_date),
    },
    {
      id: 'end_date',
      label: 'End Date',
      render: (row) => formatDate(row.end_date),
    },
    {
      id: 'status',
      label: 'Status',
      align: 'center',
      render: (row) => <StatusBadge status={row.status} />,
    },
    {
      id: 'emission_factor',
      label: 'EF (tCO2e/MWh)',
      align: 'right',
      render: (row) => row.emission_factor.toFixed(4),
      getValue: (row) => row.emission_factor,
    },
  ];

  return (
    <Box>
      {/* Summary cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">Active Instruments</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700 }}>{activeCount}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">Total Green MWh</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#2e7d32' }}>
                {formatNumber(totalGreenMwh)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">% of Consumption</Typography>
              <Typography variant="h4" sx={{ fontWeight: 700, color: '#1b5e20' }}>
                {greenPercent.toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Instruments table */}
      <DataTable
        columns={columns}
        rows={instruments}
        rowKey={(row) => row.id}
        searchPlaceholder="Search instruments..."
        defaultSort="mwh"
        defaultOrder="desc"
      />
    </Box>
  );
};

export default InstrumentTracker;
