/**
 * ActivityDetail - Detailed view of a single economic activity.
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, Chip, Box, Divider } from '@mui/material';
import StatusBadge from '../common/StatusBadge';
import { AlignmentStatus } from '../../types';
import { currencyFormat } from '../../utils/formatters';

interface ActivityDetailProps {
  activity?: {
    nace_code: string;
    name: string;
    sector: string;
    objectives: string[];
    type: string;
    status: AlignmentStatus;
    turnover: number;
    capex: number;
    opex: number;
    description: string;
  };
}

const DEMO = {
  nace_code: 'D35.11',
  name: 'Electricity generation using solar photovoltaic technology',
  sector: 'Energy',
  objectives: ['Climate Change Mitigation'],
  type: 'Own Performance',
  status: AlignmentStatus.ALIGNED,
  turnover: 45000000,
  capex: 12000000,
  opex: 3500000,
  description: 'Construction and operation of electricity generation facilities that produce electricity using solar photovoltaic (PV) technology.',
};

const ActivityDetail: React.FC<ActivityDetailProps> = ({ activity = DEMO }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>{activity.name}</Typography>
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            <Chip label={activity.nace_code} size="small" color="primary" variant="outlined" />
            <Chip label={activity.sector} size="small" />
            <Chip label={activity.type} size="small" variant="outlined" />
          </Box>
        </Box>
        <StatusBadge status={activity.status} />
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {activity.description}
      </Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle2" sx={{ mb: 1 }}>Financial Data</Typography>
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <Typography variant="caption" color="text.secondary">Turnover</Typography>
          <Typography variant="body1" sx={{ fontWeight: 600 }}>{currencyFormat(activity.turnover)}</Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography variant="caption" color="text.secondary">CapEx</Typography>
          <Typography variant="body1" sx={{ fontWeight: 600 }}>{currencyFormat(activity.capex)}</Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography variant="caption" color="text.secondary">OpEx</Typography>
          <Typography variant="body1" sx={{ fontWeight: 600 }}>{currencyFormat(activity.opex)}</Typography>
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      <Typography variant="subtitle2" sx={{ mb: 1 }}>Eligible Objectives</Typography>
      <Box sx={{ display: 'flex', gap: 1 }}>
        {activity.objectives.map(obj => (
          <Chip key={obj} label={obj} color="primary" size="small" />
        ))}
      </Box>
    </CardContent>
  </Card>
);

export default ActivityDetail;
