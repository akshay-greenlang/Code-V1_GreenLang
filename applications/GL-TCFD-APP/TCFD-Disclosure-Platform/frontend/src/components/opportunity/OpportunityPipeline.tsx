import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip, Paper } from '@mui/material';
import type { ClimateOpportunity } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface OpportunityPipelineProps { pipeline: { stage: string; opportunities: ClimateOpportunity[] }[]; }

const STAGE_COLORS: Record<string, string> = { identified: '#9E9E9E', evaluating: '#0D47A1', approved: '#F57F17', implementing: '#E65100', realized: '#1B5E20' };

const OpportunityPipeline: React.FC<OpportunityPipelineProps> = ({ pipeline }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Opportunity Pipeline</Typography>
    <Grid container spacing={2}>
      {pipeline.map((stage) => (
        <Grid item xs={12} sm={6} md key={stage.stage}>
          <Paper sx={{ p: 1.5, bgcolor: '#FAFAFA', borderTop: `3px solid ${STAGE_COLORS[stage.stage] || '#9E9E9E'}`, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, textTransform: 'capitalize' }}>{stage.stage.replace(/_/g, ' ')}</Typography>
              <Chip label={stage.opportunities.length} size="small" />
            </Box>
            {stage.opportunities.map((opp) => (
              <Box key={opp.id} sx={{ p: 1, mb: 1, bgcolor: 'white', borderRadius: 1, border: '1px solid #E0E0E0' }}>
                <Typography variant="body2" sx={{ fontWeight: 500, fontSize: 12 }}>{opp.name}</Typography>
                <Typography variant="caption" color="text.secondary">{formatCurrency(opp.revenue_potential_mid, 'USD', true)}</Typography>
              </Box>
            ))}
          </Paper>
        </Grid>
      ))}
    </Grid>
  </CardContent></Card>
);

export default OpportunityPipeline;
