/**
 * AmbitionIndicator - 1.5C vs WB2C alignment indicator.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import type { PathwayAlignment } from '../../types';
import { formatPathwayAlignment } from '../../utils/formatters';
import { getTemperatureColor } from '../../utils/pathwayHelpers';

interface AmbitionIndicatorProps { alignment: PathwayAlignment; annualRate: number; nearTermCompliant: boolean; longTermCompliant: boolean; }

const ALIGNMENTS: PathwayAlignment[] = ['1.5C', 'well_below_2C', '2C', 'above_2C', 'not_aligned'];

const AmbitionIndicator: React.FC<AmbitionIndicatorProps> = ({ alignment, annualRate, nearTermCompliant, longTermCompliant }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Ambition Level</Typography>
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5, mb: 3 }}>
        {ALIGNMENTS.map((a) => (
          <Box key={a} sx={{
            flex: 1, py: 1.5, textAlign: 'center', borderRadius: 1,
            backgroundColor: a === alignment ? getTemperatureColor(a === '1.5C' ? 1.5 : a === 'well_below_2C' ? 1.8 : a === '2C' ? 2.0 : a === 'above_2C' ? 3.0 : 4.0) : '#F5F5F5',
            color: a === alignment ? 'white' : 'text.secondary',
            fontWeight: a === alignment ? 700 : 400,
            fontSize: '0.7rem',
          }}>
            {formatPathwayAlignment(a)}
          </Box>
        ))}
      </Box>
      <Typography variant="body2" sx={{ mb: 1 }}>Annual reduction rate: <strong>{annualRate.toFixed(2)}% per year</strong></Typography>
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Chip label={nearTermCompliant ? 'Near-term compliant' : 'Near-term gap'} size="small" color={nearTermCompliant ? 'success' : 'error'} />
        <Chip label={longTermCompliant ? 'Long-term compliant' : 'Long-term gap'} size="small" color={longTermCompliant ? 'success' : 'error'} />
      </Box>
    </CardContent>
  </Card>
);

export default AmbitionIndicator;
