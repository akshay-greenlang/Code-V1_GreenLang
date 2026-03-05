/**
 * CompletenessMatrix - Matrix showing data completeness by category.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import ProgressBar from '../common/ProgressBar';

const DEMO = [
  { category: 'Activity Financial Data', completeness: 95 },
  { category: 'NACE Code Mapping', completeness: 100 },
  { category: 'SC Evidence Documents', completeness: 78 },
  { category: 'DNSH Assessment Data', completeness: 82 },
  { category: 'Safeguard Documentation', completeness: 70 },
  { category: 'Climate Risk Data', completeness: 65 },
  { category: 'Counterparty Taxonomy Data', completeness: 55 },
  { category: 'EPC/Energy Ratings', completeness: 48 },
];

const CompletenessMatrix: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Completeness Matrix</Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {DEMO.map(item => (
          <ProgressBar
            key={item.category}
            value={item.completeness}
            label={item.category}
            color={item.completeness >= 80 ? 'success' : item.completeness >= 60 ? 'warning' : 'error'}
          />
        ))}
      </Box>
    </CardContent>
  </Card>
);

export default CompletenessMatrix;
