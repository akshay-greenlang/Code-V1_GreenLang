import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip } from '@mui/material';
import { CheckCircle, Cancel, Sync } from '@mui/icons-material';
import { formatDate } from '../../utils/formatters';

interface ERMIntegrationProps { data: { category: string; integrated: boolean; status: string; last_sync: string }[]; }

const ERMIntegration: React.FC<ERMIntegrationProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>ERM Integration Status</Typography>
    <Grid container spacing={2}>
      {data.map((item) => (
        <Grid item xs={12} sm={6} md={4} key={item.category}>
          <Box sx={{ p: 2, border: '1px solid #E0E0E0', borderRadius: 1, display: 'flex', alignItems: 'center', gap: 1.5 }}>
            {item.integrated ? <CheckCircle color="success" /> : <Cancel color="error" />}
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{item.category}</Typography>
              <Typography variant="caption" color="text.secondary">Last sync: {formatDate(item.last_sync)}</Typography>
            </Box>
            <Chip label={item.status} size="small" color={item.status === 'active' ? 'success' : item.status === 'pending' ? 'warning' : 'default'} />
          </Box>
        </Grid>
      ))}
    </Grid>
    {data.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No ERM integration data</Typography>}
  </CardContent></Card>
);

export default ERMIntegration;
