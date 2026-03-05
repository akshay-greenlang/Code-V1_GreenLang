/**
 * Article8Preview - Preview of Article 8 disclosure report.
 */

import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Divider, Box } from '@mui/material';

const DEMO_DATA = {
  period: 'FY 2025',
  turnover: { eligible: 68.2, aligned: 42.5, enabling: 6.6, transitional: 4.1 },
  capex: { eligible: 72.1, aligned: 51.3, enabling: 7.4, transitional: 5.2 },
  opex: { eligible: 55.4, aligned: 38.7, enabling: 4.8, transitional: 3.1 },
};

const Article8Preview: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>Article 8 Disclosure Preview</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>Reporting Period: {DEMO_DATA.period}</Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>KPI</TableCell>
              <TableCell align="right">Eligible %</TableCell>
              <TableCell align="right">Aligned %</TableCell>
              <TableCell align="right">of which Enabling %</TableCell>
              <TableCell align="right">of which Transitional %</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {['turnover', 'capex', 'opex'].map(kpi => {
              const data = DEMO_DATA[kpi as keyof typeof DEMO_DATA] as { eligible: number; aligned: number; enabling: number; transitional: number };
              return (
                <TableRow key={kpi}>
                  <TableCell sx={{ fontWeight: 600, textTransform: 'capitalize' }}>{kpi}</TableCell>
                  <TableCell align="right">{data.eligible.toFixed(1)}%</TableCell>
                  <TableCell align="right" sx={{ color: '#1B5E20', fontWeight: 600 }}>{data.aligned.toFixed(1)}%</TableCell>
                  <TableCell align="right">{data.enabling.toFixed(1)}%</TableCell>
                  <TableCell align="right">{data.transitional.toFixed(1)}%</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default Article8Preview;
