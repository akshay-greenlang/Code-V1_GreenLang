/**
 * GapHeatmap - Heatmap showing gaps by category and severity.
 */

import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Box } from '@mui/material';
import { severityColor } from '../../utils/formatters';

const CATEGORIES = ['Substantial Contribution', 'DNSH', 'Safeguards', 'Data Quality', 'Reporting'];
const SEVERITIES = ['Critical', 'High', 'Medium', 'Low'];

const DEMO_COUNTS: Record<string, Record<string, number>> = {
  'Substantial Contribution': { Critical: 0, High: 2, Medium: 3, Low: 5 },
  'DNSH': { Critical: 1, High: 3, Medium: 4, Low: 2 },
  'Safeguards': { Critical: 0, High: 1, Medium: 2, Low: 3 },
  'Data Quality': { Critical: 2, High: 4, Medium: 6, Low: 8 },
  'Reporting': { Critical: 0, High: 1, Medium: 3, Low: 4 },
};

const cellColor = (count: number, severity: string) => {
  if (count === 0) return '#F5F5F5';
  const base = severityColor(severity.toLowerCase());
  const opacity = Math.min(0.2 + count * 0.15, 0.8);
  return `${base}${Math.round(opacity * 255).toString(16).padStart(2, '0')}`;
};

const GapHeatmap: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Gap Heatmap</Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Category</TableCell>
              {SEVERITIES.map(s => (
                <TableCell key={s} align="center" sx={{ color: severityColor(s.toLowerCase()), fontWeight: 700 }}>{s}</TableCell>
              ))}
              <TableCell align="center" sx={{ fontWeight: 700 }}>Total</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {CATEGORIES.map(cat => {
              const counts = DEMO_COUNTS[cat];
              const total = Object.values(counts).reduce((a, b) => a + b, 0);
              return (
                <TableRow key={cat}>
                  <TableCell sx={{ fontWeight: 500 }}>{cat}</TableCell>
                  {SEVERITIES.map(s => (
                    <TableCell
                      key={s}
                      align="center"
                      sx={{ backgroundColor: cellColor(counts[s], s), fontWeight: counts[s] > 0 ? 700 : 400 }}
                    >
                      {counts[s]}
                    </TableCell>
                  ))}
                  <TableCell align="center" sx={{ fontWeight: 700 }}>{total}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default GapHeatmap;
