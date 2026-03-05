/**
 * EBATemplatePreview - Preview of EBA Pillar 3 template output.
 */

import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Box, Chip } from '@mui/material';
import { currencyFormat } from '../../utils/formatters';

const DEMO_ROWS = [
  { label: 'Total on-balance sheet exposures', total: 4850, eligible: 2183, aligned: 1198, enabling: 145, transitional: 95, nonEligible: 2667 },
  { label: 'Loans and advances', total: 3200, eligible: 1600, aligned: 880, enabling: 110, transitional: 70, nonEligible: 1600 },
  { label: 'Debt securities', total: 850, eligible: 340, aligned: 170, enabling: 20, transitional: 15, nonEligible: 510 },
  { label: 'Equity instruments', total: 350, eligible: 140, aligned: 53, enabling: 8, transitional: 5, nonEligible: 210 },
  { label: 'Off-balance sheet exposures', total: 450, eligible: 103, aligned: 95, enabling: 7, transitional: 5, nonEligible: 347 },
];

const EBATemplatePreview: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>EBA Pillar 3 - GAR Template</Typography>
        <Chip label="Template 7" size="small" variant="outlined" />
      </Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Row</TableCell>
              <TableCell align="right">Total (EUR M)</TableCell>
              <TableCell align="right">Eligible</TableCell>
              <TableCell align="right">Aligned</TableCell>
              <TableCell align="right">of which Enabling</TableCell>
              <TableCell align="right">of which Transitional</TableCell>
              <TableCell align="right">Non-eligible</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {DEMO_ROWS.map((row, idx) => (
              <TableRow key={idx} sx={{ fontWeight: idx === 0 ? 700 : 400 }}>
                <TableCell sx={{ fontWeight: idx === 0 ? 700 : 400 }}>{row.label}</TableCell>
                <TableCell align="right">{row.total.toLocaleString()}</TableCell>
                <TableCell align="right">{row.eligible.toLocaleString()}</TableCell>
                <TableCell align="right" sx={{ color: '#1B5E20', fontWeight: 600 }}>{row.aligned.toLocaleString()}</TableCell>
                <TableCell align="right">{row.enabling.toLocaleString()}</TableCell>
                <TableCell align="right">{row.transitional.toLocaleString()}</TableCell>
                <TableCell align="right" sx={{ color: '#757575' }}>{row.nonEligible.toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default EBATemplatePreview;
