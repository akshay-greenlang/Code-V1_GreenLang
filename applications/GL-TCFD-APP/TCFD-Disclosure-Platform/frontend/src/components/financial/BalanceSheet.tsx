import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import type { FinancialLineItem } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface BalanceSheetProps { items: FinancialLineItem[]; }

const BalanceSheet: React.FC<BalanceSheetProps> = ({ items }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Balance Sheet Climate Impact</Typography>
    <TableContainer><Table size="small">
      <TableHead><TableRow>
        <TableCell sx={{ fontWeight: 700 }}>Category</TableCell><TableCell sx={{ fontWeight: 700 }}>Line Item</TableCell>
        <TableCell align="right" sx={{ fontWeight: 700 }}>Baseline</TableCell><TableCell align="right" sx={{ fontWeight: 700 }}>Climate Impact</TableCell>
        <TableCell align="right" sx={{ fontWeight: 700 }}>Adjusted</TableCell><TableCell align="center" sx={{ fontWeight: 700 }}>Impact %</TableCell>
      </TableRow></TableHead>
      <TableBody>{items.map((item) => (
        <TableRow key={item.id}><TableCell>{item.category}</TableCell><TableCell>{item.line_item}</TableCell>
          <TableCell align="right">{formatCurrency(item.baseline_value)}</TableCell>
          <TableCell align="right" sx={{ color: item.climate_impact >= 0 ? 'success.main' : 'error.main', fontWeight: 600 }}>{item.climate_impact >= 0 ? '+' : ''}{formatCurrency(item.climate_impact)}</TableCell>
          <TableCell align="right" sx={{ fontWeight: 600 }}>{formatCurrency(item.adjusted_value)}</TableCell>
          <TableCell align="center"><Chip label={`${item.impact_pct.toFixed(1)}%`} size="small" color={Math.abs(item.impact_pct) <= 5 ? 'success' : Math.abs(item.impact_pct) <= 15 ? 'warning' : 'error'} /></TableCell>
        </TableRow>
      ))}</TableBody>
    </Table></TableContainer>
    {items.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No balance sheet data</Typography>}
  </CardContent></Card>
);

export default BalanceSheet;
