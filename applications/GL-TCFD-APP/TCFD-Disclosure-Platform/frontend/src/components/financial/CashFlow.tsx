import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import type { FinancialLineItem } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface CashFlowProps { items: FinancialLineItem[]; }

const CashFlow: React.FC<CashFlowProps> = ({ items }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Cash Flow Climate Impact</Typography>
    <TableContainer><Table size="small">
      <TableHead><TableRow>
        <TableCell sx={{ fontWeight: 700 }}>Category</TableCell><TableCell sx={{ fontWeight: 700 }}>Line Item</TableCell>
        <TableCell align="right" sx={{ fontWeight: 700 }}>Baseline</TableCell><TableCell align="right" sx={{ fontWeight: 700 }}>Climate Impact</TableCell>
        <TableCell align="right" sx={{ fontWeight: 700 }}>Adjusted</TableCell>
      </TableRow></TableHead>
      <TableBody>{items.map((item) => (
        <TableRow key={item.id}><TableCell>{item.category}</TableCell><TableCell>{item.line_item}</TableCell>
          <TableCell align="right">{formatCurrency(item.baseline_value)}</TableCell>
          <TableCell align="right" sx={{ color: item.climate_impact >= 0 ? 'success.main' : 'error.main', fontWeight: 600 }}>{item.climate_impact >= 0 ? '+' : ''}{formatCurrency(item.climate_impact)}</TableCell>
          <TableCell align="right" sx={{ fontWeight: 600 }}>{formatCurrency(item.adjusted_value)}</TableCell>
        </TableRow>
      ))}</TableBody>
    </Table></TableContainer>
  </CardContent></Card>
);

export default CashFlow;
