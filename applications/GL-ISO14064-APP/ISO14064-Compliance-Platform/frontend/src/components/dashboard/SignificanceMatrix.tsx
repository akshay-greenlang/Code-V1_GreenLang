/**
 * GL-ISO14064-APP v1.0 - Significance Assessment Matrix
 *
 * Displays a matrix of ISO categories 2-6 with their significance
 * status (significant / not-significant / under-review) and estimated
 * magnitude per ISO 14064-1 Clause 5.2.2.
 */

import React from 'react';
import {
  Card, CardContent, Typography, Box,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Chip,
} from '@mui/material';
import type { CategoryBreakdownItem } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, SignificanceLevel } from '../../types';

interface Props {
  data: CategoryBreakdownItem[];
  title?: string;
}

function significanceChip(level: SignificanceLevel) {
  switch (level) {
    case SignificanceLevel.SIGNIFICANT:
      return <Chip label="Significant" size="small" color="error" />;
    case SignificanceLevel.NOT_SIGNIFICANT:
      return <Chip label="Not Significant" size="small" color="success" />;
    default:
      return <Chip label="Under Review" size="small" color="warning" />;
  }
}

const SignificanceMatrix: React.FC<Props> = ({ data, title = 'Significance Assessment' }) => {
  const indirectCategories = data.filter(
    (d) => d.category !== 'category_1_direct' as any,
  );

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        {indirectCategories.length === 0 ? (
          <Box display="flex" justifyContent="center" alignItems="center" height={160}>
            <Typography color="text.secondary">No significance data available</Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Category</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600 }}>Emissions (tCO2e)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600 }}>% of Total</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {indirectCategories.map((row) => (
                  <TableRow key={row.category} hover>
                    <TableCell>
                      {ISO_CATEGORY_SHORT_NAMES[row.category] ?? row.category_name}
                    </TableCell>
                    <TableCell align="right">
                      {row.emissions_tco2e.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                    </TableCell>
                    <TableCell align="right">
                      {row.percentage_of_total.toFixed(1)}%
                    </TableCell>
                    <TableCell align="center">
                      {significanceChip(row.significance)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default SignificanceMatrix;
