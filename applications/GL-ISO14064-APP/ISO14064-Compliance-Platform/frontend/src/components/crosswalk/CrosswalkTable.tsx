/**
 * CrosswalkTable - ISO 14064-1 vs GHG Protocol mapping table
 *
 * Displays the mapping between ISO 14064-1 categories (1-6) and
 * GHG Protocol scopes (Scope 1/2/3) with tCO2e values and a
 * reconciliation summary row.
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Typography,
  Box,
} from '@mui/material';
import type { CrosswalkResult, CrosswalkMapping } from '../../types';
import { ISO_CATEGORY_SHORT_NAMES, ISOCategory, CATEGORY_COLORS } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface CrosswalkTableProps {
  crosswalk: CrosswalkResult;
}

const CrosswalkTable: React.FC<CrosswalkTableProps> = ({ crosswalk }) => {
  const { mappings, iso_total_tco2e, ghg_protocol_total_tco2e, reconciliation_difference, reconciliation_pct } = crosswalk;

  return (
    <Card>
      <CardHeader
        title="ISO 14064-1 to GHG Protocol Crosswalk"
        subheader="Category mapping with emission reconciliation"
      />
      <CardContent>
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                <TableCell>ISO 14064-1 Category</TableCell>
                <TableCell>GHG Protocol Scope</TableCell>
                <TableCell>GHG Protocol Category</TableCell>
                <TableCell align="right">tCO2e</TableCell>
                <TableCell>Notes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {mappings.map((mapping, idx) => (
                <TableRow key={idx} hover>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          bgcolor: CATEGORY_COLORS[mapping.iso_category as ISOCategory] || '#9e9e9e',
                        }}
                      />
                      <Typography variant="body2">
                        {ISO_CATEGORY_SHORT_NAMES[mapping.iso_category as ISOCategory] || mapping.iso_category_name}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={mapping.ghg_scope}
                      size="small"
                      variant="outlined"
                      sx={{ fontWeight: 600 }}
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {mapping.ghg_category || '--'}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2" fontWeight={600}>
                      {formatNumber(mapping.tco2e, 2)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption" color="text.secondary">
                      {mapping.notes}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}

              {/* Reconciliation row */}
              <TableRow sx={{ bgcolor: '#fafafa' }}>
                <TableCell colSpan={2}>
                  <Typography variant="subtitle2">ISO Total</Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="subtitle2">GHG Protocol Total</Typography>
                </TableCell>
                <TableCell align="right" colSpan={2}>
                  <Box>
                    <Typography variant="body2">
                      ISO: <strong>{formatNumber(iso_total_tco2e, 2)} tCO2e</strong>
                    </Typography>
                    <Typography variant="body2">
                      GHG: <strong>{formatNumber(ghg_protocol_total_tco2e, 2)} tCO2e</strong>
                    </Typography>
                  </Box>
                </TableCell>
              </TableRow>

              {/* Difference row */}
              <TableRow sx={{ bgcolor: reconciliation_pct === 0 ? '#e8f5e9' : '#fff3e0' }}>
                <TableCell colSpan={3}>
                  <Typography variant="subtitle2">
                    Reconciliation Difference
                  </Typography>
                </TableCell>
                <TableCell align="right">
                  <Typography
                    variant="body2"
                    fontWeight={700}
                    color={reconciliation_pct === 0 ? 'success.main' : 'warning.main'}
                  >
                    {formatNumber(reconciliation_difference, 2)} tCO2e
                  </Typography>
                </TableCell>
                <TableCell>
                  <Chip
                    label={`${reconciliation_pct.toFixed(2)}%`}
                    color={Math.abs(reconciliation_pct) < 1 ? 'success' : 'warning'}
                    size="small"
                  />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default CrosswalkTable;
