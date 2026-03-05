/**
 * CommoditySelector - 11 commodity pathway selector.
 */
import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableHead, TableRow, TableContainer, Chip } from '@mui/material';
import type { CommodityData } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface CommoditySelectorProps { commodities: CommodityData[]; }

const CommoditySelector: React.FC<CommoditySelectorProps> = ({ commodities }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Commodity Pathways</Typography>
      <TableContainer>
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>Commodity</TableCell><TableCell align="right">Emissions</TableCell>
            <TableCell align="right">% of FLAG</TableCell><TableCell align="right">LUC</TableCell>
            <TableCell align="center">Pathway</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {commodities.map((c) => (
              <TableRow key={c.commodity} hover>
                <TableCell sx={{ textTransform: 'capitalize' }}>{c.commodity.replace(/_/g, ' ')}</TableCell>
                <TableCell align="right">{formatNumber(c.emissions_tco2e)} tCO2e</TableCell>
                <TableCell align="right">{c.percentage_of_flag.toFixed(1)}%</TableCell>
                <TableCell align="right">{formatNumber(c.land_use_change_emissions)} tCO2e</TableCell>
                <TableCell align="center">
                  <Chip label={c.pathway_available ? 'Available' : 'N/A'} size="small" color={c.pathway_available ? 'success' : 'default'} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default CommoditySelector;
