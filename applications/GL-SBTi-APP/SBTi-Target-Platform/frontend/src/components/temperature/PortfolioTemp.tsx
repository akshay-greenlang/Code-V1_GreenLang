/**
 * PortfolioTemp - Portfolio-level temperature display for FIs.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Table, TableBody, TableCell, TableHead, TableRow, TableContainer } from '@mui/material';
import type { PortfolioTemperature } from '../../types';
import { getTemperatureColor } from '../../utils/pathwayHelpers';

interface PortfolioTempProps { portfolio: PortfolioTemperature; }

const PortfolioTemp: React.FC<PortfolioTempProps> = ({ portfolio }) => {
  const color = getTemperatureColor(portfolio.weighted_temperature);
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Portfolio Temperature</Typography>
        <Box sx={{ textAlign: 'center', mb: 2 }}>
          <Typography variant="h3" sx={{ fontWeight: 700, color }}>{portfolio.weighted_temperature.toFixed(2)}{'\u00B0C'}</Typography>
          <Typography variant="body2" color="text.secondary">Methodology: {portfolio.methodology} | Coverage: {portfolio.coverage_pct.toFixed(0)}%</Typography>
        </Box>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>Top Contributors</Typography>
        <TableContainer>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell>Company</TableCell><TableCell align="right">Weight</TableCell>
              <TableCell align="right">Temp</TableCell><TableCell align="right">Contribution</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {portfolio.contributions.slice(0, 8).map((c) => (
                <TableRow key={c.company_id} hover>
                  <TableCell>{c.company_name}</TableCell>
                  <TableCell align="right">{c.weight_pct.toFixed(1)}%</TableCell>
                  <TableCell align="right" sx={{ color: getTemperatureColor(c.temperature) }}>{c.temperature.toFixed(2)}{'\u00B0C'}</TableCell>
                  <TableCell align="right">{c.contribution.toFixed(3)}{'\u00B0C'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default PortfolioTemp;
