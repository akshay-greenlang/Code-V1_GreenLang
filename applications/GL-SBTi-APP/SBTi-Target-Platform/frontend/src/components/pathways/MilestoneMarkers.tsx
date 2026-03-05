/**
 * MilestoneMarkers - Annual milestone annotations display.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { CheckCircle, Cancel, HorizontalRule } from '@mui/icons-material';
import type { PathwayMilestone } from '../../types';
import { formatNumber, formatPercentageAbs } from '../../utils/formatters';

interface MilestoneMarkersProps {
  milestones: PathwayMilestone[];
}

const MilestoneMarkers: React.FC<MilestoneMarkersProps> = ({ milestones }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Annual Milestones</Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Year</TableCell>
                <TableCell align="right">Expected</TableCell>
                <TableCell align="right">Actual</TableCell>
                <TableCell align="right">Reduction</TableCell>
                <TableCell align="center">Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {milestones.map((m) => (
                <TableRow key={m.year} hover>
                  <TableCell>{m.year}</TableCell>
                  <TableCell align="right">{formatNumber(m.expected_emissions)} tCO2e</TableCell>
                  <TableCell align="right">{m.actual_emissions !== null ? `${formatNumber(m.actual_emissions)} tCO2e` : '-'}</TableCell>
                  <TableCell align="right">{formatPercentageAbs(m.reduction_from_base_pct)}</TableCell>
                  <TableCell align="center">
                    {m.on_track === null ? (
                      <HorizontalRule sx={{ color: '#9E9E9E', fontSize: 18 }} />
                    ) : m.on_track ? (
                      <CheckCircle sx={{ color: '#2E7D32', fontSize: 18 }} />
                    ) : (
                      <Cancel sx={{ color: '#C62828', fontSize: 18 }} />
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default MilestoneMarkers;
