/**
 * EligibilityMatrix - Grid showing which activities are eligible for which objectives.
 */

import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import { CheckCircle, Cancel, Remove } from '@mui/icons-material';

const DEMO_MATRIX = [
  { activity: 'Solar PV generation', nace: 'D35.11', ccm: true, cca: false, wtr: null, ce: null, ppc: null, bio: null },
  { activity: 'Wind generation', nace: 'D35.11', ccm: true, cca: false, wtr: null, ce: null, ppc: null, bio: null },
  { activity: 'Building renovation', nace: 'F41.2', ccm: true, cca: true, wtr: false, ce: true, ppc: false, bio: false },
  { activity: 'Rail transport', nace: 'H49.1', ccm: true, cca: false, wtr: null, ce: null, ppc: true, bio: null },
  { activity: 'Data solutions', nace: 'J62.0', ccm: true, cca: true, wtr: null, ce: null, ppc: null, bio: null },
];

const renderStatus = (val: boolean | null) => {
  if (val === true) return <CheckCircle fontSize="small" sx={{ color: '#2E7D32' }} />;
  if (val === false) return <Cancel fontSize="small" sx={{ color: '#C62828' }} />;
  return <Remove fontSize="small" sx={{ color: '#BDBDBD' }} />;
};

const EligibilityMatrix: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Eligibility Matrix
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Activity</TableCell>
              <TableCell>NACE</TableCell>
              <TableCell align="center">CCM</TableCell>
              <TableCell align="center">CCA</TableCell>
              <TableCell align="center">WTR</TableCell>
              <TableCell align="center">CE</TableCell>
              <TableCell align="center">PPC</TableCell>
              <TableCell align="center">BIO</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {DEMO_MATRIX.map((row, idx) => (
              <TableRow key={idx} hover>
                <TableCell>{row.activity}</TableCell>
                <TableCell><Chip label={row.nace} size="small" variant="outlined" /></TableCell>
                <TableCell align="center">{renderStatus(row.ccm)}</TableCell>
                <TableCell align="center">{renderStatus(row.cca)}</TableCell>
                <TableCell align="center">{renderStatus(row.wtr)}</TableCell>
                <TableCell align="center">{renderStatus(row.ce)}</TableCell>
                <TableCell align="center">{renderStatus(row.ppc)}</TableCell>
                <TableCell align="center">{renderStatus(row.bio)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default EligibilityMatrix;
