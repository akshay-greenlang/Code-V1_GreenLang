/**
 * ObjectiveMatrix - Grid showing DNSH status for each activity x objective.
 */

import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import { DNSHStatus } from '../../types';
import { dnshStatusColor, dnshStatusLabel } from '../../utils/taxonomyHelpers';

const DEMO_MATRIX = [
  { activity: 'Solar PV', sc_obj: 'CCM', cca: DNSHStatus.PASS, wtr: DNSHStatus.PASS, ce: DNSHStatus.PASS, ppc: DNSHStatus.PASS, bio: DNSHStatus.PASS },
  { activity: 'Wind generation', sc_obj: 'CCM', cca: DNSHStatus.PASS, wtr: DNSHStatus.NOT_APPLICABLE, ce: DNSHStatus.PASS, ppc: DNSHStatus.PASS, bio: DNSHStatus.PENDING },
  { activity: 'Building renovation', sc_obj: 'CCM', cca: DNSHStatus.PASS, wtr: DNSHStatus.FAIL, ce: DNSHStatus.PASS, ppc: DNSHStatus.PASS, bio: DNSHStatus.NOT_APPLICABLE },
  { activity: 'Rail transport', sc_obj: 'CCM', cca: DNSHStatus.PENDING, wtr: DNSHStatus.NOT_APPLICABLE, ce: DNSHStatus.PASS, ppc: DNSHStatus.PASS, bio: DNSHStatus.PASS },
];

const renderCell = (status: DNSHStatus) => (
  <Chip
    label={dnshStatusLabel(status)}
    size="small"
    sx={{ backgroundColor: dnshStatusColor(status), color: '#FFF', fontWeight: 600, minWidth: 60 }}
  />
);

const ObjectiveMatrix: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        DNSH Assessment Matrix
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Activity</TableCell>
              <TableCell>SC Obj.</TableCell>
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
                <TableCell sx={{ fontWeight: 500 }}>{row.activity}</TableCell>
                <TableCell><Chip label={row.sc_obj} size="small" color="primary" variant="outlined" /></TableCell>
                <TableCell align="center">{renderCell(row.cca)}</TableCell>
                <TableCell align="center">{renderCell(row.wtr)}</TableCell>
                <TableCell align="center">{renderCell(row.ce)}</TableCell>
                <TableCell align="center">{renderCell(row.ppc)}</TableCell>
                <TableCell align="center">{renderCell(row.bio)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </CardContent>
  </Card>
);

export default ObjectiveMatrix;
