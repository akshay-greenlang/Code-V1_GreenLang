import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, Box } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';
import type { GovernanceRole } from '../../types';

interface RolesMatrixProps {
  roles: GovernanceRole[];
}

const RESPONSIBILITIES = [
  'Climate strategy oversight',
  'Risk assessment review',
  'Target setting approval',
  'Disclosure review',
  'Scenario analysis',
  'Stakeholder engagement',
  'Budget allocation',
  'Performance monitoring',
];

const RolesMatrix: React.FC<RolesMatrixProps> = ({ roles }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Roles & Responsibilities Matrix</Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 700, minWidth: 200 }}>Responsibility</TableCell>
                {roles.map((role) => (
                  <TableCell key={role.id} align="center" sx={{ fontWeight: 600, fontSize: 12, minWidth: 100 }}>
                    {role.title}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {RESPONSIBILITIES.map((resp) => (
                <TableRow key={resp}>
                  <TableCell>{resp}</TableCell>
                  {roles.map((role) => {
                    const hasResp = role.responsibilities.some(
                      (r) => r.toLowerCase().includes(resp.toLowerCase().split(' ')[0])
                    );
                    return (
                      <TableCell key={role.id} align="center">
                        {hasResp
                          ? <CheckCircle sx={{ color: 'success.main', fontSize: 20 }} />
                          : <Cancel sx={{ color: '#E0E0E0', fontSize: 20 }} />
                        }
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        {roles.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">Add governance roles to populate the matrix</Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RolesMatrix;
