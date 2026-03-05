import React from 'react';
import { Card, CardContent, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Box, LinearProgress } from '@mui/material';
import type { CompetencyEntry } from '../../types';

interface CompetencyMatrixProps {
  competencies: CompetencyEntry[];
}

const SKILLS = [
  { key: 'climate_science', label: 'Climate Science' },
  { key: 'ghg_accounting', label: 'GHG Accounting' },
  { key: 'scenario_analysis', label: 'Scenario Analysis' },
  { key: 'risk_management', label: 'Risk Management' },
  { key: 'regulatory_knowledge', label: 'Regulatory' },
  { key: 'financial_impact', label: 'Financial Impact' },
  { key: 'strategy_development', label: 'Strategy Dev.' },
  { key: 'stakeholder_engagement', label: 'Stakeholder Eng.' },
];

const getColor = (value: number): 'error' | 'warning' | 'success' | 'primary' =>
  value >= 80 ? 'success' : value >= 60 ? 'primary' : value >= 40 ? 'warning' : 'error';

const CompetencyMatrix: React.FC<CompetencyMatrixProps> = ({ competencies }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Climate Competency Matrix</Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 700 }}>Name</TableCell>
                <TableCell sx={{ fontWeight: 700 }}>Role</TableCell>
                {SKILLS.map((s) => (
                  <TableCell key={s.key} align="center" sx={{ fontWeight: 600, fontSize: 11, minWidth: 80 }}>
                    {s.label}
                  </TableCell>
                ))}
                <TableCell align="center" sx={{ fontWeight: 700 }}>Overall</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {competencies.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell sx={{ fontWeight: 500 }}>{entry.person_name}</TableCell>
                  <TableCell>{entry.role}</TableCell>
                  {SKILLS.map((s) => {
                    const val = entry[s.key as keyof CompetencyEntry] as number;
                    return (
                      <TableCell key={s.key} align="center">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <LinearProgress
                            variant="determinate"
                            value={val}
                            color={getColor(val)}
                            sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                          />
                          <Typography variant="caption" sx={{ minWidth: 24, fontSize: 10 }}>{val}</Typography>
                        </Box>
                      </TableCell>
                    );
                  })}
                  <TableCell align="center">
                    <Typography variant="body2" sx={{ fontWeight: 700, color: getColor(entry.overall_score) + '.main' }}>
                      {entry.overall_score.toFixed(0)}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        {competencies.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
            No competency data available
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default CompetencyMatrix;
