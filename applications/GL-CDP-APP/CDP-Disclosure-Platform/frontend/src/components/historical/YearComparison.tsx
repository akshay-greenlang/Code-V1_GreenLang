/**
 * YearComparison - Year-over-year category comparison
 *
 * Shows a side-by-side comparison of category scores between two
 * reporting years, with delta indicators and color-coded changes.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  CompareArrows,
} from '@mui/icons-material';
import type { YearComparison as YearComparisonType, CategoryComparisonItem } from '../../types';
import { getLevelColor } from '../../utils/scoringHelpers';

interface YearComparisonProps {
  comparison: YearComparisonType;
}

function getDeltaIcon(delta: number) {
  if (delta > 0) return <TrendingUp sx={{ fontSize: 16, color: '#2e7d32' }} />;
  if (delta < 0) return <TrendingDown sx={{ fontSize: 16, color: '#e53935' }} />;
  return <TrendingFlat sx={{ fontSize: 16, color: '#9e9e9e' }} />;
}

function getDeltaColor(delta: number): string {
  if (delta > 0) return '#2e7d32';
  if (delta < 0) return '#e53935';
  return '#9e9e9e';
}

const YearComparisonComponent: React.FC<YearComparisonProps> = ({ comparison }) => {
  const overallDelta = comparison.score_b - comparison.score_a;
  const improved = comparison.category_comparison.filter((c) => c.delta > 0).length;
  const declined = comparison.category_comparison.filter((c) => c.delta < 0).length;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <CompareArrows sx={{ color: '#1565c0' }} />
          <Typography variant="h6">
            {comparison.year_a} vs {comparison.year_b}
          </Typography>
        </Box>

        {/* Overall summary */}
        <Box sx={{ display: 'flex', gap: 3, mb: 3 }}>
          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {comparison.year_a}
            </Typography>
            <Typography variant="h5" fontWeight={700} color={getLevelColor(comparison.level_a)}>
              {comparison.level_a}
            </Typography>
            <Typography variant="body2">{comparison.score_a.toFixed(1)}%</Typography>
          </Box>

          <Box sx={{ textAlign: 'center', display: 'flex', alignItems: 'center' }}>
            {getDeltaIcon(overallDelta)}
            <Typography
              variant="h6"
              fontWeight={700}
              color={getDeltaColor(overallDelta)}
              sx={{ mx: 0.5 }}
            >
              {overallDelta > 0 ? '+' : ''}{overallDelta.toFixed(1)}
            </Typography>
            <Typography variant="caption" color="text.secondary">pts</Typography>
          </Box>

          <Box sx={{ textAlign: 'center', flex: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {comparison.year_b}
            </Typography>
            <Typography variant="h5" fontWeight={700} color={getLevelColor(comparison.level_b)}>
              {comparison.level_b}
            </Typography>
            <Typography variant="body2">{comparison.score_b.toFixed(1)}%</Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, justifyContent: 'center' }}>
          <Chip
            label={`${improved} improved`}
            size="small"
            color="success"
            variant="outlined"
          />
          <Chip
            label={`${declined} declined`}
            size="small"
            color="error"
            variant="outlined"
          />
          <Chip
            label={`${comparison.category_comparison.length - improved - declined} unchanged`}
            size="small"
            variant="outlined"
          />
        </Box>

        {/* Category table */}
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Category</TableCell>
                <TableCell align="right">{comparison.year_a}</TableCell>
                <TableCell align="right">{comparison.year_b}</TableCell>
                <TableCell align="right">Change</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {comparison.category_comparison.map((cat: CategoryComparisonItem) => (
                <TableRow key={cat.category}>
                  <TableCell>
                    <Typography variant="body2">{cat.category_name}</Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">{cat.score_a.toFixed(1)}%</Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">{cat.score_b.toFixed(1)}%</Typography>
                  </TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                      {getDeltaIcon(cat.delta)}
                      <Typography
                        variant="body2"
                        fontWeight={500}
                        color={getDeltaColor(cat.delta)}
                      >
                        {cat.delta > 0 ? '+' : ''}{cat.delta.toFixed(1)}
                      </Typography>
                    </Box>
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

export default YearComparisonComponent;
