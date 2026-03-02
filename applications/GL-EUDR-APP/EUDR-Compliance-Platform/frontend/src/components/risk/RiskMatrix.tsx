/**
 * RiskMatrix - Country x Commodity risk matrix visualization.
 *
 * MUI Table with countries as rows and 7 EUDR commodities as columns.
 * Cell color indicates risk score on a green-yellow-orange-red gradient.
 * Supports hover tooltips, click-to-filter, and column sorting.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Typography,
  Tooltip,
  Stack,
  Chip,
} from '@mui/material';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RiskMatrixProps {
  /** Country code -> Commodity -> risk score (0-1). */
  heatmapData: Record<string, Record<string, number>>;
  /** Mapping of country codes to display names. */
  countryNames?: Record<string, string>;
  /** Called when user clicks a cell. */
  onCellClick?: (country: string, commodity: string, score: number) => void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const COMMODITIES = [
  { key: 'cattle', label: 'Cattle' },
  { key: 'cocoa', label: 'Cocoa' },
  { key: 'coffee', label: 'Coffee' },
  { key: 'oil_palm', label: 'Oil Palm' },
  { key: 'rubber', label: 'Rubber' },
  { key: 'soya', label: 'Soya' },
  { key: 'wood', label: 'Wood' },
];

// ---------------------------------------------------------------------------
// Color interpolation
// ---------------------------------------------------------------------------

function riskColor(score: number): string {
  if (score < 0) return '#f5f5f5';
  if (score <= 0.3) return interpolate('#c8e6c9', '#fff9c4', score / 0.3);
  if (score <= 0.5) return interpolate('#fff9c4', '#ffe0b2', (score - 0.3) / 0.2);
  if (score <= 0.7) return interpolate('#ffe0b2', '#ffccbc', (score - 0.5) / 0.2);
  return interpolate('#ffccbc', '#ef9a9a', Math.min((score - 0.7) / 0.3, 1));
}

function interpolate(c1: string, c2: string, t: number): string {
  const parse = (c: string) => [
    parseInt(c.slice(1, 3), 16),
    parseInt(c.slice(3, 5), 16),
    parseInt(c.slice(5, 7), 16),
  ];
  const [r1, g1, b1] = parse(c1);
  const [r2, g2, b2] = parse(c2);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

function riskLabel(score: number): string {
  if (score < 0) return 'N/A';
  if (score <= 0.3) return 'Low';
  if (score <= 0.5) return 'Standard';
  if (score <= 0.7) return 'High';
  return 'Critical';
}

function textColor(score: number): string {
  if (score < 0) return '#9e9e9e';
  if (score >= 0.7) return '#b71c1c';
  if (score >= 0.5) return '#e65100';
  return '#333';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const RiskMatrix: React.FC<RiskMatrixProps> = ({
  heatmapData,
  countryNames = {},
  onCellClick,
}) => {
  const [sortBy, setSortBy] = useState<string>('country');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Compute rows
  const rows = useMemo(() => {
    const entries = Object.entries(heatmapData).map(([country, commodityScores]) => {
      const avgScore =
        COMMODITIES.reduce((sum, c) => sum + (commodityScores[c.key] ?? 0), 0) /
        COMMODITIES.filter((c) => commodityScores[c.key] !== undefined).length || 0;

      return {
        country,
        displayName: countryNames[country] ?? country,
        commodityScores,
        avgScore,
      };
    });

    entries.sort((a, b) => {
      let cmp: number;
      if (sortBy === 'country') {
        cmp = a.displayName.localeCompare(b.displayName);
      } else if (sortBy === 'overall') {
        cmp = a.avgScore - b.avgScore;
      } else {
        const scoreA = a.commodityScores[sortBy] ?? -1;
        const scoreB = b.commodityScores[sortBy] ?? -1;
        cmp = scoreA - scoreB;
      }
      return sortOrder === 'asc' ? cmp : -cmp;
    });

    return entries;
  }, [heatmapData, countryNames, sortBy, sortOrder]);

  const handleSort = useCallback(
    (col: string) => {
      if (sortBy === col) {
        setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortBy(col);
        setSortOrder('asc');
      }
    },
    [sortBy]
  );

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Country x Commodity Risk Matrix</Typography>
        <Typography variant="body2" color="text.secondary">
          Risk scores from 0 (low) to 1 (critical). Click a cell to filter suppliers.
        </Typography>
      </Box>

      <TableContainer sx={{ maxHeight: 600 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  fontWeight: 700,
                  position: 'sticky',
                  left: 0,
                  backgroundColor: 'background.paper',
                  zIndex: 3,
                  minWidth: 160,
                }}
              >
                <TableSortLabel
                  active={sortBy === 'country'}
                  direction={sortBy === 'country' ? sortOrder : 'asc'}
                  onClick={() => handleSort('country')}
                >
                  Country
                </TableSortLabel>
              </TableCell>
              {COMMODITIES.map((c) => (
                <TableCell
                  key={c.key}
                  align="center"
                  sx={{ fontWeight: 600, minWidth: 80 }}
                >
                  <TableSortLabel
                    active={sortBy === c.key}
                    direction={sortBy === c.key ? sortOrder : 'asc'}
                    onClick={() => handleSort(c.key)}
                  >
                    {c.label}
                  </TableSortLabel>
                </TableCell>
              ))}
              <TableCell align="center" sx={{ fontWeight: 700, minWidth: 80 }}>
                <TableSortLabel
                  active={sortBy === 'overall'}
                  direction={sortBy === 'overall' ? sortOrder : 'asc'}
                  onClick={() => handleSort('overall')}
                >
                  Overall
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.country} hover>
                <TableCell
                  sx={{
                    fontWeight: 500,
                    position: 'sticky',
                    left: 0,
                    backgroundColor: 'background.paper',
                    zIndex: 1,
                  }}
                >
                  {row.displayName}
                </TableCell>

                {COMMODITIES.map((c) => {
                  const score = row.commodityScores[c.key] ?? -1;
                  const bg = riskColor(score);
                  const label = riskLabel(score);
                  const pct = score >= 0 ? `${(score * 100).toFixed(0)}%` : '-';

                  return (
                    <Tooltip
                      key={c.key}
                      title={
                        <Box>
                          <Typography variant="body2" fontWeight={600}>
                            {row.displayName} - {c.label}
                          </Typography>
                          <Typography variant="body2">
                            Score: {pct} ({label})
                          </Typography>
                        </Box>
                      }
                      arrow
                    >
                      <TableCell
                        align="center"
                        sx={{
                          backgroundColor: bg,
                          color: textColor(score),
                          fontWeight: 600,
                          fontSize: 12,
                          cursor: onCellClick ? 'pointer' : 'default',
                          transition: 'opacity 0.15s',
                          '&:hover': { opacity: 0.8 },
                          py: 1,
                        }}
                        onClick={() => {
                          if (onCellClick && score >= 0) {
                            onCellClick(row.country, c.key, score);
                          }
                        }}
                      >
                        {pct}
                      </TableCell>
                    </Tooltip>
                  );
                })}

                {/* Overall column */}
                <TableCell
                  align="center"
                  sx={{
                    backgroundColor: riskColor(row.avgScore),
                    color: textColor(row.avgScore),
                    fontWeight: 700,
                    fontSize: 12,
                    py: 1,
                  }}
                >
                  {(row.avgScore * 100).toFixed(0)}%
                </TableCell>
              </TableRow>
            ))}

            {rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={COMMODITIES.length + 2} align="center" sx={{ py: 4 }}>
                  <Typography color="text.secondary">No risk data available.</Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Legend */}
      <Box sx={{ p: 1.5, borderTop: 1, borderColor: 'divider' }}>
        <Stack direction="row" spacing={2} alignItems="center" justifyContent="center">
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Box sx={{ width: 14, height: 14, borderRadius: '2px', backgroundColor: '#c8e6c9' }} />
            <Typography variant="caption">Low (0-30%)</Typography>
          </Stack>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Box sx={{ width: 14, height: 14, borderRadius: '2px', backgroundColor: '#fff9c4' }} />
            <Typography variant="caption">Standard (30-50%)</Typography>
          </Stack>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Box sx={{ width: 14, height: 14, borderRadius: '2px', backgroundColor: '#ffe0b2' }} />
            <Typography variant="caption">High (50-70%)</Typography>
          </Stack>
          <Stack direction="row" alignItems="center" spacing={0.5}>
            <Box sx={{ width: 14, height: 14, borderRadius: '2px', backgroundColor: '#ef9a9a' }} />
            <Typography variant="caption">Critical (70-100%)</Typography>
          </Stack>
        </Stack>
      </Box>
    </Paper>
  );
};

export default RiskMatrix;
