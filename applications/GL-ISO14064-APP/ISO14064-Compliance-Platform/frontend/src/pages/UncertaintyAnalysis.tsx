/**
 * UncertaintyAnalysis Page - Monte Carlo results
 *
 * Displays confidence intervals table, distribution chart,
 * and category/gas uncertainty breakdown.
 */

import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Alert,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
} from '@mui/material';
import { PlayArrow } from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ErrorBar,
  ResponsiveContainer,
} from 'recharts';
import type { AppDispatch, AppRootState } from '../store';
import { runUncertainty } from '../store/slices/emissionsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { formatNumber, formatTCO2e } from '../utils/formatters';

const UncertaintyAnalysis: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { uncertainty, loading, error } = useSelector(
    (s: AppRootState) => s.emissions,
  );

  const handleRun = () => {
    if (inventoryId) {
      dispatch(runUncertainty(inventoryId));
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Uncertainty Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            ISO 14064-1 Clause 6.3 - Monte Carlo uncertainty quantification
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PlayArrow />}
          onClick={handleRun}
          disabled={loading}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          {loading ? 'Running...' : 'Run Analysis'}
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {!uncertainty && !loading && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              No uncertainty analysis results available.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Click "Run Analysis" to perform a Monte Carlo simulation of the inventory uncertainty.
            </Typography>
          </CardContent>
        </Card>
      )}

      {loading && <LoadingSpinner message="Running Monte Carlo simulation..." />}

      {uncertainty && (
        <>
          {/* Summary stats */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Mean
                  </Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {formatTCO2e(uncertainty.mean)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Std Deviation
                  </Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {formatTCO2e(uncertainty.std_dev)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    CV (%)
                  </Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {uncertainty.cv_percent.toFixed(2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Iterations
                  </Typography>
                  <Typography variant="h6" fontWeight={700}>
                    {formatNumber(uncertainty.iterations, 0)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Confidence intervals table */}
          <Grid container spacing={3} sx={{ mb: 3 }}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Confidence Intervals
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Confidence Level</TableCell>
                          <TableCell align="right">Lower Bound</TableCell>
                          <TableCell align="right">Upper Bound</TableCell>
                          <TableCell align="right">Half-Width</TableCell>
                          <TableCell align="right">Half-Width %</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {uncertainty.intervals.map((interval, idx) => (
                          <TableRow key={idx}>
                            <TableCell>
                              <Chip
                                label={`${interval.confidence_level}%`}
                                size="small"
                                color={
                                  interval.confidence_level === 95
                                    ? 'primary'
                                    : 'default'
                                }
                              />
                            </TableCell>
                            <TableCell align="right">
                              {formatNumber(interval.lower_bound, 2)}
                            </TableCell>
                            <TableCell align="right">
                              {formatNumber(interval.upper_bound, 2)}
                            </TableCell>
                            <TableCell align="right">
                              {formatNumber(interval.half_width, 2)}
                            </TableCell>
                            <TableCell align="right">
                              <Typography
                                variant="body2"
                                fontWeight={600}
                                color={
                                  interval.half_width_pct < 10
                                    ? 'success.main'
                                    : interval.half_width_pct < 25
                                    ? 'warning.main'
                                    : 'error.main'
                                }
                              >
                                +/-{interval.half_width_pct.toFixed(1)}%
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Uncertainty by Category
                  </Typography>
                  {Object.keys(uncertainty.by_category).length > 0 ? (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={Object.entries(uncertainty.by_category).map(
                          ([cat, stats]) => ({
                            name: cat.replace(/_/g, ' ').substring(0, 15),
                            mean: (stats as Record<string, number>).mean || 0,
                            std_dev: (stats as Record<string, number>).std_dev || 0,
                          }),
                        )}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                        <YAxis />
                        <Tooltip
                          formatter={(value: number) => formatNumber(value, 2) + ' tCO2e'}
                        />
                        <Bar dataKey="mean" fill="#1b5e20" radius={[4, 4, 0, 0]}>
                          <ErrorBar dataKey="std_dev" width={4} stroke="#e53935" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
                      No category-level uncertainty data.
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default UncertaintyAnalysis;
