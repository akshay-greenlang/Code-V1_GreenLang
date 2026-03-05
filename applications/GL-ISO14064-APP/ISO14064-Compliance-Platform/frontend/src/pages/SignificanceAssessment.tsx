/**
 * SignificanceAssessment Page - Multi-criteria assessment for Cat 3-6
 *
 * Displays significance assessments for indirect categories with
 * score visualization, threshold comparison, and assessment actions.
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
  CardHeader,
  Button,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { Assessment } from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import type { AppDispatch, AppRootState } from '../store';
import { fetchAssessments, assessCategory } from '../store/slices/significanceSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { ISOCategory, ISO_CATEGORY_SHORT_NAMES, CATEGORY_COLORS } from '../types';
import { formatTCO2e, formatNumber, getStatusColor } from '../utils/formatters';

const INDIRECT_CATEGORIES = [
  ISOCategory.CATEGORY_3_TRANSPORT,
  ISOCategory.CATEGORY_4_PRODUCTS_USED,
  ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  ISOCategory.CATEGORY_6_OTHER,
];

const SignificanceAssessment: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { assessments, loading, error } = useSelector(
    (s: AppRootState) => s.significance,
  );

  useEffect(() => {
    if (inventoryId) {
      dispatch(fetchAssessments(inventoryId));
    }
  }, [dispatch, inventoryId]);

  const handleAssess = (category: ISOCategory) => {
    if (inventoryId) {
      dispatch(assessCategory({ inventoryId, category }));
    }
  };

  const chartData = INDIRECT_CATEGORIES.map((cat) => {
    const assessment = assessments.find((a) => a.category === cat);
    return {
      name: ISO_CATEGORY_SHORT_NAMES[cat],
      score: assessment?.total_weighted_score ?? 0,
      threshold: assessment?.threshold ?? 50,
      fill: CATEGORY_COLORS[cat],
    };
  });

  if (loading && assessments.length === 0) {
    return <LoadingSpinner message="Loading significance assessments..." />;
  }

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Significance Assessment
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        ISO 14064-1 Clause 5.2.2 - Multi-criteria assessment for indirect categories (3-6)
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Score visualization */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Significance Scores
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip
                    formatter={(value: number) => [`${value.toFixed(1)}`, 'Score']}
                  />
                  <ReferenceLine
                    y={50}
                    stroke="#e53935"
                    strokeDasharray="5 5"
                    label={{ value: 'Threshold', position: 'right', fill: '#e53935', fontSize: 11 }}
                  />
                  <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, idx) => (
                      <Bar key={idx} dataKey="score" fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Assessment Summary
              </Typography>
              {INDIRECT_CATEGORIES.map((cat) => {
                const a = assessments.find((x) => x.category === cat);
                return (
                  <Box key={cat} sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                      <Typography variant="body2" fontWeight={500}>
                        {ISO_CATEGORY_SHORT_NAMES[cat]}
                      </Typography>
                      {a ? (
                        <Chip
                          label={a.result.replace(/_/g, ' ')}
                          color={getStatusColor(a.result)}
                          size="small"
                        />
                      ) : (
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => handleAssess(cat)}
                          disabled={loading}
                        >
                          Assess
                        </Button>
                      )}
                    </Box>
                    {a && (
                      <>
                        <LinearProgress
                          variant="determinate"
                          value={Math.min(a.total_weighted_score, 100)}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            backgroundColor: '#e0e0e0',
                            '& .MuiLinearProgress-bar': {
                              borderRadius: 3,
                              backgroundColor:
                                a.total_weighted_score >= a.threshold ? '#e53935' : '#43a047',
                            },
                          }}
                        />
                        <Typography variant="caption" color="text.secondary">
                          Score: {a.total_weighted_score.toFixed(1)} / Threshold: {a.threshold}
                          {a.estimated_magnitude_tco2e != null && (
                            <> | Est: {formatTCO2e(a.estimated_magnitude_tco2e)}</>
                          )}
                        </Typography>
                      </>
                    )}
                  </Box>
                );
              })}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Criteria details */}
      {assessments.map((assessment) => (
        <Card key={assessment.id} sx={{ mb: 2 }}>
          <CardHeader
            title={ISO_CATEGORY_SHORT_NAMES[assessment.category]}
            subheader={`Assessed by: ${assessment.assessed_by}`}
            action={
              <Chip
                label={assessment.result.replace(/_/g, ' ')}
                color={getStatusColor(assessment.result)}
                size="small"
              />
            }
          />
          <CardContent sx={{ pt: 0 }}>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Criterion</TableCell>
                    <TableCell align="right">Weight</TableCell>
                    <TableCell align="right">Score</TableCell>
                    <TableCell align="right">Weighted</TableCell>
                    <TableCell>Rationale</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {assessment.criteria.map((c, idx) => (
                    <TableRow key={idx}>
                      <TableCell>{c.criterion}</TableCell>
                      <TableCell align="right">{c.weight.toFixed(2)}</TableCell>
                      <TableCell align="right">{c.score.toFixed(1)}</TableCell>
                      <TableCell align="right">
                        {(c.weight * c.score).toFixed(2)}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">{c.rationale}</Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                  <TableRow sx={{ bgcolor: '#fafafa' }}>
                    <TableCell colSpan={3}>
                      <Typography variant="subtitle2">Total</Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="subtitle2">
                        {assessment.total_weighted_score.toFixed(2)}
                      </Typography>
                    </TableCell>
                    <TableCell />
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      ))}
    </Box>
  );
};

export default SignificanceAssessment;
