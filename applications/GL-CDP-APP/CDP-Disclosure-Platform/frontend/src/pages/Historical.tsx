/**
 * Historical Page - Historical score analysis and year comparison
 *
 * Composes ScoreProgression, YearComparison, and ChangeLog for
 * multi-year CDP score tracking and change analysis.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
} from '@mui/material';
import { CompareArrows } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchHistoricalScores,
  fetchYearComparison,
} from '../store/slices/historicalSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ScoreProgression from '../components/historical/ScoreProgression';
import YearComparisonComponent from '../components/historical/YearComparison';
import ChangeLog from '../components/historical/ChangeLog';

const DEMO_ORG_ID = 'demo-org';

const HistoricalPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { scores, comparison, loading, error } = useAppSelector(
    (s) => s.historical,
  );

  const currentYear = new Date().getFullYear();
  const [yearA, setYearA] = useState(currentYear - 1);
  const [yearB, setYearB] = useState(currentYear);

  useEffect(() => {
    dispatch(fetchHistoricalScores(DEMO_ORG_ID));
  }, [dispatch]);

  const handleCompare = () => {
    dispatch(fetchYearComparison({
      orgId: DEMO_ORG_ID,
      yearA,
      yearB,
    }));
  };

  // Generate year options
  const yearOptions = Array.from({ length: 10 }, (_, i) => currentYear - i);

  if (loading && scores.length === 0) {
    return <LoadingSpinner message="Loading historical data..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Historical Analysis
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Track CDP score progression and compare year-over-year changes
      </Typography>

      <Grid container spacing={3}>
        {/* Score progression chart */}
        <Grid item xs={12}>
          <ScoreProgression scores={scores} />
        </Grid>

        {/* Year comparison controls */}
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 2 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Year A</InputLabel>
              <Select
                value={yearA}
                label="Year A"
                onChange={(e) => setYearA(Number(e.target.value))}
              >
                {yearOptions.map((y) => (
                  <MenuItem key={y} value={y}>{y}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <CompareArrows color="action" />

            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Year B</InputLabel>
              <Select
                value={yearB}
                label="Year B"
                onChange={(e) => setYearB(Number(e.target.value))}
              >
                {yearOptions.map((y) => (
                  <MenuItem key={y} value={y}>{y}</MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              variant="contained"
              startIcon={<CompareArrows />}
              onClick={handleCompare}
              disabled={yearA === yearB}
            >
              Compare
            </Button>
          </Box>
        </Grid>

        {/* Year comparison */}
        {comparison && (
          <>
            <Grid item xs={12} md={6}>
              <YearComparisonComponent comparison={comparison} />
            </Grid>
            <Grid item xs={12} md={6}>
              <ChangeLog
                changes={comparison.changes}
                yearA={comparison.year_a}
                yearB={comparison.year_b}
              />
            </Grid>
          </>
        )}

        {scores.length === 0 && (
          <Grid item xs={12}>
            <Alert severity="info">
              No historical data available. Historical scores will appear here
              after your first CDP submission.
            </Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default HistoricalPage;
