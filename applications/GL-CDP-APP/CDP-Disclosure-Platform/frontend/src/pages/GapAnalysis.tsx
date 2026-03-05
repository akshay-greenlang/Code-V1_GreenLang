/**
 * GapAnalysis Page - Gap identification and resolution workflow
 *
 * Composes GapList, GapDetail, PriorityMatrix, UpliftCalculator,
 * and RecommendationCard for comprehensive gap analysis.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
} from '@mui/material';
import { Search, Refresh, ArrowUpward } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  runGapAnalysis,
  fetchGapAnalysis,
  fetchRecommendations,
  resolveGap,
} from '../store/slices/gapAnalysisSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import GapList from '../components/gaps/GapList';
import GapDetail from '../components/gaps/GapDetail';
import PriorityMatrix from '../components/gaps/PriorityMatrix';
import UpliftCalculator from '../components/gaps/UpliftCalculator';
import RecommendationCard from '../components/gaps/RecommendationCard';
import { ScoringLevel } from '../types';
import type { GapItem } from '../types';

const DEMO_QUESTIONNAIRE_ID = 'demo-questionnaire';

const GapAnalysisPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { analysis, recommendations, loading, error } = useAppSelector(
    (s) => s.gapAnalysis,
  );
  const { result: scoringResult } = useAppSelector((s) => s.scoring);

  const [targetLevel, setTargetLevel] = useState<ScoringLevel>(ScoringLevel.A);
  const [selectedGap, setSelectedGap] = useState<GapItem | null>(null);

  useEffect(() => {
    dispatch(fetchGapAnalysis(DEMO_QUESTIONNAIRE_ID));
  }, [dispatch]);

  const handleRunAnalysis = () => {
    dispatch(runGapAnalysis({
      questionnaire_id: DEMO_QUESTIONNAIRE_ID,
      target_level: targetLevel,
    }));
  };

  const handleResolve = (gapId: string) => {
    dispatch(resolveGap({ questionnaireId: DEMO_QUESTIONNAIRE_ID, gapId }));
  };

  const handleGapSelect = (gap: GapItem) => {
    setSelectedGap(gap);
    dispatch(fetchRecommendations({
      questionnaireId: DEMO_QUESTIONNAIRE_ID,
      gapId: gap.id,
    }));
  };

  if (loading && !analysis) return <LoadingSpinner message="Analyzing gaps..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Gap Analysis
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Identify and resolve gaps to improve your CDP score
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Target Level</InputLabel>
            <Select
              value={targetLevel}
              label="Target Level"
              onChange={(e) => setTargetLevel(e.target.value as ScoringLevel)}
            >
              {Object.values(ScoringLevel).map((level) => (
                <MenuItem key={level} value={level}>{level}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="contained"
            startIcon={<Search />}
            onClick={handleRunAnalysis}
            disabled={loading}
          >
            Analyze
          </Button>
        </Box>
      </Box>

      {analysis && (
        <>
          {/* Summary chips */}
          <Box sx={{ display: 'flex', gap: 1, mb: 3, flexWrap: 'wrap' }}>
            <Chip label={`${analysis.total_gaps} total gaps`} variant="outlined" />
            <Chip
              label={`${analysis.critical_count} critical`}
              color="error"
              size="small"
            />
            <Chip
              label={`${analysis.high_count} high`}
              color="warning"
              size="small"
            />
            <Chip
              label={`${analysis.medium_count} medium`}
              color="info"
              size="small"
            />
            <Chip
              label={`${analysis.low_count} low`}
              size="small"
              variant="outlined"
            />
            <Chip
              icon={<ArrowUpward sx={{ fontSize: 14 }} />}
              label={`${analysis.total_uplift_potential.toFixed(1)} pts uplift potential`}
              color="success"
              size="small"
            />
          </Box>

          <Grid container spacing={3}>
            {/* Priority matrix */}
            <Grid item xs={12} md={6}>
              <PriorityMatrix
                gaps={analysis.gaps}
                onGapSelect={handleGapSelect}
              />
            </Grid>

            {/* Uplift calculator */}
            <Grid item xs={12} md={6}>
              <UpliftCalculator
                currentScore={scoringResult?.overall_score || 0}
                totalUplift={analysis.total_uplift_potential}
                gaps={analysis.gaps}
              />
            </Grid>

            {/* Gap list */}
            <Grid item xs={12} md={selectedGap ? 6 : 12}>
              <GapList
                gaps={analysis.gaps}
                onSelect={handleGapSelect}
                onResolve={handleResolve}
              />
            </Grid>

            {/* Gap detail + recommendations */}
            {selectedGap && (
              <Grid item xs={12} md={6}>
                <GapDetail gap={selectedGap} />
                {recommendations.map((rec) => (
                  <Box key={rec.id} sx={{ mt: 2 }}>
                    <RecommendationCard recommendation={rec} />
                  </Box>
                ))}
              </Grid>
            )}
          </Grid>
        </>
      )}
    </Box>
  );
};

export default GapAnalysisPage;
