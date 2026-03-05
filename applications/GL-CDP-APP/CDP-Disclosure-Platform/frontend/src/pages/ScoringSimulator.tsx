/**
 * ScoringSimulator Page - CDP score simulation and what-if analysis
 *
 * Composes ScoreGaugeDetail, CategoryBreakdown, WhatIfBuilder,
 * ScoreDelta, and ARequirementsCheck for interactive scoring.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Button,
  Card,
  CardContent,
} from '@mui/material';
import { Calculate, Refresh } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  simulateScore,
  runWhatIf,
  checkALevel,
} from '../store/slices/scoringSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ScoreGaugeDetail from '../components/scoring/ScoreGaugeDetail';
import CategoryBreakdown from '../components/scoring/CategoryBreakdown';
import WhatIfBuilder from '../components/scoring/WhatIfBuilder';
import ScoreDelta from '../components/scoring/ScoreDelta';
import ARequirementsCheck from '../components/scoring/ARequirementsCheck';
import type { WhatIfImprovement } from '../types';

const DEMO_QUESTIONNAIRE_ID = 'demo-questionnaire';

const ScoringSimulatorPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { result, whatIfScenarios, currentScenario, simulating, loading, error } =
    useAppSelector((s) => s.scoring);

  const [improvements, setImprovements] = useState<WhatIfImprovement[]>([]);

  useEffect(() => {
    dispatch(simulateScore(DEMO_QUESTIONNAIRE_ID));
    dispatch(checkALevel(DEMO_QUESTIONNAIRE_ID));
  }, [dispatch]);

  const handleSimulate = () => {
    dispatch(simulateScore(DEMO_QUESTIONNAIRE_ID));
  };

  const handleRunWhatIf = () => {
    dispatch(runWhatIf({
      questionnaire_id: DEMO_QUESTIONNAIRE_ID,
      improvements: improvements.map((i) => ({
        question_id: i.question_id,
        improved_score: i.improved_score,
      })),
    }));
  };

  if (loading && !result) return <LoadingSpinner message="Simulating score..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Scoring Simulator
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Simulate your CDP score and explore what-if scenarios
          </Typography>
        </Box>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={handleSimulate}
          disabled={simulating}
        >
          {simulating ? 'Simulating...' : 'Re-simulate'}
        </Button>
      </Box>

      {result && (
        <Grid container spacing={3}>
          {/* Score gauge */}
          <Grid item xs={12} md={4}>
            <ScoreGaugeDetail
              score={result.overall_score}
              level={result.scoring_level}
              band={result.scoring_band}
              confidence={result.confidence}
              aLevelEligible={result.a_level_eligible}
            />
          </Grid>

          {/* Score delta */}
          <Grid item xs={12} md={4}>
            {result.previous_score != null && result.previous_level && (
              <ScoreDelta
                currentScore={result.overall_score}
                currentLevel={result.scoring_level}
                previousScore={result.previous_score}
                previousLevel={result.previous_level}
              />
            )}
            {currentScenario && (
              <Box sx={{ mt: 2 }}>
                <ScoreDelta
                  currentScore={result.overall_score}
                  currentLevel={result.scoring_level}
                  previousScore={currentScenario.projected_score}
                  previousLevel={currentScenario.projected_level}
                />
              </Box>
            )}
          </Grid>

          {/* A-Level requirements */}
          <Grid item xs={12} md={4}>
            <ARequirementsCheck
              requirements={result.a_level_requirements}
              eligible={result.a_level_eligible}
            />
          </Grid>

          {/* Category breakdown */}
          <Grid item xs={12}>
            <CategoryBreakdown categories={result.category_scores} />
          </Grid>

          {/* What-if builder */}
          <Grid item xs={12}>
            <WhatIfBuilder
              categories={result.category_scores}
              onRunScenario={handleRunWhatIf}
              running={simulating}
            />
          </Grid>

          {/* Previous what-if scenarios */}
          {whatIfScenarios.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Saved Scenarios
                  </Typography>
                  {whatIfScenarios.map((scenario) => (
                    <Box
                      key={scenario.id}
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        py: 1,
                        borderBottom: '1px solid #f0f0f0',
                      }}
                    >
                      <Typography variant="body2" fontWeight={500}>
                        {scenario.name}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                        <Typography variant="body2">
                          {scenario.projected_level} ({scenario.projected_score.toFixed(1)}%)
                        </Typography>
                        <Typography
                          variant="body2"
                          fontWeight={600}
                          color={scenario.score_delta > 0 ? 'success.main' : 'error.main'}
                        >
                          {scenario.score_delta > 0 ? '+' : ''}{scenario.score_delta.toFixed(1)} pts
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default ScoringSimulatorPage;
