/**
 * QuestionnaireWizard Page - Module navigation and questionnaire overview
 *
 * Shows all 13 CDP modules with completion status, auto-populate
 * controls, and navigation to individual module detail pages.
 */

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  Button,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  PlayArrow,
  AutoFixHigh,
  CheckCircle,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchQuestionnaires,
  fetchModules,
  fetchProgress,
  autoPopulateModule,
} from '../store/slices/questionnaireSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ModuleNav from '../components/questionnaire/ModuleNav';
import { CDP_MODULE_NAMES, MODULE_COLORS, CDPModule } from '../types';

const DEMO_ORG_ID = 'demo-org';

const QuestionnaireWizardPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();
  const { currentQuestionnaire, modules, loading, error } = useAppSelector(
    (s) => s.questionnaire,
  );
  const [autoPopulating, setAutoPopulating] = useState<string | null>(null);

  useEffect(() => {
    dispatch(fetchQuestionnaires(DEMO_ORG_ID));
  }, [dispatch]);

  useEffect(() => {
    if (currentQuestionnaire) {
      dispatch(fetchModules(currentQuestionnaire.id));
      dispatch(fetchProgress(currentQuestionnaire.id));
    }
  }, [dispatch, currentQuestionnaire]);

  const handleModuleClick = (moduleId: string) => {
    navigate(`/questionnaire/${moduleId}`);
  };

  const handleAutoPopulate = async (moduleId: string) => {
    if (!currentQuestionnaire) return;
    setAutoPopulating(moduleId);
    try {
      await dispatch(
        autoPopulateModule({
          questionnaireId: currentQuestionnaire.id,
          moduleId,
        }),
      ).unwrap();
    } finally {
      setAutoPopulating(null);
    }
  };

  if (loading && !currentQuestionnaire) {
    return <LoadingSpinner message="Loading questionnaire..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        CDP Climate Change Questionnaire
      </Typography>

      {currentQuestionnaire && (
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
            <Chip
              label={`Version: ${currentQuestionnaire.questionnaire_version}`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`${currentQuestionnaire.answered_questions}/${currentQuestionnaire.total_questions} answered`}
              size="small"
              color="primary"
            />
            <Chip
              label={`${currentQuestionnaire.reviewed_questions} reviewed`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`${currentQuestionnaire.approved_questions} approved`}
              size="small"
              color="success"
              variant="outlined"
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <LinearProgress
              variant="determinate"
              value={currentQuestionnaire.completion_pct}
              sx={{ flex: 1, height: 8, borderRadius: 4 }}
            />
            <Typography variant="body2" fontWeight={600}>
              {currentQuestionnaire.completion_pct.toFixed(0)}%
            </Typography>
          </Box>
        </Box>
      )}

      <Grid container spacing={3}>
        {/* Module list */}
        <Grid item xs={12} md={4}>
          <ModuleNav
            modules={modules}
            onModuleSelect={(mod) => handleModuleClick(mod.id)}
          />
        </Grid>

        {/* Module cards grid */}
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            {modules.map((mod) => {
              const moduleColor =
                MODULE_COLORS[mod.module_code as CDPModule] || '#9e9e9e';
              const isComplete = mod.completion_pct >= 100;

              return (
                <Grid item xs={12} sm={6} key={mod.id}>
                  <Card
                    sx={{
                      cursor: 'pointer',
                      borderLeft: `4px solid ${moduleColor}`,
                      opacity: mod.is_applicable ? 1 : 0.5,
                      '&:hover': { boxShadow: 2 },
                    }}
                    onClick={() => mod.is_applicable && handleModuleClick(mod.id)}
                  >
                    <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="subtitle2" fontWeight={600}>
                          {mod.module_code}
                        </Typography>
                        {isComplete && (
                          <CheckCircle sx={{ fontSize: 18, color: '#2e7d32' }} />
                        )}
                      </Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        {CDP_MODULE_NAMES[mod.module_code as CDPModule] || mod.name}
                      </Typography>

                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <LinearProgress
                          variant="determinate"
                          value={mod.completion_pct}
                          sx={{
                            flex: 1,
                            height: 4,
                            borderRadius: 2,
                            '& .MuiLinearProgress-bar': { bgcolor: moduleColor },
                          }}
                        />
                        <Typography variant="caption">{mod.completion_pct.toFixed(0)}%</Typography>
                      </Box>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="caption" color="text.secondary">
                          {mod.answered_count}/{mod.question_count} questions
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 0.5 }}>
                          {mod.is_applicable && (
                            <Button
                              size="small"
                              startIcon={<AutoFixHigh sx={{ fontSize: 14 }} />}
                              disabled={autoPopulating === mod.id}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleAutoPopulate(mod.id);
                              }}
                              sx={{ fontSize: 10, minWidth: 0, px: 0.5 }}
                            >
                              {autoPopulating === mod.id ? '...' : 'Auto'}
                            </Button>
                          )}
                          {mod.is_applicable && (
                            <Button
                              size="small"
                              startIcon={<PlayArrow sx={{ fontSize: 14 }} />}
                              sx={{ fontSize: 10, minWidth: 0, px: 0.5 }}
                            >
                              Open
                            </Button>
                          )}
                        </Box>
                      </Box>

                      {!mod.is_applicable && (
                        <Chip
                          label="Not Applicable"
                          size="small"
                          variant="outlined"
                          sx={{ mt: 0.5, fontSize: 10 }}
                        />
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </Grid>
      </Grid>
    </Box>
  );
};

export default QuestionnaireWizardPage;
