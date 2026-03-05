/**
 * ModuleDetail Page - Single module question-by-question view
 *
 * Shows all questions for a selected module with response editing,
 * evidence upload, guidance panel, auto-populated data, and
 * assignment management.
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Button,
  Chip,
  LinearProgress,
  IconButton,
} from '@mui/material';
import { ArrowBack, NavigateBefore, NavigateNext } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchModules } from '../store/slices/questionnaireSlice';
import {
  fetchModuleResponses,
  saveResponse,
  submitForReview,
  uploadEvidence,
  fetchComments,
} from '../store/slices/responseSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import QuestionCard from '../components/questionnaire/QuestionCard';
import ResponseEditor from '../components/questionnaire/ResponseEditor';
import EvidencePanel from '../components/questionnaire/EvidencePanel';
import GuidancePanel from '../components/questionnaire/GuidancePanel';
import AutoPopulatedData from '../components/questionnaire/AutoPopulatedData';
import AssignmentPanel from '../components/questionnaire/AssignmentPanel';
import type { Question, Response as CDPResponse } from '../types';
import { CDP_MODULE_NAMES, CDPModule, ResponseStatus } from '../types';

const ModuleDetailPage: React.FC = () => {
  const { moduleId } = useParams<{ moduleId: string }>();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const { modules, currentQuestionnaire } = useAppSelector((s) => s.questionnaire);
  const { responses, evidence, comments, saving, loading, error } = useAppSelector(
    (s) => s.response,
  );

  const [selectedQuestionIdx, setSelectedQuestionIdx] = useState(0);
  const [questions, setQuestions] = useState<Question[]>([]);

  const currentModule = modules.find((m) => m.id === moduleId);

  useEffect(() => {
    if (moduleId && currentQuestionnaire) {
      dispatch(fetchModuleResponses({
        questionnaireId: currentQuestionnaire.id,
        moduleId,
      }));
    }
  }, [dispatch, moduleId, currentQuestionnaire]);

  // For demo, use questions from the API response or empty
  useEffect(() => {
    // In a real implementation, questions would be fetched from the API
    // For now, we derive from responses
    const qs: Question[] = Object.values(responses).map((r) => ({
      id: r.question_id,
      module_id: moduleId || '',
      module_code: currentModule?.module_code || CDPModule.M0_INTRODUCTION,
      question_number: '',
      question_text: '',
      guidance_text: '',
      question_type: 'text' as const,
      scoring_category: null,
      scoring_weight: 0,
      is_required: false,
      is_conditional: false,
      depends_on_question_id: null,
      depends_on_answer: null,
      options: null,
      table_columns: null,
      example_response: null,
      previous_year_response: null,
      auto_populated_data: null,
      assigned_to: null,
      display_order: 0,
    }));
    setQuestions(qs);
  }, [responses, moduleId, currentModule]);

  const currentQuestion = questions[selectedQuestionIdx] || null;
  const currentResponse = currentQuestion
    ? responses[currentQuestion.id] || null
    : null;

  const handleSave = (responseText: string, status: ResponseStatus) => {
    if (!currentQuestion || !currentQuestionnaire) return;
    dispatch(saveResponse({
      questionnaireId: currentQuestionnaire.id,
      questionId: currentQuestion.id,
      data: { response_text: responseText, status },
    }));
  };

  const handleSubmitForReview = () => {
    if (!currentResponse || !currentQuestionnaire) return;
    dispatch(submitForReview({
      questionnaireId: currentQuestionnaire.id,
      data: {
        response_ids: [currentResponse.id],
        reviewer: '',
      },
    }));
  };

  const handleUploadEvidence = (file: File, description: string) => {
    if (!currentResponse || !currentQuestionnaire) return;
    dispatch(uploadEvidence({
      questionnaireId: currentQuestionnaire.id,
      responseId: currentResponse.id,
      data: { file, description },
    }));
  };

  if (loading && Object.keys(responses).length === 0) {
    return <LoadingSpinner message="Loading module..." />;
  }
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <IconButton onClick={() => navigate('/questionnaire')}>
          <ArrowBack />
        </IconButton>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h5" fontWeight={700}>
            {currentModule
              ? `${currentModule.module_code}: ${CDP_MODULE_NAMES[currentModule.module_code as CDPModule] || currentModule.name}`
              : 'Module Detail'}
          </Typography>
          {currentModule && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <LinearProgress
                variant="determinate"
                value={currentModule.completion_pct}
                sx={{ flex: 1, maxWidth: 200, height: 6, borderRadius: 3 }}
              />
              <Typography variant="caption">
                {currentModule.answered_count}/{currentModule.question_count} answered
              </Typography>
            </Box>
          )}
        </Box>
        <Chip
          label={`Question ${selectedQuestionIdx + 1}/${questions.length}`}
          variant="outlined"
        />
      </Box>

      {questions.length === 0 ? (
        <Alert severity="info">
          No questions loaded for this module. This may be due to the module being
          inapplicable or data still loading.
        </Alert>
      ) : (
        <Grid container spacing={3}>
          {/* Left column: Question + Response */}
          <Grid item xs={12} md={8}>
            {currentQuestion && (
              <>
                <QuestionCard question={currentQuestion} />
                <Box sx={{ mt: 2 }}>
                  <ResponseEditor
                    question={currentQuestion}
                    response={currentResponse}
                    saving={saving}
                    onSave={handleSave}
                    onSubmitForReview={handleSubmitForReview}
                  />
                </Box>
                <Box sx={{ mt: 2 }}>
                  <EvidencePanel
                    evidence={evidence}
                    onUpload={handleUploadEvidence}
                  />
                </Box>

                {/* Navigation */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                  <Button
                    startIcon={<NavigateBefore />}
                    disabled={selectedQuestionIdx === 0}
                    onClick={() => setSelectedQuestionIdx((i) => i - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    endIcon={<NavigateNext />}
                    disabled={selectedQuestionIdx >= questions.length - 1}
                    onClick={() => setSelectedQuestionIdx((i) => i + 1)}
                  >
                    Next
                  </Button>
                </Box>
              </>
            )}
          </Grid>

          {/* Right column: Guidance + Auto-populated + Assignment */}
          <Grid item xs={12} md={4}>
            {currentQuestion && (
              <>
                <GuidancePanel guidance={currentQuestion.guidance_text} />
                {currentQuestion.auto_populated_data &&
                 currentQuestion.auto_populated_data.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <AutoPopulatedData
                      fields={currentQuestion.auto_populated_data}
                      onAccept={() => {}}
                    />
                  </Box>
                )}
                <Box sx={{ mt: 2 }}>
                  <AssignmentPanel
                    assignedTo={currentQuestion.assigned_to}
                    teamMembers={[]}
                    onAssign={() => {}}
                  />
                </Box>
              </>
            )}
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default ModuleDetailPage;
