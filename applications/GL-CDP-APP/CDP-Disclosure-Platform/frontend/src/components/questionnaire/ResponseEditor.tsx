/**
 * ResponseEditor - Rich text editor for responses
 *
 * Provides a text area for composing CDP responses with
 * character count, auto-save indicator, and submission controls.
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Box, TextField, Button, Typography, Chip, Alert } from '@mui/material';
import { Save, Send, History } from '@mui/icons-material';
import type { Question, Response as CDPResponse, SaveResponseRequest } from '../../types';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { saveResponse } from '../../store/slices/responseSlice';
import { ResponseStatus } from '../../types';

interface ResponseEditorProps {
  question: Question;
  response: CDPResponse | null;
  questionnaireId: string;
}

const ResponseEditor: React.FC<ResponseEditorProps> = ({
  question,
  response,
  questionnaireId,
}) => {
  const dispatch = useAppDispatch();
  const { saving } = useAppSelector((s) => s.response);
  const [text, setText] = useState(response?.response_text || '');
  const [isDirty, setIsDirty] = useState(false);

  useEffect(() => {
    setText(response?.response_text || '');
    setIsDirty(false);
  }, [response]);

  const handleTextChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    setIsDirty(true);
  }, []);

  const handleSave = useCallback((status: ResponseStatus = ResponseStatus.DRAFT) => {
    const payload: SaveResponseRequest = {
      response_text: text,
      status,
    };
    dispatch(saveResponse({ questionId: question.id, payload }));
    setIsDirty(false);
  }, [dispatch, question.id, text]);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" fontWeight={600}>
          Response
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {isDirty && (
            <Chip label="Unsaved changes" size="small" color="warning" variant="outlined" sx={{ height: 22 }} />
          )}
          {response?.version && (
            <Chip
              icon={<History sx={{ fontSize: 14 }} />}
              label={`v${response.version}`}
              size="small"
              variant="outlined"
              sx={{ height: 22 }}
            />
          )}
        </Box>
      </Box>

      {question.example_response && (
        <Alert severity="info" sx={{ mb: 1.5, py: 0 }}>
          <Typography variant="caption">
            <strong>Example:</strong> {question.example_response.substring(0, 200)}
            {question.example_response.length > 200 ? '...' : ''}
          </Typography>
        </Alert>
      )}

      {question.previous_year_response && (
        <Alert severity="info" icon={<History />} sx={{ mb: 1.5, py: 0 }}>
          <Typography variant="caption">
            <strong>Previous year:</strong> {question.previous_year_response.substring(0, 150)}
            {question.previous_year_response.length > 150 ? '...' : ''}
          </Typography>
        </Alert>
      )}

      <TextField
        multiline
        minRows={6}
        maxRows={20}
        fullWidth
        value={text}
        onChange={handleTextChange}
        placeholder="Enter your response..."
        variant="outlined"
        sx={{ mb: 1 }}
      />

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="caption" color="text.secondary">
          {text.length.toLocaleString()} characters
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<Save />}
            onClick={() => handleSave(ResponseStatus.DRAFT)}
            disabled={saving || !isDirty}
          >
            Save Draft
          </Button>
          <Button
            variant="contained"
            size="small"
            startIcon={<Send />}
            onClick={() => handleSave(ResponseStatus.IN_REVIEW)}
            disabled={saving || !text.trim()}
          >
            Submit for Review
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default ResponseEditor;
