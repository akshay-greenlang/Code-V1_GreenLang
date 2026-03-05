/**
 * QuestionCard - Individual question display
 *
 * Renders a CDP question with its type indicator, scoring category,
 * required badge, assigned user, and response status.
 */

import React from 'react';
import { Card, CardContent, Box, Typography, Chip, IconButton, Tooltip } from '@mui/material';
import { Edit, PersonOutline, Star, AutoAwesome } from '@mui/icons-material';
import type { Question, Response as CDPResponse } from '../../types';
import { SCORING_CATEGORY_NAMES, ScoringCategory } from '../../types';
import StatusChip from '../common/StatusChip';

interface QuestionCardProps {
  question: Question;
  response: CDPResponse | null;
  onEdit: (questionId: string) => void;
  isSelected?: boolean;
}

const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  response,
  onEdit,
  isSelected = false,
}) => {
  return (
    <Card
      sx={{
        mb: 1,
        cursor: 'pointer',
        borderLeft: isSelected ? '3px solid #1b5e20' : '3px solid transparent',
        '&:hover': { backgroundColor: '#f5f7f5' },
        transition: 'background-color 0.15s ease',
      }}
      onClick={() => onEdit(question.id)}
    >
      <CardContent sx={{ py: 1.5, px: 2, '&:last-child': { pb: 1.5 } }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                {question.question_number}
              </Typography>
              {question.is_required && (
                <Chip label="Required" size="small" color="error" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
              )}
              {question.scoring_category && (
                <Chip
                  label={SCORING_CATEGORY_NAMES[question.scoring_category as ScoringCategory]?.split(' ').slice(0, 2).join(' ') || question.scoring_category}
                  size="small"
                  variant="outlined"
                  sx={{ height: 18, fontSize: '0.6rem' }}
                />
              )}
              {question.auto_populated_data && question.auto_populated_data.length > 0 && (
                <Tooltip title="Auto-populated from MRV agents">
                  <AutoAwesome fontSize="small" sx={{ color: '#7b1fa2', fontSize: 14 }} />
                </Tooltip>
              )}
            </Box>
            <Typography variant="body2" sx={{ mb: 0.5 }}>
              {question.question_text.length > 200
                ? question.question_text.substring(0, 200) + '...'
                : question.question_text}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label={question.question_type.replace('_', ' ')} size="small" sx={{ height: 18, fontSize: '0.6rem' }} />
              {question.assigned_to && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                  <PersonOutline sx={{ fontSize: 14, color: 'text.secondary' }} />
                  <Typography variant="caption" color="text.secondary">
                    {question.assigned_to}
                  </Typography>
                </Box>
              )}
              {question.scoring_weight > 0 && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                  <Star sx={{ fontSize: 14, color: '#ef6c00' }} />
                  <Typography variant="caption" color="text.secondary">
                    {question.scoring_weight.toFixed(1)}pts
                  </Typography>
                </Box>
              )}
            </Box>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 1 }}>
            <StatusChip status={response?.status || 'not_started'} />
            <IconButton size="small" onClick={(e) => { e.stopPropagation(); onEdit(question.id); }}>
              <Edit fontSize="small" />
            </IconButton>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default QuestionCard;
