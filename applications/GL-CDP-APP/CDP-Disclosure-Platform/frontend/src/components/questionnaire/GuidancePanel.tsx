/**
 * GuidancePanel - CDP guidance display
 *
 * Shows the official CDP guidance text and scoring criteria
 * for the selected question.
 */

import React from 'react';
import { Box, Typography, Paper, Divider, Chip } from '@mui/material';
import { Info, Star } from '@mui/icons-material';
import type { Question } from '../../types';
import { SCORING_CATEGORY_NAMES, ScoringCategory } from '../../types';

interface GuidancePanelProps {
  question: Question;
}

const GuidancePanel: React.FC<GuidancePanelProps> = ({ question }) => {
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <Info sx={{ color: '#1565c0' }} />
        <Typography variant="subtitle2" fontWeight={600}>
          CDP Guidance
        </Typography>
      </Box>

      {question.guidance_text ? (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, whiteSpace: 'pre-line' }}>
          {question.guidance_text}
        </Typography>
      ) : (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontStyle: 'italic' }}>
          No guidance text available for this question.
        </Typography>
      )}

      <Divider sx={{ my: 1.5 }} />

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="caption" color="text.secondary" fontWeight={600}>
            Question Type:
          </Typography>
          <Chip
            label={question.question_type.replace('_', ' ')}
            size="small"
            sx={{ height: 20, fontSize: '0.65rem' }}
          />
        </Box>

        {question.scoring_category && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" color="text.secondary" fontWeight={600}>
              Scoring Category:
            </Typography>
            <Typography variant="caption">
              {SCORING_CATEGORY_NAMES[question.scoring_category as ScoringCategory] || question.scoring_category}
            </Typography>
          </Box>
        )}

        {question.scoring_weight > 0 && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Star fontSize="small" sx={{ color: '#ef6c00', fontSize: 14 }} />
            <Typography variant="caption" color="text.secondary">
              Scoring Weight: {question.scoring_weight.toFixed(1)} points
            </Typography>
          </Box>
        )}

        {question.is_conditional && question.depends_on_question_id && (
          <Typography variant="caption" color="warning.main">
            Conditional: depends on Q{question.depends_on_question_id}
          </Typography>
        )}
      </Box>

      {question.options && question.options.length > 0 && (
        <>
          <Divider sx={{ my: 1.5 }} />
          <Typography variant="caption" color="text.secondary" fontWeight={600}>
            Options:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
            {question.options.map((opt, idx) => (
              <Chip key={idx} label={opt} size="small" variant="outlined" sx={{ height: 22, fontSize: '0.65rem' }} />
            ))}
          </Box>
        </>
      )}
    </Paper>
  );
};

export default GuidancePanel;
