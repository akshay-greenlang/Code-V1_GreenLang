/**
 * AutoPopulatedData - Auto-populated MRV data display
 *
 * Shows data that has been auto-populated from GreenLang MRV agents
 * with source attribution and confidence indicators.
 */

import React from 'react';
import { Box, Typography, Paper, Chip, Tooltip, LinearProgress } from '@mui/material';
import { AutoAwesome, Info } from '@mui/icons-material';
import type { AutoPopulatedField } from '../../types';
import { formatDate } from '../../utils/formatters';

interface AutoPopulatedDataProps {
  fields: AutoPopulatedField[];
}

const AutoPopulatedData: React.FC<AutoPopulatedDataProps> = ({ fields }) => {
  if (!fields || fields.length === 0) return null;

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 90) return '#2e7d32';
    if (confidence >= 70) return '#1565c0';
    if (confidence >= 50) return '#ef6c00';
    return '#c62828';
  };

  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <AutoAwesome sx={{ color: '#7b1fa2' }} />
        <Typography variant="subtitle2" fontWeight={600}>
          Auto-Populated Data
        </Typography>
        <Chip label={`${fields.length} fields`} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
        {fields.map((field, idx) => (
          <Box
            key={idx}
            sx={{
              p: 1.5,
              borderRadius: 1,
              backgroundColor: '#f3e5f5' + '30',
              border: '1px solid #e1bee7',
            }}
          >
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
              <Typography variant="body2" fontWeight={600}>
                {field.field_name}
              </Typography>
              <Tooltip title={`Confidence: ${field.confidence.toFixed(0)}%`}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <LinearProgress
                    variant="determinate"
                    value={field.confidence}
                    sx={{
                      width: 60,
                      height: 4,
                      borderRadius: 2,
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: getConfidenceColor(field.confidence),
                      },
                    }}
                  />
                  <Typography variant="caption" sx={{ color: getConfidenceColor(field.confidence) }}>
                    {field.confidence.toFixed(0)}%
                  </Typography>
                </Box>
              </Tooltip>
            </Box>
            <Typography variant="body2" color="text.primary" sx={{ mb: 0.5 }}>
              {String(field.value)}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Info sx={{ fontSize: 12, color: 'text.secondary' }} />
              <Typography variant="caption" color="text.secondary">
                Source: {field.source_description} ({field.source_agent})
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Updated: {formatDate(field.last_updated)}
              </Typography>
            </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default AutoPopulatedData;
