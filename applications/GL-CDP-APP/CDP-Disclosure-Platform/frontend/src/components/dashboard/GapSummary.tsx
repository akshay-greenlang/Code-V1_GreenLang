/**
 * GapSummary - Gap count by severity
 *
 * Displays gap counts for critical, high, medium, and low
 * severity levels in a compact card format.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { Warning, Error as ErrorIcon, Info, CheckCircle } from '@mui/icons-material';
import type { GapSummaryData } from '../../types';

interface GapSummaryProps {
  data: GapSummaryData;
}

const GapSummary: React.FC<GapSummaryProps> = ({ data }) => {
  const items = [
    { label: 'Critical', count: data.critical, color: '#c62828', icon: <ErrorIcon fontSize="small" /> },
    { label: 'High', count: data.high, color: '#e53935', icon: <Warning fontSize="small" /> },
    { label: 'Medium', count: data.medium, color: '#ef6c00', icon: <Info fontSize="small" /> },
    { label: 'Low', count: data.low, color: '#1565c0', icon: <Info fontSize="small" /> },
  ];

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
          <Typography variant="h6">Gap Summary</Typography>
          <Chip
            label={`${data.total} total`}
            size="small"
            variant="outlined"
          />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {items.map((item) => (
            <Box
              key={item.label}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 1,
                borderRadius: 1,
                backgroundColor: item.color + '08',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ color: item.color }}>{item.icon}</Box>
                <Typography variant="body2" fontWeight={500}>
                  {item.label}
                </Typography>
              </Box>
              <Typography variant="h6" fontWeight={700} sx={{ color: item.color }}>
                {item.count}
              </Typography>
            </Box>
          ))}
        </Box>
        {data.resolved > 0 && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1.5 }}>
            <CheckCircle fontSize="small" sx={{ color: '#2e7d32' }} />
            <Typography variant="body2" color="success.main" fontWeight={500}>
              {data.resolved} gaps resolved
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default GapSummary;
