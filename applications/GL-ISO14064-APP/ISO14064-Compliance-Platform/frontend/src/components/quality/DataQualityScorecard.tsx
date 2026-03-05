/**
 * DataQualityScorecard - Spider/radar chart of quality dimensions
 *
 * Renders a Recharts radar chart showing 5 data quality dimensions
 * (completeness, accuracy, consistency, timeliness, methodology)
 * plus an overall score display.
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Box,
  Grid,
  LinearProgress,
} from '@mui/material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { DataQualityIndicator } from '../../types';

interface DataQualityScorecardProps {
  quality: DataQualityIndicator | null;
}

const DIMENSION_LABELS: Record<string, string> = {
  completeness: 'Completeness',
  accuracy: 'Accuracy',
  consistency: 'Consistency',
  timeliness: 'Timeliness',
  methodology: 'Methodology',
};

const DataQualityScorecard: React.FC<DataQualityScorecardProps> = ({
  quality,
}) => {
  if (!quality) {
    return (
      <Card>
        <CardHeader title="Data Quality Scorecard" />
        <CardContent>
          <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
            No data quality assessment available. Run the data quality analysis first.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const radarData = [
    { dimension: 'Completeness', value: quality.completeness, fullMark: 100 },
    { dimension: 'Accuracy', value: quality.accuracy, fullMark: 100 },
    { dimension: 'Consistency', value: quality.consistency, fullMark: 100 },
    { dimension: 'Timeliness', value: quality.timeliness, fullMark: 100 },
    { dimension: 'Methodology', value: quality.methodology, fullMark: 100 },
  ];

  const overallColor =
    quality.overall_score >= 90
      ? '#1b5e20'
      : quality.overall_score >= 70
      ? '#ef6c00'
      : '#e53935';

  return (
    <Card>
      <CardHeader
        title="Data Quality Scorecard"
        subheader="ISO 14064-1 Clause 7 - Data Quality Assessment"
      />
      <CardContent>
        <Grid container spacing={3}>
          {/* Radar Chart */}
          <Grid item xs={12} md={7}>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid gridType="polygon" />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 10 }}
                />
                <Radar
                  name="Quality Score"
                  dataKey="value"
                  stroke="#1b5e20"
                  fill="#1b5e20"
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Grid>

          {/* Score breakdown */}
          <Grid item xs={12} md={5}>
            {/* Overall score */}
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <Typography variant="caption" color="text.secondary">
                Overall Quality Score
              </Typography>
              <Typography variant="h2" fontWeight={700} sx={{ color: overallColor }}>
                {quality.overall_score.toFixed(0)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                out of 100
              </Typography>
            </Box>

            {/* Individual dimensions */}
            {Object.entries(DIMENSION_LABELS).map(([key, label]) => {
              const val = quality[key as keyof DataQualityIndicator] as number;
              return (
                <Box key={key} sx={{ mb: 1.5 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.25 }}>
                    <Typography variant="body2" color="text.secondary">
                      {label}
                    </Typography>
                    <Typography variant="body2" fontWeight={600}>
                      {val.toFixed(0)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={val}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 3,
                        backgroundColor:
                          val >= 90 ? '#1b5e20' : val >= 70 ? '#ef6c00' : '#e53935',
                      },
                    }}
                  />
                </Box>
              );
            })}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default DataQualityScorecard;
