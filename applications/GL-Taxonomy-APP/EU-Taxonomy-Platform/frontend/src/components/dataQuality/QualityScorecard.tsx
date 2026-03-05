/**
 * QualityScorecard - 5-dimension data quality scorecard with radar chart.
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip } from '@mui/material';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import ScoreGauge from '../common/ScoreGauge';
import { gradeColor } from '../../utils/formatters';

const DEMO = {
  overall: 82,
  grade: 'B',
  dimensions: [
    { dimension: 'Completeness', score: 88 },
    { dimension: 'Accuracy', score: 85 },
    { dimension: 'Timeliness', score: 78 },
    { dimension: 'Consistency', score: 82 },
    { dimension: 'Verifiability', score: 76 },
  ],
};

const QualityScorecard: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Data Quality Scorecard</Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Box sx={{ textAlign: 'center' }}>
            <ScoreGauge value={DEMO.overall} label="Overall Score" size={120} />
            <Chip
              label={`Grade: ${DEMO.grade}`}
              sx={{ mt: 1, backgroundColor: gradeColor(DEMO.grade), color: '#FFF', fontWeight: 700 }}
            />
          </Box>
        </Grid>
        <Grid item xs={12} md={8}>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={DEMO.dimensions}>
              <PolarGrid />
              <PolarAngleAxis dataKey="dimension" />
              <PolarRadiusAxis domain={[0, 100]} />
              <Radar dataKey="score" stroke="#1B5E20" fill="#C8E6C9" fillOpacity={0.5} />
            </RadarChart>
          </ResponsiveContainer>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default QualityScorecard;
