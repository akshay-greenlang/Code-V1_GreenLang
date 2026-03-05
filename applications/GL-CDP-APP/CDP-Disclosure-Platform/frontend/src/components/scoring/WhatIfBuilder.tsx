/**
 * WhatIfBuilder - What-if scenario builder
 */
import React, { useState } from 'react';
import { Card, CardContent, Typography, Box, Slider, Button, Chip } from '@mui/material';
import { Calculate } from '@mui/icons-material';
import type { CategoryScore } from '../../types';
import { SCORING_CATEGORY_NAMES, ScoringCategory } from '../../types';

interface WhatIfBuilderProps {
  categories: CategoryScore[];
  onSimulate: (improvements: Array<{ category: ScoringCategory; target: number }>) => void;
  simulating: boolean;
}

const WhatIfBuilder: React.FC<WhatIfBuilderProps> = ({ categories, onSimulate, simulating }) => {
  const [adjustments, setAdjustments] = useState<Record<string, number>>({});

  const handleSliderChange = (category: string, value: number) => {
    setAdjustments((prev) => ({ ...prev, [category]: value }));
  };

  const handleSimulate = () => {
    const improvements = Object.entries(adjustments)
      .filter(([_, target]) => target > 0)
      .map(([category, target]) => ({ category: category as ScoringCategory, target }));
    onSimulate(improvements);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>What-If Scenario Builder</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Adjust category scores to see projected overall score impact.
        </Typography>
        <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
          {categories.map((cat) => (
            <Box key={cat.category} sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" fontWeight={500}>
                  {SCORING_CATEGORY_NAMES[cat.category]?.split(' ').slice(0, 3).join(' ')}
                </Typography>
                <Chip
                  label={`${(adjustments[cat.category] ?? cat.percentage).toFixed(0)}%`}
                  size="small"
                  sx={{ height: 20, fontSize: '0.65rem' }}
                />
              </Box>
              <Slider
                value={adjustments[cat.category] ?? cat.percentage}
                onChange={(_, val) => handleSliderChange(cat.category, val as number)}
                min={0}
                max={100}
                step={5}
                size="small"
                sx={{ color: '#1b5e20' }}
              />
            </Box>
          ))}
        </Box>
        <Button
          variant="contained"
          fullWidth
          startIcon={<Calculate />}
          onClick={handleSimulate}
          disabled={simulating || Object.keys(adjustments).length === 0}
          sx={{ mt: 1 }}
        >
          {simulating ? 'Simulating...' : 'Simulate Score'}
        </Button>
      </CardContent>
    </Card>
  );
};

export default WhatIfBuilder;
