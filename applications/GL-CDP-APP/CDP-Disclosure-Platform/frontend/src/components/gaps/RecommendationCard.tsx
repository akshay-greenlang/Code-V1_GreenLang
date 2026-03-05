/**
 * RecommendationCard - Single gap recommendation display
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { Lightbulb, CheckCircleOutline, TrendingUp } from '@mui/icons-material';
import type { GapRecommendation } from '../../types';

interface RecommendationCardProps { recommendation: GapRecommendation; }

const RecommendationCard: React.FC<RecommendationCardProps> = ({ recommendation }) => (
  <Card sx={{ mb: 1.5 }}>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Lightbulb sx={{ color: '#ef6c00' }} />
        <Typography variant="subtitle2" fontWeight={600}>{recommendation.title}</Typography>
        <Chip label={`Priority #${recommendation.priority_rank}`} size="small" sx={{ height: 20, fontSize: '0.6rem' }} />
        <Chip label={recommendation.effort} size="small" variant="outlined" sx={{ height: 20, fontSize: '0.6rem' }} />
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>{recommendation.description}</Typography>
      <List dense disablePadding>
        {recommendation.action_items.map((item, idx) => (
          <ListItem key={idx} disablePadding>
            <ListItemIcon sx={{ minWidth: 28 }}><CheckCircleOutline sx={{ fontSize: 16, color: '#1565c0' }} /></ListItemIcon>
            <ListItemText primary={item} primaryTypographyProps={{ variant: 'body2' }} />
          </ListItem>
        ))}
      </List>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 1 }}>
        <TrendingUp fontSize="small" sx={{ color: '#2e7d32' }} />
        <Typography variant="caption" color="success.main" fontWeight={600}>
          Estimated uplift: +{recommendation.estimated_uplift.toFixed(1)} pts
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

export default RecommendationCard;
