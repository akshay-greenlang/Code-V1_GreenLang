import React from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, LinearProgress } from '@mui/material';
import RiskBadge from '../common/RiskBadge';
import type { AssetLocation } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface AssetRiskCardsProps { assets: AssetLocation[]; }

const AssetRiskCards: React.FC<AssetRiskCardsProps> = ({ assets }) => (
  <Box>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Asset Risk Profiles</Typography>
    <Grid container spacing={2}>
      {assets.map((asset) => (
        <Grid item xs={12} sm={6} md={4} key={asset.id}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{asset.name}</Typography>
                <RiskBadge level={asset.risk_level} />
              </Box>
              <Typography variant="caption" color="text.secondary">{asset.asset_type.replace(/_/g, ' ')} | {asset.country}</Typography>
              <Box sx={{ mt: 1.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="caption">Risk Score</Typography>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>{asset.overall_risk_score.toFixed(0)}/100</Typography>
                </Box>
                <LinearProgress variant="determinate" value={asset.overall_risk_score}
                  color={asset.overall_risk_score >= 70 ? 'error' : asset.overall_risk_score >= 40 ? 'warning' : 'success'}
                  sx={{ height: 6, borderRadius: 3 }} />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1.5 }}>
                <Typography variant="caption">Book Value: {formatCurrency(asset.book_value, 'USD', true)}</Typography>
                <Typography variant="caption">Revenue: {formatCurrency(asset.annual_revenue, 'USD', true)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                {asset.hazards.slice(0, 4).map((h) => (
                  <Chip key={h.hazard_type} label={h.hazard_type.replace(/_/g, ' ')} size="small" variant="outlined" sx={{ fontSize: 10, height: 20 }} />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
    {assets.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No assets registered</Typography>}
  </Box>
);

export default AssetRiskCards;
