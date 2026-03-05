/**
 * HotspotHeatmap - Category hotspot visualization.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Tooltip as MuiTooltip } from '@mui/material';

interface Hotspot { category_number: number; category_name: string; emissions: number; significance: string; hotspot_rank: number; }

interface HotspotHeatmapProps { hotspots: Hotspot[]; }

const SIGNIFICANCE_COLORS: Record<string, string> = { high: '#B71C1C', medium: '#EF6C00', low: '#2E7D32' };

const HotspotHeatmap: React.FC<HotspotHeatmapProps> = ({ hotspots }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Emission Hotspots</Typography>
      <Grid container spacing={1}>
        {hotspots.sort((a, b) => a.hotspot_rank - b.hotspot_rank).map((h) => (
          <Grid item xs={4} sm={3} md={2} key={h.category_number}>
            <MuiTooltip title={`${h.category_name}: ${h.emissions.toLocaleString()} tCO2e`} arrow>
              <Box sx={{
                p: 1.5, borderRadius: 1, textAlign: 'center', cursor: 'pointer',
                backgroundColor: SIGNIFICANCE_COLORS[h.significance] + '20',
                border: `2px solid ${SIGNIFICANCE_COLORS[h.significance]}`,
              }}>
                <Typography variant="subtitle2" fontWeight={700}>Cat {h.category_number}</Typography>
                <Typography variant="caption" sx={{ color: SIGNIFICANCE_COLORS[h.significance] }}>{h.significance.toUpperCase()}</Typography>
              </Box>
            </MuiTooltip>
          </Grid>
        ))}
      </Grid>
    </CardContent>
  </Card>
);

export default HotspotHeatmap;
