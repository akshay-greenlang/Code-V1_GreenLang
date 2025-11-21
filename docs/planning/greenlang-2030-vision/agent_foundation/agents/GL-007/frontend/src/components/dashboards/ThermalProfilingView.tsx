/**
 * Thermal Profiling View Component
 *
 * Advanced thermal visualization with heatmaps, hot spot detection, and multi-zone analysis
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import { useQuery } from '@tanstack/react-query';

import { useFurnaceStore } from '../../store/furnaceStore';
import { apiClient } from '../../services/apiClient';
import type { ThermalProfile, HotSpot, ColdSpot } from '../../types';

const ThermalProfilingView: React.FC = () => {
  const { selectedFurnaceId } = useFurnaceStore();
  const [viewMode, setViewMode] = useState<'heatmap' | '3d' | 'zones'>('heatmap');

  // Fetch thermal profile
  const { data: thermalProfile, isLoading, error } = useQuery({
    queryKey: ['thermal-profile', selectedFurnaceId],
    queryFn: () =>
      selectedFurnaceId ? apiClient.getThermalProfile(selectedFurnaceId) : null,
    enabled: !!selectedFurnaceId,
    refetchInterval: 10000,
  });

  // Fetch performance for hot/cold spots
  const { data: performance } = useQuery({
    queryKey: ['performance', selectedFurnaceId],
    queryFn: () =>
      selectedFurnaceId ? apiClient.getPerformance(selectedFurnaceId) : null,
    enabled: !!selectedFurnaceId,
  });

  if (!selectedFurnaceId) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">Please select a furnace to view thermal profiling.</Alert>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !thermalProfile) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">Failed to load thermal profile data.</Alert>
      </Box>
    );
  }

  // Convert thermal data to heatmap format
  const heatmapData = prepareHeatmapData(thermalProfile);

  const hotSpots = performance?.thermal.hotSpots || [];
  const coldSpots = performance?.thermal.coldSpots || [];

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h4" gutterBottom>
            Thermal Profiling
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Last updated: {new Date(thermalProfile.timestamp).toLocaleString()}
          </Typography>
        </Box>
        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(_, value) => value && setViewMode(value)}
          size="small"
        >
          <ToggleButton value="heatmap">Heatmap</ToggleButton>
          <ToggleButton value="zones">By Zone</ToggleButton>
          <ToggleButton value="3d">3D View</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Uniformity Index */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Temperature Uniformity Index
              </Typography>
              <Typography variant="h3" color="primary.main">
                {thermalProfile.uniformityIndex.toFixed(1)}%
              </Typography>
              <Chip
                label={
                  thermalProfile.uniformityIndex >= 95
                    ? 'Excellent'
                    : thermalProfile.uniformityIndex >= 90
                    ? 'Good'
                    : 'Needs Attention'
                }
                color={
                  thermalProfile.uniformityIndex >= 95
                    ? 'success'
                    : thermalProfile.uniformityIndex >= 90
                    ? 'info'
                    : 'warning'
                }
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Hot Spots Detected
              </Typography>
              <Typography variant="h3" color="error.main">
                {hotSpots.length}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {hotSpots.filter((h) => h.severity === 'critical').length} critical
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Cold Spots Detected
              </Typography>
              <Typography variant="h3" color="info.main">
                {coldSpots.length}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {coldSpots.filter((c) => c.impact === 'high').length} high impact
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Avg Temperature
              </Typography>
              <Typography variant="h3">
                {performance?.thermal.averageTemperature.toFixed(0)}°C
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Across all zones
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Thermal Visualization */}
      {viewMode === 'heatmap' && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Temperature Distribution Heatmap
            </Typography>
            <Box sx={{ height: 500 }}>
              <ResponsiveHeatMap
                data={heatmapData}
                margin={{ top: 60, right: 90, bottom: 60, left: 90 }}
                valueFormat=">-.0f"
                axisTop={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: -45,
                  legend: '',
                  legendOffset: 46,
                }}
                axisRight={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: 'Zone',
                  legendPosition: 'middle',
                  legendOffset: 70,
                }}
                axisLeft={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: 'Zone',
                  legendPosition: 'middle',
                  legendOffset: -72,
                }}
                colors={{
                  type: 'diverging',
                  scheme: 'red_yellow_blue',
                  divergeAt: 0.5,
                  minValue: 800,
                  maxValue: 1400,
                }}
                emptyColor="#555555"
                legends={[
                  {
                    anchor: 'bottom',
                    translateX: 0,
                    translateY: 30,
                    length: 400,
                    thickness: 8,
                    direction: 'row',
                    tickPosition: 'after',
                    tickSize: 3,
                    tickSpacing: 4,
                    tickOverlap: false,
                    title: 'Temperature (°C) →',
                    titleAlign: 'start',
                    titleOffset: 4,
                  },
                ]}
                annotations={[
                  ...hotSpots.map((spot, idx) => ({
                    type: 'circle' as const,
                    match: { id: spot.zoneId },
                    note: `Hot: ${spot.temperature}°C`,
                    noteX: 100,
                    noteY: idx * 30,
                    offset: 6,
                    noteTextOffset: 5,
                  })),
                ]}
              />
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Hot Spots Table */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="error.main">
                Hot Spots
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Location</TableCell>
                      <TableCell align="right">Temp (°C)</TableCell>
                      <TableCell>Severity</TableCell>
                      <TableCell>Zone</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {hotSpots.map((spot: HotSpot, idx: number) => (
                      <TableRow key={idx}>
                        <TableCell>{spot.location}</TableCell>
                        <TableCell align="right">{spot.temperature.toFixed(0)}</TableCell>
                        <TableCell>
                          <Chip
                            label={spot.severity}
                            size="small"
                            color={
                              spot.severity === 'critical'
                                ? 'error'
                                : spot.severity === 'high'
                                ? 'warning'
                                : 'default'
                            }
                          />
                        </TableCell>
                        <TableCell>{spot.zoneId}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom color="info.main">
                Cold Spots
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Location</TableCell>
                      <TableCell align="right">Temp (°C)</TableCell>
                      <TableCell>Impact</TableCell>
                      <TableCell>Zone</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {coldSpots.map((spot: ColdSpot, idx: number) => (
                      <TableRow key={idx}>
                        <TableCell>{spot.location}</TableCell>
                        <TableCell align="right">{spot.temperature.toFixed(0)}</TableCell>
                        <TableCell>
                          <Chip
                            label={spot.impact}
                            size="small"
                            color={
                              spot.impact === 'high'
                                ? 'error'
                                : spot.impact === 'medium'
                                ? 'warning'
                                : 'default'
                            }
                          />
                        </TableCell>
                        <TableCell>{spot.zoneId}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recommendations */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Thermal Control Recommendations
          </Typography>
          <Grid container spacing={2}>
            {hotSpots.slice(0, 3).map((spot, idx) => (
              <Grid item xs={12} md={4} key={idx}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="error.main" gutterBottom>
                      {spot.location}
                    </Typography>
                    <Typography variant="body2">{spot.recommendation}</Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

// Helper function to prepare heatmap data
function prepareHeatmapData(profile: ThermalProfile) {
  // Group data points by position to create 2D grid
  const zones = new Set(profile.data.map((d) => d.zoneId));

  return Array.from(zones).map((zoneId) => {
    const zoneData = profile.data.filter((d) => d.zoneId === zoneId);
    const sections = ['Section 1', 'Section 2', 'Section 3', 'Section 4'];

    const data: any = { id: zoneId };
    sections.forEach((section, idx) => {
      const sectionData = zoneData.filter((_, i) => i % 4 === idx);
      const avgTemp = sectionData.length > 0
        ? sectionData.reduce((sum, d) => sum + d.temperature, 0) / sectionData.length
        : 0;
      data[section] = Math.round(avgTemp);
    });

    return data;
  });
}

export default ThermalProfilingView;
