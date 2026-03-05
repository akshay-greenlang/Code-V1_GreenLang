/**
 * Benchmarking Page - Sector benchmarking and peer comparison
 *
 * Composes SectorSelector, ScoreHistogram, SpiderChart, PeerTable,
 * and AListRate for comprehensive benchmarking analysis.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchSectorBenchmark,
  fetchPeerComparison,
} from '../store/slices/benchmarkingSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import SectorSelector from '../components/benchmarking/SectorSelector';
import ScoreHistogram from '../components/benchmarking/ScoreHistogram';
import SpiderChart from '../components/benchmarking/SpiderChart';
import PeerTable from '../components/benchmarking/PeerTable';
import AListRate from '../components/benchmarking/AListRate';

const DEMO_ORG_ID = 'demo-org';
const REPORTING_YEAR = new Date().getFullYear();

const BenchmarkingPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { benchmark, peerComparison, loading, error } = useAppSelector(
    (s) => s.benchmarking,
  );
  const { result: scoringResult } = useAppSelector((s) => s.scoring);

  const [selectedSector, setSelectedSector] = useState('Energy');
  const [selectedRegion, setSelectedRegion] = useState('Global');

  useEffect(() => {
    dispatch(fetchSectorBenchmark({
      sector: selectedSector,
      region: selectedRegion,
      year: REPORTING_YEAR,
    }));
    dispatch(fetchPeerComparison({
      orgId: DEMO_ORG_ID,
      sector: selectedSector,
      year: REPORTING_YEAR,
    }));
  }, [dispatch, selectedSector, selectedRegion]);

  const handleSectorChange = (sector: string) => {
    setSelectedSector(sector);
  };

  if (loading && !benchmark) return <LoadingSpinner message="Loading benchmarks..." />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Sector Benchmarking
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Compare your CDP performance against sector peers
      </Typography>

      {/* Sector selector */}
      <Box sx={{ mb: 3 }}>
        <SectorSelector
          selectedSector={selectedSector}
          onSectorChange={handleSectorChange}
        />
      </Box>

      {benchmark && (
        <Grid container spacing={3}>
          {/* Summary stats */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip label={`Sector: ${benchmark.sector}`} variant="outlined" />
              <Chip label={`Region: ${benchmark.region}`} variant="outlined" />
              <Chip label={`Sample: ${benchmark.sample_size} companies`} variant="outlined" />
              <Chip
                label={`Avg Score: ${benchmark.avg_score.toFixed(1)}%`}
                color="primary"
              />
              <Chip
                label={`Median: ${benchmark.median_score.toFixed(1)}%`}
                color="primary"
                variant="outlined"
              />
            </Box>
          </Grid>

          {/* Score histogram */}
          <Grid item xs={12} md={6}>
            <ScoreHistogram
              distribution={benchmark.score_distribution}
              yourScore={scoringResult?.overall_score || null}
            />
          </Grid>

          {/* A-List rate */}
          <Grid item xs={12} md={6}>
            <AListRate
              aListCount={benchmark.a_list_count}
              totalCompanies={benchmark.sample_size}
              aListRate={benchmark.a_list_rate}
            />
          </Grid>

          {/* Spider chart */}
          <Grid item xs={12} md={6}>
            <SpiderChart
              sectorAverages={benchmark.category_averages}
              yourScores={scoringResult?.category_scores || []}
            />
          </Grid>

          {/* Peer table */}
          <Grid item xs={12} md={6}>
            {peerComparison && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Your Position
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" fontWeight={700} color="primary">
                        #{peerComparison.rank}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Rank
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" fontWeight={700}>
                        {peerComparison.percentile.toFixed(0)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Percentile
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" fontWeight={700} color="primary">
                        {peerComparison.level}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Level
                      </Typography>
                    </Box>
                  </Box>
                  <PeerTable
                    peerComparison={peerComparison}
                    sectorAvg={benchmark.avg_score}
                  />
                </CardContent>
              </Card>
            )}
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default BenchmarkingPage;
