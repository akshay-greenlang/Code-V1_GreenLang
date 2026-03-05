/**
 * PeerTable - Peer ranking table
 */
import React from 'react';
import { Card, CardContent, Typography, Chip, Box } from '@mui/material';
import type { PeerComparison } from '../../types';
import { getScoringLevelColor } from '../../utils/formatters';

interface PeerTableProps { peer: PeerComparison; }

const PeerTable: React.FC<PeerTableProps> = ({ peer }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" gutterBottom>Your Position</Typography>
      <Box sx={{ display: 'flex', gap: 3, mb: 2 }}>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">Rank</Typography>
          <Typography variant="h4" fontWeight={700}>#{peer.rank}</Typography>
          <Typography variant="caption" color="text.secondary">of {peer.total_peers}</Typography>
        </Box>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">Score</Typography>
          <Typography variant="h4" fontWeight={700} sx={{ color: getScoringLevelColor(peer.level) }}>{peer.level}</Typography>
          <Typography variant="caption">{peer.score.toFixed(0)}%</Typography>
        </Box>
        <Box sx={{ textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">Percentile</Typography>
          <Typography variant="h4" fontWeight={700} color="primary">{peer.percentile.toFixed(0)}th</Typography>
        </Box>
      </Box>
      <Chip label={`Sector: ${peer.sector}`} size="small" sx={{ mr: 1 }} />
      <Chip label={`Region: ${peer.region}`} size="small" variant="outlined" />
    </CardContent>
  </Card>
);

export default PeerTable;
