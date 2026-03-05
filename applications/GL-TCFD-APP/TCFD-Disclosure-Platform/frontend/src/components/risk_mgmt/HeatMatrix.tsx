import React from 'react';
import { Card, CardContent, Typography, Box, Tooltip as MuiTooltip } from '@mui/material';
import type { HeatMapCell } from '../../types';

interface HeatMatrixProps { data: HeatMapCell[]; }

const LABELS_Y = ['Almost Certain', 'Likely', 'Possible', 'Unlikely', 'Rare'];
const LABELS_X = ['Insignificant', 'Minor', 'Moderate', 'Major', 'Catastrophic'];

const getColor = (l: number, i: number): string => {
  const score = l * i;
  if (score >= 20) return '#B71C1C';
  if (score >= 12) return '#E65100';
  if (score >= 6) return '#F57F17';
  if (score >= 3) return '#FDD835';
  return '#4CAF50';
};

const HeatMatrix: React.FC<HeatMatrixProps> = ({ data }) => {
  const getCell = (l: number, i: number) => data.find((c) => c.likelihood_score === l && c.impact_score === i);
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Risk Heat Map (5x5)</Typography>
      <Box sx={{ display: 'flex' }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-around', pr: 1, width: 100 }}>
          {LABELS_Y.map((label, i) => (
            <Typography key={label} variant="caption" sx={{ textAlign: 'right', fontSize: 10, height: 60, display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>{label}</Typography>
          ))}
        </Box>
        <Box sx={{ flexGrow: 1 }}>
          {[5, 4, 3, 2, 1].map((l) => (
            <Box key={l} sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
              {[1, 2, 3, 4, 5].map((i) => {
                const cell = getCell(l, i);
                return (
                  <MuiTooltip key={`${l}-${i}`} title={cell ? `${cell.risk_count} risk(s): ${cell.risks.map((r) => r.name).join(', ')}` : 'No risks'}>
                    <Box sx={{ flex: 1, height: 56, bgcolor: getColor(l, i), borderRadius: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', opacity: cell?.risk_count ? 1 : 0.3, transition: 'opacity 0.2s' }}>
                      <Typography variant="body2" sx={{ color: 'white', fontWeight: 700 }}>{cell?.risk_count || 0}</Typography>
                    </Box>
                  </MuiTooltip>
                );
              })}
            </Box>
          ))}
          <Box sx={{ display: 'flex', mt: 0.5 }}>
            {LABELS_X.map((label) => (
              <Box key={label} sx={{ flex: 1, textAlign: 'center' }}><Typography variant="caption" sx={{ fontSize: 10 }}>{label}</Typography></Box>
            ))}
          </Box>
        </Box>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
        <Typography variant="caption" color="text.secondary">Likelihood (vertical) x Impact (horizontal)</Typography>
      </Box>
    </CardContent></Card>
  );
};

export default HeatMatrix;
