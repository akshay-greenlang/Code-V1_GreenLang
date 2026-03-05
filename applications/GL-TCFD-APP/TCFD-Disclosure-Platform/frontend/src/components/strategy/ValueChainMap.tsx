import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip, List, ListItem, ListItemText } from '@mui/material';
import RiskBadge from '../common/RiskBadge';
import type { ValueChainNode } from '../../types';
import { formatCurrency } from '../../utils/formatters';

interface ValueChainMapProps {
  nodes: ValueChainNode[];
}

const POSITION_CONFIG: Record<string, { label: string; color: string }> = {
  upstream: { label: 'Upstream', color: '#0D47A1' },
  direct_operations: { label: 'Direct Operations', color: '#1B5E20' },
  downstream: { label: 'Downstream', color: '#E65100' },
};

const ValueChainMap: React.FC<ValueChainMapProps> = ({ nodes }) => {
  const positions = ['upstream', 'direct_operations', 'downstream'];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Value Chain Risk & Opportunity Map</Typography>
        <Grid container spacing={2}>
          {positions.map((pos) => {
            const config = POSITION_CONFIG[pos];
            const posNodes = nodes.filter((n) => n.position === pos);
            return (
              <Grid item xs={12} md={4} key={pos}>
                <Box sx={{ border: `2px solid ${config.color}`, borderRadius: 2, p: 2, height: '100%' }}>
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, color: config.color, mb: 1.5, textAlign: 'center' }}>
                    {config.label}
                  </Typography>
                  {posNodes.map((node) => (
                    <Box key={node.id} sx={{ mb: 2, p: 1.5, bgcolor: '#FAFAFA', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{node.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Exposure: {formatCurrency(node.financial_exposure, 'USD', true)}
                      </Typography>
                      {node.risks.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" sx={{ fontWeight: 600, color: 'error.main' }}>Risks:</Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                            {node.risks.slice(0, 3).map((r) => (
                              <RiskBadge key={r.id} level={r.level} size="small" />
                            ))}
                          </Box>
                        </Box>
                      )}
                      {node.opportunities.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" sx={{ fontWeight: 600, color: 'success.main' }}>Opportunities:</Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                            {node.opportunities.slice(0, 3).map((o) => (
                              <Chip key={o.id} label={o.name} size="small" color="success" variant="outlined" sx={{ fontSize: 10, height: 20 }} />
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  ))}
                  {posNodes.length === 0 && (
                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                      No nodes defined
                    </Typography>
                  )}
                </Box>
              </Grid>
            );
          })}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ValueChainMap;
