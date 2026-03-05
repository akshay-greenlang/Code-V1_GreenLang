import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Avatar, Grid } from '@mui/material';
import { Person, Groups } from '@mui/icons-material';
import type { GovernanceRole } from '../../types';

interface BoardStructureProps {
  roles: GovernanceRole[];
}

const ROLE_COLORS: Record<string, string> = {
  board: '#1B5E20',
  committee: '#0D47A1',
  executive: '#E65100',
  management: '#757575',
};

const BoardStructure: React.FC<BoardStructureProps> = ({ roles }) => {
  const grouped = roles.reduce<Record<string, GovernanceRole[]>>((acc, role) => {
    const key = role.role_type;
    if (!acc[key]) acc[key] = [];
    acc[key].push(role);
    return acc;
  }, {});

  const tiers = ['board', 'committee', 'executive', 'management'];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Groups /> Board & Management Structure
        </Typography>
        {tiers.map((tier) => {
          const tierRoles = grouped[tier] || [];
          if (tierRoles.length === 0) return null;
          return (
            <Box key={tier} sx={{ mb: 3 }}>
              <Typography variant="subtitle2" sx={{ color: ROLE_COLORS[tier], fontWeight: 600, mb: 1, textTransform: 'capitalize' }}>
                {tier} Level ({tierRoles.length})
              </Typography>
              <Grid container spacing={1.5}>
                {tierRoles.map((role) => (
                  <Grid item xs={12} sm={6} md={4} key={role.id}>
                    <Box sx={{ p: 1.5, border: `2px solid ${ROLE_COLORS[tier]}20`, borderRadius: 1, borderLeft: `4px solid ${ROLE_COLORS[tier]}` }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                        <Avatar sx={{ width: 28, height: 28, fontSize: 12, bgcolor: ROLE_COLORS[tier] }}>
                          {role.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
                        </Avatar>
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>{role.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{role.title}</Typography>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                        {role.climate_competencies.slice(0, 3).map((comp) => (
                          <Chip key={comp} label={comp} size="small" variant="outlined" sx={{ fontSize: 10, height: 20 }} />
                        ))}
                      </Box>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>
          );
        })}
        {roles.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
            No governance roles defined. Add roles to build the organizational structure.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default BoardStructure;
