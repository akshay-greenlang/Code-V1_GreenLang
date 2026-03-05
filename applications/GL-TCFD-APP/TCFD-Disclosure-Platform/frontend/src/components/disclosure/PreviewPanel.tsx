import React from 'react';
import { Card, CardContent, Typography, Box, Divider } from '@mui/material';
import type { DisclosureSection } from '../../types';

interface PreviewPanelProps { sections: DisclosureSection[]; title?: string; }

const PILLAR_LABELS: Record<string, string> = { governance: 'Governance', strategy: 'Strategy', risk_management: 'Risk Management', metrics_targets: 'Metrics & Targets' };

const PreviewPanel: React.FC<PreviewPanelProps> = ({ sections, title = 'TCFD Disclosure Report' }) => {
  const pillars = ['governance', 'strategy', 'risk_management', 'metrics_targets'];
  return (
    <Card><CardContent>
      <Typography variant="h5" sx={{ fontWeight: 700, mb: 1, textAlign: 'center' }}>{title}</Typography>
      <Divider sx={{ mb: 3 }} />
      {pillars.map((pillar) => {
        const pillarSections = sections.filter((s) => s.pillar === pillar).sort((a, b) => a.order - b.order);
        if (pillarSections.length === 0) return null;
        return (
          <Box key={pillar} sx={{ mb: 4 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, color: 'primary.main', mb: 2, borderBottom: '2px solid', borderColor: 'primary.main', pb: 0.5 }}>
              {PILLAR_LABELS[pillar] || pillar}
            </Typography>
            {pillarSections.map((section) => (
              <Box key={section.id} sx={{ mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>{section.tcfd_code}: {section.title}</Typography>
                <Typography variant="body2" sx={{ lineHeight: 1.8, whiteSpace: 'pre-wrap', fontFamily: '"Georgia", serif' }}>
                  {section.content || <em style={{ color: '#999' }}>Content not yet provided.</em>}
                </Typography>
              </Box>
            ))}
          </Box>
        );
      })}
    </CardContent></Card>
  );
};

export default PreviewPanel;
