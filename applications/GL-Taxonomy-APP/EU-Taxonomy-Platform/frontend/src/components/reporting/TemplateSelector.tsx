/**
 * TemplateSelector - Select report template (Article 8 or EBA Pillar 3).
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, Box, Radio, RadioGroup, FormControlLabel, Chip } from '@mui/material';

const TEMPLATES = [
  { id: 'article_8_nfrd', label: 'Article 8 (NFRD)', group: 'Article 8', desc: 'Non-Financial Reporting Directive format' },
  { id: 'article_8_csrd', label: 'Article 8 (CSRD)', group: 'Article 8', desc: 'Corporate Sustainability Reporting Directive format' },
  { id: 'article_8_simplified', label: 'Article 8 (Simplified)', group: 'Article 8', desc: 'Simplified format for SMEs under Omnibus' },
  { id: 'eba_pillar_3_gar', label: 'EBA GAR Template', group: 'EBA Pillar 3', desc: 'Template 7 - GAR (stock and flow)' },
  { id: 'eba_pillar_3_btar', label: 'EBA BTAR Template', group: 'EBA Pillar 3', desc: 'Template 8 - BTAR' },
  { id: 'eba_pillar_3_stock', label: 'EBA Stock Template', group: 'EBA Pillar 3', desc: 'Template 7a - Stock breakdown' },
  { id: 'eba_pillar_3_flow', label: 'EBA Flow Template', group: 'EBA Pillar 3', desc: 'Template 7b - Flow breakdown' },
  { id: 'eba_pillar_3_summary', label: 'EBA Summary', group: 'EBA Pillar 3', desc: 'Template 10 - Summary KPIs' },
];

interface TemplateSelectorProps {
  onSelect?: (templateId: string) => void;
}

const TemplateSelector: React.FC<TemplateSelectorProps> = ({ onSelect }) => {
  const [selected, setSelected] = useState('article_8_csrd');

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Select Report Template</Typography>
        <RadioGroup value={selected} onChange={(e) => { setSelected(e.target.value); onSelect?.(e.target.value); }}>
          <Grid container spacing={1}>
            {TEMPLATES.map(t => (
              <Grid item xs={12} sm={6} key={t.id}>
                <Box sx={{ p: 1.5, border: '1px solid', borderColor: selected === t.id ? 'primary.main' : '#E0E0E0', borderRadius: 1 }}>
                  <FormControlLabel
                    value={t.id}
                    control={<Radio size="small" />}
                    label={
                      <Box>
                        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>{t.label}</Typography>
                          <Chip label={t.group} size="small" variant="outlined" />
                        </Box>
                        <Typography variant="caption" color="text.secondary">{t.desc}</Typography>
                      </Box>
                    }
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        </RadioGroup>
      </CardContent>
    </Card>
  );
};

export default TemplateSelector;
