/**
 * Reporting - Template selector, Article 8 preview, EBA view, export.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import TemplateSelector from '../components/reporting/TemplateSelector';
import Article8Preview from '../components/reporting/Article8Preview';
import EBAPillar3View from '../components/reporting/EBAPillar3View';
import ExportDialog from '../components/reporting/ExportDialog';

const Reporting: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Reporting & Disclosure
    </Typography>

    <Box sx={{ mb: 3 }}>
      <TemplateSelector />
    </Box>

    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} md={7}>
        <Article8Preview />
      </Grid>
      <Grid item xs={12} md={5}>
        <ExportDialog />
      </Grid>
    </Grid>

    <Box>
      <EBAPillar3View />
    </Box>
  </Box>
);

export default Reporting;
