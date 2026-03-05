/**
 * PortfolioManagement - Portfolio CRUD, holdings, upload, counterparty search.
 */

import React from 'react';
import { Typography, Grid, Box } from '@mui/material';
import PortfolioTable from '../components/portfolio/PortfolioTable';
import HoldingsEditor from '../components/portfolio/HoldingsEditor';
import ExposureUpload from '../components/portfolio/ExposureUpload';
import CounterpartySearch from '../components/portfolio/CounterpartySearch';

const PortfolioManagement: React.FC = () => (
  <Box>
    <Typography variant="h4" sx={{ fontWeight: 700, mb: 3 }}>
      Portfolio Management
    </Typography>

    <Box sx={{ mb: 3 }}>
      <PortfolioTable />
    </Box>

    <Box sx={{ mb: 3 }}>
      <HoldingsEditor />
    </Box>

    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <ExposureUpload />
      </Grid>
      <Grid item xs={12} md={6}>
        <CounterpartySearch />
      </Grid>
    </Grid>
  </Box>
);

export default PortfolioManagement;
