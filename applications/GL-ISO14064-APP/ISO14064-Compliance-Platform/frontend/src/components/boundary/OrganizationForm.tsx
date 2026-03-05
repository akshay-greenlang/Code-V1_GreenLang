/**
 * OrganizationForm - Create/edit organization form
 *
 * Provides a form for creating or editing an organization profile
 * with name, industry, country, and description fields.  Submits
 * via the organization Redux slice async thunks.
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  TextField,
  Button,
  Grid,
  MenuItem,
  Box,
  CircularProgress,
} from '@mui/material';
import { Save, Add } from '@mui/icons-material';
import type { Organization, CreateOrganizationRequest } from '../../types';

const INDUSTRIES = [
  'Energy',
  'Manufacturing',
  'Technology',
  'Finance',
  'Healthcare',
  'Transportation',
  'Agriculture',
  'Construction',
  'Mining',
  'Retail',
  'Utilities',
  'Other',
];

const COUNTRIES = [
  'United States',
  'United Kingdom',
  'Germany',
  'France',
  'Canada',
  'Australia',
  'Japan',
  'China',
  'India',
  'Brazil',
  'Netherlands',
  'Switzerland',
  'Singapore',
  'Other',
];

interface OrganizationFormProps {
  organization?: Organization | null;
  onSubmit: (data: CreateOrganizationRequest) => void;
  loading?: boolean;
}

const OrganizationForm: React.FC<OrganizationFormProps> = ({
  organization,
  onSubmit,
  loading = false,
}) => {
  const [name, setName] = useState('');
  const [industry, setIndustry] = useState('');
  const [country, setCountry] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    if (organization) {
      setName(organization.name);
      setIndustry(organization.industry);
      setCountry(organization.country);
      setDescription(organization.description || '');
    }
  }, [organization]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      name,
      industry,
      country,
      description: description || null,
    });
  };

  const isEdit = Boolean(organization);

  return (
    <Card>
      <CardHeader
        title={isEdit ? 'Edit Organization' : 'Create Organization'}
        subheader="Define the reporting organization for ISO 14064-1"
      />
      <CardContent>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                required
                label="Organization Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Acme Corporation"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                required
                select
                label="Industry"
                value={industry}
                onChange={(e) => setIndustry(e.target.value)}
              >
                {INDUSTRIES.map((ind) => (
                  <MenuItem key={ind} value={ind}>
                    {ind}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                required
                select
                label="Country"
                value={country}
                onChange={(e) => setCountry(e.target.value)}
              >
                {COUNTRIES.map((c) => (
                  <MenuItem key={c} value={c}>
                    {c}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional description"
              />
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading || !name || !industry || !country}
                  startIcon={
                    loading ? (
                      <CircularProgress size={18} />
                    ) : isEdit ? (
                      <Save />
                    ) : (
                      <Add />
                    )
                  }
                  sx={{
                    bgcolor: '#1b5e20',
                    '&:hover': { bgcolor: '#2e7d32' },
                  }}
                >
                  {isEdit ? 'Save Changes' : 'Create Organization'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </form>
      </CardContent>
    </Card>
  );
};

export default OrganizationForm;
