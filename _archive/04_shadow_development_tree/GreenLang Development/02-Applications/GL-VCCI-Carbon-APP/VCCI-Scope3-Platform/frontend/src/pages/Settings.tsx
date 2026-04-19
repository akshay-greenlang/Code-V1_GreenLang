import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
} from '@mui/material';
import { Save } from '@mui/icons-material';

const Settings: React.FC = () => {
  const [settings, setSettings] = useState({
    // User Profile
    name: 'John Doe',
    email: 'john.doe@example.com',
    company: 'Green Industries Ltd.',

    // Preferences
    theme: 'light',
    language: 'en',
    timezone: 'UTC',
    dateFormat: 'YYYY-MM-DD',
    currency: 'USD',

    // Notifications
    emailNotifications: true,
    inAppNotifications: true,
    weeklyReport: true,

    // Dashboard
    defaultView: 'overview',
    defaultPeriod: '90d',
  });

  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleChange = (field: string, value: any) => {
    setSettings((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = () => {
    // In a real app, this would save to backend
    console.log('Saving settings:', settings);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Settings saved successfully!
        </Alert>
      )}

      {/* User Profile Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          User Profile
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Full Name"
              value={settings.name}
              onChange={(e) => handleChange('name', e.target.value)}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Email"
              type="email"
              value={settings.email}
              onChange={(e) => handleChange('email', e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Company"
              value={settings.company}
              onChange={(e) => handleChange('company', e.target.value)}
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Preferences Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Preferences
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Theme</InputLabel>
              <Select
                value={settings.theme}
                label="Theme"
                onChange={(e) => handleChange('theme', e.target.value)}
              >
                <MenuItem value="light">Light</MenuItem>
                <MenuItem value="dark">Dark</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Language</InputLabel>
              <Select
                value={settings.language}
                label="Language"
                onChange={(e) => handleChange('language', e.target.value)}
              >
                <MenuItem value="en">English</MenuItem>
                <MenuItem value="es">Spanish</MenuItem>
                <MenuItem value="fr">French</MenuItem>
                <MenuItem value="de">German</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Timezone</InputLabel>
              <Select
                value={settings.timezone}
                label="Timezone"
                onChange={(e) => handleChange('timezone', e.target.value)}
              >
                <MenuItem value="UTC">UTC</MenuItem>
                <MenuItem value="America/New_York">Eastern Time</MenuItem>
                <MenuItem value="America/Chicago">Central Time</MenuItem>
                <MenuItem value="America/Los_Angeles">Pacific Time</MenuItem>
                <MenuItem value="Europe/London">London</MenuItem>
                <MenuItem value="Europe/Paris">Paris</MenuItem>
                <MenuItem value="Asia/Tokyo">Tokyo</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Date Format</InputLabel>
              <Select
                value={settings.dateFormat}
                label="Date Format"
                onChange={(e) => handleChange('dateFormat', e.target.value)}
              >
                <MenuItem value="YYYY-MM-DD">YYYY-MM-DD</MenuItem>
                <MenuItem value="MM/DD/YYYY">MM/DD/YYYY</MenuItem>
                <MenuItem value="DD/MM/YYYY">DD/MM/YYYY</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Currency</InputLabel>
              <Select
                value={settings.currency}
                label="Currency"
                onChange={(e) => handleChange('currency', e.target.value)}
              >
                <MenuItem value="USD">USD ($)</MenuItem>
                <MenuItem value="EUR">EUR (€)</MenuItem>
                <MenuItem value="GBP">GBP (£)</MenuItem>
                <MenuItem value="JPY">JPY (¥)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Notifications Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Notifications
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.emailNotifications}
                  onChange={(e) => handleChange('emailNotifications', e.target.checked)}
                />
              }
              label="Email Notifications"
            />
            <Typography variant="caption" display="block" color="textSecondary" sx={{ ml: 4 }}>
              Receive email updates about important platform events and alerts
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.inAppNotifications}
                  onChange={(e) => handleChange('inAppNotifications', e.target.checked)}
                />
              }
              label="In-App Notifications"
            />
            <Typography variant="caption" display="block" color="textSecondary" sx={{ ml: 4 }}>
              Show notifications within the application
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.weeklyReport}
                  onChange={(e) => handleChange('weeklyReport', e.target.checked)}
                />
              }
              label="Weekly Summary Report"
            />
            <Typography variant="caption" display="block" color="textSecondary" sx={{ ml: 4 }}>
              Receive a weekly summary of your emissions and activity
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Dashboard Defaults Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Dashboard Defaults
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Default View</InputLabel>
              <Select
                value={settings.defaultView}
                label="Default View"
                onChange={(e) => handleChange('defaultView', e.target.value)}
              >
                <MenuItem value="overview">Overview</MenuItem>
                <MenuItem value="detailed">Detailed</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Default Period</InputLabel>
              <Select
                value={settings.defaultPeriod}
                label="Default Period"
                onChange={(e) => handleChange('defaultPeriod', e.target.value)}
              >
                <MenuItem value="30d">Last 30 Days</MenuItem>
                <MenuItem value="90d">Last 90 Days</MenuItem>
                <MenuItem value="1y">Last Year</MenuItem>
                <MenuItem value="ytd">Year to Date</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Save Button */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="contained"
          size="large"
          startIcon={<Save />}
          onClick={handleSave}
        >
          Save Settings
        </Button>
      </Box>
    </Box>
  );
};

export default Settings;
