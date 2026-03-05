/**
 * Settings Page - Application and organization settings
 *
 * Manages organization profile, reporting configuration, team members,
 * MRV agent connections, and notification preferences.
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Alert,
  Card,
  CardContent,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  Chip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Save,
  PersonAdd,
  Delete,
  Sync,
  CheckCircle,
  Cancel,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchSettings,
  updateSettings,
  addTeamMember,
  removeTeamMember,
} from '../store/slices/settingsSlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { CDP_MODULE_NAMES, CDPModule } from '../types';
import { formatDateTime } from '../utils/formatters';

const DEMO_ORG_ID = 'demo-org';

const SettingsPage: React.FC = () => {
  const dispatch = useAppDispatch();
  const { settings, loading, saving, error } = useAppSelector(
    (s) => s.settings,
  );

  const [addMemberOpen, setAddMemberOpen] = useState(false);
  const [newMember, setNewMember] = useState({
    name: '',
    email: '',
    role: 'contributor',
  });

  useEffect(() => {
    dispatch(fetchSettings(DEMO_ORG_ID));
  }, [dispatch]);

  const handleSave = () => {
    if (!settings) return;
    dispatch(updateSettings({
      orgId: DEMO_ORG_ID,
      data: {
        reporting_year: settings.reporting_year,
        reporting_boundary: settings.reporting_boundary,
        gics_sector: settings.gics_sector,
        notification_email: settings.notification_email,
        auto_populate_enabled: settings.auto_populate_enabled,
        submission_deadline: settings.submission_deadline,
      },
    }));
  };

  const handleAddMember = () => {
    dispatch(addTeamMember({
      orgId: DEMO_ORG_ID,
      data: newMember,
    }));
    setAddMemberOpen(false);
    setNewMember({ name: '', email: '', role: 'contributor' });
  };

  const handleRemoveMember = (memberId: string) => {
    dispatch(removeTeamMember({ orgId: DEMO_ORG_ID, memberId }));
  };

  if (loading && !settings) return <LoadingSpinner message="Loading settings..." />;
  if (error) return <Alert severity="error">{error}</Alert>;
  if (!settings) return null;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h5" fontWeight={700}>
          Settings
        </Typography>
        <Button
          variant="contained"
          startIcon={<Save />}
          onClick={handleSave}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* Organization settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Organization
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Reporting Year"
                  type="number"
                  value={settings.reporting_year}
                  size="small"
                  fullWidth
                  InputProps={{ readOnly: true }}
                />
                <TextField
                  label="Reporting Boundary"
                  value={settings.reporting_boundary}
                  size="small"
                  fullWidth
                />
                <TextField
                  label="GICS Sector"
                  value={settings.gics_sector}
                  size="small"
                  fullWidth
                />
                <TextField
                  label="GICS Industry Group"
                  value={settings.gics_industry_group}
                  size="small"
                  fullWidth
                  InputProps={{ readOnly: true }}
                />
                <TextField
                  label="Notification Email"
                  type="email"
                  value={settings.notification_email}
                  size="small"
                  fullWidth
                />
                <TextField
                  label="Submission Deadline"
                  type="date"
                  value={settings.submission_deadline?.split('T')[0] || ''}
                  size="small"
                  fullWidth
                  InputLabelProps={{ shrink: true }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.auto_populate_enabled}
                      color="primary"
                    />
                  }
                  label="Auto-populate from MRV agents"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* MRV connections */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">MRV Agent Connections</Typography>
                <Chip
                  label={`${settings.mrv_connections.filter((c) => c.connected).length}/${settings.mrv_connections.length} connected`}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              </Box>

              {settings.mrv_connections.map((conn) => (
                <Box
                  key={conn.agent_id}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1.5,
                    py: 1,
                    borderBottom: '1px solid #f0f0f0',
                  }}
                >
                  {conn.connected ? (
                    <CheckCircle sx={{ fontSize: 18, color: '#2e7d32' }} />
                  ) : (
                    <Cancel sx={{ fontSize: 18, color: '#9e9e9e' }} />
                  )}
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="body2" fontWeight={500}>
                      {conn.agent_name}
                    </Typography>
                    {conn.last_sync && (
                      <Typography variant="caption" color="text.secondary">
                        Last sync: {formatDateTime(conn.last_sync)}
                      </Typography>
                    )}
                  </Box>
                  <Chip
                    label={conn.data_freshness}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: 10 }}
                  />
                  <IconButton size="small" disabled={!conn.connected}>
                    <Sync fontSize="small" />
                  </IconButton>
                </Box>
              ))}

              {settings.mrv_connections.length === 0 && (
                <Typography variant="body2" color="text.secondary">
                  No MRV agent connections configured.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Team members */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">Team Members</Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<PersonAdd />}
                  onClick={() => setAddMemberOpen(true)}
                >
                  Add Member
                </Button>
              </Box>

              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Email</TableCell>
                      <TableCell>Role</TableCell>
                      <TableCell>Assigned Modules</TableCell>
                      <TableCell align="right">Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {settings.team_members.map((member) => (
                      <TableRow key={member.id}>
                        <TableCell>
                          <Typography variant="body2" fontWeight={500}>
                            {member.name}
                          </Typography>
                        </TableCell>
                        <TableCell>{member.email}</TableCell>
                        <TableCell>
                          <Chip label={member.role} size="small" variant="outlined" />
                        </TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                            {member.modules_assigned.map((mod) => (
                              <Chip
                                key={mod}
                                label={CDP_MODULE_NAMES[mod as CDPModule] || mod}
                                size="small"
                                sx={{ fontSize: 10 }}
                              />
                            ))}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <IconButton
                            size="small"
                            onClick={() => handleRemoveMember(member.id)}
                            sx={{ color: '#9e9e9e', '&:hover': { color: '#e53935' } }}
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                    {settings.team_members.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={5}>
                          <Typography variant="body2" color="text.secondary" textAlign="center">
                            No team members added.
                          </Typography>
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Add member dialog */}
      <Dialog open={addMemberOpen} onClose={() => setAddMemberOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Team Member</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Name"
              value={newMember.name}
              onChange={(e) => setNewMember({ ...newMember, name: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={newMember.email}
              onChange={(e) => setNewMember({ ...newMember, email: e.target.value })}
              fullWidth
              required
            />
            <TextField
              label="Role"
              value={newMember.role}
              onChange={(e) => setNewMember({ ...newMember, role: e.target.value })}
              fullWidth
              placeholder="e.g., contributor, reviewer, admin"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddMemberOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddMember}
            disabled={!newMember.name || !newMember.email}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SettingsPage;
