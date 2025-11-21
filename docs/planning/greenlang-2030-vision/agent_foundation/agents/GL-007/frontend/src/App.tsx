/**
 * Main App Component
 *
 * Root component with routing and layout
 */

import React, { useState } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useLocation,
} from 'react-router-dom';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Avatar,
  Menu,
  Badge,
  Tooltip,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  Settings,
  Assessment,
  Build,
  Notifications,
  Thermostat,
  Report,
  Warning,
  Brightness4,
  Brightness7,
  ChevronLeft,
} from '@mui/icons-material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { useFurnaceStore } from './store/furnaceStore';
import ExecutiveDashboard from './components/dashboards/ExecutiveDashboard';
import OperationsDashboard from './components/dashboards/OperationsDashboard';
import ThermalProfilingView from './components/dashboards/ThermalProfilingView';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 5000,
    },
  },
});

const DRAWER_WIDTH = 260;

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  path: string;
  badge?: number;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'executive',
    label: 'Executive Dashboard',
    icon: <Dashboard />,
    path: '/executive',
  },
  {
    id: 'operations',
    label: 'Operations',
    icon: <Assessment />,
    path: '/operations',
  },
  {
    id: 'thermal',
    label: 'Thermal Profiling',
    icon: <Thermostat />,
    path: '/thermal',
  },
  {
    id: 'maintenance',
    label: 'Maintenance',
    icon: <Build />,
    path: '/maintenance',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: <Assessment />,
    path: '/analytics',
  },
  {
    id: 'alerts',
    label: 'Alerts',
    icon: <Warning />,
    path: '/alerts',
  },
  {
    id: 'reports',
    label: 'Reports',
    icon: <Report />,
    path: '/reports',
  },
  {
    id: 'settings',
    label: 'Configuration',
    icon: <Settings />,
    path: '/settings',
  },
];

function AppContent() {
  const navigate = useNavigate();
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const { selectedFurnaceId, setSelectedFurnace, furnaces, unacknowledgedCount } =
    useFurnaceStore();

  // Create theme
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
    },
  });

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleThemeToggle = () => {
    setDarkMode(!darkMode);
  };

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  // Mock furnaces if none available
  const availableFurnaces = furnaces.length > 0 ? furnaces : [
    {
      id: 'furnace-1',
      name: 'Blast Furnace #1',
      type: 'blast_furnace' as const,
      manufacturer: 'Siemens',
      model: 'BF-5000',
      capacity: 500,
      zones: [],
      sensors: [],
      fuelType: 'coal' as const,
      installDate: '2020-01-01',
      location: { plant: 'Main Plant', site: 'Site A', building: 'Building 1' },
      status: 'running' as const,
      specifications: {
        maxTemperature: 1500,
        minTemperature: 800,
        optimalTemperature: 1200,
        maxPressure: 5,
        maxFuelConsumption: 10000,
        thermalEfficiency: 85,
        emissionLimits: { co2: 400, nox: 50, sox: 30, particulates: 20 },
      },
    },
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        {/* App Bar */}
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            transition: theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              onClick={handleDrawerToggle}
              edge="start"
              sx={{ mr: 2 }}
            >
              {drawerOpen ? <ChevronLeft /> : <MenuIcon />}
            </IconButton>

            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 0, mr: 3 }}>
              GL-007 Furnace Monitor
            </Typography>

            {/* Furnace Selector */}
            <FormControl size="small" sx={{ minWidth: 250, mr: 'auto' }}>
              <InputLabel sx={{ color: 'white' }}>Select Furnace</InputLabel>
              <Select
                value={selectedFurnaceId || ''}
                onChange={(e) => setSelectedFurnace(e.target.value)}
                label="Select Furnace"
                sx={{
                  color: 'white',
                  '.MuiOutlinedInput-notchedOutline': { borderColor: 'white' },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'white',
                  },
                  '.MuiSvgIcon-root': { color: 'white' },
                }}
              >
                {availableFurnaces.map((furnace) => (
                  <MenuItem key={furnace.id} value={furnace.id}>
                    {furnace.name} - {furnace.type}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Action Icons */}
            <Tooltip title="Toggle dark mode">
              <IconButton color="inherit" onClick={handleThemeToggle}>
                {darkMode ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Tooltip>

            <Tooltip title="Notifications">
              <IconButton color="inherit">
                <Badge badgeContent={unacknowledgedCount} color="error">
                  <Notifications />
                </Badge>
              </IconButton>
            </Tooltip>

            <Tooltip title="Account">
              <IconButton onClick={handleProfileMenuOpen} sx={{ ml: 1 }}>
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'secondary.main' }}>
                  U
                </Avatar>
              </IconButton>
            </Tooltip>

            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleProfileMenuClose}
            >
              <MenuItem onClick={handleProfileMenuClose}>Profile</MenuItem>
              <MenuItem onClick={handleProfileMenuClose}>Settings</MenuItem>
              <Divider />
              <MenuItem onClick={handleProfileMenuClose}>Logout</MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        {/* Drawer */}
        <Drawer
          variant="persistent"
          open={drawerOpen}
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto', mt: 1 }}>
            <List>
              {navigationItems.map((item) => (
                <ListItem key={item.id} disablePadding>
                  <ListItemButton
                    selected={location.pathname === item.path}
                    onClick={() => navigate(item.path)}
                  >
                    <ListItemIcon>
                      {item.badge !== undefined ? (
                        <Badge badgeContent={item.badge} color="error">
                          {item.icon}
                        </Badge>
                      ) : (
                        item.icon
                      )}
                    </ListItemIcon>
                    <ListItemText primary={item.label} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          </Box>
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 0,
            width: `calc(100% - ${drawerOpen ? DRAWER_WIDTH : 0}px)`,
            transition: theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
          }}
        >
          <Toolbar />
          <Routes>
            <Route path="/" element={<Navigate to="/executive" replace />} />
            <Route path="/executive" element={<ExecutiveDashboard />} />
            <Route path="/operations" element={<OperationsDashboard />} />
            <Route path="/thermal" element={<ThermalProfilingView />} />
            <Route
              path="/maintenance"
              element={
                <Box sx={{ p: 3 }}>
                  <Typography variant="h4">Maintenance Dashboard</Typography>
                  <Typography>Coming soon...</Typography>
                </Box>
              }
            />
            <Route
              path="/analytics"
              element={
                <Box sx={{ p: 3 }}>
                  <Typography variant="h4">Analytics Dashboard</Typography>
                  <Typography>Coming soon...</Typography>
                </Box>
              }
            />
            <Route
              path="/alerts"
              element={
                <Box sx={{ p: 3 }}>
                  <Typography variant="h4">Alert Management</Typography>
                  <Typography>Coming soon...</Typography>
                </Box>
              }
            />
            <Route
              path="/reports"
              element={
                <Box sx={{ p: 3 }}>
                  <Typography variant="h4">Reporting Module</Typography>
                  <Typography>Coming soon...</Typography>
                </Box>
              }
            />
            <Route
              path="/settings"
              element={
                <Box sx={{ p: 3 }}>
                  <Typography variant="h4">Configuration</Typography>
                  <Typography>Coming soon...</Typography>
                </Box>
              }
            />
          </Routes>
        </Box>
      </Box>

      {/* Toast Notifications */}
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme={darkMode ? 'dark' : 'light'}
      />
    </ThemeProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppContent />
      </Router>
    </QueryClientProvider>
  );
}

export default App;
