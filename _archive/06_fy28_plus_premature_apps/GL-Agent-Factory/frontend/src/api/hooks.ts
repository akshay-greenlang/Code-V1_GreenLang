/**
 * GreenLang Agent Factory - React Query Hooks
 *
 * Custom hooks for data fetching with React Query.
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  UseQueryOptions,
  UseMutationOptions,
} from '@tanstack/react-query';
import { toast } from 'sonner';
import { apiClient } from './client';
import type {
  Agent,
  AgentLog,
  User,
  Tenant,
  UserCreateRequest,
  UserUpdateRequest,
  FuelAnalysisRequest,
  FuelAnalysisResult,
  CBAMCalculationRequest,
  CBAMCalculationResult,
  BuildingEnergyRequest,
  BuildingEnergyResult,
  EUDRComplianceRequest,
  EUDRComplianceResult,
  Report,
  ReportGenerateRequest,
  DashboardMetrics,
  SystemAlert,
  PaginatedResponse,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
} from './types';

// ============================================================================
// Query Keys
// ============================================================================

export const queryKeys = {
  // Auth
  currentUser: ['currentUser'] as const,

  // Agents
  agents: ['agents'] as const,
  agent: (id: string) => ['agents', id] as const,
  agentLogs: (id: string) => ['agents', id, 'logs'] as const,

  // Users
  users: ['users'] as const,
  user: (id: string) => ['users', id] as const,

  // Tenants
  tenants: ['tenants'] as const,
  tenant: (id: string) => ['tenants', id] as const,

  // Calculations
  fuelHistory: ['calculations', 'fuel'] as const,
  cbamHistory: ['calculations', 'cbam'] as const,
  buildingHistory: ['calculations', 'building'] as const,
  eudrHistory: ['compliance', 'eudr'] as const,

  // Reports
  reports: ['reports'] as const,
  report: (id: string) => ['reports', id] as const,

  // Dashboard
  dashboardMetrics: ['dashboard', 'metrics'] as const,
  systemAlerts: ['dashboard', 'alerts'] as const,
};

// ============================================================================
// Auth Hooks
// ============================================================================

export function useCurrentUser(
  options?: Omit<UseQueryOptions<User>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.currentUser,
    queryFn: () => apiClient.getCurrentUser(),
    staleTime: 1000 * 60 * 5, // 5 minutes
    ...options,
  });
}

export function useLogin(
  options?: UseMutationOptions<LoginResponse, Error, LoginRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (credentials: LoginRequest) => apiClient.login(credentials),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.currentUser, data.user);
      toast.success('Welcome back!');
    },
    onError: (error) => {
      toast.error(error.message || 'Login failed');
    },
    ...options,
  });
}

export function useRegister(
  options?: UseMutationOptions<LoginResponse, Error, RegisterRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: RegisterRequest) => apiClient.register(data),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.currentUser, data.user);
      toast.success('Account created successfully!');
    },
    onError: (error) => {
      toast.error(error.message || 'Registration failed');
    },
    ...options,
  });
}

export function useLogout(options?: UseMutationOptions<void, Error, void>) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiClient.logout(),
    onSuccess: () => {
      queryClient.clear();
      toast.success('Logged out successfully');
    },
    ...options,
  });
}

export function useForgotPassword(
  options?: UseMutationOptions<void, Error, string>
) {
  return useMutation({
    mutationFn: (email: string) => apiClient.forgotPassword(email),
    onSuccess: () => {
      toast.success('Password reset email sent');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to send reset email');
    },
    ...options,
  });
}

// ============================================================================
// Agent Hooks
// ============================================================================

export function useAgents(
  params?: Parameters<typeof apiClient.getAgents>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<Agent>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.agents, params],
    queryFn: () => apiClient.getAgents(params),
    ...options,
  });
}

export function useAgent(
  agentId: string,
  options?: Omit<UseQueryOptions<Agent>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.agent(agentId),
    queryFn: () => apiClient.getAgent(agentId),
    enabled: !!agentId,
    ...options,
  });
}

export function useAgentLogs(
  agentId: string,
  params?: Parameters<typeof apiClient.getAgentLogs>[1],
  options?: Omit<UseQueryOptions<PaginatedResponse<AgentLog>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.agentLogs(agentId), params],
    queryFn: () => apiClient.getAgentLogs(agentId, params),
    enabled: !!agentId,
    ...options,
  });
}

export function useUpdateAgentConfig(
  options?: UseMutationOptions<Agent, Error, { agentId: string; config: Partial<Agent['config']> }>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, config }) => apiClient.updateAgentConfig(agentId, config),
    onSuccess: (data, variables) => {
      queryClient.setQueryData(queryKeys.agent(variables.agentId), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.agents });
      toast.success('Agent configuration updated');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to update configuration');
    },
    ...options,
  });
}

export function useRestartAgent(
  options?: UseMutationOptions<Agent, Error, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (agentId: string) => apiClient.restartAgent(agentId),
    onSuccess: (data, agentId) => {
      queryClient.setQueryData(queryKeys.agent(agentId), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.agents });
      toast.success('Agent restarted successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to restart agent');
    },
    ...options,
  });
}

// ============================================================================
// User Management Hooks
// ============================================================================

export function useUsers(
  params?: Parameters<typeof apiClient.getUsers>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<User>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.users, params],
    queryFn: () => apiClient.getUsers(params),
    ...options,
  });
}

export function useUser(
  userId: string,
  options?: Omit<UseQueryOptions<User>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.user(userId),
    queryFn: () => apiClient.getUser(userId),
    enabled: !!userId,
    ...options,
  });
}

export function useCreateUser(
  options?: UseMutationOptions<User, Error, UserCreateRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: UserCreateRequest) => apiClient.createUser(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      toast.success('User created successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to create user');
    },
    ...options,
  });
}

export function useUpdateUser(
  options?: UseMutationOptions<User, Error, { userId: string; data: UserUpdateRequest }>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ userId, data }) => apiClient.updateUser(userId, data),
    onSuccess: (data, variables) => {
      queryClient.setQueryData(queryKeys.user(variables.userId), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      toast.success('User updated successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to update user');
    },
    ...options,
  });
}

export function useDeleteUser(options?: UseMutationOptions<void, Error, string>) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (userId: string) => apiClient.deleteUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.users });
      toast.success('User deleted successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to delete user');
    },
    ...options,
  });
}

// ============================================================================
// Tenant Management Hooks
// ============================================================================

export function useTenants(
  params?: Parameters<typeof apiClient.getTenants>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<Tenant>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.tenants, params],
    queryFn: () => apiClient.getTenants(params),
    ...options,
  });
}

export function useTenant(
  tenantId: string,
  options?: Omit<UseQueryOptions<Tenant>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.tenant(tenantId),
    queryFn: () => apiClient.getTenant(tenantId),
    enabled: !!tenantId,
    ...options,
  });
}

export function useCreateTenant(
  options?: UseMutationOptions<Tenant, Error, Partial<Tenant>>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: Partial<Tenant>) => apiClient.createTenant(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.tenants });
      toast.success('Tenant created successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to create tenant');
    },
    ...options,
  });
}

export function useUpdateTenant(
  options?: UseMutationOptions<Tenant, Error, { tenantId: string; data: Partial<Tenant> }>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ tenantId, data }) => apiClient.updateTenant(tenantId, data),
    onSuccess: (data, variables) => {
      queryClient.setQueryData(queryKeys.tenant(variables.tenantId), data);
      queryClient.invalidateQueries({ queryKey: queryKeys.tenants });
      toast.success('Tenant updated successfully');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to update tenant');
    },
    ...options,
  });
}

// ============================================================================
// Calculation Hooks
// ============================================================================

export function useCalculateFuel(
  options?: UseMutationOptions<FuelAnalysisResult, Error, FuelAnalysisRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: FuelAnalysisRequest) => apiClient.calculateFuelEmissions(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.fuelHistory });
      toast.success('Fuel emissions calculated');
    },
    onError: (error) => {
      toast.error(error.message || 'Calculation failed');
    },
    ...options,
  });
}

export function useFuelHistory(
  params?: Parameters<typeof apiClient.getFuelCalculationHistory>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<FuelAnalysisResult>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.fuelHistory, params],
    queryFn: () => apiClient.getFuelCalculationHistory(params),
    ...options,
  });
}

export function useCalculateCBAM(
  options?: UseMutationOptions<CBAMCalculationResult, Error, CBAMCalculationRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CBAMCalculationRequest) => apiClient.calculateCBAM(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.cbamHistory });
      toast.success('CBAM calculation completed');
    },
    onError: (error) => {
      toast.error(error.message || 'Calculation failed');
    },
    ...options,
  });
}

export function useCBAMHistory(
  params?: Parameters<typeof apiClient.getCBAMHistory>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<CBAMCalculationResult>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.cbamHistory, params],
    queryFn: () => apiClient.getCBAMHistory(params),
    ...options,
  });
}

export function useCalculateBuildingEnergy(
  options?: UseMutationOptions<BuildingEnergyResult, Error, BuildingEnergyRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: BuildingEnergyRequest) => apiClient.calculateBuildingEnergy(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.buildingHistory });
      toast.success('Building energy calculated');
    },
    onError: (error) => {
      toast.error(error.message || 'Calculation failed');
    },
    ...options,
  });
}

export function useBuildingHistory(
  params?: Parameters<typeof apiClient.getBuildingEnergyHistory>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<BuildingEnergyResult>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.buildingHistory, params],
    queryFn: () => apiClient.getBuildingEnergyHistory(params),
    ...options,
  });
}

export function useCheckEUDRCompliance(
  options?: UseMutationOptions<EUDRComplianceResult, Error, EUDRComplianceRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: EUDRComplianceRequest) => apiClient.checkEUDRCompliance(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.eudrHistory });
      toast.success('EUDR compliance check completed');
    },
    onError: (error) => {
      toast.error(error.message || 'Compliance check failed');
    },
    ...options,
  });
}

export function useEUDRHistory(
  params?: Parameters<typeof apiClient.getEUDRComplianceHistory>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<EUDRComplianceResult>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.eudrHistory, params],
    queryFn: () => apiClient.getEUDRComplianceHistory(params),
    ...options,
  });
}

// ============================================================================
// Report Hooks
// ============================================================================

export function useReports(
  params?: Parameters<typeof apiClient.getReports>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<Report>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.reports, params],
    queryFn: () => apiClient.getReports(params),
    ...options,
  });
}

export function useReport(
  reportId: string,
  options?: Omit<UseQueryOptions<Report>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.report(reportId),
    queryFn: () => apiClient.getReport(reportId),
    enabled: !!reportId,
    ...options,
  });
}

export function useGenerateReport(
  options?: UseMutationOptions<Report, Error, ReportGenerateRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ReportGenerateRequest) => apiClient.generateReport(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.reports });
      toast.success('Report generation started');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to generate report');
    },
    ...options,
  });
}

export function useDeleteReport(options?: UseMutationOptions<void, Error, string>) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (reportId: string) => apiClient.deleteReport(reportId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.reports });
      toast.success('Report deleted');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to delete report');
    },
    ...options,
  });
}

// ============================================================================
// Dashboard Hooks
// ============================================================================

export function useDashboardMetrics(
  params?: Parameters<typeof apiClient.getDashboardMetrics>[0],
  options?: Omit<UseQueryOptions<DashboardMetrics>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.dashboardMetrics, params],
    queryFn: () => apiClient.getDashboardMetrics(params),
    refetchInterval: 1000 * 60, // Refresh every minute
    ...options,
  });
}

export function useSystemAlerts(
  params?: Parameters<typeof apiClient.getSystemAlerts>[0],
  options?: Omit<UseQueryOptions<PaginatedResponse<SystemAlert>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...queryKeys.systemAlerts, params],
    queryFn: () => apiClient.getSystemAlerts(params),
    refetchInterval: 1000 * 30, // Refresh every 30 seconds
    ...options,
  });
}

export function useAcknowledgeAlert(
  options?: UseMutationOptions<SystemAlert, Error, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (alertId: string) => apiClient.acknowledgeAlert(alertId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.systemAlerts });
      toast.success('Alert acknowledged');
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to acknowledge alert');
    },
    ...options,
  });
}
