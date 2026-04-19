/**
 * Admin Dashboard Page
 *
 * Overview of system status, metrics, and alerts.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  Bot,
  Users,
  Building2,
  Activity,
  TrendingUp,
  AlertCircle,
  ArrowRight,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { MetricCard, MetricGrid } from '@/components/widgets/MetricCard';
import { AgentStatusCard, AgentStatusCardSkeleton } from '@/components/widgets/AgentStatusCard';
import { AlertsList } from '@/components/widgets/AlertsList';
import { EmissionsTrendChart, EmissionsByCategoryChart } from '@/components/charts/EmissionsChart';
import { useDashboardMetrics, useAgents, useSystemAlerts, useAcknowledgeAlert } from '@/api/hooks';
import { formatEmissions, formatNumber, formatRelativeTime } from '@/utils/format';

export default function AdminDashboard() {
  // Fetch data
  const { data: metrics, isLoading: metricsLoading } = useDashboardMetrics();
  const { data: agentsResponse, isLoading: agentsLoading } = useAgents({ perPage: 4 });
  const { data: alertsResponse, isLoading: alertsLoading } = useSystemAlerts({ perPage: 5 });

  const acknowledgeAlert = useAcknowledgeAlert();

  // Mock data for development (remove when API is ready)
  const mockMetrics = {
    totalEmissions: 125430,
    emissionsTrend: -5.2,
    totalCalculations: 15834,
    activeAgents: 5,
    dataQualityScore: 94.5,
    complianceRate: 98.2,
    recentActivity: [
      {
        id: '1',
        type: 'calculation' as const,
        title: 'CBAM Calculation Completed',
        description: 'Steel imports from China - 450 tonnes',
        timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
      },
      {
        id: '2',
        type: 'report' as const,
        title: 'Quarterly Report Generated',
        description: 'Q3 2024 Emissions Report ready for download',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      },
      {
        id: '3',
        type: 'alert' as const,
        title: 'Data Quality Alert',
        description: 'Missing supplier data for 3 shipments',
        timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
      },
    ],
    emissionsByCategory: [
      { category: 'Iron & Steel', value: 45200, percentage: 36 },
      { category: 'Aluminum', value: 28100, percentage: 22.4 },
      { category: 'Cement', value: 21500, percentage: 17.2 },
      { category: 'Fertilizers', value: 18400, percentage: 14.7 },
      { category: 'Electricity', value: 12230, percentage: 9.7 },
    ],
    emissionsTrendData: Array.from({ length: 12 }, (_, i) => ({
      date: new Date(2024, i, 1).toLocaleDateString('en-US', { month: 'short' }),
      emissions: Math.floor(10000 + Math.random() * 5000),
    })),
    topCountries: [
      { country: 'China', emissions: 45000, imports: 1250 },
      { country: 'India', emissions: 28000, imports: 850 },
      { country: 'Turkey', emissions: 18500, imports: 620 },
    ],
  };

  const mockAgents = [
    {
      id: '1',
      name: 'CBAM Agent',
      description: 'Carbon Border Adjustment Mechanism calculations and reporting',
      version: '2.1.0',
      status: 'active' as const,
      type: 'cbam' as const,
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
      createdAt: '2024-01-01',
      updatedAt: '2024-07-15',
      metrics: { requestsToday: 1234, requestsThisMonth: 28500, avgResponseTime: 245, errorRate: 0.5, uptime: 99.9 },
      config: { endpoint: '/api/cbam', maxConcurrency: 100, timeout: 30000, retryAttempts: 3, rateLimit: 1000, enableCache: true },
    },
    {
      id: '2',
      name: 'EUDR Agent',
      description: 'EU Deforestation Regulation compliance verification',
      version: '1.5.2',
      status: 'active' as const,
      type: 'eudr' as const,
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
      createdAt: '2024-02-15',
      updatedAt: '2024-07-10',
      metrics: { requestsToday: 856, requestsThisMonth: 18200, avgResponseTime: 380, errorRate: 1.2, uptime: 99.5 },
      config: { endpoint: '/api/eudr', maxConcurrency: 50, timeout: 60000, retryAttempts: 3, rateLimit: 500, enableCache: true },
    },
    {
      id: '3',
      name: 'Fuel Emissions Agent',
      description: 'Fuel consumption and emissions calculations',
      version: '3.0.1',
      status: 'active' as const,
      type: 'fuel' as const,
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(),
      createdAt: '2023-06-01',
      updatedAt: '2024-07-18',
      metrics: { requestsToday: 2145, requestsThisMonth: 45600, avgResponseTime: 120, errorRate: 0.3, uptime: 99.95 },
      config: { endpoint: '/api/fuel', maxConcurrency: 200, timeout: 15000, retryAttempts: 3, rateLimit: 2000, enableCache: true },
    },
    {
      id: '4',
      name: 'Building Energy Agent',
      description: 'Building energy consumption analysis and benchmarking',
      version: '1.8.0',
      status: 'maintenance' as const,
      type: 'building' as const,
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 10).toISOString(),
      createdAt: '2023-09-01',
      updatedAt: '2024-07-05',
      metrics: { requestsToday: 0, requestsThisMonth: 12300, avgResponseTime: 0, errorRate: 0, uptime: 95.0 },
      config: { endpoint: '/api/building', maxConcurrency: 75, timeout: 45000, retryAttempts: 3, rateLimit: 750, enableCache: true },
    },
  ];

  const mockAlerts = [
    {
      id: '1',
      severity: 'error' as const,
      title: 'High Error Rate Detected',
      message: 'EUDR Agent experiencing elevated error rates (5.2%). Investigation recommended.',
      agentId: '2',
      createdAt: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
    },
    {
      id: '2',
      severity: 'warning' as const,
      title: 'Rate Limit Approaching',
      message: 'Fuel Emissions Agent at 85% of rate limit capacity.',
      agentId: '3',
      createdAt: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
    },
    {
      id: '3',
      severity: 'info' as const,
      title: 'Scheduled Maintenance',
      message: 'Building Energy Agent maintenance window: July 20, 2024 02:00-04:00 UTC',
      agentId: '4',
      createdAt: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
      acknowledgedAt: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
    },
  ];

  const displayMetrics = metrics || mockMetrics;
  const displayAgents = agentsResponse?.items || mockAgents;
  const displayAlerts = alertsResponse?.items || mockAlerts;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-muted-foreground">
            System overview and key metrics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="gap-1">
            <Clock className="h-3 w-3" />
            Last updated: {formatRelativeTime(new Date().toISOString())}
          </Badge>
        </div>
      </div>

      {/* Key metrics */}
      <MetricGrid columns={4}>
        <MetricCard
          title="Total Emissions"
          value={formatEmissions(displayMetrics.totalEmissions)}
          trend={{ value: displayMetrics.emissionsTrend, label: 'vs last month', isPositiveGood: false }}
          icon={<Activity className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Active Agents"
          value={displayMetrics.activeAgents}
          subtitle={`${displayMetrics.activeAgents} of 6 running`}
          icon={<Bot className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Data Quality Score"
          value={`${displayMetrics.dataQualityScore}%`}
          trend={{ value: 2.1, label: 'vs last week' }}
          icon={<TrendingUp className="h-5 w-5" />}
          loading={metricsLoading}
        />
        <MetricCard
          title="Compliance Rate"
          value={`${displayMetrics.complianceRate}%`}
          subtitle="Across all regulations"
          icon={<AlertCircle className="h-5 w-5" />}
          loading={metricsLoading}
        />
      </MetricGrid>

      {/* Charts row */}
      <div className="grid gap-6 lg:grid-cols-2">
        <EmissionsTrendChart
          data={displayMetrics.emissionsTrendData}
          title="Emissions Trend"
          description="Monthly emissions over the past year"
          loading={metricsLoading}
          showTarget
        />
        <EmissionsByCategoryChart
          data={displayMetrics.emissionsByCategory}
          title="Emissions by Category"
          description="Distribution across product categories"
          loading={metricsLoading}
        />
      </div>

      {/* Agents and Alerts row */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Agents */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Active Agents</h2>
            <Button variant="ghost" size="sm" asChild>
              <Link to="/admin/agents" className="flex items-center gap-1">
                View All <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {agentsLoading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <AgentStatusCardSkeleton key={i} />
              ))
            ) : (
              displayAgents.map((agent) => (
                <AgentStatusCard
                  key={agent.id}
                  agent={agent}
                  onRestart={(id) => console.log('Restart agent', id)}
                  onToggle={(id) => console.log('Toggle agent', id)}
                />
              ))
            )}
          </div>
        </div>

        {/* Alerts */}
        <div>
          <AlertsList
            alerts={displayAlerts}
            loading={alertsLoading}
            onAcknowledge={(id) => acknowledgeAlert.mutate(id)}
            maxItems={5}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Recent Activity</CardTitle>
          <Button variant="ghost" size="sm">
            View All
          </Button>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {displayMetrics.recentActivity.map((activity) => (
              <div
                key={activity.id}
                className="flex items-start gap-4 pb-4 border-b last:border-0 last:pb-0"
              >
                <div
                  className={`rounded-full p-2 ${
                    activity.type === 'calculation'
                      ? 'bg-greenlang-100 text-greenlang-600'
                      : activity.type === 'report'
                        ? 'bg-blue-100 text-blue-600'
                        : 'bg-amber-100 text-amber-600'
                  }`}
                >
                  {activity.type === 'calculation' && <Activity className="h-4 w-4" />}
                  {activity.type === 'report' && <Building2 className="h-4 w-4" />}
                  {activity.type === 'alert' && <AlertCircle className="h-4 w-4" />}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium">{activity.title}</p>
                  <p className="text-sm text-muted-foreground truncate">
                    {activity.description}
                  </p>
                </div>
                <p className="text-sm text-muted-foreground whitespace-nowrap">
                  {formatRelativeTime(activity.timestamp)}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Quick Stats Row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-100 p-2 text-blue-600">
              <Users className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Users</p>
              <p className="text-xl font-bold">{formatNumber(1247)}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-purple-100 p-2 text-purple-600">
              <Building2 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Active Tenants</p>
              <p className="text-xl font-bold">{formatNumber(48)}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-greenlang-100 p-2 text-greenlang-600">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">API Calls Today</p>
              <p className="text-xl font-bold">{formatNumber(156789)}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-amber-100 p-2 text-amber-600">
              <AlertCircle className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Pending Alerts</p>
              <p className="text-xl font-bold">{displayAlerts.filter(a => !a.acknowledgedAt).length}</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
