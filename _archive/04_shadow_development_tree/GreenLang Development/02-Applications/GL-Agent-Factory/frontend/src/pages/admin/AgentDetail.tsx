/**
 * AgentDetail Page
 *
 * View and configure a single agent with metrics, logs, and settings.
 */

import * as React from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  RefreshCw,
  Play,
  Pause,
  Settings,
  Activity,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Download,
  Copy,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/Table';
import { MetricCard, MetricGrid } from '@/components/widgets/MetricCard';
import { EmissionsTrendChart } from '@/components/charts/EmissionsChart';
import { useAgent, useAgentLogs, useRestartAgent, useUpdateAgentConfig } from '@/api/hooks';
import { formatNumber, formatDateTime, formatRelativeTime } from '@/utils/format';
import { cn } from '@/utils/cn';
import type { Agent, AgentLog } from '@/api/types';

export default function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const [activeTab, setActiveTab] = React.useState('overview');

  // Fetch data
  const { data: agent, isLoading } = useAgent(agentId || '');
  const { data: logsResponse } = useAgentLogs(agentId || '', { perPage: 50 });

  const restartAgent = useRestartAgent();
  const updateConfig = useUpdateAgentConfig();

  // Mock data for development
  const mockAgent: Agent = {
    id: agentId || '1',
    name: 'CBAM Agent',
    description: 'Carbon Border Adjustment Mechanism calculations and CBAM quarterly report generation. Supports all CBAM product categories including iron, steel, aluminum, cement, fertilizers, and electricity.',
    version: '2.1.0',
    status: 'active',
    type: 'cbam',
    lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-07-15T14:30:00Z',
    metrics: {
      requestsToday: 1234,
      requestsThisMonth: 28500,
      avgResponseTime: 245,
      errorRate: 0.5,
      uptime: 99.9,
    },
    config: {
      endpoint: 'https://api.greenlang.io/v1/cbam',
      maxConcurrency: 100,
      timeout: 30000,
      retryAttempts: 3,
      rateLimit: 1000,
      enableCache: true,
    },
  };

  const mockLogs: AgentLog[] = [
    { id: '1', agentId: agentId || '1', level: 'info', message: 'Agent started successfully', timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString() },
    { id: '2', agentId: agentId || '1', level: 'info', message: 'Processing CBAM calculation request', metadata: { requestId: 'req-123' }, timestamp: new Date(Date.now() - 1000 * 60 * 4).toISOString() },
    { id: '3', agentId: agentId || '1', level: 'warn', message: 'Rate limit threshold reached (85%)', timestamp: new Date(Date.now() - 1000 * 60 * 3).toISOString() },
    { id: '4', agentId: agentId || '1', level: 'info', message: 'CBAM calculation completed', metadata: { requestId: 'req-123', duration: 234 }, timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString() },
    { id: '5', agentId: agentId || '1', level: 'error', message: 'Failed to connect to EU ETS price feed', metadata: { error: 'Connection timeout' }, timestamp: new Date(Date.now() - 1000 * 60 * 1).toISOString() },
    { id: '6', agentId: agentId || '1', level: 'info', message: 'Retry successful for EU ETS price feed', timestamp: new Date(Date.now() - 1000 * 30).toISOString() },
  ];

  const mockRequestData = Array.from({ length: 24 }, (_, i) => ({
    date: `${i}:00`,
    emissions: Math.floor(40 + Math.random() * 60),
  }));

  const displayAgent = agent || mockAgent;
  const displayLogs = logsResponse?.items || mockLogs;

  const handleRestart = () => {
    if (agentId) {
      restartAgent.mutate(agentId);
    }
  };

  const statusConfig = {
    active: { variant: 'active' as const, label: 'Active', icon: CheckCircle },
    inactive: { variant: 'inactive' as const, label: 'Inactive', icon: XCircle },
    error: { variant: 'error' as const, label: 'Error', icon: AlertTriangle },
    maintenance: { variant: 'pending' as const, label: 'Maintenance', icon: Clock },
    deploying: { variant: 'processing' as const, label: 'Deploying', icon: RefreshCw },
  };

  const config = statusConfig[displayAgent.status];
  const StatusIcon = config.icon;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-4">
          <Link to="/admin/agents">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold">{displayAgent.name}</h1>
              <Badge variant={config.variant} className="gap-1">
                <StatusIcon className="h-3 w-3" />
                {config.label}
              </Badge>
            </div>
            <p className="text-muted-foreground mt-1">{displayAgent.description}</p>
            <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
              <span>Version {displayAgent.version}</span>
              <span>|</span>
              <span>Last deployed {formatRelativeTime(displayAgent.lastDeployedAt)}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {displayAgent.status === 'active' ? (
            <Button variant="outline">
              <Pause className="h-4 w-4 mr-2" />
              Pause
            </Button>
          ) : (
            <Button variant="outline">
              <Play className="h-4 w-4 mr-2" />
              Start
            </Button>
          )}
          <Button
            variant="outline"
            onClick={handleRestart}
            loading={restartAgent.isPending}
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Restart
          </Button>
          <Button>
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
        </div>
      </div>

      {/* Metrics */}
      <MetricGrid columns={4}>
        <MetricCard
          title="Requests Today"
          value={formatNumber(displayAgent.metrics.requestsToday)}
          trend={{ value: 12.5, label: 'vs yesterday' }}
          icon={<Activity className="h-5 w-5" />}
        />
        <MetricCard
          title="Avg Response Time"
          value={`${displayAgent.metrics.avgResponseTime}ms`}
          trend={{ value: -8, label: 'vs avg', isPositiveGood: false }}
          icon={<Clock className="h-5 w-5" />}
        />
        <MetricCard
          title="Error Rate"
          value={`${displayAgent.metrics.errorRate}%`}
          trend={{ value: -0.3, label: 'vs yesterday', isPositiveGood: false }}
          icon={<AlertTriangle className="h-5 w-5" />}
        />
        <MetricCard
          title="Uptime"
          value={`${displayAgent.metrics.uptime}%`}
          subtitle="Last 30 days"
          icon={<CheckCircle className="h-5 w-5" />}
        />
      </MetricGrid>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
          <TabsTrigger value="configuration">Configuration</TabsTrigger>
          <TabsTrigger value="deployments">Deployments</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <EmissionsTrendChart
              data={mockRequestData}
              title="Requests (Last 24 Hours)"
              description="Hourly request volume"
              height={250}
            />

            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Response time distribution</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>P50 Response Time</span>
                      <span className="font-medium">180ms</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-greenlang-500 rounded-full" style={{ width: '36%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>P90 Response Time</span>
                      <span className="font-medium">320ms</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 rounded-full" style={{ width: '64%' }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>P99 Response Time</span>
                      <span className="font-medium">450ms</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-red-500 rounded-full" style={{ width: '90%' }} />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Logs Preview */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Recent Logs</CardTitle>
                <CardDescription>Latest agent activity</CardDescription>
              </div>
              <Button variant="ghost" size="sm" onClick={() => setActiveTab('logs')}>
                View All
              </Button>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {displayLogs.slice(0, 5).map((log) => (
                  <div
                    key={log.id}
                    className={cn(
                      'flex items-start gap-3 p-2 rounded-lg text-sm',
                      log.level === 'error' && 'bg-red-50',
                      log.level === 'warn' && 'bg-amber-50',
                      log.level === 'info' && 'bg-muted/50'
                    )}
                  >
                    <span
                      className={cn(
                        'px-1.5 py-0.5 rounded text-xs font-medium uppercase',
                        log.level === 'error' && 'bg-red-100 text-red-700',
                        log.level === 'warn' && 'bg-amber-100 text-amber-700',
                        log.level === 'info' && 'bg-blue-100 text-blue-700',
                        log.level === 'debug' && 'bg-gray-100 text-gray-700'
                      )}
                    >
                      {log.level}
                    </span>
                    <span className="flex-1">{log.message}</span>
                    <span className="text-muted-foreground">
                      {formatRelativeTime(log.timestamp)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Logs Tab */}
        <TabsContent value="logs" className="space-y-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Agent Logs</CardTitle>
                <CardDescription>Real-time log stream</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="bg-carbon-900 rounded-lg p-4 font-mono text-sm max-h-[600px] overflow-y-auto">
                {displayLogs.map((log) => (
                  <div
                    key={log.id}
                    className={cn(
                      'py-1 border-b border-carbon-800 last:border-0',
                      log.level === 'error' && 'text-red-400',
                      log.level === 'warn' && 'text-amber-400',
                      log.level === 'info' && 'text-green-400',
                      log.level === 'debug' && 'text-gray-400'
                    )}
                  >
                    <span className="text-carbon-400">
                      {formatDateTime(log.timestamp)}
                    </span>
                    {' '}
                    <span className="uppercase">[{log.level}]</span>
                    {' '}
                    <span className="text-white">{log.message}</span>
                    {log.metadata && (
                      <span className="text-carbon-400">
                        {' '}
                        {JSON.stringify(log.metadata)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="configuration" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Agent Configuration</CardTitle>
              <CardDescription>Runtime settings and parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-2">
                <Input
                  label="API Endpoint"
                  value={displayAgent.config.endpoint}
                  readOnly
                  rightIcon={
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => navigator.clipboard.writeText(displayAgent.config.endpoint)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  }
                />
                <Input
                  label="Max Concurrency"
                  type="number"
                  defaultValue={displayAgent.config.maxConcurrency}
                  helperText="Maximum concurrent requests"
                />
                <Input
                  label="Timeout (ms)"
                  type="number"
                  defaultValue={displayAgent.config.timeout}
                  helperText="Request timeout in milliseconds"
                />
                <Input
                  label="Retry Attempts"
                  type="number"
                  defaultValue={displayAgent.config.retryAttempts}
                  helperText="Number of retry attempts on failure"
                />
                <Input
                  label="Rate Limit"
                  type="number"
                  defaultValue={displayAgent.config.rateLimit}
                  helperText="Maximum requests per minute"
                />
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <p className="font-medium">Enable Cache</p>
                    <p className="text-sm text-muted-foreground">Cache responses for repeated requests</p>
                  </div>
                  <input
                    type="checkbox"
                    defaultChecked={displayAgent.config.enableCache}
                    className="h-5 w-5"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Deployments Tab */}
        <TabsContent value="deployments" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Deployment History</CardTitle>
              <CardDescription>Past deployments and version changes</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Version</TableHead>
                    <TableHead>Deployed At</TableHead>
                    <TableHead>Deployed By</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Duration</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  <TableRow>
                    <TableCell className="font-medium">v2.1.0</TableCell>
                    <TableCell>{formatDateTime(displayAgent.lastDeployedAt)}</TableCell>
                    <TableCell>system</TableCell>
                    <TableCell><Badge variant="success">Success</Badge></TableCell>
                    <TableCell>45s</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">v2.0.5</TableCell>
                    <TableCell>{formatDateTime(new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString())}</TableCell>
                    <TableCell>admin@greenlang.io</TableCell>
                    <TableCell><Badge variant="success">Success</Badge></TableCell>
                    <TableCell>38s</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell className="font-medium">v2.0.4</TableCell>
                    <TableCell>{formatDateTime(new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString())}</TableCell>
                    <TableCell>admin@greenlang.io</TableCell>
                    <TableCell><Badge variant="error">Failed</Badge></TableCell>
                    <TableCell>12s</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
