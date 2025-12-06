/**
 * AgentList Page
 *
 * List and manage all AI agents in the system.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  Bot,
  Search,
  Filter,
  MoreVertical,
  Play,
  Pause,
  RefreshCw,
  Settings,
  ExternalLink,
  Plus,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  TableSkeleton,
  TableEmpty,
} from '@/components/ui/Table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select';
import { Pagination, PaginationInfo } from '@/components/ui/Pagination';
import { AgentStatusCard, AgentStatusCardSkeleton } from '@/components/widgets/AgentStatusCard';
import { useAgents, useRestartAgent } from '@/api/hooks';
import { formatNumber, formatRelativeTime } from '@/utils/format';
import { cn } from '@/utils/cn';
import type { Agent, AgentStatus } from '@/api/types';

const statusOptions = [
  { value: 'all', label: 'All Status' },
  { value: 'active', label: 'Active' },
  { value: 'inactive', label: 'Inactive' },
  { value: 'error', label: 'Error' },
  { value: 'maintenance', label: 'Maintenance' },
];

const typeOptions = [
  { value: 'all', label: 'All Types' },
  { value: 'cbam', label: 'CBAM' },
  { value: 'eudr', label: 'EUDR' },
  { value: 'fuel', label: 'Fuel' },
  { value: 'building', label: 'Building' },
  { value: 'sb253', label: 'SB253' },
];

const statusConfig: Record<AgentStatus, { variant: 'active' | 'inactive' | 'error' | 'pending' | 'processing'; label: string }> = {
  active: { variant: 'active', label: 'Active' },
  inactive: { variant: 'inactive', label: 'Inactive' },
  error: { variant: 'error', label: 'Error' },
  maintenance: { variant: 'pending', label: 'Maintenance' },
  deploying: { variant: 'processing', label: 'Deploying' },
};

export default function AgentList() {
  // State
  const [view, setView] = React.useState<'grid' | 'table'>('grid');
  const [search, setSearch] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState('all');
  const [typeFilter, setTypeFilter] = React.useState('all');
  const [page, setPage] = React.useState(1);
  const perPage = 12;

  // Fetch data
  const { data: response, isLoading } = useAgents({
    status: statusFilter !== 'all' ? statusFilter : undefined,
    type: typeFilter !== 'all' ? typeFilter : undefined,
    page,
    perPage,
  });

  const restartAgent = useRestartAgent();

  // Mock data for development
  const mockAgents: Agent[] = [
    {
      id: '1',
      name: 'CBAM Agent',
      description: 'Carbon Border Adjustment Mechanism calculations and CBAM quarterly report generation',
      version: '2.1.0',
      status: 'active',
      type: 'cbam',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
      createdAt: '2024-01-01',
      updatedAt: '2024-07-15',
      metrics: { requestsToday: 1234, requestsThisMonth: 28500, avgResponseTime: 245, errorRate: 0.5, uptime: 99.9 },
      config: { endpoint: '/api/cbam', maxConcurrency: 100, timeout: 30000, retryAttempts: 3, rateLimit: 1000, enableCache: true },
    },
    {
      id: '2',
      name: 'EUDR Agent',
      description: 'EU Deforestation Regulation compliance verification and satellite imagery analysis',
      version: '1.5.2',
      status: 'active',
      type: 'eudr',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
      createdAt: '2024-02-15',
      updatedAt: '2024-07-10',
      metrics: { requestsToday: 856, requestsThisMonth: 18200, avgResponseTime: 380, errorRate: 1.2, uptime: 99.5 },
      config: { endpoint: '/api/eudr', maxConcurrency: 50, timeout: 60000, retryAttempts: 3, rateLimit: 500, enableCache: true },
    },
    {
      id: '3',
      name: 'Fuel Emissions Agent',
      description: 'Fuel consumption tracking and Scope 1/2/3 emissions calculations with multiple emission factor databases',
      version: '3.0.1',
      status: 'active',
      type: 'fuel',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(),
      createdAt: '2023-06-01',
      updatedAt: '2024-07-18',
      metrics: { requestsToday: 2145, requestsThisMonth: 45600, avgResponseTime: 120, errorRate: 0.3, uptime: 99.95 },
      config: { endpoint: '/api/fuel', maxConcurrency: 200, timeout: 15000, retryAttempts: 3, rateLimit: 2000, enableCache: true },
    },
    {
      id: '4',
      name: 'Building Energy Agent',
      description: 'Building energy consumption analysis, benchmarking against industry standards, and efficiency recommendations',
      version: '1.8.0',
      status: 'maintenance',
      type: 'building',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 10).toISOString(),
      createdAt: '2023-09-01',
      updatedAt: '2024-07-05',
      metrics: { requestsToday: 0, requestsThisMonth: 12300, avgResponseTime: 0, errorRate: 0, uptime: 95.0 },
      config: { endpoint: '/api/building', maxConcurrency: 75, timeout: 45000, retryAttempts: 3, rateLimit: 750, enableCache: true },
    },
    {
      id: '5',
      name: 'SB253 Agent',
      description: 'California SB253 Climate Corporate Data Accountability Act compliance and reporting',
      version: '1.2.0',
      status: 'active',
      type: 'sb253',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(),
      createdAt: '2024-03-01',
      updatedAt: '2024-07-12',
      metrics: { requestsToday: 567, requestsThisMonth: 9800, avgResponseTime: 290, errorRate: 0.8, uptime: 99.7 },
      config: { endpoint: '/api/sb253', maxConcurrency: 60, timeout: 30000, retryAttempts: 3, rateLimit: 600, enableCache: true },
    },
    {
      id: '6',
      name: 'Legacy CBAM Agent',
      description: 'Deprecated CBAM agent - use CBAM Agent v2.1.0 instead',
      version: '1.0.0',
      status: 'inactive',
      type: 'cbam',
      lastDeployedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 90).toISOString(),
      createdAt: '2023-01-01',
      updatedAt: '2024-04-01',
      metrics: { requestsToday: 0, requestsThisMonth: 0, avgResponseTime: 0, errorRate: 0, uptime: 0 },
      config: { endpoint: '/api/cbam-legacy', maxConcurrency: 50, timeout: 30000, retryAttempts: 3, rateLimit: 500, enableCache: false },
    },
  ];

  const agents = response?.items || mockAgents;
  const totalItems = response?.pagination?.totalItems || mockAgents.length;
  const totalPages = response?.pagination?.totalPages || Math.ceil(mockAgents.length / perPage);

  // Filter agents by search
  const filteredAgents = React.useMemo(() => {
    if (!search) return agents;
    const searchLower = search.toLowerCase();
    return agents.filter(
      (agent) =>
        agent.name.toLowerCase().includes(searchLower) ||
        agent.description.toLowerCase().includes(searchLower) ||
        agent.type.toLowerCase().includes(searchLower)
    );
  }, [agents, search]);

  const handleRestart = (agentId: string) => {
    restartAgent.mutate(agentId);
  };

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Agents</h1>
          <p className="text-muted-foreground">
            Manage AI agents and their configurations
          </p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Deploy New Agent
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search agents..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <div className="flex gap-2">
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Status" />
                </SelectTrigger>
                <SelectContent>
                  {statusOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-[140px]">
                  <SelectValue placeholder="Type" />
                </SelectTrigger>
                <SelectContent>
                  {typeOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* View toggle */}
              <div className="flex border rounded-md">
                <Button
                  variant={view === 'grid' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setView('grid')}
                  className="rounded-r-none"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="7" height="7" />
                    <rect x="14" y="3" width="7" height="7" />
                    <rect x="14" y="14" width="7" height="7" />
                    <rect x="3" y="14" width="7" height="7" />
                  </svg>
                </Button>
                <Button
                  variant={view === 'table' ? 'secondary' : 'ghost'}
                  size="icon"
                  onClick={() => setView('table')}
                  className="rounded-l-none"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="8" y1="6" x2="21" y2="6" />
                    <line x1="8" y1="12" x2="21" y2="12" />
                    <line x1="8" y1="18" x2="21" y2="18" />
                    <line x1="3" y1="6" x2="3.01" y2="6" />
                    <line x1="3" y1="12" x2="3.01" y2="12" />
                    <line x1="3" y1="18" x2="3.01" y2="18" />
                  </svg>
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Grid View */}
      {view === 'grid' && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {isLoading ? (
            Array.from({ length: 6 }).map((_, i) => (
              <AgentStatusCardSkeleton key={i} />
            ))
          ) : filteredAgents.length === 0 ? (
            <Card className="col-span-full p-12">
              <div className="flex flex-col items-center justify-center text-center">
                <Bot className="h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="font-semibold">No agents found</h3>
                <p className="text-sm text-muted-foreground">
                  {search ? 'Try adjusting your search or filters' : 'Deploy a new agent to get started'}
                </p>
              </div>
            </Card>
          ) : (
            filteredAgents.map((agent) => (
              <AgentStatusCard
                key={agent.id}
                agent={agent}
                onRestart={handleRestart}
                onToggle={(id) => console.log('Toggle agent', id)}
              />
            ))
          )}
        </div>
      )}

      {/* Table View */}
      {view === 'table' && (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Agent</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Requests/Day</TableHead>
                <TableHead className="text-right">Avg Response</TableHead>
                <TableHead className="text-right">Error Rate</TableHead>
                <TableHead className="text-right">Uptime</TableHead>
                <TableHead className="w-[80px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <TableSkeleton rows={6} columns={8} />
              ) : filteredAgents.length === 0 ? (
                <TableEmpty
                  icon={<Bot className="h-12 w-12" />}
                  title="No agents found"
                  description={search ? 'Try adjusting your search or filters' : 'Deploy a new agent to get started'}
                  action={
                    <Button size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Deploy Agent
                    </Button>
                  }
                />
              ) : (
                filteredAgents.map((agent) => (
                  <TableRow key={agent.id}>
                    <TableCell>
                      <Link
                        to={`/admin/agents/${agent.id}`}
                        className="flex items-center gap-3 hover:underline"
                      >
                        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary font-semibold text-xs">
                          {agent.type.toUpperCase().slice(0, 4)}
                        </div>
                        <div>
                          <p className="font-medium">{agent.name}</p>
                          <p className="text-xs text-muted-foreground">v{agent.version}</p>
                        </div>
                      </Link>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">{agent.type.toUpperCase()}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={statusConfig[agent.status].variant} dot>
                        {statusConfig[agent.status].label}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-medium">
                      {formatNumber(agent.metrics.requestsToday)}
                    </TableCell>
                    <TableCell className="text-right">
                      {agent.metrics.avgResponseTime}ms
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={cn(agent.metrics.errorRate > 5 && 'text-destructive')}>
                        {agent.metrics.errorRate.toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={cn(agent.metrics.uptime < 99 && 'text-amber-600')}>
                        {agent.metrics.uptime.toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => handleRestart(agent.id)}
                          disabled={restartAgent.isPending}
                        >
                          <RefreshCw className={cn('h-4 w-4', restartAgent.isPending && 'animate-spin')} />
                        </Button>
                        <Button variant="ghost" size="icon-sm" asChild>
                          <Link to={`/admin/agents/${agent.id}`}>
                            <Settings className="h-4 w-4" />
                          </Link>
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Pagination */}
      {!isLoading && filteredAgents.length > 0 && (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <PaginationInfo
            currentPage={page}
            pageSize={perPage}
            totalItems={totalItems}
          />
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            onPageChange={setPage}
          />
        </div>
      )}
    </div>
  );
}
