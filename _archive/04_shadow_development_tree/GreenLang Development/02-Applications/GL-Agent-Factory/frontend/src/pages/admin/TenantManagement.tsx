/**
 * TenantManagement Page
 *
 * Manage multi-tenant configuration.
 */

import * as React from 'react';
import {
  Search,
  Plus,
  Building2,
  Users,
  Activity,
  HardDrive,
  Edit,
  Trash2,
  Settings,
  ExternalLink,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/Dialog';
import { Pagination, PaginationInfo } from '@/components/ui/Pagination';
import { useTenants, useCreateTenant, useUpdateTenant } from '@/api/hooks';
import { formatNumber, formatDateTime, formatFileSize } from '@/utils/format';
import type { Tenant, TenantPlan } from '@/api/types';

const planOptions = [
  { value: 'all', label: 'All Plans' },
  { value: 'starter', label: 'Starter' },
  { value: 'professional', label: 'Professional' },
  { value: 'enterprise', label: 'Enterprise' },
];

const planConfig: Record<TenantPlan, { variant: 'secondary' | 'default' | 'destructive'; label: string; color: string }> = {
  starter: { variant: 'secondary', label: 'Starter', color: 'bg-gray-100 text-gray-700' },
  professional: { variant: 'default', label: 'Professional', color: 'bg-blue-100 text-blue-700' },
  enterprise: { variant: 'destructive', label: 'Enterprise', color: 'bg-purple-100 text-purple-700' },
};

export default function TenantManagement() {
  // State
  const [search, setSearch] = React.useState('');
  const [planFilter, setPlanFilter] = React.useState('all');
  const [page, setPage] = React.useState(1);
  const [showCreateDialog, setShowCreateDialog] = React.useState(false);
  const perPage = 10;

  // Form state
  const [formData, setFormData] = React.useState({
    name: '',
    slug: '',
    plan: 'starter' as TenantPlan,
  });

  // Fetch data
  const { data: response, isLoading } = useTenants({
    plan: planFilter !== 'all' ? planFilter : undefined,
    search: search || undefined,
    page,
    perPage,
  });

  const createTenant = useCreateTenant();
  const updateTenant = useUpdateTenant();

  // Mock data for development
  const mockTenants: Tenant[] = [
    {
      id: 't1',
      name: 'TechCorp Industries',
      slug: 'techcorp',
      plan: 'enterprise',
      isActive: true,
      settings: {
        maxUsers: 500,
        maxApiCalls: 1000000,
        enabledAgents: ['cbam', 'eudr', 'fuel', 'building', 'sb253'],
        customDomain: 'compliance.techcorp.com',
        ssoEnabled: true,
        dataRetentionDays: 365,
      },
      usage: {
        currentUsers: 156,
        apiCallsThisMonth: 456789,
        storageUsedMb: 2450,
        reportsGenerated: 234,
      },
      createdAt: '2023-06-01T00:00:00Z',
      updatedAt: '2024-07-15T14:30:00Z',
    },
    {
      id: 't2',
      name: 'Green Manufacturing Co',
      slug: 'greenmanufacturing',
      plan: 'professional',
      isActive: true,
      settings: {
        maxUsers: 50,
        maxApiCalls: 100000,
        enabledAgents: ['cbam', 'fuel', 'building'],
        ssoEnabled: false,
        dataRetentionDays: 180,
      },
      usage: {
        currentUsers: 28,
        apiCallsThisMonth: 45678,
        storageUsedMb: 890,
        reportsGenerated: 56,
      },
      createdAt: '2024-01-15T00:00:00Z',
      updatedAt: '2024-07-10T09:15:00Z',
    },
    {
      id: 't3',
      name: 'EcoStartup',
      slug: 'ecostartup',
      plan: 'starter',
      isActive: true,
      settings: {
        maxUsers: 5,
        maxApiCalls: 10000,
        enabledAgents: ['fuel'],
        ssoEnabled: false,
        dataRetentionDays: 90,
      },
      usage: {
        currentUsers: 3,
        apiCallsThisMonth: 1234,
        storageUsedMb: 45,
        reportsGenerated: 8,
      },
      createdAt: '2024-05-01T00:00:00Z',
      updatedAt: '2024-07-05T11:20:00Z',
    },
    {
      id: 't4',
      name: 'Global Logistics Ltd',
      slug: 'globallogistics',
      plan: 'enterprise',
      isActive: true,
      settings: {
        maxUsers: 1000,
        maxApiCalls: 5000000,
        enabledAgents: ['cbam', 'eudr', 'fuel', 'building', 'sb253'],
        customDomain: 'sustainability.globallogistics.com',
        ssoEnabled: true,
        dataRetentionDays: 730,
      },
      usage: {
        currentUsers: 423,
        apiCallsThisMonth: 2345678,
        storageUsedMb: 8900,
        reportsGenerated: 567,
      },
      createdAt: '2023-03-15T00:00:00Z',
      updatedAt: '2024-07-18T16:45:00Z',
    },
    {
      id: 't5',
      name: 'Suspended Corp',
      slug: 'suspendedcorp',
      plan: 'professional',
      isActive: false,
      settings: {
        maxUsers: 50,
        maxApiCalls: 100000,
        enabledAgents: ['cbam', 'fuel'],
        ssoEnabled: false,
        dataRetentionDays: 180,
      },
      usage: {
        currentUsers: 0,
        apiCallsThisMonth: 0,
        storageUsedMb: 234,
        reportsGenerated: 0,
      },
      createdAt: '2023-09-01T00:00:00Z',
      updatedAt: '2024-06-01T08:00:00Z',
    },
  ];

  const tenants = response?.items || mockTenants;
  const totalItems = response?.pagination?.totalItems || mockTenants.length;
  const totalPages = response?.pagination?.totalPages || 1;

  // Filter tenants by search
  const filteredTenants = React.useMemo(() => {
    if (!search) return tenants;
    const searchLower = search.toLowerCase();
    return tenants.filter(
      (tenant) =>
        tenant.name.toLowerCase().includes(searchLower) ||
        tenant.slug.toLowerCase().includes(searchLower)
    );
  }, [tenants, search]);

  const handleCreateTenant = () => {
    createTenant.mutate(formData, {
      onSuccess: () => {
        setShowCreateDialog(false);
        setFormData({ name: '', slug: '', plan: 'starter' });
      },
    });
  };

  // Calculate summary stats
  const stats = React.useMemo(() => {
    return {
      totalTenants: tenants.length,
      activeTenants: tenants.filter((t) => t.isActive).length,
      totalUsers: tenants.reduce((acc, t) => acc + t.usage.currentUsers, 0),
      totalApiCalls: tenants.reduce((acc, t) => acc + t.usage.apiCallsThisMonth, 0),
    };
  }, [tenants]);

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Tenant Management</h1>
          <p className="text-muted-foreground">
            Manage organizations and their configurations
          </p>
        </div>
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Tenant
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Tenant</DialogTitle>
              <DialogDescription>
                Add a new organization to the platform.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-4">
              <Input
                label="Organization Name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                required
              />
              <Input
                label="Slug"
                value={formData.slug}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    slug: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ''),
                  })
                }
                helperText="URL-friendly identifier (lowercase, no spaces)"
                required
              />
              <div>
                <label className="text-sm font-medium">Plan</label>
                <Select
                  value={formData.plan}
                  onValueChange={(value) => setFormData({ ...formData, plan: value as TenantPlan })}
                >
                  <SelectTrigger className="mt-1.5">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="starter">Starter</SelectItem>
                    <SelectItem value="professional">Professional</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateTenant} loading={createTenant.isPending}>
                Create Tenant
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Summary stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-blue-100 p-2 text-blue-600">
              <Building2 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Tenants</p>
              <p className="text-xl font-bold">{stats.totalTenants}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-greenlang-100 p-2 text-greenlang-600">
              <Building2 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Active Tenants</p>
              <p className="text-xl font-bold">{stats.activeTenants}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-purple-100 p-2 text-purple-600">
              <Users className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Users</p>
              <p className="text-xl font-bold">{formatNumber(stats.totalUsers)}</p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center gap-3">
            <div className="rounded-lg bg-amber-100 p-2 text-amber-600">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">API Calls (Month)</p>
              <p className="text-xl font-bold">{formatNumber(stats.totalApiCalls)}</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search tenants..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <Select value={planFilter} onValueChange={setPlanFilter}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Plan" />
              </SelectTrigger>
              <SelectContent>
                {planOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Tenants table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Tenant</TableHead>
              <TableHead>Plan</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Users</TableHead>
              <TableHead className="text-right">API Calls</TableHead>
              <TableHead className="text-right">Storage</TableHead>
              <TableHead>Agents</TableHead>
              <TableHead className="w-[80px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableSkeleton rows={5} columns={8} />
            ) : filteredTenants.length === 0 ? (
              <TableEmpty
                icon={<Building2 className="h-12 w-12" />}
                title="No tenants found"
                description={search ? 'Try adjusting your search' : 'Add a tenant to get started'}
                action={
                  <Button size="sm" onClick={() => setShowCreateDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Tenant
                  </Button>
                }
              />
            ) : (
              filteredTenants.map((tenant) => (
                <TableRow key={tenant.id}>
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary font-semibold">
                        {tenant.name.slice(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <p className="font-medium">{tenant.name}</p>
                        <p className="text-xs text-muted-foreground">{tenant.slug}</p>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge className={planConfig[tenant.plan].color}>
                      {planConfig[tenant.plan].label}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={tenant.isActive ? 'active' : 'inactive'} dot>
                      {tenant.isActive ? 'Active' : 'Suspended'}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <span className="font-medium">{tenant.usage.currentUsers}</span>
                    <span className="text-muted-foreground">/{tenant.settings.maxUsers}</span>
                  </TableCell>
                  <TableCell className="text-right">
                    {formatNumber(tenant.usage.apiCallsThisMonth)}
                  </TableCell>
                  <TableCell className="text-right">
                    {formatFileSize(tenant.usage.storageUsedMb * 1024 * 1024)}
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {tenant.settings.enabledAgents.slice(0, 3).map((agent) => (
                        <Badge key={agent} variant="secondary" size="sm">
                          {agent.toUpperCase()}
                        </Badge>
                      ))}
                      {tenant.settings.enabledAgents.length > 3 && (
                        <Badge variant="secondary" size="sm">
                          +{tenant.settings.enabledAgents.length - 3}
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center justify-end gap-1">
                      <Button variant="ghost" size="icon-sm" title="Settings">
                        <Settings className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon-sm" title="Edit">
                        <Edit className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      {/* Pagination */}
      {!isLoading && filteredTenants.length > 0 && (
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

      {/* Plan comparison */}
      <Card>
        <CardHeader>
          <CardTitle>Plan Comparison</CardTitle>
          <CardDescription>Features and limits by plan tier</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Feature</TableHead>
                <TableHead className="text-center">Starter</TableHead>
                <TableHead className="text-center">Professional</TableHead>
                <TableHead className="text-center">Enterprise</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell>Max Users</TableCell>
                <TableCell className="text-center">5</TableCell>
                <TableCell className="text-center">50</TableCell>
                <TableCell className="text-center">Unlimited</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>API Calls / Month</TableCell>
                <TableCell className="text-center">10,000</TableCell>
                <TableCell className="text-center">100,000</TableCell>
                <TableCell className="text-center">Unlimited</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Available Agents</TableCell>
                <TableCell className="text-center">1</TableCell>
                <TableCell className="text-center">3</TableCell>
                <TableCell className="text-center">All</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Data Retention</TableCell>
                <TableCell className="text-center">90 days</TableCell>
                <TableCell className="text-center">180 days</TableCell>
                <TableCell className="text-center">Custom</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>SSO Support</TableCell>
                <TableCell className="text-center">-</TableCell>
                <TableCell className="text-center">-</TableCell>
                <TableCell className="text-center">Yes</TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Custom Domain</TableCell>
                <TableCell className="text-center">-</TableCell>
                <TableCell className="text-center">-</TableCell>
                <TableCell className="text-center">Yes</TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
