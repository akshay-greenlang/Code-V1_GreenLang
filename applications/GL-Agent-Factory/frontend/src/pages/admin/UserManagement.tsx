/**
 * UserManagement Page
 *
 * Manage users, roles, and permissions.
 */

import * as React from 'react';
import {
  Search,
  Plus,
  MoreVertical,
  Edit,
  Trash2,
  Mail,
  Shield,
  UserCheck,
  UserX,
  Download,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Avatar } from '@/components/ui/Avatar';
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
import { useUsers, useCreateUser, useUpdateUser, useDeleteUser } from '@/api/hooks';
import { formatDateTime, formatRelativeTime } from '@/utils/format';
import type { User, UserRole } from '@/api/types';

const roleOptions = [
  { value: 'all', label: 'All Roles' },
  { value: 'admin', label: 'Admin' },
  { value: 'analyst', label: 'Analyst' },
  { value: 'viewer', label: 'Viewer' },
  { value: 'api_user', label: 'API User' },
];

const roleConfig: Record<UserRole, { variant: 'default' | 'secondary' | 'destructive'; label: string }> = {
  admin: { variant: 'destructive', label: 'Admin' },
  analyst: { variant: 'default', label: 'Analyst' },
  viewer: { variant: 'secondary', label: 'Viewer' },
  api_user: { variant: 'secondary', label: 'API User' },
};

export default function UserManagement() {
  // State
  const [search, setSearch] = React.useState('');
  const [roleFilter, setRoleFilter] = React.useState('all');
  const [page, setPage] = React.useState(1);
  const [showCreateDialog, setShowCreateDialog] = React.useState(false);
  const [selectedUser, setSelectedUser] = React.useState<User | null>(null);
  const perPage = 10;

  // Form state
  const [formData, setFormData] = React.useState({
    email: '',
    firstName: '',
    lastName: '',
    role: 'viewer' as UserRole,
    password: '',
  });

  // Fetch data
  const { data: response, isLoading } = useUsers({
    role: roleFilter !== 'all' ? roleFilter : undefined,
    search: search || undefined,
    page,
    perPage,
  });

  const createUser = useCreateUser();
  const updateUser = useUpdateUser();
  const deleteUser = useDeleteUser();

  // Mock data for development
  const mockUsers: User[] = [
    {
      id: '1',
      email: 'admin@greenlang.io',
      firstName: 'John',
      lastName: 'Admin',
      role: 'admin',
      tenantId: 't1',
      isActive: true,
      lastLoginAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      createdAt: '2024-01-01T00:00:00Z',
      updatedAt: '2024-07-15T14:30:00Z',
    },
    {
      id: '2',
      email: 'sarah.chen@techcorp.com',
      firstName: 'Sarah',
      lastName: 'Chen',
      role: 'analyst',
      tenantId: 't1',
      isActive: true,
      lastLoginAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
      createdAt: '2024-02-15T00:00:00Z',
      updatedAt: '2024-07-10T09:15:00Z',
    },
    {
      id: '3',
      email: 'mike.johnson@company.com',
      firstName: 'Mike',
      lastName: 'Johnson',
      role: 'viewer',
      tenantId: 't1',
      isActive: true,
      lastLoginAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
      createdAt: '2024-03-01T00:00:00Z',
      updatedAt: '2024-07-05T11:20:00Z',
    },
    {
      id: '4',
      email: 'api-integration@company.com',
      firstName: 'API',
      lastName: 'Integration',
      role: 'api_user',
      tenantId: 't1',
      isActive: true,
      lastLoginAt: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
      createdAt: '2024-04-10T00:00:00Z',
      updatedAt: '2024-07-18T16:45:00Z',
    },
    {
      id: '5',
      email: 'inactive.user@example.com',
      firstName: 'Inactive',
      lastName: 'User',
      role: 'viewer',
      tenantId: 't1',
      isActive: false,
      lastLoginAt: null,
      createdAt: '2024-05-20T00:00:00Z',
      updatedAt: '2024-06-01T08:00:00Z',
    },
  ];

  const users = response?.items || mockUsers;
  const totalItems = response?.pagination?.totalItems || mockUsers.length;
  const totalPages = response?.pagination?.totalPages || 1;

  // Filter users by search
  const filteredUsers = React.useMemo(() => {
    if (!search) return users;
    const searchLower = search.toLowerCase();
    return users.filter(
      (user) =>
        user.email.toLowerCase().includes(searchLower) ||
        user.firstName.toLowerCase().includes(searchLower) ||
        user.lastName.toLowerCase().includes(searchLower)
    );
  }, [users, search]);

  const handleCreateUser = () => {
    createUser.mutate(formData, {
      onSuccess: () => {
        setShowCreateDialog(false);
        setFormData({
          email: '',
          firstName: '',
          lastName: '',
          role: 'viewer',
          password: '',
        });
      },
    });
  };

  const handleDeleteUser = (userId: string) => {
    if (confirm('Are you sure you want to delete this user?')) {
      deleteUser.mutate(userId);
    }
  };

  const handleToggleActive = (user: User) => {
    updateUser.mutate({
      userId: user.id,
      data: { isActive: !user.isActive },
    });
  };

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">User Management</h1>
          <p className="text-muted-foreground">
            Manage users, roles, and permissions
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Add User
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create New User</DialogTitle>
                <DialogDescription>
                  Add a new user to the system. They will receive an email invitation.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 py-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <Input
                    label="First Name"
                    value={formData.firstName}
                    onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
                    required
                  />
                  <Input
                    label="Last Name"
                    value={formData.lastName}
                    onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
                    required
                  />
                </div>
                <Input
                  label="Email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  required
                />
                <div>
                  <label className="text-sm font-medium">Role</label>
                  <Select
                    value={formData.role}
                    onValueChange={(value) => setFormData({ ...formData, role: value as UserRole })}
                  >
                    <SelectTrigger className="mt-1.5">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="admin">Admin</SelectItem>
                      <SelectItem value="analyst">Analyst</SelectItem>
                      <SelectItem value="viewer">Viewer</SelectItem>
                      <SelectItem value="api_user">API User</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Input
                  label="Temporary Password"
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  helperText="User will be required to change password on first login"
                  required
                />
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                  Cancel
                </Button>
                <Button onClick={handleCreateUser} loading={createUser.isPending}>
                  Create User
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search users..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            <Select value={roleFilter} onValueChange={setRoleFilter}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Role" />
              </SelectTrigger>
              <SelectContent>
                {roleOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Users table */}
      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>User</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last Login</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="w-[80px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableSkeleton rows={5} columns={6} />
            ) : filteredUsers.length === 0 ? (
              <TableEmpty
                icon={<Search className="h-12 w-12" />}
                title="No users found"
                description={search ? 'Try adjusting your search' : 'Add a user to get started'}
                action={
                  <Button size="sm" onClick={() => setShowCreateDialog(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add User
                  </Button>
                }
              />
            ) : (
              filteredUsers.map((user) => (
                <TableRow key={user.id}>
                  <TableCell>
                    <div className="flex items-center gap-3">
                      <Avatar
                        src={user.avatar}
                        alt={`${user.firstName} ${user.lastName}`}
                        size="sm"
                      />
                      <div>
                        <p className="font-medium">
                          {user.firstName} {user.lastName}
                        </p>
                        <p className="text-sm text-muted-foreground">{user.email}</p>
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant={roleConfig[user.role].variant}>
                      {roleConfig[user.role].label}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant={user.isActive ? 'active' : 'inactive'} dot>
                      {user.isActive ? 'Active' : 'Inactive'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {user.lastLoginAt ? (
                      <span className="text-sm">
                        {formatRelativeTime(user.lastLoginAt)}
                      </span>
                    ) : (
                      <span className="text-sm text-muted-foreground">Never</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <span className="text-sm text-muted-foreground">
                      {formatDateTime(user.createdAt)}
                    </span>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => handleToggleActive(user)}
                        title={user.isActive ? 'Deactivate' : 'Activate'}
                      >
                        {user.isActive ? (
                          <UserX className="h-4 w-4" />
                        ) : (
                          <UserCheck className="h-4 w-4" />
                        )}
                      </Button>
                      <Button variant="ghost" size="icon-sm" title="Edit">
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => handleDeleteUser(user.id)}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
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
      {!isLoading && filteredUsers.length > 0 && (
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

      {/* Role descriptions */}
      <Card>
        <CardHeader>
          <CardTitle>Role Permissions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-5 w-5 text-destructive" />
                <h3 className="font-semibold">Admin</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Full system access including user management, agent configuration, and tenant settings.
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Analyst</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Can run calculations, generate reports, and view all data. Cannot manage users.
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-5 w-5 text-muted-foreground" />
                <h3 className="font-semibold">Viewer</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Read-only access to dashboards, reports, and calculation history.
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="h-5 w-5 text-muted-foreground" />
                <h3 className="font-semibold">API User</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                API-only access for integrations. Cannot access web interface.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
