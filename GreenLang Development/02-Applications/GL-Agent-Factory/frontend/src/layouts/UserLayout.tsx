/**
 * UserLayout Component
 *
 * Layout wrapper for user portal pages with navigation.
 */

import * as React from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  Home,
  Fuel,
  Calculator,
  Building,
  TreePine,
  FileText,
  Menu,
  X,
  Bell,
  ChevronDown,
  Moon,
  Sun,
  HelpCircle,
  LogOut,
  Settings,
  User,
} from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Avatar } from '@/components/ui/Avatar';
import { Badge } from '@/components/ui/Badge';
import { useAuthStore } from '@/stores/authStore';
import { useUIStore } from '@/stores/uiStore';
import { cn } from '@/utils/cn';

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Fuel Analyzer', href: '/fuel-analyzer', icon: Fuel },
  { name: 'CBAM Calculator', href: '/cbam-calculator', icon: Calculator },
  { name: 'Building Energy', href: '/building-energy', icon: Building },
  { name: 'EUDR Compliance', href: '/eudr-compliance', icon: TreePine },
  { name: 'Reports', href: '/reports', icon: FileText },
];

export default function UserLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const { sidebarOpen, setSidebarOpen, theme, setTheme } = useUIStore();

  const [showUserMenu, setShowUserMenu] = React.useState(false);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  // Close menus when clicking outside
  React.useEffect(() => {
    const handleClickOutside = () => {
      setShowUserMenu(false);
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile navigation backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Mobile sidebar */}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-72 bg-card border-r transform transition-transform duration-300 lg:hidden',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <div className="flex h-16 items-center justify-between border-b px-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold">
              GL
            </div>
            <span className="font-semibold">GreenLang</span>
          </div>
          <Button variant="ghost" size="icon-sm" onClick={() => setSidebarOpen(false)}>
            <X className="h-5 w-5" />
          </Button>
        </div>

        <nav className="p-4">
          <ul className="space-y-1">
            {navigation.map((item) => {
              const isActive =
                location.pathname === item.href ||
                (item.href !== '/' && location.pathname.startsWith(item.href));

              return (
                <li key={item.name}>
                  <NavLink
                    to={item.href}
                    onClick={() => setSidebarOpen(false)}
                    className={cn(
                      'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    )}
                  >
                    <item.icon className="h-5 w-5" />
                    {item.name}
                  </NavLink>
                </li>
              );
            })}
          </ul>
        </nav>
      </aside>

      {/* Desktop header */}
      <header className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            {/* Logo and mobile menu */}
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                className="lg:hidden"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu className="h-5 w-5" />
              </Button>

              <NavLink to="/" className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground font-bold">
                  GL
                </div>
                <span className="font-semibold hidden sm:block">GreenLang</span>
              </NavLink>
            </div>

            {/* Desktop navigation */}
            <nav className="hidden lg:flex items-center gap-1">
              {navigation.map((item) => {
                const isActive =
                  location.pathname === item.href ||
                  (item.href !== '/' && location.pathname.startsWith(item.href));

                return (
                  <NavLink
                    key={item.name}
                    to={item.href}
                    className={cn(
                      'flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary/10 text-primary'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    )}
                  >
                    <item.icon className="h-4 w-4" />
                    {item.name}
                  </NavLink>
                );
              })}
            </nav>

            {/* Right side actions */}
            <div className="flex items-center gap-2">
              {/* Help */}
              <Button variant="ghost" size="icon">
                <HelpCircle className="h-5 w-5" />
              </Button>

              {/* Theme toggle */}
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>

              {/* Notifications */}
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-5 w-5" />
                <Badge
                  variant="destructive"
                  className="absolute -right-1 -top-1 h-4 w-4 rounded-full p-0 text-[10px]"
                >
                  2
                </Badge>
              </Button>

              {/* User menu */}
              <div className="relative">
                <Button
                  variant="ghost"
                  className="flex items-center gap-2 px-2"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowUserMenu(!showUserMenu);
                  }}
                >
                  <Avatar
                    src={user?.avatar}
                    alt={`${user?.firstName} ${user?.lastName}`}
                    size="sm"
                  />
                  <span className="hidden md:block text-sm font-medium">
                    {user?.firstName}
                  </span>
                  <ChevronDown className="h-4 w-4" />
                </Button>

                {showUserMenu && (
                  <div className="absolute right-0 top-full mt-2 w-56 rounded-lg border bg-card shadow-lg">
                    <div className="border-b p-4">
                      <p className="font-medium">
                        {user?.firstName} {user?.lastName}
                      </p>
                      <p className="text-sm text-muted-foreground">{user?.email}</p>
                      <Badge variant="secondary" size="sm" className="mt-2">
                        {user?.role}
                      </Badge>
                    </div>
                    <div className="p-2">
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          navigate('/profile');
                        }}
                        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted"
                      >
                        <User className="h-4 w-4" />
                        Profile
                      </button>
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          navigate('/settings');
                        }}
                        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted"
                      >
                        <Settings className="h-4 w-4" />
                        Settings
                      </button>
                      {user?.role === 'admin' && (
                        <button
                          onClick={() => {
                            setShowUserMenu(false);
                            navigate('/admin');
                          }}
                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted"
                        >
                          <Settings className="h-4 w-4" />
                          Admin Portal
                        </button>
                      )}
                      <hr className="my-2" />
                      <button
                        onClick={handleLogout}
                        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-destructive hover:bg-muted"
                      >
                        <LogOut className="h-4 w-4" />
                        Log out
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Page content */}
      <main className="mx-auto max-w-7xl px-4 py-6 lg:px-8">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="mx-auto max-w-7xl px-4 py-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground text-xs font-bold">
                GL
              </div>
              <span>GreenLang Agent Factory</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a href="#" className="hover:text-foreground">Documentation</a>
              <a href="#" className="hover:text-foreground">API Reference</a>
              <a href="#" className="hover:text-foreground">Support</a>
              <a href="#" className="hover:text-foreground">Privacy</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
