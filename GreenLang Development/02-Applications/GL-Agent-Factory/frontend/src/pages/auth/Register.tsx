/**
 * Register Page
 *
 * New user registration form.
 */

import * as React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { User, Mail, Lock, Building2, ArrowRight, Check } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useAuthStore } from '@/stores/authStore';
import { useRegister } from '@/api/hooks';

const registerSchema = z.object({
  firstName: z.string().min(1, 'First name is required'),
  lastName: z.string().min(1, 'Last name is required'),
  email: z.string().email('Please enter a valid email address'),
  organizationName: z.string().optional(),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
  acceptTerms: z.boolean().refine(val => val === true, 'You must accept the terms'),
}).refine(data => data.password === data.confirmPassword, {
  message: 'Passwords do not match',
  path: ['confirmPassword'],
});

type RegisterFormData = z.infer<typeof registerSchema>;

export default function Register() {
  const navigate = useNavigate();
  const { setUser } = useAuthStore();
  const registerMutation = useRegister();

  const form = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      firstName: '',
      lastName: '',
      email: '',
      organizationName: '',
      password: '',
      confirmPassword: '',
      acceptTerms: false,
    },
  });

  const password = form.watch('password');

  const passwordRequirements = [
    { label: 'At least 8 characters', met: password.length >= 8 },
    { label: 'One uppercase letter', met: /[A-Z]/.test(password) },
    { label: 'One number', met: /[0-9]/.test(password) },
  ];

  const onSubmit = async (data: RegisterFormData) => {
    try {
      const response = await registerMutation.mutateAsync({
        email: data.email,
        password: data.password,
        firstName: data.firstName,
        lastName: data.lastName,
        organizationName: data.organizationName,
      });
      setUser(response.user);
      navigate('/', { replace: true });
    } catch {
      // Error handled by mutation
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold">Create your account</h1>
        <p className="text-muted-foreground mt-2">
          Start your free trial today
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="relative">
                <User className="absolute left-3 top-9 h-4 w-4 text-muted-foreground" />
                <Input
                  label="First Name"
                  placeholder="John"
                  className="pl-10"
                  {...form.register('firstName')}
                  error={form.formState.errors.firstName?.message}
                />
              </div>
              <Input
                label="Last Name"
                placeholder="Doe"
                {...form.register('lastName')}
                error={form.formState.errors.lastName?.message}
              />
            </div>

            <div className="relative">
              <Mail className="absolute left-3 top-9 h-4 w-4 text-muted-foreground" />
              <Input
                label="Email"
                type="email"
                placeholder="you@company.com"
                className="pl-10"
                {...form.register('email')}
                error={form.formState.errors.email?.message}
              />
            </div>

            <div className="relative">
              <Building2 className="absolute left-3 top-9 h-4 w-4 text-muted-foreground" />
              <Input
                label="Organization Name (Optional)"
                placeholder="Your Company Inc."
                className="pl-10"
                {...form.register('organizationName')}
                helperText="Leave blank to create a personal account"
              />
            </div>

            <div className="relative">
              <Lock className="absolute left-3 top-9 h-4 w-4 text-muted-foreground" />
              <Input
                label="Password"
                type="password"
                placeholder="Create a strong password"
                className="pl-10"
                showPasswordToggle
                {...form.register('password')}
                error={form.formState.errors.password?.message}
              />
            </div>

            {/* Password requirements */}
            {password.length > 0 && (
              <div className="space-y-2 p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground font-medium">Password requirements:</p>
                {passwordRequirements.map((req, index) => (
                  <div key={index} className="flex items-center gap-2 text-xs">
                    <Check className={`h-3 w-3 ${req.met ? 'text-greenlang-500' : 'text-muted-foreground'}`} />
                    <span className={req.met ? 'text-greenlang-600' : 'text-muted-foreground'}>
                      {req.label}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="relative">
              <Lock className="absolute left-3 top-9 h-4 w-4 text-muted-foreground" />
              <Input
                label="Confirm Password"
                type="password"
                placeholder="Confirm your password"
                className="pl-10"
                showPasswordToggle
                {...form.register('confirmPassword')}
                error={form.formState.errors.confirmPassword?.message}
              />
            </div>

            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                {...form.register('acceptTerms')}
                className="rounded border-gray-300 mt-0.5"
              />
              <span className="text-muted-foreground">
                I agree to the{' '}
                <a href="#" className="text-primary hover:underline">Terms of Service</a>
                {' '}and{' '}
                <a href="#" className="text-primary hover:underline">Privacy Policy</a>
              </span>
            </label>
            {form.formState.errors.acceptTerms && (
              <p className="text-sm text-destructive -mt-2">
                {form.formState.errors.acceptTerms.message}
              </p>
            )}

            <Button
              type="submit"
              className="w-full"
              loading={registerMutation.isPending}
            >
              Create Account
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </form>

          {/* Divider */}
          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-card px-2 text-muted-foreground">Or sign up with</span>
            </div>
          </div>

          {/* Social signup */}
          <div className="grid gap-2">
            <Button variant="outline" type="button">
              <svg className="h-4 w-4 mr-2" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M12.545,10.239v3.821h5.445c-0.712,2.315-2.647,3.972-5.445,3.972c-3.332,0-6.033-2.701-6.033-6.032s2.701-6.032,6.033-6.032c1.498,0,2.866,0.549,3.921,1.453l2.814-2.814C17.503,2.988,15.139,2,12.545,2C7.021,2,2.543,6.477,2.543,12s4.478,10,10.002,10c8.396,0,10.249-7.85,9.426-11.748L12.545,10.239z"
                />
              </svg>
              Sign up with Google
            </Button>
          </div>
        </CardContent>
      </Card>

      <p className="text-center text-sm text-muted-foreground">
        Already have an account?{' '}
        <Link to="/login" className="text-primary hover:underline font-medium">
          Sign in
        </Link>
      </p>
    </div>
  );
}
