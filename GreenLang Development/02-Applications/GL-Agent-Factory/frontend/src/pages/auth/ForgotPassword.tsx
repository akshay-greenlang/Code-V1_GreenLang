/**
 * ForgotPassword Page
 *
 * Password reset request form.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Mail, ArrowLeft, ArrowRight, CheckCircle } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useForgotPassword } from '@/api/hooks';

const forgotPasswordSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
});

type ForgotPasswordFormData = z.infer<typeof forgotPasswordSchema>;

export default function ForgotPassword() {
  const [isSubmitted, setIsSubmitted] = React.useState(false);
  const forgotPasswordMutation = useForgotPassword();

  const form = useForm<ForgotPasswordFormData>({
    resolver: zodResolver(forgotPasswordSchema),
    defaultValues: {
      email: '',
    },
  });

  const onSubmit = async (data: ForgotPasswordFormData) => {
    try {
      await forgotPasswordMutation.mutateAsync(data.email);
      setIsSubmitted(true);
    } catch {
      // Error handled by mutation
    }
  };

  if (isSubmitted) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <div className="mx-auto w-12 h-12 rounded-full bg-greenlang-100 flex items-center justify-center mb-4">
            <CheckCircle className="h-6 w-6 text-greenlang-600" />
          </div>
          <h1 className="text-2xl font-bold">Check your email</h1>
          <p className="text-muted-foreground mt-2">
            We've sent a password reset link to{' '}
            <strong>{form.getValues('email')}</strong>
          </p>
        </div>

        <Card>
          <CardContent className="pt-6 space-y-4">
            <p className="text-sm text-muted-foreground text-center">
              Didn't receive the email? Check your spam folder or try again.
            </p>

            <Button
              variant="outline"
              className="w-full"
              onClick={() => {
                setIsSubmitted(false);
                form.reset();
              }}
            >
              Try another email
            </Button>
          </CardContent>
        </Card>

        <p className="text-center text-sm text-muted-foreground">
          <Link to="/login" className="text-primary hover:underline inline-flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back to sign in
          </Link>
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold">Forgot your password?</h1>
        <p className="text-muted-foreground mt-2">
          No worries, we'll send you reset instructions.
        </p>
      </div>

      <Card>
        <CardContent className="pt-6">
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
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

            <Button
              type="submit"
              className="w-full"
              loading={forgotPasswordMutation.isPending}
            >
              Send Reset Link
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </form>
        </CardContent>
      </Card>

      <p className="text-center text-sm text-muted-foreground">
        <Link to="/login" className="text-primary hover:underline inline-flex items-center gap-1">
          <ArrowLeft className="h-4 w-4" />
          Back to sign in
        </Link>
      </p>
    </div>
  );
}
