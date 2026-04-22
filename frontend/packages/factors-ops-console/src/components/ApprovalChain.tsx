import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { User, ShieldAlert, CheckCircle2, XCircle, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/input";
import { canApprove, type Identity } from "@/lib/auth";
import type { ReviewItem } from "@/types/ops";

/**
 * Two-stage approval chain with Segregation-of-Duties enforcement.
 *
 * SoD rule (spec §3.3): `author.sub !== current_user.sub`. If violated, the
 * Approve button is not rendered at all (we still show a "you authored this"
 * banner so the user understands why the control is missing).
 *
 * Every action sends an audit reason (>= 10 chars) via buildAuditHeaders in
 * the api layer.
 */
const ReasonSchema = z.object({
  reason: z.string().min(10, "Reason must be at least 10 characters").max(500),
});
type ReasonInput = z.infer<typeof ReasonSchema>;

interface Props {
  review: ReviewItem;
  identity: Identity;
  onApprove: (reason: string) => Promise<void>;
  onReject: (reason: string) => Promise<void>;
  onRequestChanges?: (reason: string) => Promise<void>;
}

export function ApprovalChain({
  review,
  identity,
  onApprove,
  onReject,
  onRequestChanges,
}: Props) {
  const [submitting, setSubmitting] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
    reset,
  } = useForm<ReasonInput>({
    resolver: zodResolver(ReasonSchema),
    mode: "onChange",
    defaultValues: { reason: "" },
  });

  const isAuthor = identity.sub === review.author.sub;
  const mayApprove = canApprove(identity, review.author.sub);

  const submit = (action: "approve" | "reject" | "changes") =>
    handleSubmit(async ({ reason }) => {
      setSubmitting(true);
      try {
        if (action === "approve") await onApprove(reason);
        else if (action === "reject") await onReject(reason);
        else if (action === "changes" && onRequestChanges) await onRequestChanges(reason);
        reset();
      } finally {
        setSubmitting(false);
      }
    });

  return (
    <section className="space-y-4" aria-label="Approval chain">
      <div
        className="flex flex-wrap items-center gap-2 text-sm"
        aria-label="Review chain"
        data-testid="approval-chain-visual"
      >
        <ChainNode
          label={review.author.display_name}
          role="author"
          status="done"
          icon={User}
        />
        {review.steps.map((step, idx) => (
          <ChainNode
            key={idx}
            label={step.approver?.display_name ?? "unassigned"}
            role={`approver ${idx + 1}`}
            status={step.status === "approved" ? "done" : step.status === "rejected" ? "rejected" : "pending"}
            icon={
              step.status === "approved"
                ? CheckCircle2
                : step.status === "rejected"
                  ? XCircle
                  : Clock
            }
          />
        ))}
      </div>

      {isAuthor && (
        <div
          role="alert"
          data-testid="sod-banner"
          className="flex items-start gap-2 rounded-md border border-factor-preview-500 bg-factor-preview-50 p-3 text-sm text-factor-preview-700"
        >
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <strong>You authored this submission.</strong>{" "}
            Segregation of duties blocks you from approving your own work. Another reviewer
            must approve.
          </div>
        </div>
      )}

      <form className="space-y-2">
        <label htmlFor="reason" className="text-sm font-medium">
          Reason <span className="text-muted-foreground">(required, min 10 chars)</span>
        </label>
        <Textarea
          id="reason"
          rows={3}
          aria-invalid={Boolean(errors.reason)}
          placeholder="Matches QA acceptance criteria v4; gas breakdown audited…"
          {...register("reason")}
        />
        {errors.reason && (
          <p role="alert" className="text-xs text-factor-deprecated-700">
            {errors.reason.message}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2">
          {mayApprove ? (
            <Button
              type="button"
              variant="default"
              disabled={!isValid || submitting}
              onClick={submit("approve")}
              data-testid="approve-btn"
            >
              Approve
            </Button>
          ) : (
            <Badge variant="muted" data-testid="approve-disabled">
              Approve blocked by SoD
            </Badge>
          )}
          {onRequestChanges && (
            <Button
              type="button"
              variant="outline"
              disabled={!isValid || submitting}
              onClick={submit("changes")}
            >
              Request changes
            </Button>
          )}
          <Button
            type="button"
            variant="destructive"
            disabled={!isValid || submitting}
            onClick={submit("reject")}
            data-testid="reject-btn"
          >
            Reject
          </Button>
        </div>
      </form>
    </section>
  );
}

function ChainNode({
  label,
  role,
  status,
  icon: Icon,
}: {
  label: string;
  role: string;
  status: "done" | "pending" | "rejected";
  icon: typeof User;
}) {
  const variant = status === "done" ? "success" : status === "rejected" ? "danger" : "muted";
  return (
    <div className="flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1">
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      <span className="text-sm">{label}</span>
      <Badge variant={variant} className="text-2xs capitalize">
        {role}
      </Badge>
    </div>
  );
}
