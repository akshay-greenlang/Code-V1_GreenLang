import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { ApprovalChain } from "@/components/ApprovalChain";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { queryKeys } from "@/lib/query";
import { approveRelease, listReviewQueue, rejectRelease } from "@/lib/api";
import { formatAge } from "@/lib/utils";
import type { Identity } from "@/lib/auth";

export const Route = createFileRoute("/approvals")({
  component: ApprovalsPage,
});

function ApprovalsPage() {
  return (
    <AuthGuard requiredAction="review.approve">
      {(identity) => <ApprovalsWorkflow identity={identity} />}
    </AuthGuard>
  );
}

function ApprovalsWorkflow({ identity }: { identity: Identity }) {
  const qc = useQueryClient();
  const { data: queue } = useQuery({
    queryKey: queryKeys.reviews.queue(),
    queryFn: listReviewQueue,
  });
  const [focused, setFocused] = useState<string | null>(null);
  const active = (queue ?? []).find((r) => r.review_id === focused);

  const approve = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      approveRelease(identity, id, reason),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.reviews.queue() }),
  });
  const reject = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      rejectRelease(identity, id, reason),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.reviews.queue() }),
  });

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_2fr]">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Review queue ({queue?.length ?? 0})</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <ul className="divide-y divide-border text-sm">
            {(queue ?? []).map((r) => (
              <li key={r.review_id}>
                <button
                  type="button"
                  className={`flex w-full items-center justify-between px-3 py-2 text-left hover:bg-muted/30 ${
                    focused === r.review_id ? "bg-muted/30" : ""
                  }`}
                  onClick={() => setFocused(r.review_id)}
                >
                  <div className="flex flex-col">
                    <span className="font-mono text-xs">{r.review_id}</span>
                    <span className="text-xs text-muted-foreground">
                      {r.kind} · author {r.author.display_name}
                    </span>
                  </div>
                  <Badge variant="muted">{formatAge(r.age_hours)}</Badge>
                </button>
              </li>
            ))}
            {(!queue || queue.length === 0) && (
              <li className="px-3 py-6 text-center text-xs text-muted-foreground">
                Queue is empty.
              </li>
            )}
          </ul>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {active ? (
          <Card>
            <CardHeader>
              <CardTitle>
                Review {active.review_id}
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {active.kind}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <ApprovalChain
                review={active}
                identity={identity}
                onApprove={(reason) => approve.mutateAsync({ id: active.review_id, reason })}
                onReject={(reason) => reject.mutateAsync({ id: active.review_id, reason })}
              />
            </CardContent>
          </Card>
        ) : (
          <p className="text-sm text-muted-foreground">
            Select a review from the queue to see its approval chain.
          </p>
        )}
      </div>
    </div>
  );
}
