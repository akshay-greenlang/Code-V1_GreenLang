import { useState } from "react";
import { Check, Copy, ShieldCheck } from "lucide-react";
import { Link } from "@tanstack/react-router";
import type { SignedReceipt as SignedReceiptType } from "@/types/factors";
import { copyToClipboard, formatDate, truncateHash } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface SignedReceiptProps {
  receipt: SignedReceiptType;
}

/** Signature hash + algorithm + key_id + signed_at, with a copy button. */
export function SignedReceipt({ receipt }: SignedReceiptProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const ok = await copyToClipboard(receipt.content_hash);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <div
      data-testid="signed-receipt"
      className="flex flex-wrap items-center gap-3 rounded-md border border-border bg-muted/40 p-3 text-sm"
    >
      <ShieldCheck
        className="h-5 w-5 shrink-0 text-factor-certified-700"
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1">
        <p className="flex items-center gap-2 font-medium">
          Signed receipt
          <span className="rounded bg-white px-1 py-px font-mono text-[10px] text-muted-foreground">
            {receipt.algorithm}
          </span>
        </p>
        <p className="font-mono text-xs text-muted-foreground">
          {truncateHash(receipt.content_hash, 16, 8)}
        </p>
        <p className="text-xs text-muted-foreground">
          {receipt.key_id ? (
            <>
              key <code className="font-mono">{receipt.key_id}</code>
              {" • "}
            </>
          ) : null}
          {receipt.signed_at ? `signed ${formatDate(receipt.signed_at)}` : null}
        </p>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopy}
          aria-label="Copy receipt hash"
        >
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy receipt"}
        </Button>
        <Link
          to="/verify"
          className="text-xs underline underline-offset-2 hover:text-foreground"
        >
          How to verify
        </Link>
      </div>
    </div>
  );
}
