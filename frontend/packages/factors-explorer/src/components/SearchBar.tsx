import { useEffect, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { searchSuggestions } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SearchBarProps {
  value: string;
  onChange: (q: string) => void;
  onSubmit: (q: string) => void;
  placeholder?: string;
  compact?: boolean;
}

/** Autocomplete input hitting /factors/search?suggest=1 with debounce 200ms. */
export function SearchBar({
  value,
  onChange,
  onSubmit,
  placeholder = "Search factors…",
  compact = false,
}: SearchBarProps) {
  const [draft, setDraft] = useState(value);
  const [open, setOpen] = useState(false);
  const [debounced, setDebounced] = useState(value);
  const listRef = useRef<HTMLUListElement | null>(null);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(() => {
    const handle = setTimeout(() => setDebounced(draft), 200);
    return () => clearTimeout(handle);
  }, [draft]);

  const { data: suggestions = [] } = useQuery({
    queryKey: queryKeys.suggestions(debounced),
    queryFn: () => searchSuggestions(debounced),
    enabled: debounced.length >= 2,
    staleTime: 60_000,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onChange(draft);
    onSubmit(draft);
    setOpen(false);
  };

  const handlePick = (s: string) => {
    setDraft(s);
    onChange(s);
    onSubmit(s);
    setOpen(false);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={cn("relative flex items-center gap-2", compact ? "" : "w-full")}
      role="search"
    >
      <div className="relative flex-1">
        <Search
          className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden="true"
        />
        <Input
          value={draft}
          onChange={(e) => {
            setDraft(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(draft.length >= 2)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          placeholder={placeholder}
          aria-label="Search factors"
          aria-autocomplete="list"
          aria-expanded={open}
          className="pl-8 pr-8"
        />
        {draft.length > 0 ? (
          <button
            type="button"
            onClick={() => {
              setDraft("");
              onChange("");
            }}
            aria-label="Clear search"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}

        {open && suggestions.length > 0 ? (
          <ul
            ref={listRef}
            role="listbox"
            className="absolute left-0 right-0 top-full z-10 mt-1 max-h-72 overflow-y-auto rounded-md border border-border bg-card shadow-lg"
          >
            {suggestions.map((s, i) => (
              <li key={`${s}-${i}`}>
                <button
                  type="button"
                  role="option"
                  aria-selected={false}
                  onClick={() => handlePick(s)}
                  className="w-full px-3 py-1.5 text-left text-sm hover:bg-accent"
                >
                  {s}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
      </div>

      <Button type="submit" size={compact ? "sm" : "default"}>
        Search
      </Button>
    </form>
  );
}
