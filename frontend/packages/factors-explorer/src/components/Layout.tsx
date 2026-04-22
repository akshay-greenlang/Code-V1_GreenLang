import { useEffect, useState } from "react";
import { Link, Outlet } from "@tanstack/react-router";
import { BookOpen, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * Root layout: nav bar (Home, Search, Sources, Method Packs, Editions, API
 * docs link), footer with link to developer portal, dark-mode toggle.
 */
export function Layout() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("gl-factors-theme");
    const initial =
      saved === "dark" ||
      (saved === null &&
        window.matchMedia?.("(prefers-color-scheme: dark)").matches);
    setDark(Boolean(initial));
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("gl-factors-theme", dark ? "dark" : "light");
  }, [dark]);

  return (
    <div className="flex min-h-full flex-col bg-background">
      <header className="sticky top-0 z-30 border-b border-border bg-background/95 backdrop-blur">
        <div className="container flex h-14 items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <span
              aria-hidden="true"
              className="inline-block h-6 w-6 rounded bg-factor-certified-500"
            />
            <span>GreenLang Factors</span>
          </Link>
          <nav aria-label="Primary" className="flex items-center gap-1 text-sm">
            <NavLink to="/">Home</NavLink>
            <NavLink to="/search">Search</NavLink>
            <NavLink to="/sources">Sources</NavLink>
            <NavLink to="/method-packs">Method Packs</NavLink>
            <NavLink to="/editions">Editions</NavLink>
            <a
              href="https://developer.greenlang.io/docs/factors"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <BookOpen className="h-3.5 w-3.5" />
              API docs
            </a>
          </nav>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setDark((d) => !d)}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </header>

      <main className="container flex-1 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-border bg-muted/30">
        <div className="container flex flex-col items-center justify-between gap-2 py-4 text-xs text-muted-foreground md:flex-row">
          <p>
            GreenLang Factors • Canonical climate reference system.{" "}
            <a
              className="underline underline-offset-2 hover:text-foreground"
              href="https://developer.greenlang.io"
              target="_blank"
              rel="noopener noreferrer"
            >
              Developer portal
            </a>
          </p>
          <p>
            <Link
              to="/verify"
              className="underline underline-offset-2 hover:text-foreground"
            >
              Verify a signed receipt
            </Link>
            {" · "}
            <a
              className="underline underline-offset-2 hover:text-foreground"
              href="https://greenlang.io/legal/terms"
              target="_blank"
              rel="noopener noreferrer"
            >
              Terms
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}

function NavLink({
  to,
  children,
}: {
  to: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      to={to}
      className={cn(
        "rounded-md px-2 py-1 text-muted-foreground hover:bg-accent hover:text-foreground",
        "data-[status=active]:bg-accent data-[status=active]:text-foreground"
      )}
      activeProps={{ "data-status": "active" } as never}
    >
      {children}
    </Link>
  );
}
