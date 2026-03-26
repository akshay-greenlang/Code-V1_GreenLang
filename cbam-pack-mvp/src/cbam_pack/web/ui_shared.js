(() => {
  const path = window.location.pathname || "";
  const appId = path.includes("/apps/csrd")
    ? "csrd"
    : path.includes("/apps/vcci")
      ? "vcci"
      : path.includes("/apps/cbam")
        ? "cbam"
        : "";

  function createPalette() {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = `
      <div id="glCmdPalette" style="display:none;position:fixed;inset:0;background:rgba(2,8,23,.72);z-index:9999;">
        <div style="width:min(680px,96%);margin:8vh auto;padding:16px;border-radius:14px;border:1px solid rgba(255,255,255,.18);background:rgba(10,15,26,.95);">
          <div style="font-weight:700;margin-bottom:10px;color:#e8efff;">Command Palette</div>
          <input id="glCmdSearch" placeholder="Type to filter commands..." style="width:100%;padding:10px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.05);color:#e8efff;" />
          <div id="glCmdList" style="margin-top:10px;display:grid;gap:8px;"></div>
          <div style="margin-top:10px;color:#a8b6d8;font-size:.82rem;">Esc to close • Ctrl/Cmd+K to open</div>
        </div>
      </div>
    `;
    document.body.appendChild(wrapper);
    const palette = document.getElementById("glCmdPalette");
    const list = document.getElementById("glCmdList");
    const input = document.getElementById("glCmdSearch");

    const commands = [
      { label: "Open Apps Home", action: () => (window.location.href = "/apps") },
      { label: "Open Runs Center", action: () => (window.location.href = "/runs") },
      { label: "Open CBAM Workspace", action: () => (window.location.href = "/apps/cbam") },
      { label: "Open CSRD Workspace", action: () => (window.location.href = "/apps/csrd") },
      { label: "Open VCCI Workspace", action: () => (window.location.href = "/apps/vcci") },
      { label: "Open API Docs", action: () => (window.location.href = "/docs") },
      { label: "Run Demo Data", action: () => runDemo() },
    ];

    function render(filter) {
      const q = (filter || "").toLowerCase();
      const filtered = commands.filter((c) => c.label.toLowerCase().includes(q));
      list.innerHTML = filtered
        .map(
          (c, i) => `<button data-idx="${i}" style="text-align:left;padding:10px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.04);color:#e8efff;cursor:pointer;">${c.label}</button>`
        )
        .join("");
      [...list.querySelectorAll("button")].forEach((btn) => {
        btn.addEventListener("click", () => {
          const idx = Number(btn.getAttribute("data-idx"));
          filtered[idx]?.action();
          close();
        });
      });
    }

    function open() {
      palette.style.display = "block";
      render("");
      input.value = "";
      input.focus();
    }
    function close() {
      palette.style.display = "none";
    }
    input.addEventListener("input", () => render(input.value));
    palette.addEventListener("click", (e) => {
      if (e.target === palette) close();
    });
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        open();
      } else if (e.key === "Escape") {
        close();
      }
    });
  }

  async function runDemo() {
    if (!appId) return;
    const endpoint = `/api/v1/apps/${appId}/demo-run`;
    try {
      const res = await fetch(endpoint, { method: "POST" });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload?.detail || `HTTP ${res.status}`);
      if (payload?.run_id) {
        window.location.href = `/runs`;
      }
    } catch (err) {
      console.error("Demo run failed:", err);
      alert(`Demo run failed: ${err?.message || String(err)}`);
    }
  }

  function mountDemoButton() {
    if (!appId) return;
    const runBtn = document.getElementById("runBtn") || document.getElementById("processBtn");
    if (!runBtn || document.getElementById("glDemoBtn")) return;
    const btn = document.createElement("button");
    btn.id = "glDemoBtn";
    btn.textContent = "Try Demo Data";
    btn.style.cssText =
      "margin-top:10px;width:100%;padding:10px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.2);background:rgba(12,20,40,.35);color:#e8efff;font-weight:700;cursor:pointer;";
    btn.addEventListener("click", runDemo);
    runBtn.parentElement?.insertBefore(btn, runBtn.nextSibling);
  }

  function mountOnboarding() {
    const key = "gl_onboarding_seen_v1";
    if (localStorage.getItem(key)) return;
    const modal = document.createElement("div");
    modal.innerHTML = `
      <div style="position:fixed;inset:0;z-index:9998;background:rgba(2,8,23,.7);display:grid;place-items:center;">
        <div style="width:min(760px,96%);padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,.18);background:rgba(10,15,26,.96);color:#e8efff;">
          <h2 style="margin-bottom:8px;">Welcome to GreenLang</h2>
          <p style="color:#a8b6d8;margin-bottom:12px;">Use one shell for CBAM, CSRD, and VCCI. Press Ctrl/Cmd+K to open command palette anytime.</p>
          <ul style="color:#a8b6d8;line-height:1.6;margin-left:16px;">
            <li>Run with your own files, or use <strong>Try Demo Data</strong>.</li>
            <li>Review runs in <strong>/runs</strong>.</li>
            <li>Download audit bundle for deterministic evidence.</li>
          </ul>
          <div style="margin-top:14px;display:flex;justify-content:flex-end;gap:8px;">
            <button id="glOnboardClose" style="padding:8px 12px;border-radius:10px;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.04);color:#e8efff;cursor:pointer;">Got it</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    document.getElementById("glOnboardClose")?.addEventListener("click", () => {
      localStorage.setItem(key, "1");
      modal.remove();
    });
  }

  function wireClientErrorTelemetry() {
    const send = async (payload) => {
      try {
        await fetch("/api/telemetry/client-error", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (e) {
        // Ignore telemetry failures by design.
      }
    };
    window.addEventListener("error", (evt) => {
      send({
        type: "window_error",
        message: evt?.message || "unknown",
        source: evt?.filename || "",
        line: evt?.lineno || 0,
        column: evt?.colno || 0,
        path: window.location.pathname,
      });
    });
    window.addEventListener("unhandledrejection", (evt) => {
      send({
        type: "unhandled_rejection",
        message: String(evt?.reason || "unknown"),
        path: window.location.pathname,
      });
    });
  }

  createPalette();
  mountDemoButton();
  mountOnboarding();
  wireClientErrorTelemetry();
})();
