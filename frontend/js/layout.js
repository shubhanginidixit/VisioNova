/**
 * VisioNova Layout Manager v3.0
 * Injects the top navigation header and collapsible sidebar.
 * Typography: Archivo Black / DM Mono / Archivo
 */

const Layout = {
    init() {
        this.injectFonts();
        this.injectHeader();
        this.injectSidebar();
        this.highlightCurrentNav();
    },

    injectFonts() {
        if (document.getElementById('vn-fonts')) return;
        const link = document.createElement('link');
        link.id = 'vn-fonts';
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Archivo:wght@400;500;600&display=swap';
        document.head.appendChild(link);
    },

    injectHeader() {
        const el = document.querySelector('header[data-inject], header:empty');
        if (!el) return;

        el.style.cssText = `
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 46px;
      padding: 0 20px;
      background: var(--header-bg, #111318);
      border-bottom: 1px solid var(--color-border, rgba(255,255,255,0.06));
      position: sticky;
      top: 0;
      z-index: 50;
      flex-shrink: 0;
    `;

        el.innerHTML = `
      <div style="display:flex;align-items:center;gap:20px;">
        <a href="homepage.html" style="display:flex;align-items:center;gap:8px;text-decoration:none;">
          <div style="width:22px;height:22px;border:1.5px solid var(--signal,#C8FF57);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            <div style="width:8px;height:8px;background:var(--signal,#C8FF57);"></div>
          </div>
          <span style="font-family:'Archivo Black',sans-serif;font-size:13px;color:var(--paper,#F4F2ED);letter-spacing:0.06em;text-transform:uppercase;">VisioNova</span>
          <span style="font-family:'DM Mono',monospace;font-size:9px;color:var(--signal,#C8FF57);border:1px solid rgba(200,255,87,0.3);padding:1px 5px;letter-spacing:0.08em;text-transform:uppercase;">v2.1</span>
        </a>
        <nav style="display:flex;align-items:center;">
          <a class="vn-nav-link" href="homepage.html"   style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted,#5C6270);text-decoration:none;padding:0 14px;height:46px;line-height:46px;border-left:1px solid var(--color-border);letter-spacing:0.06em;text-transform:uppercase;transition:all 0.15s;">Home</a>
          <a class="vn-nav-link" href="AnalysisDashboard.html" style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted,#5C6270);text-decoration:none;padding:0 14px;height:46px;line-height:46px;border-left:1px solid var(--color-border);letter-spacing:0.06em;text-transform:uppercase;transition:all 0.15s;">Analyze</a>
          <a class="vn-nav-link" href="FactCheckPage.html"     style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted,#5C6270);text-decoration:none;padding:0 14px;height:46px;line-height:46px;border-left:1px solid var(--color-border);letter-spacing:0.06em;text-transform:uppercase;transition:all 0.15s;">Fact Check</a>
          <a class="vn-nav-link" href="ReportPage.html"        style="font-family:'DM Mono',monospace;font-size:10px;color:var(--muted,#5C6270);text-decoration:none;padding:0 14px;height:46px;line-height:46px;border-left:1px solid var(--color-border);letter-spacing:0.06em;text-transform:uppercase;transition:all 0.15s;">Reports</a>
        </nav>
      </div>

      <div style="display:flex;align-items:center;gap:12px;">
        <div style="display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:var(--signal,#C8FF57);padding:4px 10px;border:1px solid rgba(200,255,87,0.2);background:rgba(200,255,87,0.06);letter-spacing:0.08em;text-transform:uppercase;">
          <div style="width:5px;height:5px;background:var(--signal,#C8FF57);border-radius:50%;animation:vn-blink 2.8s ease-in-out infinite;"></div>
          Online
        </div>
        <button id="themeToggle" style="display:flex;align-items:center;gap:6px;padding:5px 10px;background:transparent;border:1px solid var(--color-border);color:var(--muted,#5C6270);cursor:pointer;font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;transition:all 0.15s;" title="Toggle Theme">
          <span class="material-symbols-outlined theme-icon-dark" style="font-size:13px;color:#FFCC44;">light_mode</span>
          <span class="material-symbols-outlined theme-icon-light" style="font-size:13px;display:none;">dark_mode</span>
          Theme
        </button>
      </div>

      <style>
        @keyframes vn-blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
        .vn-nav-link:hover { color: var(--paper,#F4F2ED) !important; background: rgba(255,255,255,0.025) !important; }
        .vn-nav-link.active { color: var(--signal,#C8FF57) !important; border-bottom: 2px solid var(--signal,#C8FF57) !important; }
      </style>
    `;
    },

    injectSidebar() {
        const el = document.querySelector('aside[data-inject], aside:empty');
        if (!el) return;

        el.style.cssText = `
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 48px;
      background: var(--sidebar-bg, #111318);
      border-right: 1px solid var(--color-border, rgba(255,255,255,0.06));
      padding: 16px 0;
      gap: 2px;
      flex-shrink: 0;
      overflow: hidden;
      transition: width 0.22s ease;
      position: relative;
      z-index: 40;
    `;

        el.addEventListener('mouseenter', () => { el.style.width = '180px'; });
        el.addEventListener('mouseleave', () => { el.style.width = '48px'; });

        const itemBase = `
      display:flex;align-items:center;gap:10px;width:100%;
      padding:9px 14px;text-decoration:none;
      color:var(--ghost,#3A3F4A);
      font-family:'DM Mono',monospace;font-size:10px;white-space:nowrap;
      letter-spacing:0.05em;text-transform:uppercase;
      transition:all 0.15s;border-left:2px solid transparent;
    `;

        el.innerHTML = `
      <a class="vn-sb-link" href="homepage.html"           style="${itemBase}"><span class="material-symbols-outlined" style="font-size:16px;flex-shrink:0;">add_circle</span><span class="vn-sb-label" style="opacity:0;transition:opacity 0.18s;">New Scan</span></a>
      <div style="width:28px;height:1px;background:var(--color-border);margin:4px auto;transition:width 0.22s ease;" class="vn-sb-divider"></div>
      <a class="vn-sb-link" href="AnalysisDashboard.html"  style="${itemBase}"><span class="material-symbols-outlined" style="font-size:16px;flex-shrink:0;">dashboard</span><span class="vn-sb-label" style="opacity:0;transition:opacity 0.18s;">Dashboard</span></a>
      <a class="vn-sb-link" href="FactCheckPage.html"      style="${itemBase}"><span class="material-symbols-outlined" style="font-size:16px;flex-shrink:0;">shield</span><span class="vn-sb-label" style="opacity:0;transition:opacity 0.18s;">Fact Check</span></a>
      <a class="vn-sb-link" href="ReportPage.html"         style="${itemBase}"><span class="material-symbols-outlined" style="font-size:16px;flex-shrink:0;">description</span><span class="vn-sb-label" style="opacity:0;transition:opacity 0.18s;">Reports</span></a>
      <style>
        .vn-sb-link:hover { color:var(--paper,#F4F2ED)!important; background:rgba(255,255,255,0.025)!important; }
        .vn-sb-link.active { color:var(--signal,#C8FF57)!important; border-left-color:var(--signal,#C8FF57)!important; background:rgba(200,255,87,0.06)!important; }
      </style>
    `;

        el.addEventListener('mouseenter', () => {
            el.querySelectorAll('.vn-sb-label').forEach(l => l.style.opacity = '1');
            el.querySelectorAll('.vn-sb-divider').forEach(d => d.style.width = '148px');
        });
        el.addEventListener('mouseleave', () => {
            el.querySelectorAll('.vn-sb-label').forEach(l => l.style.opacity = '0');
            el.querySelectorAll('.vn-sb-divider').forEach(d => d.style.width = '28px');
        });
    },

    highlightCurrentNav() {
        const current = window.location.pathname.split('/').pop();
        document.querySelectorAll('.vn-nav-link').forEach(link => {
            if (link.getAttribute('href') === current) {
                link.style.color = 'var(--signal, #C8FF57)';
                link.style.borderBottom = '2px solid var(--signal, #C8FF57)';
            }
        });
        document.querySelectorAll('.vn-sb-link').forEach(link => {
            if (link.getAttribute('href') === current) {
                link.classList.add('active');
            }
        });
    }
};

document.addEventListener('DOMContentLoaded', () => Layout.init());