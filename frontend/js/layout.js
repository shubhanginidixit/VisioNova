/**
 * VisioNova Layout Manager
 * Injects common UI components (Header, Sidebar) into pages to avoid duplication.
 */

const Layout = {
    /**
     * Inject the layout into the current page
     */
    init: function () {
        this.injectHeader();
        this.injectSidebar();
        this.highlightCurrentNav();
    },

    injectHeader: function () {
        const headerPlaceholder = document.querySelector('header');
        if (!headerPlaceholder) return;

        // If header already has content, assume it's custom and don't overwrite
        if (headerPlaceholder.children.length > 0 && !headerPlaceholder.hasAttribute('data-inject')) return;

        headerPlaceholder.className = "z-50 flex items-center justify-between whitespace-nowrap border-b border-[var(--color-border)] bg-[var(--header-bg)] backdrop-blur-md px-6 py-3 sticky top-0";
        headerPlaceholder.innerHTML = `
            <div class="flex items-center gap-4 text-white">
                <div class="size-8 text-primary">
                    <span class="material-symbols-outlined !text-[32px]">hub</span>
                </div>
                <h2 class="text-white text-xl font-bold leading-tight tracking-[-0.015em] hidden sm:block">VisioNova</h2>
            </div>
            <div class="flex flex-1 justify-center hidden md:flex">
                <nav class="flex items-center gap-1 bg-[var(--color-bg-card)]/50 p-1 rounded-full border border-[var(--color-border)] ${window.location.pathname.includes('homepage.html') ? 'hidden' : ''}">
                    <a class="nav-link text-[var(--color-text-muted)] hover:text-white px-4 py-1.5 rounded-full text-sm font-medium transition-colors"
                        href="homepage.html">Home</a>
                    <a class="nav-link text-[var(--color-text-muted)] hover:text-white px-4 py-1.5 rounded-full text-sm font-medium transition-colors"
                        href="AnalysisDashboard.html">Analyze</a>
                    <a class="nav-link text-[var(--color-text-muted)] hover:text-white px-4 py-1.5 rounded-full text-sm font-medium transition-colors"
                        href="FactCheckPage.html">Fact Check</a>
                </nav>
            </div>
            <div class="flex items-center gap-3">
                <button id="themeToggle"
                    class="flex items-center gap-2 px-3 py-2 rounded-xl bg-[var(--color-bg-card)] hover:bg-[var(--color-border-hover)] transition-colors border border-[var(--color-border)]"
                    title="Toggle Dark/Light Mode">
                    <span class="material-symbols-outlined !text-[20px] theme-icon-dark text-yellow-400">light_mode</span>
                    <span
                        class="material-symbols-outlined !text-[20px] theme-icon-light hidden text-slate-700">dark_mode</span>
                </button>
            </div>
        `;
    },

    injectSidebar: function () {
        const sidebarPlaceholder = document.querySelector('aside');
        if (!sidebarPlaceholder) return;

        // If sidebar already has content, assume it's custom and don't overwrite
        if (sidebarPlaceholder.children.length > 0 && !sidebarPlaceholder.hasAttribute('data-inject')) return;

        sidebarPlaceholder.className = "hidden md:flex w-20 flex-col items-center justify-between bg-[var(--sidebar-bg)] border-r border-[var(--color-border)] py-6 z-40 hover:w-64 transition-all duration-300 group/sidebar absolute md:relative h-full";
        sidebarPlaceholder.innerHTML = `
            <div class="flex flex-col gap-6 w-full px-3">
                <div class="w-full flex flex-col gap-1">
                    <a href="homepage.html"
                        class="flex items-center gap-4 px-3 py-3 rounded-xl bg-primary text-white w-full justify-start overflow-hidden whitespace-nowrap shadow-glow">
                        <span class="material-symbols-outlined shrink-0">add_circle</span>
                        <span class="text-sm font-semibold opacity-0 group-hover/sidebar:opacity-100 transition-opacity duration-300">New Scan</span>
                    </a>
                </div>
                <div class="w-full h-px bg-[var(--color-border)]"></div>
                <div class="flex flex-col gap-2 w-full">
                    <a href="AnalysisDashboard.html"
                        class="sidebar-link flex items-center gap-4 px-3 py-3 rounded-xl text-[var(--color-text-muted)] hover:text-white hover:bg-[var(--color-bg-card)] w-full justify-start overflow-hidden whitespace-nowrap transition-colors">
                        <span class="material-symbols-outlined shrink-0">dashboard</span>
                        <span class="text-sm font-medium opacity-0 group-hover/sidebar:opacity-100 transition-opacity duration-300">Dashboard</span>
                    </a>

                    <a href="FactCheckPage.html"
                        class="sidebar-link flex items-center gap-4 px-3 py-3 rounded-xl text-[var(--color-text-muted)] hover:text-white hover:bg-[var(--color-bg-card)] w-full justify-start overflow-hidden whitespace-nowrap transition-colors">
                        <span class="material-symbols-outlined shrink-0">shield</span>
                        <span class="text-sm font-medium opacity-0 group-hover/sidebar:opacity-100 transition-opacity duration-300">Threat Intel</span>
                    </a>
                </div>
            </div>

        `;
    },

    highlightCurrentNav: function () {
        const currentPath = window.location.pathname.split('/').pop();

        // Highlight Header Nav
        document.querySelectorAll('.nav-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('bg-primary/20', 'text-white');
                link.classList.remove('text-[var(--color-text-muted)]');
            }
        });

        // Highlight Sidebar Nav
        document.querySelectorAll('.sidebar-link').forEach(link => {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('text-white', 'bg-[var(--color-bg-card)]');
                link.classList.remove('text-[var(--color-text-muted)]');
            }
        });
    }
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => Layout.init());
