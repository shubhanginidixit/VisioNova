/**
 * VisioNova - Theme Toggle (Dark/Light Mode)
 * Add this script to all pages that have the theme toggle button.
 */

(function () {
    'use strict';

    const THEME_KEY = 'visioNova_theme';

    /**
     * Get the current theme from localStorage or system preference.
     * @returns {'dark' | 'light'}
     */
    function getStoredTheme() {
        const stored = localStorage.getItem(THEME_KEY);
        if (stored === 'light' || stored === 'dark') {
            return stored;
        }
        // Default to dark if no preference
        return 'dark';
    }

    /**
     * Apply the theme to the document.
     * @param {'dark' | 'light'} theme
     */
    function applyTheme(theme) {
        const html = document.documentElement;
        const body = document.body;

        if (theme === 'light') {
            html.classList.remove('dark');
            html.classList.add('light');
            body.classList.remove('dark');
            body.classList.add('light');
        } else {
            html.classList.remove('light');
            html.classList.add('dark');
            body.classList.remove('light');
            body.classList.add('dark');
        }

        // Update button icons
        updateToggleIcons(theme);
        localStorage.setItem(THEME_KEY, theme);
    }

    /**
     * Update the visibility of sun/moon icons.
     * @param {'dark' | 'light'} theme
     */
    function updateToggleIcons(theme) {
        const darkIcons = document.querySelectorAll('.theme-icon-dark');
        const lightIcons = document.querySelectorAll('.theme-icon-light');

        if (theme === 'dark') {
            // In dark mode, show sun icon (to switch to light)
            darkIcons.forEach(icon => icon.classList.remove('hidden'));
            lightIcons.forEach(icon => icon.classList.add('hidden'));
        } else {
            // In light mode, show moon icon (to switch to dark)
            darkIcons.forEach(icon => icon.classList.add('hidden'));
            lightIcons.forEach(icon => icon.classList.remove('hidden'));
        }
    }

    /**
     * Toggle between dark and light mode.
     */
    function toggleTheme() {
        const currentTheme = getStoredTheme();
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        applyTheme(newTheme);
    }

    // Initialize on page load
    function init() {
        // Theme class is already applied by blocking script in <head>
        // Just sync body class and update icons
        const currentTheme = getStoredTheme();
        const body = document.body;
        
        // Ensure body has the correct class
        if (currentTheme === 'light') {
            body.classList.remove('dark');
            body.classList.add('light');
        } else {
            body.classList.remove('light');
            body.classList.add('dark');
        }
        
        // Update toggle button icons
        updateToggleIcons(currentTheme);

        // Setup toggle button
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', toggleTheme);
        }
    }

    // Run on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
