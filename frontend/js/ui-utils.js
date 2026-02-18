/**
 * VisioNova UI Utilities
 * Shared helper functions for UI interactions, tabs, and layout.
 */

const UI = {
    /**
     * Initialize tab system
     * @param {string} defaultTab - ID of the tab to open by default
     * @param {Function} onTabSwitch - Optional callback when tab is switched
     */
    initTabs: function(defaultTab, onTabSwitch) {
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const tabName = btn.dataset.tab || btn.id.replace('tab-', '');
                this.switchTab(tabName, onTabSwitch);
            });
        });

        // Open default tab if specified
        if (defaultTab) {
            this.switchTab(defaultTab, onTabSwitch);
        }
    },

    /**
     * Switch between tabs
     * @param {string} tabName - Name/ID of the tab to switch to
     * @param {Function} onTabSwitch - Optional callback
     */
    switchTab: function(tabName, onTabSwitch) {
        // Update tab buttons
        const tabButtons = document.querySelectorAll('.tab-btn');
        tabButtons.forEach(btn => {
            const btnTabName = btn.dataset.tab || btn.id.replace('tab-', '');
            if (btnTabName === tabName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Callback to render content
        if (typeof onTabSwitch === 'function') {
            onTabSwitch(tabName);
        }
    },

    /**
     * Show a notification message
     * @param {string} message - Message text
     * @param {'success'|'error'|'warning'|'info'} type - Notification type
     */
    showNotification: function(message, type = 'info') {
        const id = 'visio-notification-' + Date.now();
        const notification = document.createElement('div');
        notification.id = id;
        
        let bgClass, icon;
        switch(type) {
            case 'success': bgClass = 'bg-accent-green'; icon = 'check_circle'; break;
            case 'error': bgClass = 'bg-accent-danger'; icon = 'error'; break;
            case 'warning': bgClass = 'bg-accent-warning'; icon = 'warning'; break;
            default: bgClass = 'bg-primary'; icon = 'info';
        }

        notification.className = `fixed bottom-4 right-4 z-50 flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl text-white transform translate-y-10 opacity-0 transition-all duration-300 ${bgClass}`;
        notification.innerHTML = `
            <span class="material-symbols-outlined">${icon}</span>
            <span class="font-medium text-sm">${message}</span>
        `;

        document.body.appendChild(notification);

        // Animate in
        requestAnimationFrame(() => {
            notification.classList.remove('translate-y-10', 'opacity-0');
        });

        // Remove after delay
        setTimeout(() => {
            notification.classList.add('translate-y-10', 'opacity-0');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    },

    /**
     * Format a number with commas
     * @param {number} num 
     */
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    /**
     * Copy text to clipboard
     * @param {string} text 
     */
    copyToClipboard: async function(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showNotification('Copied to clipboard', 'success');
        } catch (err) {
            console.error('Failed to copy keys: ', err);
            this.showNotification('Failed to copy', 'error');
        }
    }
};

// Make globally available
window.UI = UI;
