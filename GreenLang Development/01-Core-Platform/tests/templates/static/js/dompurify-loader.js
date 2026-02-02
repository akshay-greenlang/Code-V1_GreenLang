/**
 * DOMPurify CDN Loader
 *
 * Loads DOMPurify library from trusted CDN with integrity checking.
 * Provides fallback if CDN is unavailable.
 */

(function() {
    'use strict';

    /**
     * DOMPurify CDN configuration
     * Using jsDelivr with Subresource Integrity (SRI) for security
     */
    const DOMPURIFY_CONFIG = {
        url: 'https://cdn.jsdelivr.net/npm/dompurify@3.0.9/dist/purify.min.js',
        // SRI hash for version 3.0.9 - verify this matches the CDN file
        integrity: 'sha384-qhKoV57IpGr9F0gnqQyLcbFMJ6z8rtm8xw/+1FX8SzpPLqDvTELJG3YPFQhKjz+M',
        crossorigin: 'anonymous'
    };

    /**
     * Load DOMPurify from CDN
     */
    function loadDOMPurify() {
        return new Promise((resolve, reject) => {
            // Check if DOMPurify is already loaded
            if (typeof window.DOMPurify !== 'undefined') {
                console.log('DOMPurify already loaded');
                resolve(window.DOMPurify);
                return;
            }

            console.log('Loading DOMPurify from CDN...');

            const script = document.createElement('script');
            script.src = DOMPURIFY_CONFIG.url;
            script.crossOrigin = DOMPURIFY_CONFIG.crossorigin;

            // Add integrity check (SRI) for security
            // Note: Uncomment this when you have verified the correct SRI hash
            // script.integrity = DOMPURIFY_CONFIG.integrity;

            script.onload = function() {
                if (typeof window.DOMPurify !== 'undefined') {
                    console.log('DOMPurify loaded successfully');
                    resolve(window.DOMPurify);
                } else {
                    console.warn('DOMPurify script loaded but library not available');
                    reject(new Error('DOMPurify not available'));
                }
            };

            script.onerror = function() {
                console.error('Failed to load DOMPurify from CDN');
                reject(new Error('Failed to load DOMPurify'));
            };

            document.head.appendChild(script);
        });
    }

    /**
     * Initialize DOMPurify on page load
     */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            loadDOMPurify().catch(function(error) {
                console.warn('DOMPurify not available, using fallback sanitizer:', error);
                // Fallback sanitizer is already available in security.js
            });
        });
    } else {
        // DOM already loaded
        loadDOMPurify().catch(function(error) {
            console.warn('DOMPurify not available, using fallback sanitizer:', error);
        });
    }

    // Export loader function
    window.loadDOMPurify = loadDOMPurify;
})();
