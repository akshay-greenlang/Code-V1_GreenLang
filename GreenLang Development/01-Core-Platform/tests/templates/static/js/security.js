/**
 * GreenLang Security Module
 *
 * Provides XSS protection, input sanitization, and secure DOM manipulation utilities.
 * This module implements OWASP best practices for web application security.
 */

/**
 * HTML entity encoding map for manual escaping
 */
const HTML_ENTITIES = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;'
};

/**
 * Escapes HTML entities to prevent XSS attacks
 *
 * @param {string} str - The string to escape
 * @returns {string} The escaped string
 */
function escapeHTML(str) {
    if (typeof str !== 'string') {
        return '';
    }
    return str.replace(/[&<>"'/]/g, char => HTML_ENTITIES[char]);
}

/**
 * Lightweight HTML sanitizer (fallback for when DOMPurify is not available)
 * Strips all HTML tags and only keeps text content
 *
 * @param {string} html - The HTML string to sanitize
 * @returns {string} The sanitized text
 */
function stripHTML(html) {
    if (typeof html !== 'string') {
        return '';
    }
    const doc = new DOMParser().parseFromString(html, 'text/html');
    return doc.body.textContent || '';
}

/**
 * DOMPurify configuration
 * Allows only safe HTML tags and attributes
 */
const DOMPURIFY_CONFIG = {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'u', 'p', 'br', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'a', 'pre', 'code', 'details', 'summary'],
    ALLOWED_ATTR: ['class', 'style', 'href'],
    ALLOW_DATA_ATTR: false,
    ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|sms|cid|xmpp):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
    KEEP_CONTENT: true,
    RETURN_DOM: false,
    RETURN_DOM_FRAGMENT: false,
    RETURN_TRUSTED_TYPE: false,
    FORCE_BODY: false,
    SANITIZE_DOM: true,
    IN_PLACE: false
};

/**
 * Lightweight DOMPurify-like sanitizer (when DOMPurify is not loaded)
 * This is a basic implementation that handles common XSS vectors
 *
 * @param {string} dirty - The HTML to sanitize
 * @param {object} config - Configuration options
 * @returns {string} Sanitized HTML
 */
function basicSanitize(dirty, config = {}) {
    if (typeof dirty !== 'string') {
        return '';
    }

    const allowedTags = config.ALLOWED_TAGS || DOMPURIFY_CONFIG.ALLOWED_TAGS;
    const allowedAttrs = config.ALLOWED_ATTR || DOMPURIFY_CONFIG.ALLOWED_ATTR;

    // Create a temporary DOM element
    const doc = new DOMParser().parseFromString(dirty, 'text/html');

    // Remove all script tags and event handlers
    const removeElements = doc.querySelectorAll('script, iframe, object, embed, link[rel="import"]');
    removeElements.forEach(el => el.remove());

    // Clean all elements
    const allElements = doc.body.querySelectorAll('*');
    allElements.forEach(el => {
        // Remove if tag not allowed
        if (!allowedTags.includes(el.tagName.toLowerCase())) {
            // Keep content but remove tag
            while (el.firstChild) {
                el.parentNode.insertBefore(el.firstChild, el);
            }
            el.remove();
            return;
        }

        // Remove dangerous attributes
        const attrs = Array.from(el.attributes);
        attrs.forEach(attr => {
            const attrName = attr.name.toLowerCase();

            // Remove event handlers
            if (attrName.startsWith('on')) {
                el.removeAttribute(attr.name);
                return;
            }

            // Remove javascript: URLs
            if (attrName === 'href' || attrName === 'src') {
                const value = attr.value.toLowerCase().trim();
                if (value.startsWith('javascript:') || value.startsWith('data:')) {
                    el.removeAttribute(attr.name);
                    return;
                }
            }

            // Only keep allowed attributes
            if (!allowedAttrs.includes(attrName)) {
                el.removeAttribute(attr.name);
            }
        });
    });

    return doc.body.innerHTML;
}

/**
 * Sanitizes HTML using DOMPurify (if available) or fallback sanitizer
 *
 * @param {string} dirty - The HTML to sanitize
 * @param {object} config - Optional DOMPurify configuration
 * @returns {string} Sanitized HTML
 */
function sanitizeHTML(dirty, config = DOMPURIFY_CONFIG) {
    // Use DOMPurify if available (we'll add it via CDN)
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(dirty, config);
    }

    // Fallback to basic sanitizer
    return basicSanitize(dirty, config);
}

/**
 * Safely sets text content (never uses innerHTML)
 *
 * @param {HTMLElement} element - The target element
 * @param {string} text - The text content to set
 */
function safeSetText(element, text) {
    if (!element || !(element instanceof HTMLElement)) {
        console.error('Invalid element provided to safeSetText');
        return;
    }
    element.textContent = escapeHTML(text);
}

/**
 * Safely sets HTML content with sanitization
 *
 * @param {HTMLElement} element - The target element
 * @param {string} html - The HTML content to set
 * @param {object} config - Optional sanitization config
 */
function safeSetHTML(element, html, config = DOMPURIFY_CONFIG) {
    if (!element || !(element instanceof HTMLElement)) {
        console.error('Invalid element provided to safeSetHTML');
        return;
    }

    const sanitized = sanitizeHTML(html, config);
    element.innerHTML = sanitized;
}

/**
 * Creates a safe DOM element with sanitized content
 *
 * @param {string} tag - The tag name (e.g., 'div', 'span')
 * @param {object} attributes - Object of attributes to set
 * @param {string} content - Text or HTML content
 * @param {boolean} isHTML - Whether content is HTML (will be sanitized)
 * @returns {HTMLElement} The created element
 */
function createSafeElement(tag, attributes = {}, content = '', isHTML = false) {
    const element = document.createElement(tag);

    // Set attributes safely
    for (const [key, value] of Object.entries(attributes)) {
        // Prevent setting dangerous attributes
        if (key.toLowerCase().startsWith('on')) {
            console.warn(`Blocked attempt to set event handler attribute: ${key}`);
            continue;
        }

        // Validate href and src attributes
        if ((key === 'href' || key === 'src') && typeof value === 'string') {
            const lowercaseValue = value.toLowerCase().trim();
            if (lowercaseValue.startsWith('javascript:') || lowercaseValue.startsWith('data:text/html')) {
                console.warn(`Blocked dangerous URL in ${key}: ${value}`);
                continue;
            }
        }

        element.setAttribute(key, value);
    }

    // Set content safely
    if (content) {
        if (isHTML) {
            safeSetHTML(element, content);
        } else {
            safeSetText(element, content);
        }
    }

    return element;
}

/**
 * Validates and sanitizes numeric input
 *
 * @param {any} value - The value to validate
 * @param {object} options - Validation options (min, max, default)
 * @returns {number} The validated number
 */
function sanitizeNumber(value, options = {}) {
    const {
        min = -Infinity,
        max = Infinity,
        defaultValue = 0,
        allowFloat = true
    } = options;

    const num = allowFloat ? parseFloat(value) : parseInt(value, 10);

    if (isNaN(num)) {
        return defaultValue;
    }

    if (num < min) {
        return min;
    }

    if (num > max) {
        return max;
    }

    return num;
}

/**
 * Validates and sanitizes string input
 *
 * @param {any} value - The value to validate
 * @param {object} options - Validation options
 * @returns {string} The validated string
 */
function sanitizeString(value, options = {}) {
    const {
        maxLength = 1000,
        allowedPattern = null,
        defaultValue = '',
        trim = true
    } = options;

    if (typeof value !== 'string') {
        return defaultValue;
    }

    let sanitized = trim ? value.trim() : value;

    // Limit length
    if (sanitized.length > maxLength) {
        sanitized = sanitized.substring(0, maxLength);
    }

    // Validate against pattern if provided
    if (allowedPattern && !allowedPattern.test(sanitized)) {
        return defaultValue;
    }

    return sanitized;
}

/**
 * Validates URL parameters and returns sanitized values
 *
 * @param {string} paramName - The URL parameter name
 * @param {object} options - Validation options
 * @returns {string|null} The sanitized parameter value or null
 */
function getSafeURLParam(paramName, options = {}) {
    const params = new URLSearchParams(window.location.search);
    const value = params.get(paramName);

    if (value === null) {
        return null;
    }

    return sanitizeString(value, options);
}

/**
 * Content Security Policy helper
 * Validates that inline scripts/styles are not being used
 */
const CSP = {
    /**
     * Checks if CSP is properly configured
     */
    checkConfiguration() {
        const metaCSP = document.querySelector('meta[http-equiv="Content-Security-Policy"]');
        if (!metaCSP) {
            console.warn('Content Security Policy not found in meta tags. Add CSP headers for better security.');
            return false;
        }
        return true;
    },

    /**
     * Recommended CSP policy for GreenLang applications
     */
    recommended: "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';"
};

/**
 * Input validation patterns
 */
const ValidationPatterns = {
    email: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/,
    alphanumeric: /^[a-zA-Z0-9]+$/,
    alphanumericWithSpaces: /^[a-zA-Z0-9\s]+$/,
    numeric: /^[0-9]+$/,
    float: /^[0-9]*\.?[0-9]+$/,
    url: /^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/,
    uuid: /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
};

/**
 * Form validation helper
 *
 * @param {HTMLFormElement} form - The form to validate
 * @param {object} rules - Validation rules for each field
 * @returns {object} { isValid: boolean, errors: object }
 */
function validateForm(form, rules) {
    const errors = {};
    let isValid = true;

    for (const [fieldName, fieldRules] of Object.entries(rules)) {
        const field = form.elements[fieldName];
        if (!field) continue;

        const value = field.value;

        // Required validation
        if (fieldRules.required && !value.trim()) {
            errors[fieldName] = 'This field is required';
            isValid = false;
            continue;
        }

        // Pattern validation
        if (fieldRules.pattern && value && !fieldRules.pattern.test(value)) {
            errors[fieldName] = fieldRules.message || 'Invalid format';
            isValid = false;
            continue;
        }

        // Min/max length
        if (fieldRules.minLength && value.length < fieldRules.minLength) {
            errors[fieldName] = `Minimum length is ${fieldRules.minLength}`;
            isValid = false;
            continue;
        }

        if (fieldRules.maxLength && value.length > fieldRules.maxLength) {
            errors[fieldName] = `Maximum length is ${fieldRules.maxLength}`;
            isValid = false;
            continue;
        }

        // Custom validator
        if (fieldRules.validator && !fieldRules.validator(value)) {
            errors[fieldName] = fieldRules.message || 'Invalid value';
            isValid = false;
        }
    }

    return { isValid, errors };
}

// Export security utilities
window.GreenLangSecurity = {
    escapeHTML,
    stripHTML,
    sanitizeHTML,
    safeSetText,
    safeSetHTML,
    createSafeElement,
    sanitizeNumber,
    sanitizeString,
    getSafeURLParam,
    CSP,
    ValidationPatterns,
    validateForm,
    DOMPURIFY_CONFIG
};

// Check CSP on load
document.addEventListener('DOMContentLoaded', () => {
    CSP.checkConfiguration();
});
