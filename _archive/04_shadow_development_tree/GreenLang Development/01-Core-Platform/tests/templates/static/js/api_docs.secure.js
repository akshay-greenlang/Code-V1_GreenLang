/**
 * API Documentation Interactive Functions - SECURE VERSION
 *
 * Security improvements:
 * - Lines 6, 55, 58, 70, 90-100, 169-171: Replaced innerHTML with safe alternatives
 * - All user inputs sanitized
 * - All outputs properly escaped
 * - No inline event handlers
 * - URL validation for fetch calls
 */

// Import security utilities
const {
    escapeHTML,
    sanitizeHTML,
    safeSetText,
    safeSetHTML,
    createSafeElement,
    sanitizeNumber,
    sanitizeString,
    ValidationPatterns
} = window.GreenLangSecurity;

/**
 * Try endpoint functionality - SECURE VERSION
 */
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
        let response;
        let url;

        switch(sanitizedEndpoint) {
            case 'calculate':
                url = '/api/calculate';
                response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        fuels: [
                            { type: 'electricity', consumption: 1000, unit: 'kWh' },
                            { type: 'natural_gas', consumption: 50, unit: 'therms' }
                        ],
                        region: 'US',
                        building_info: {
                            area: 10000,
                            type: 'commercial_office'
                        }
                    })
                });
                break;

            case 'quick-calc':
                // Validate query parameters
                url = '/api/quick-calc?electricity=1000&gas=50';
                response = await fetch(url);
                break;

            case 'agents':
                url = '/api/agents';
                response = await fetch(url);
                break;

            case 'fuel-types':
                url = '/api/fuel-types';
                response = await fetch(url);
                break;

            case 'regions':
                url = '/api/regions';
                response = await fetch(url);
                break;

            case 'building-types':
                url = '/api/building-types';
                response = await fetch(url);
                break;

            default:
                throw new Error('Invalid endpoint');
        }

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

/**
 * Run interactive example - SECURE VERSION
 */
async function runExample() {
    // Validate and sanitize inputs
    const fuelTypeSelect = document.getElementById('example-fuel-type');
    const fuelType = sanitizeString(fuelTypeSelect.value, {
        maxLength: 50,
        allowedPattern: /^[a-z_]+$/,
        defaultValue: 'electricity'
    });

    const consumptionInput = document.getElementById('example-consumption');
    const consumption = sanitizeNumber(consumptionInput.value, {
        min: 0,
        max: 1000000,
        defaultValue: 1000,
        allowFloat: true
    });

    const unitSelect = document.getElementById('example-unit');
    const unit = sanitizeString(unitSelect.value, {
        maxLength: 20,
        allowedPattern: /^[a-zA-Z_]+$/,
        defaultValue: 'kWh'
    });

    const regionSelect = document.getElementById('example-region');
    const region = sanitizeString(regionSelect.value, {
        maxLength: 10,
        allowedPattern: /^[A-Z]{2,3}$/,
        defaultValue: 'US'
    });

    const responseContainer = document.getElementById('example-response');

    // Clear existing content
    while (responseContainer.firstChild) {
        responseContainer.removeChild(responseContainer.firstChild);
    }

    // Show processing state
    const processingDiv = createSafeElement('div', {}, 'Processing...', false);
    responseContainer.appendChild(processingDiv);
    responseContainer.classList.add('show');

    try {
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                fuels: [
                    { type: fuelType, consumption: consumption, unit: unit }
                ],
                region: region
            })
        });

        const data = await response.json();

        // Clear processing
        while (responseContainer.firstChild) {
            responseContainer.removeChild(responseContainer.firstChild);
        }

        if (data.success) {
            // Success message
            const successDiv = createSafeElement('div', {
                style: 'color: green; margin-bottom: 10px;'
            }, '✓ Success', false);
            responseContainer.appendChild(successDiv);

            // Total emissions
            const totalKg = sanitizeNumber(data.emissions.total_co2e_kg, {
                min: 0,
                max: 1000000000,
                allowFloat: true
            });
            const emissionsDiv = createSafeElement('div', {},
                `Total Emissions: ${totalKg.toFixed(2)} kg CO2e`, false);
            responseContainer.appendChild(emissionsDiv);

            // In tons
            const totalTons = sanitizeNumber(data.emissions.total_co2e_tons, {
                min: 0,
                max: 1000000,
                allowFloat: true
            });
            const tonsDiv = createSafeElement('div', {},
                `In Tons: ${totalTons.toFixed(4)} metric tons`, false);
            responseContainer.appendChild(tonsDiv);

            // Full response in details
            const details = createSafeElement('details', {
                style: 'margin-top: 10px;'
            });
            const summary = createSafeElement('summary', {}, 'Full Response', false);
            details.appendChild(summary);

            const jsonString = JSON.stringify(data, null, 2);
            const pre = createSafeElement('pre', {}, jsonString, false);
            details.appendChild(pre);

            responseContainer.appendChild(details);
        } else {
            const errorMsg = sanitizeString(data.error || 'Unknown error', {
                maxLength: 500
            });
            const errorDiv = createSafeElement('div', {
                style: 'color: red;'
            }, `Error: ${errorMsg}`, false);
            responseContainer.appendChild(errorDiv);
        }

    } catch (error) {
        // Clear processing
        while (responseContainer.firstChild) {
            responseContainer.removeChild(responseContainer.firstChild);
        }

        const errorMsg = sanitizeString(error.message, { maxLength: 500 });
        const errorDiv = createSafeElement('div', {
            style: 'color: red;'
        }, `Error: ${errorMsg}`, false);
        responseContainer.appendChild(errorDiv);
    }
}

/**
 * Generate code snippets - SECURE VERSION
 */
function generateCode() {
    // Validate and sanitize inputs
    const fuelTypeSelect = document.getElementById('example-fuel-type');
    const fuelType = sanitizeString(fuelTypeSelect.value, {
        maxLength: 50,
        allowedPattern: /^[a-z_]+$/,
        defaultValue: 'electricity'
    });

    const consumptionInput = document.getElementById('example-consumption');
    const consumption = sanitizeNumber(consumptionInput.value, {
        min: 0,
        max: 1000000,
        defaultValue: 1000,
        allowFloat: true
    });

    const unitSelect = document.getElementById('example-unit');
    const unit = sanitizeString(unitSelect.value, {
        maxLength: 20,
        allowedPattern: /^[a-zA-Z_]+$/,
        defaultValue: 'kWh'
    });

    const regionSelect = document.getElementById('example-region');
    const region = sanitizeString(regionSelect.value, {
        maxLength: 10,
        allowedPattern: /^[A-Z]{2,3}$/,
        defaultValue: 'US'
    });

    // Python snippet - using template literals with escaped values
    const pythonCode = `from greenlang.sdk import GreenLangClient

client = GreenLangClient()

result = client.calculate_emissions(
    fuel_type="${escapeHTML(fuelType)}",
    consumption=${consumption},
    unit="${escapeHTML(unit)}",
    region="${escapeHTML(region)}"
)

print(f"CO2 emissions: {result['data']['co2e_kg']} kg")`;

    // JavaScript snippet
    const jsCode = `async function calculateEmissions() {
  const response = await fetch('http://localhost:5000/api/calculate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      fuels: [
        {
          type: '${escapeHTML(fuelType)}',
          consumption: ${consumption},
          unit: '${escapeHTML(unit)}'
        }
      ],
      region: '${escapeHTML(region)}'
    })
  });

  const data = await response.json();
  console.log('CO2 emissions:', data.emissions.total_co2e_kg, 'kg');
  return data;
}

calculateEmissions();`;

    // cURL snippet
    const curlCode = `curl -X POST http://localhost:5000/api/calculate \\
  -H "Content-Type: application/json" \\
  -d '{
    "fuels": [
      {
        "type": "${escapeHTML(fuelType)}",
        "consumption": ${consumption},
        "unit": "${escapeHTML(unit)}"
      }
    ],
    "region": "${escapeHTML(region)}"
  }'`;

    // Set code snippets safely using textContent
    safeSetText(document.getElementById('python-snippet'), pythonCode);
    safeSetText(document.getElementById('javascript-snippet'), jsCode);
    safeSetText(document.getElementById('curl-snippet'), curlCode);
}

/**
 * Tab functionality - SECURE VERSION
 */
function showTab(language, event) {
    // Validate language parameter
    const allowedLanguages = ['python', 'javascript', 'curl'];
    const sanitizedLanguage = sanitizeString(language, {
        maxLength: 20,
        allowedPattern: /^[a-z]+$/,
        defaultValue: 'python'
    });

    if (!allowedLanguages.includes(sanitizedLanguage)) {
        console.error('Invalid language tab:', language);
        return;
    }

    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });

    // Show selected tab
    const tabContent = document.getElementById(`${sanitizedLanguage}-code`);
    if (tabContent) {
        tabContent.classList.add('active');
    }

    if (event && event.target) {
        event.target.classList.add('active');
    }
}

/**
 * Initialize on page load - SECURE VERSION
 */
document.addEventListener('DOMContentLoaded', function() {
    // Set default code snippets
    generateCode();

    // Update units based on fuel type
    const fuelTypeSelect = document.getElementById('example-fuel-type');
    if (fuelTypeSelect) {
        fuelTypeSelect.addEventListener('change', function() {
            const unitSelect = document.getElementById('example-unit');
            if (!unitSelect) return;

            const fuelType = sanitizeString(this.value, {
                maxLength: 50,
                allowedPattern: /^[a-z_]+$/,
                defaultValue: 'electricity'
            });

            // Clear existing options safely
            while (unitSelect.firstChild) {
                unitSelect.removeChild(unitSelect.firstChild);
            }

            // Define allowed units for each fuel type
            const units = {
                'electricity': ['kWh', 'MWh'],
                'natural_gas': ['therms', 'cubic_meters', 'mcf'],
                'diesel': ['gallons', 'liters'],
                'gasoline': ['gallons', 'liters']
            };

            const fuelUnits = units[fuelType] || ['gallons', 'liters'];
            fuelUnits.forEach(unit => {
                const option = createSafeElement('option', {
                    value: unit
                }, unit, false);
                unitSelect.appendChild(option);
            });

            // Update code snippets with new units
            generateCode();
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();

            // Validate href
            const href = this.getAttribute('href');
            if (!href || href.length > 100) {
                console.error('Invalid anchor href');
                return;
            }

            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

/**
 * Copy code functionality - SECURE VERSION
 */
function copyCode(elementId, event) {
    // Validate element ID
    const sanitizedId = sanitizeString(elementId, {
        maxLength: 50,
        allowedPattern: /^[a-z-]+$/,
        defaultValue: ''
    });

    if (!sanitizedId) {
        console.error('Invalid element ID');
        return;
    }

    const codeElement = document.getElementById(sanitizedId);
    if (!codeElement) {
        console.error('Code element not found:', sanitizedId);
        return;
    }

    const text = codeElement.textContent;

    navigator.clipboard.writeText(text).then(() => {
        // Show success message
        if (event && event.target) {
            const button = event.target;
            const originalText = button.textContent;
            button.textContent = 'Copied!';
            button.style.background = '#4caf50';

            setTimeout(() => {
                button.textContent = originalText;
                button.style.background = '';
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy code to clipboard');
    });
}
