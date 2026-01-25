/**
 * GreenLang Web Interface JavaScript - SECURE VERSION
 *
 * This version implements comprehensive XSS protection using:
 * - Input sanitization and validation
 * - Safe DOM manipulation (no innerHTML with user data)
 * - HTML entity escaping
 * - DOMPurify for trusted HTML content
 *
 * Security improvements over original:
 * - Lines 106, 132, 134, 140, 153, 171: Replaced innerHTML with safe alternatives
 * - All numeric inputs validated and sanitized
 * - All string outputs properly escaped
 * - No inline event handlers (onclick removed)
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
    ValidationPatterns,
    validateForm
} = window.GreenLangSecurity;

document.addEventListener('DOMContentLoaded', function() {
    loadAgents();

    // Calculate button
    document.getElementById('calculate-btn').addEventListener('click', calculateEmissions);

    // Add custom fuel button
    document.getElementById('add-fuel-btn').addEventListener('click', addCustomFuel);

    // Enter key support
    document.querySelectorAll('input').forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') calculateEmissions();
        });
    });
});

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

    // Collect and validate custom fuels
    const customFuels = [];
    const allowedFuelTypes = ['diesel', 'gasoline', 'propane', 'fuel_oil', 'coal'];
    const allowedUnits = ['gallons', 'liters', 'kg', 'tons'];

    document.querySelectorAll('.custom-fuel-row').forEach(row => {
        const typeSelect = row.querySelector('.fuel-type');
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

        const consumption = sanitizeNumber(row.querySelector('.fuel-consumption').value, {
            min: 0,
            max: 1000000,
            defaultValue: 0,
            allowFloat: true
        });

        const unitSelect = row.querySelector('.fuel-unit');
        const unit = sanitizeString(unitSelect.value, {
            maxLength: 20,
            allowedPattern: /^[a-z_]+$/,
            defaultValue: 'gallons'
        });

        // Ensure unit is in allowed list
        if (!allowedUnits.includes(unit)) {
            console.warn(`Invalid unit: ${unit}`);
            return;
        }

        if (consumption > 0) {
            customFuels.push({ type, consumption, unit });
        }
    });

    // Build request data
    const fuels = [];
    if (electricity > 0) {
        fuels.push({ type: 'electricity', consumption: electricity, unit: 'kWh' });
    }
    if (gas > 0) {
        fuels.push({ type: 'natural_gas', consumption: gas, unit: 'therms' });
    }
    fuels.push(...customFuels);

    if (fuels.length === 0) {
        // Use safe alert or better UX with modal
        showNotification('Please enter at least one fuel consumption value', 'warning');
        return;
    }

    const requestData = {
        fuels: fuels,
        region: 'US',
        period_months: 1
    };

    if (buildingArea > 0) {
        requestData.building_info = {
            area: buildingArea,
            type: buildingType
        };
    }

    try {
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            const errorMsg = sanitizeString(data.error || 'Unknown error', {
                maxLength: 500
            });
            showNotification('Error: ' + errorMsg, 'error');
        }
    } catch (error) {
        const errorMsg = sanitizeString(error.message, {
            maxLength: 500
        });
        showNotification('Error calculating emissions: ' + errorMsg, 'error');
    }
}

/**
 * Displays results with secure DOM manipulation
 * SECURITY: All innerHTML replaced with safe alternatives
 */
function displayResults(data) {
    // Show results section
    document.getElementById('results').style.display = 'block';

    // Display totals using textContent (safe)
    const totalEmissions = sanitizeNumber(data.emissions.total_co2e_tons, {
        min: 0,
        max: 1000000000,
        defaultValue: 0,
        allowFloat: true
    });

    const totalKg = sanitizeNumber(data.emissions.total_co2e_kg, {
        min: 0,
        max: 1000000000000,
        defaultValue: 0,
        allowFloat: true
    });

    document.getElementById('total-emissions').textContent = totalEmissions.toFixed(3);
    document.getElementById('total-kg').textContent = totalKg.toFixed(2);

    // Display breakdown - SECURE VERSION (replaces line 106)
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

            // Create source span
            const sourceSpan = createSafeElement('span', {}, source, false);
            breakdownItem.appendChild(sourceSpan);

            // Create value span
            const valueText = `${co2eTons.toFixed(3)} tons (${percentage.toFixed(1)}%)`;
            const valueSpan = createSafeElement('span', {}, valueText, false);
            breakdownItem.appendChild(valueSpan);

            breakdownDiv.appendChild(breakdownItem);
        });
    }

    // Display benchmark if available - SECURE VERSION (replaces lines 132, 134)
    const benchmarkDiv = document.getElementById('benchmark');

    // Clear existing content safely
    while (benchmarkDiv.firstChild) {
        benchmarkDiv.removeChild(benchmarkDiv.firstChild);
    }

    if (data.benchmark) {
        // Sanitize benchmark data
        const rating = sanitizeString(data.benchmark.rating, {
            maxLength: 50,
            allowedPattern: /^[A-Za-z\s]+$/,
            defaultValue: 'Unknown'
        });

        const carbonIntensity = sanitizeNumber(data.benchmark.carbon_intensity, {
            min: 0,
            max: 10000,
            allowFloat: true
        });

        const percentile = sanitizeNumber(data.benchmark.percentile, {
            min: 0,
            max: 100,
            allowFloat: true
        });

        // Determine rating class
        let ratingClass = 'benchmark-average';
        if (rating === 'Excellent' || rating === 'Good') {
            ratingClass = 'benchmark-good';
        } else if (rating === 'Poor' || rating === 'Below Average') {
            ratingClass = 'benchmark-poor';
        }

        // Create benchmark content
        const heading = createSafeElement('h3', {}, 'Benchmark Analysis', false);
        benchmarkDiv.appendChild(heading);

        // Rating paragraph
        const ratingP = createSafeElement('p', {});
        const ratingLabel = document.createTextNode('Rating: ');
        const ratingSpan = createSafeElement('span', { class: ratingClass }, rating, false);
        ratingP.appendChild(ratingLabel);
        ratingP.appendChild(ratingSpan);
        benchmarkDiv.appendChild(ratingP);

        // Carbon intensity paragraph
        const intensityP = createSafeElement('p', {},
            `Carbon Intensity: ${carbonIntensity.toFixed(2)} kg CO2e/sqft/year`, false);
        benchmarkDiv.appendChild(intensityP);

        // Percentile paragraph
        const percentileP = createSafeElement('p', {},
            `Percentile: Top ${percentile.toFixed(0)}%`, false);
        benchmarkDiv.appendChild(percentileP);

        // Recommendations
        if (data.benchmark.recommendations && Array.isArray(data.benchmark.recommendations) &&
            data.benchmark.recommendations.length > 0) {
            const recHeading = createSafeElement('h4', {}, 'Recommendations:', false);
            benchmarkDiv.appendChild(recHeading);

            const recList = createSafeElement('ul', {});
            data.benchmark.recommendations.slice(0, 3).forEach(rec => {
                const sanitizedRec = sanitizeString(rec, { maxLength: 500 });
                const listItem = createSafeElement('li', {}, sanitizedRec, false);
                recList.appendChild(listItem);
            });
            benchmarkDiv.appendChild(recList);
        }
    }

    // Display report - SECURE VERSION (replaces line 140)
    const reportDiv = document.getElementById('report');

    // Clear existing content safely
    while (reportDiv.firstChild) {
        reportDiv.removeChild(reportDiv.firstChild);
    }

    if (data.report) {
        const heading = createSafeElement('h3', {}, 'Detailed Report', false);
        reportDiv.appendChild(heading);

        // Sanitize HTML report content using DOMPurify
        // Only allow safe tags for formatting
        const reportContainer = createSafeElement('div', { class: 'report-content' });
        safeSetHTML(reportContainer, data.report, {
            ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 'h4', 'h5', 'ul', 'ol', 'li'],
            ALLOWED_ATTR: ['class'],
            ALLOW_DATA_ATTR: false
        });
        reportDiv.appendChild(reportContainer);
    }

    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Loads available agents - SECURE VERSION
 */
async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        const agents = await response.json();

        const agentsList = document.getElementById('agents-list');

        // Clear existing content safely
        while (agentsList.firstChild) {
            agentsList.removeChild(agentsList.firstChild);
        }

        // SECURE VERSION: Replaced line 153 innerHTML with safe DOM manipulation
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

let customFuelCount = 0;

/**
 * Adds custom fuel input row - SECURE VERSION
 * SECURITY: Removed inline onclick handler, replaced line 171 innerHTML
 */
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

/**
 * Removes custom fuel input row - SECURE VERSION
 */
function removeCustomFuel(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}

/**
 * Shows notification to user (replaces alert)
 * Better UX and more secure than alert()
 */
function showNotification(message, type = 'info') {
    // Create notification element if it doesn't exist
    let notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
        notificationContainer = createSafeElement('div', {
            id: 'notification-container',
            class: 'notification-container'
        });
        document.body.appendChild(notificationContainer);
    }

    // Sanitize message
    const sanitizedMessage = sanitizeString(message, { maxLength: 1000 });

    // Create notification
    const notification = createSafeElement('div', {
        class: `notification notification-${type}`
    }, sanitizedMessage, false);

    notificationContainer.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('notification-fade-out');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}
