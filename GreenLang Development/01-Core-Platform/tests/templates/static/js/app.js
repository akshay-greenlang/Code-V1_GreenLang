// GreenLang Web Interface JavaScript

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
    const electricity = parseFloat(document.getElementById('electricity').value) || 0;
    const gas = parseFloat(document.getElementById('gas').value) || 0;
    const buildingArea = parseFloat(document.getElementById('building-area').value) || 0;
    const buildingType = document.getElementById('building-type').value;
    
    // Collect custom fuels
    const customFuels = [];
    document.querySelectorAll('.custom-fuel-row').forEach(row => {
        const type = row.querySelector('.fuel-type').value;
        const consumption = parseFloat(row.querySelector('.fuel-consumption').value) || 0;
        const unit = row.querySelector('.fuel-unit').value;
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
        alert('Please enter at least one fuel consumption value');
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
            alert('Error: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        alert('Error calculating emissions: ' + error.message);
    }
}

function displayResults(data) {
    // Show results section
    document.getElementById('results').style.display = 'block';
    
    // Display totals
    document.getElementById('total-emissions').textContent = data.emissions.total_co2e_tons.toFixed(3);
    document.getElementById('total-kg').textContent = data.emissions.total_co2e_kg.toFixed(2);
    
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
        breakdownDiv.innerHTML = breakdownHTML;
    }
    
    // Display benchmark if available
    const benchmarkDiv = document.getElementById('benchmark');
    if (data.benchmark) {
        let ratingClass = 'benchmark-average';
        if (data.benchmark.rating === 'Excellent' || data.benchmark.rating === 'Good') {
            ratingClass = 'benchmark-good';
        } else if (data.benchmark.rating === 'Poor' || data.benchmark.rating === 'Below Average') {
            ratingClass = 'benchmark-poor';
        }
        
        let benchmarkHTML = '<h3>Benchmark Analysis</h3>';
        benchmarkHTML += `<p>Rating: <span class="${ratingClass}">${data.benchmark.rating}</span></p>`;
        benchmarkHTML += `<p>Carbon Intensity: ${data.benchmark.carbon_intensity.toFixed(2)} kg CO2e/sqft/year</p>`;
        benchmarkHTML += `<p>Percentile: Top ${data.benchmark.percentile}%</p>`;
        
        if (data.benchmark.recommendations && data.benchmark.recommendations.length > 0) {
            benchmarkHTML += '<h4>Recommendations:</h4><ul>';
            data.benchmark.recommendations.slice(0, 3).forEach(rec => {
                benchmarkHTML += `<li>${rec}</li>`;
            });
            benchmarkHTML += '</ul>';
        }
        
        benchmarkDiv.innerHTML = benchmarkHTML;
    } else {
        benchmarkDiv.innerHTML = '';
    }
    
    // Display report
    const reportDiv = document.getElementById('report');
    if (data.report) {
        reportDiv.innerHTML = '<h3>Detailed Report</h3>' + data.report;
    }
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

async function loadAgents() {
    try {
        const response = await fetch('/api/agents');
        const agents = await response.json();
        
        const agentsList = document.getElementById('agents-list');
        agentsList.innerHTML = agents.map(agent => 
            `<li><strong>${agent.name}</strong>: ${agent.description}</li>`
        ).join('');
    } catch (error) {
        console.error('Error loading agents:', error);
    }
}

let customFuelCount = 0;

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
    `;
    
    customFuelsDiv.appendChild(fuelRow);
}

function removeCustomFuel(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}