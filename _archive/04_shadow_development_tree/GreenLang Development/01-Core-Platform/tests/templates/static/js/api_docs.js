// API Documentation Interactive Functions

// Try endpoint functionality
async function tryEndpoint(endpoint) {
    const responseContainer = document.getElementById(`${endpoint}-response`);
    responseContainer.innerHTML = '<div>Loading...</div>';
    responseContainer.classList.add('show');
    
    try {
        let response;
        
        switch(endpoint) {
            case 'calculate':
                response = await fetch('/api/calculate', {
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
                response = await fetch('/api/quick-calc?electricity=1000&gas=50');
                break;
                
            case 'agents':
                response = await fetch('/api/agents');
                break;
                
            case 'fuel-types':
                response = await fetch('/api/fuel-types');
                break;
                
            case 'regions':
                response = await fetch('/api/regions');
                break;
                
            case 'building-types':
                response = await fetch('/api/building-types');
                break;
        }
        
        const data = await response.json();
        responseContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        
    } catch (error) {
        responseContainer.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
    }
}

// Run interactive example
async function runExample() {
    const fuelType = document.getElementById('example-fuel-type').value;
    const consumption = document.getElementById('example-consumption').value;
    const unit = document.getElementById('example-unit').value;
    const region = document.getElementById('example-region').value;
    
    const responseContainer = document.getElementById('example-response');
    responseContainer.innerHTML = '<div>Processing...</div>';
    responseContainer.classList.add('show');
    
    try {
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                fuels: [
                    { type: fuelType, consumption: parseFloat(consumption), unit: unit }
                ],
                region: region
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            responseContainer.innerHTML = `
                <div style="color: green; margin-bottom: 10px;">✓ Success</div>
                <div><strong>Total Emissions:</strong> ${data.emissions.total_co2e_kg.toFixed(2)} kg CO2e</div>
                <div><strong>In Tons:</strong> ${data.emissions.total_co2e_tons.toFixed(4)} metric tons</div>
                <details style="margin-top: 10px;">
                    <summary>Full Response</summary>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </details>
            `;
        } else {
            responseContainer.innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
        }
        
    } catch (error) {
        responseContainer.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
    }
}

// Generate code snippets
function generateCode() {
    const fuelType = document.getElementById('example-fuel-type').value;
    const consumption = document.getElementById('example-consumption').value;
    const unit = document.getElementById('example-unit').value;
    const region = document.getElementById('example-region').value;
    
    // Python snippet
    const pythonCode = `from greenlang.sdk import GreenLangClient

client = GreenLangClient()

result = client.calculate_emissions(
    fuel_type="${fuelType}",
    consumption=${consumption},
    unit="${unit}",
    region="${region}"
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
          type: '${fuelType}', 
          consumption: ${consumption}, 
          unit: '${unit}' 
        }
      ],
      region: '${region}'
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
        "type": "${fuelType}",
        "consumption": ${consumption},
        "unit": "${unit}"
      }
    ],
    "region": "${region}"
  }'`;
    
    document.getElementById('python-snippet').textContent = pythonCode;
    document.getElementById('javascript-snippet').textContent = jsCode;
    document.getElementById('curl-snippet').textContent = curlCode;
}

// Tab functionality
function showTab(language) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${language}-code`).classList.add('active');
    event.target.classList.add('active');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set default code snippets
    generateCode();
    
    // Update units based on fuel type
    document.getElementById('example-fuel-type').addEventListener('change', function() {
        const unitSelect = document.getElementById('example-unit');
        const fuelType = this.value;
        
        // Clear existing options
        unitSelect.innerHTML = '';
        
        // Add appropriate units based on fuel type
        const units = {
            'electricity': ['kWh', 'MWh'],
            'natural_gas': ['therms', 'cubic_meters', 'mcf'],
            'diesel': ['gallons', 'liters'],
            'gasoline': ['gallons', 'liters']
        };
        
        const fuelUnits = units[fuelType] || ['gallons', 'liters'];
        fuelUnits.forEach(unit => {
            const option = document.createElement('option');
            option.value = unit;
            option.textContent = unit;
            unitSelect.appendChild(option);
        });
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Copy code functionality
function copyCode(elementId) {
    const codeElement = document.getElementById(elementId);
    const text = codeElement.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        // Show success message
        const button = event.target;
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.style.background = '#4caf50';
        
        setTimeout(() => {
            button.textContent = originalText;
            button.style.background = '';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}