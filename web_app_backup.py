#!/usr/bin/env python3
"""
GreenLang Web Interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from greenlang.sdk import GreenLangClient
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize GreenLang client
client = GreenLangClient()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_emissions():
    """API endpoint to calculate emissions"""
    try:
        data = request.json
        
        # Process each fuel
        emissions_list = []
        for fuel in data.get('fuels', []):
            result = client.calculate_emissions(
                fuel_type=fuel['type'],
                consumption=fuel['consumption'],
                unit=fuel['unit'],
                region=data.get('region', 'US')
            )
            if result['success']:
                emissions_list.append(result['data'])
        
        # Aggregate emissions
        agg_result = client.aggregate_emissions(emissions_list)
        
        # Generate report
        report_result = client.generate_report(
            carbon_data=agg_result['data'],
            format='text',
            building_info=data.get('building_info', {})
        )
        
        # Benchmark if building info provided
        benchmark_data = None
        if data.get('building_info', {}).get('area'):
            benchmark_result = client.benchmark_emissions(
                total_emissions_kg=agg_result['data']['total_co2e_kg'],
                building_area=data['building_info']['area'],
                building_type=data['building_info'].get('type', 'commercial_office'),
                period_months=data.get('period_months', 1)
            )
            if benchmark_result['success']:
                benchmark_data = benchmark_result['data']
        
        return jsonify({
            'success': True,
            'emissions': agg_result['data'],
            'report': report_result['data']['report'],
            'benchmark': benchmark_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/agents', methods=['GET'])
def list_agents():
    """List available agents"""
    agents = client.list_agents()
    agent_info = []
    for agent_id in agents:
        info = client.get_agent_info(agent_id)
        if info:
            agent_info.append(info)
    return jsonify(agent_info)

@app.route('/api/quick-calc', methods=['GET'])
def quick_calc():
    """Quick calculation for demo"""
    # Parse query parameters
    electricity = float(request.args.get('electricity', 0))
    gas = float(request.args.get('gas', 0))
    
    emissions_list = []
    
    if electricity > 0:
        result = client.calculate_emissions('electricity', electricity, 'kWh')
        if result['success']:
            emissions_list.append(result['data'])
    
    if gas > 0:
        result = client.calculate_emissions('natural_gas', gas, 'therms')
        if result['success']:
            emissions_list.append(result['data'])
    
    if emissions_list:
        agg_result = client.aggregate_emissions(emissions_list)
        return jsonify({
            'success': True,
            'total_co2e_tons': agg_result['data']['total_co2e_tons'],
            'total_co2e_kg': agg_result['data']['total_co2e_kg'],
            'breakdown': agg_result['data']['emissions_breakdown']
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No valid emissions data'
        })

@app.route('/api/docs')
def api_docs():
    """Serve API documentation page"""
    return render_template('api_docs.html')

@app.route('/api/test-endpoint', methods=['POST'])
def test_endpoint():
    """Test endpoint for API documentation examples"""
    try:
        data = request.json
        endpoint = data.get('endpoint', '')
        method = data.get('method', 'GET')
        payload = data.get('payload', {})
        
        # Simulate API calls for documentation
        if endpoint == '/api/calculate':
            # Use actual calculation logic
            result = calculate_emissions()
            return result
        elif endpoint == '/api/quick-calc':
            # Simulate quick calc
            electricity = payload.get('electricity', 0)
            gas = payload.get('gas', 0)
            
            emissions_list = []
            if electricity > 0:
                result = client.calculate_emissions('electricity', electricity, 'kWh')
                if result['success']:
                    emissions_list.append(result['data'])
            
            if gas > 0:
                result = client.calculate_emissions('natural_gas', gas, 'therms')
                if result['success']:
                    emissions_list.append(result['data'])
            
            if emissions_list:
                agg_result = client.aggregate_emissions(emissions_list)
                return jsonify({
                    'success': True,
                    'total_co2e_tons': agg_result['data']['total_co2e_tons'],
                    'total_co2e_kg': agg_result['data']['total_co2e_kg'],
                    'breakdown': agg_result['data']['emissions_breakdown']
                })
        elif endpoint == '/api/agents':
            return list_agents()
        
        return jsonify({
            'success': False,
            'error': 'Unknown endpoint'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/fuel-types', methods=['GET'])
def get_fuel_types():
    """Get available fuel types and their units"""
    fuel_types = {
        'electricity': {'units': ['kWh', 'MWh'], 'default': 'kWh'},
        'natural_gas': {'units': ['therms', 'cubic_meters', 'mcf'], 'default': 'therms'},
        'diesel': {'units': ['gallons', 'liters'], 'default': 'gallons'},
        'gasoline': {'units': ['gallons', 'liters'], 'default': 'gallons'},
        'propane': {'units': ['gallons', 'liters'], 'default': 'gallons'},
        'coal': {'units': ['tons', 'kg'], 'default': 'tons'},
        'fuel_oil': {'units': ['gallons', 'liters'], 'default': 'gallons'}
    }
    return jsonify(fuel_types)

@app.route('/api/regions', methods=['GET'])
def get_regions():
    """Get available regions"""
    regions = [
        {'code': 'US', 'name': 'United States'},
        {'code': 'EU', 'name': 'European Union'},
        {'code': 'UK', 'name': 'United Kingdom'},
        {'code': 'IN', 'name': 'India'},
        {'code': 'CN', 'name': 'China'},
        {'code': 'JP', 'name': 'Japan'},
        {'code': 'AU', 'name': 'Australia'},
        {'code': 'CA', 'name': 'Canada'}
    ]
    return jsonify(regions)

@app.route('/api/building-types', methods=['GET'])
def get_building_types():
    """Get available building types"""
    building_types = [
        {'value': 'commercial_office', 'label': 'Commercial Office'},
        {'value': 'retail', 'label': 'Retail Store'},
        {'value': 'warehouse', 'label': 'Warehouse/Storage'},
        {'value': 'residential', 'label': 'Residential'},
        {'value': 'industrial', 'label': 'Industrial Facility'},
        {'value': 'hospital', 'label': 'Healthcare Facility'},
        {'value': 'school', 'label': 'Educational Institution'},
        {'value': 'hotel', 'label': 'Hotel/Hospitality'},
        {'value': 'restaurant', 'label': 'Restaurant/Food Service'},
        {'value': 'data_center', 'label': 'Data Center'}
    ]
    return jsonify(building_types)

if __name__ == '__main__':
    print("=" * 60)
    print("GreenLang Web Interface")
    print("=" * 60)
    print("Server starting at: http://localhost:5000")
    print("API endpoints:")
    print("  - POST /api/calculate - Calculate emissions")
    print("  - GET  /api/agents - List available agents")
    print("  - GET  /api/quick-calc - Quick calculation")
    print("  - GET  /api/docs - API documentation")
    print("  - POST /api/test-endpoint - Test API endpoints")
    print("  - GET  /api/fuel-types - Get fuel types")
    print("  - GET  /api/regions - Get available regions")
    print("  - GET  /api/building-types - Get building types")
    print("=" * 60)
    app.run(debug=True, port=5000)