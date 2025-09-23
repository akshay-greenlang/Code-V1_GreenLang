"""
Generate sample dairy hourly load data for 3 years

Requirements:
    pip install greenlang[analytics]
"""

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError(
        "pandas and numpy are required for this script. "
        "Install them with: pip install greenlang[analytics]"
    )

from datetime import datetime, timedelta


def generate_dairy_load_profile(year):
    """
    Generate realistic dairy plant hourly load data.
    Dairy plants typically have:
    - Peak production during daytime (6 AM - 6 PM)
    - Reduced operations at night
    - Higher demand during summer months
    - Weekly patterns (reduced on weekends)
    """
    
    # Create hourly timestamps for the entire year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23, 0, 0)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Initialize flow data
    flow_data = []
    
    for timestamp in timestamps:
        hour = timestamp.hour
        month = timestamp.month
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Base flow rate (kg/s) for dairy plant steam demand
        base_flow = 2.5
        
        # Hour-of-day factor (peak during daytime operations)
        if 6 <= hour < 18:  # Daytime production hours
            hour_factor = 1.0 + 0.3 * np.sin((hour - 6) * np.pi / 12)
        else:  # Night shift with reduced operations
            hour_factor = 0.4
        
        # Day-of-week factor (reduced on weekends)
        if weekday < 5:  # Weekdays
            day_factor = 1.0
        elif weekday == 5:  # Saturday
            day_factor = 0.7
        else:  # Sunday
            day_factor = 0.5
        
        # Seasonal factor (higher in summer for cooling/pasteurization)
        # Peak in June-August
        seasonal_factor = 1.0 + 0.2 * np.sin((month - 3) * np.pi / 6)
        
        # Add some random variation (±10%)
        random_factor = 1.0 + np.random.uniform(-0.1, 0.1)
        
        # Calculate final flow rate
        flow_rate = base_flow * hour_factor * day_factor * seasonal_factor * random_factor
        
        # Ensure non-negative flow
        flow_rate = max(0, flow_rate)
        
        flow_data.append({
            'timestamp': timestamp,
            'flow_kg_s': round(flow_rate, 3)
        })
    
    return pd.DataFrame(flow_data)


# Generate data for 2024 (current year for testing)
df_2024 = generate_dairy_load_profile(2024)
df_2024.to_csv('data/dairy_hourly_load_2024.csv', index=False)

# Generate data for 2023 (historical year 1)
df_2023 = generate_dairy_load_profile(2023)
df_2023.to_csv('data/dairy_hourly_load_2023.csv', index=False)

# Generate data for 2022 (historical year 2)
df_2022 = generate_dairy_load_profile(2022)
df_2022.to_csv('data/dairy_hourly_load_2022.csv', index=False)

print("Generated dairy load profiles for 2022, 2023, and 2024")
print(f"2024 Summary: {len(df_2024)} hours, Average flow: {df_2024['flow_kg_s'].mean():.2f} kg/s")
print(f"2023 Summary: {len(df_2023)} hours, Average flow: {df_2023['flow_kg_s'].mean():.2f} kg/s")
print(f"2022 Summary: {len(df_2022)} hours, Average flow: {df_2022['flow_kg_s'].mean():.2f} kg/s")