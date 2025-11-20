"""Verify emission factors database"""
import sqlite3

conn = sqlite3.connect('greenlang/data/emission_factors.db')
cursor = conn.cursor()

# Total count
cursor.execute('SELECT COUNT(*) FROM emission_factors')
total = cursor.fetchone()[0]
print(f'Total emission factors: {total}')

# Categories
cursor.execute('SELECT COUNT(DISTINCT category) FROM emission_factors')
categories = cursor.fetchone()[0]
print(f'Unique categories: {categories}')

# Scopes
cursor.execute('SELECT scope, COUNT(*) FROM emission_factors WHERE scope IS NOT NULL GROUP BY scope')
print('\nBy Scope:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} factors')

# Top categories
cursor.execute('SELECT category, COUNT(*) as cnt FROM emission_factors GROUP BY category ORDER BY cnt DESC LIMIT 10')
print('\nTop 10 categories:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} factors')

# Sample factors
cursor.execute('SELECT factor_id, name, category, emission_factor_value, unit FROM emission_factors LIMIT 5')
print('\nSample emission factors:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} = {row[2]} kg CO2e/{row[4]}')

conn.close()
print('\nDatabase verification complete!')
