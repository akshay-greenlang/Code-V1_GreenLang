# -*- coding: utf-8 -*-
import json

with open('coverage.json', 'r') as f:
    data = json.load(f)

agent_coverage = {}
for file_path, file_data in data['files'].items():
    if 'greenlang\\agents\\' in file_path or 'greenlang/agents/' in file_path:
        filename = file_path.split('\\')[-1].split('/')[-1]
        agent_coverage[filename] = {
            'covered_lines': file_data['summary']['covered_lines'],
            'num_statements': file_data['summary']['num_statements'],
            'percent_covered': file_data['summary']['percent_covered'],
            'missing_lines': file_data['summary']['missing_lines']
        }

for filename, coverage in sorted(agent_coverage.items()):
    print(f"{filename}: {coverage['percent_covered']:.2f}% ({coverage['covered_lines']}/{coverage['num_statements']})")
