# Fix imports for security patches

# Fix reasoning.py
$reasoning = "C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/capabilities/reasoning.py"
$content = Get-Content $reasoning -Raw
$content = $content -replace "import asyncio", "import ast`nimport asyncio"
$content = $content -replace "__import__\('ast'\).literal_eval", "ast.literal_eval"
Set-Content $reasoning $content

# Fix pipeline.py
$pipeline = "C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/orchestration/pipeline.py"
$content = Get-Content $pipeline -Raw
$content = $content -replace "import asyncio", "import asyncio`nfrom simpleeval import simple_eval"
$content = $content -replace "__import__\('simpleeval'\).simple_eval", "simple_eval"
Set-Content $pipeline $content

# Fix routing.py
$routing = "C:/Users/aksha/Code-V1_GreenLang/GreenLang_2030/agent_foundation/orchestration/routing.py"
$content = Get-Content $routing -Raw
$content = $content -replace "import asyncio", "import asyncio`nfrom simpleeval import simple_eval"
$content = $content -replace "__import__\('simpleeval'\).simple_eval", "simple_eval"
Set-Content $routing $content

Write-Host "Imports fixed successfully"
