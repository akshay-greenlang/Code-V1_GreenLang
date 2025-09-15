"""Demo command for GreenLang CLI."""
import json
import time
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

console = Console()
app = typer.Typer(help="Run a tiny offline demo pipeline")


@app.callback()
def callback():
    """Demo command callback."""
    pass


@app.command()
def run():
    """Runs an embedded example: loads sample inputs → executes DemoAgent → prints summary."""
    console.print(Panel.fit("🌱 GreenLang Demo Pipeline", style="green bold"))

    # For now, run a simple demo calculation
    console.print("\n[cyan]Running demo emissions calculation...[/cyan]")

    start = time.time()

    # Demo calculation
    load_kwh = 125.0
    grid_emission_factor = 0.7  # kgCO2/kWh
    emissions = load_kwh * grid_emission_factor

    elapsed = time.time() - start

    result = {
        "ok": True,
        "elapsed_sec": round(elapsed, 2),
        "result": {
            "input": {
                "load_kwh": load_kwh,
                "grid_emission_factor": grid_emission_factor
            },
            "output": {
                "emissions_kgco2": round(emissions, 3)
            }
        }
    }

    console.print("\n[green]✓ Demo pipeline completed successfully![/green]")
    console.print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    app()