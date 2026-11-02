"""Executable example demonstrating FRAC estimation without demographics.

The example runs two estimations from TOML configuration files in `frac_blp/configs/`:
first on simulated data (`simulated.toml`), then on a data file read through the
real-data interface (`example_config.toml`). Edit those files to change the dimensions,
parameters and estimation options.
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from frac_blp.config import build_frac_data, load_config
from frac_blp.frac_nodemog_estimate import nodemog_estimate

console = Console()

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def run_example(
    simulated_config: Path = CONFIG_DIR / "simulated.toml",
    real_config: Path = CONFIG_DIR / "example_config.toml",
) -> None:
    """Estimate FRAC without demographics on simulated data, then on a data file.

    Args:
        simulated_config (Path): TOML file describing the simulation and the
            estimation options. Default is `configs/simulated.toml`.
        real_config (Path): TOML file describing the data file and the estimation
            options. Default is `configs/example_config.toml`.
    """
    console.print(
        Panel(
            Text("Hello from frac_blp!", justify="center", style="bold magenta"),
            border_style="magenta",
            padding=(1, 2),
        )
    )
    config = load_config(simulated_config)
    simulated_frac_data = build_frac_data(config)
    console.print(
        Panel(
            Text("Simulated data:", justify="center", style="bold"),
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print(
        Panel(Text(str(simulated_frac_data).rstrip()), title="Dataset", expand=False)
    )

    console.rule("Estimating with FRAC")
    est = config.estimation
    _, _ = nodemog_estimate(
        simulated_frac_data,
        degree_Z=est.degree_Z,
        degree_X1=est.degree_X1,
        optimal_instruments=est.optimal_instruments,
        run_projections=est.run_projections,
    )

    console.print(
        Panel(
            Text(
                "Example with the real data interface:", justify="center", style="bold"
            ),
            border_style="blue",
            padding=(1, 2),
        )
    )
    config = load_config(real_config)
    real_frac_data = build_frac_data(config)
    console.print(
        Panel(Text(str(real_frac_data).rstrip()), title="Dataset", expand=False)
    )
    console.rule("Estimating with FRAC")
    est = config.estimation
    _, _ = nodemog_estimate(
        real_frac_data,
        degree_Z=est.degree_Z,
        degree_X1=est.degree_X1,
        optimal_instruments=est.optimal_instruments,
        run_projections=est.run_projections,
    )


if __name__ == "__main__":
    run_example()
