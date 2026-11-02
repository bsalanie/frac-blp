"""FRAC estimation on macro-BLP, without demographics."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from frac_blp.config import build_frac_data, load_config
from frac_blp.frac_nodemog_estimate import nodemog_estimate

console = Console()

EXAMPLES_DIR = Path(__file__).parent / "configs"


def copy_examples(destination: Path) -> None:
    """Copy the packaged example configurations and data into a directory.

    Files keep their relative layout, so a copied configuration still finds its data.
    Existing files are left untouched.

    Args:
        destination (Path): Directory to copy the examples into.
    """
    for source in sorted(EXAMPLES_DIR.rglob("*")):
        if source.is_dir():
            continue
        target = destination / source.relative_to(EXAMPLES_DIR)
        if target.exists():
            console.print(f"Skipping {target}: already exists")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
        console.print(f"Copied {target}")


def main():
    """Load the configuration, estimate FRAC, and return the estimates.

    The configuration file path is read from the command line. With ``--example``,
    the example configurations and data are copied into the current directory instead.
    """
    parser = argparse.ArgumentParser(
        description="Estimate FRAC without demographics using a TOML configuration file."
    )
    parser.add_argument(
        "config", type=Path, nargs="?", help="path to the TOML configuration file"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="copy the example configuration files and data into the current directory and exit",
    )
    args = parser.parse_args()

    if args.example:
        copy_examples(Path.cwd())
        args.config = Path("example_config.toml")
        console.print(f"Running frac_nodemog {args.config}")
    if args.config is None:
        parser.error("the configuration file is required (or use --example)")

    config = load_config(args.config)
    frac_data = build_frac_data(config)
    console.print(Panel(Text(str(frac_data).rstrip()), title="Dataset", expand=False))

    est = config.estimation
    _, _ = nodemog_estimate(
        frac_data,
        degree_Z=est.degree_Z,
        degree_X1=est.degree_X1,
        optimal_instruments=est.optimal_instruments,
        run_projections=est.run_projections,
    )

    if config.data.source == "simulated":
        console.rule("The true coefficients were")
        console.print("betas:", config.simulation.betas)
        console.print("sigmas2:", np.array(config.simulation.sigmas) ** 2)


if __name__ == "__main__":
    main()
