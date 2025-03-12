"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Coefficient Maker."""


if __name__ == "__main__":
    main(prog_name="ssb-coefficient-maker")  # pragma: no cover
