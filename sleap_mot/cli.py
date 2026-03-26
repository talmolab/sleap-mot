"""CLI entry point for sleap-mot.

Usage:
    sleap-mot run config.yaml [--labels PATH] [--video PATH] [--output PATH] [--save-slp]
"""

import click

from sleap_mot.pipeline import load_config, run_pipeline


@click.group()
def main():
    """SLEAP Multi-Object Tracking CLI."""
    pass


@main.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--labels", type=click.Path(), default=None,
              help="Override input.labels path.")
@click.option("--video", type=click.Path(), default=None,
              help="Override input.video path.")
@click.option("--output", type=click.Path(), default=None,
              help="Override output.path.")
@click.option("--save-slp", is_flag=True, default=False,
              help="Also save a plain .slp file alongside .slpt.")
def run(config, labels, video, output, save_slp):
    """Run a tracking pipeline from a YAML config file."""
    overrides = {}
    if labels:
        overrides["labels"] = labels
    if video:
        overrides["video"] = video
    if output:
        overrides["output"] = output
    if save_slp:
        overrides["save_slp"] = True

    cfg = load_config(config, overrides=overrides or None)
    output_path = run_pipeline(cfg)
    click.echo(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
