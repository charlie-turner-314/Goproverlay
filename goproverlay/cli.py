from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer

from .core.datatypes import Metric
from .core.fit_loader import FitData
from .core.video_meta import get_video_start_utc
from .core.gpx_loader import get_gpx_time_bounds
from .overlay.renderer import OverlayRenderer
from .utils.logging import get_logger


app = typer.Typer(
    add_completion=False, help="Overlay fitness widgets onto videos using FIT files."
)
log = get_logger(__name__)


def _metric_choices() -> List[str]:
    return [m.value for m in Metric]


@app.command()
def render(
    video: Path = typer.Option(
        ..., exists=True, dir_okay=False, readable=True, help="Input video file path"
    ),
    fit: Path = typer.Option(
        ..., exists=True, dir_okay=False, readable=True, help="Input FIT activity file"
    ),
    widget: List[str] = typer.Option(
        ...,
        "--widget",
        help="Widget to overlay (repeat up to 4)",
        case_sensitive=False,
    ),
    output: Optional[Path] = typer.Option(None, help="Output video path (mp4)"),
    theme: str = typer.Option("dark", help="Visual theme: dark or light"),
    font_path: Optional[Path] = typer.Option(
        None, help="Custom TTF font path for widgets"
    ),
    gpx: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional GPX file for this clip; takes precedence for time sync",
    ),
    ftp: Optional[float] = typer.Option(
        None, help="Functional Threshold Power (W) for power zones"
    ),
    power_zones: Optional[str] = typer.Option(
        None,
        help=(
            "Comma-separated 5 upper bounds for power zones. "
            "Examples: '150,200,250,300,350' or percentages with --ftp like '55%,75%,90%,105%,120%'"
        ),
    ),
    speed_max: Optional[float] = typer.Option(
        None, help="Max speed (km/h) for speed gauge; default computed from data"
    ),
):
    """Render the overlayed video."""

    if len(widget) == 0:
        typer.echo("Please specify at least one --widget")
        raise typer.Exit(code=2)
    if len(widget) > 4:
        typer.echo("You can specify at most 4 --widget options")
        raise typer.Exit(code=2)

    # Normalize and validate widgets
    try:
        selected = [Metric(w.lower()) for w in widget]
    except ValueError as e:
        valid = ", ".join(_metric_choices())
        typer.echo(f"Invalid widget in {widget}. Valid choices: {valid}")
        raise typer.Exit(code=2)

    if output is None:
        output = video.with_name(f"{video.stem}_overlay.mp4")

    # Determine wall-clock start for alignment
    vid_start = None
    if gpx is not None:
        log.info("Using GPX to determine clip start/end...")
        gpx_start, gpx_end = get_gpx_time_bounds(gpx)
        if gpx_start is not None:
            vid_start = gpx_start
            log.info(
                f"GPX start (UTC): {gpx_start.isoformat()} — end: {gpx_end.isoformat() if gpx_end else 'unknown'}"
            )
        else:
            log.warning(
                "Failed to extract timestamps from GPX; falling back to video metadata."
            )
    if vid_start is None:
        log.info("Probing video metadata for start time...")
        vid_start = get_video_start_utc(video)
        if vid_start is None:
            log.warning(
                "Could not determine video start time; overlay will use FIT relative start."
            )
        else:
            log.info(f"Video start (UTC): {vid_start.isoformat()}")

    log.info("Loading FIT data and aligning...")
    fit_data = FitData.from_fit_file(fit, align_to_datetime=vid_start)
    if fit_data.is_empty():
        typer.echo("No usable records found in FIT file.")
        raise typer.Exit(code=1)

    # Parse power zones if provided
    zones_watts = None
    if power_zones is not None:
        try:
            raw = [z.strip() for z in power_zones.split(",") if z.strip()]
            if len(raw) != 5:
                raise ValueError("Expected exactly 5 values for --power-zones")
            if any(s.endswith("%") for s in raw):
                if ftp is None:
                    raise ValueError("--power-zones given in % requires --ftp")
                vals = []
                for s in raw:
                    if not s.endswith("%"):
                        raise ValueError(
                            "All power zones must be % when mixing percentages"
                        )
                    p = float(s[:-1]) / 100.0
                    vals.append(p * float(ftp))
            else:
                vals = [float(s) for s in raw]
            # Ensure strictly increasing
            for i in range(1, len(vals)):
                if vals[i] <= vals[i - 1]:
                    raise ValueError("--power-zones must be strictly increasing")
            zones_watts = vals
        except Exception as e:
            typer.echo(f"Invalid --power-zones: {e}")
            raise typer.Exit(code=2)

    log.info("Rendering video with overlays...")
    renderer = OverlayRenderer(
        video_path=video,
        fit_data=fit_data,
        metrics=selected,
        theme=theme,
        font_path=font_path,
        power_zones=zones_watts,
        ftp=ftp,
        speed_max_kmh=speed_max,
    )
    renderer.render_to(output)

    typer.echo(f"Done. Wrote: {output}")


def main(argv: Optional[List[str]] = None) -> None:
    app(standalone_mode=True)  # Typer handles sys.exit


if __name__ == "__main__":
    main(sys.argv[1:])
