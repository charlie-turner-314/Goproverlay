goproverlay
============

CLI tool to overlay fitness metrics (power, speed, pace, elevation, cadence) and a GPS track onto a video, using a FIT activity file.

Features
- Parse FIT files from common watches/bike computers.
- Overlay up to 4 widgets in the video corners.
- GPS track widget shows full route with current position marker.
- Modular architecture to add new widgets/themes.

Requirements
- Python 3.9+
- FFmpeg (MoviePy uses it under the hood). Install via your OS package manager.

Install
```
pip install .
```

Usage
```
goproverlay \
  --video /path/to/video.mp4 \
  --fit /path/to/activity.fit \
  [--gpx /path/to/video_clip.gpx] \
  --widget power --widget speed --widget gps \
  --output /path/to/output.mp4
```

Options
- `--widget`: repeat up to 4 times. Choices: `power`, `speed`, `pace`, `elevation`, `cadence`, `gps`.
- Time sync precedence:
  - If `--gpx` is provided (exported for this specific GoPro clip), its first trackpoint time is used as the clip start for aligning FIT data.
  - Otherwise, sync uses the MP4 wall-clock start (ffprobe: `creation_time + start_time`).
  - If neither is available, FIT data is used relative to its own start.
- `--theme`: `dark` (default) or `light`.
- `--font-path`: custom TTF font path for text rendering.
- `--ftp`: Functional Threshold Power in watts (for power gauge zones).
- `--power-zones`: comma-separated 5 upper bounds for zones. Examples: `150,200,250,300,350` or
  percentages with `--ftp` like `55%,75%,90%,105%,120%`.
- `--speed-max`: max speed (km/h) for speed gauge scale (auto-computed if omitted).

Notes
- Speed is rendered in km/h. Pace is min/km.
- If a metric is absent in the FIT stream, the widget shows `—`.
- The GPS track widget fits your route into its box and marks the current position each frame.
- Power and Speed are rendered as gauges (segmented zones for power, progress arc for speed) with large, readable values.

Dev harnesses
- Solid color + synthetic data: `python3 scripts/harness_solid.py` (writes `harness_overlay.mp4`).
- Solid color + real FIT: `python3 scripts/harness_fit.py /path/to/activity.fit --duration 15 --fps 30`.

Development
- Project uses Typer for CLI, MoviePy for video, Pillow for drawing, fitdecode for FIT parsing.
- Code is organized into `core/` (data) and `overlay/` (rendering/widgets).
