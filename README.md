# Joe_research_26

## Python Environment

Create and activate a local virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

To check that the installed packages are consistent:

```bash
python -m pip check
```

## Docker

Build the research image from the project root:

```bash
docker build -t tidal-stream-research .
```

Run the default tidal-stream example headlessly. The final plot is written to
`outputs/tidal_stream.png` on the host:

```bash
docker run --rm -v "$PWD/outputs:/workspace/outputs" tidal-stream-research
```

You can also use Compose:

```bash
docker compose up --build
```

Useful runtime knobs:

```bash
docker run --rm \
  -e TIDAL_STREAM_PLOT=0 \
  -e TIDAL_STREAM_OUTPUT=outputs/tidal_stream.png \
  -e TIDAL_STREAM_NBODY=2000 \
  -e TIDAL_STREAM_TEND=0.25 \
  -v "$PWD/outputs:/workspace/outputs" \
  tidal-stream-research
```

Set `TIDAL_STREAM_PLOT=1` only when running with a display-capable Matplotlib
backend.
