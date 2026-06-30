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

The image uses Python 3.12 because the pinned scientific stack has narrower
Python support than the local venv. `gala==1.11.0` is installed from its tagged
GitHub source because the PyPI source metadata resolves incorrectly in the
Linux/aarch64 Docker build. `agama` is installed in a non-interactive step that
declines optional UNSIO downloads. The Docker build skips `pyfalcon` on Apple
Silicon because its source currently includes x86-specific headers; the example
script already falls back to the restricted N-body simulation when `pyfalcon` is
absent.

Run a shell inside the container when you want a Linux environment with the
right packages already installed:

```bash
docker compose run --rm tidal-stream
```

Run any Python file through that environment:

```bash
docker compose run --rm tidal-stream python agama_stream_sim/example_tidal_stream.py
```

The Compose setup mounts the repo into `/workspace`, so edits on your machine
are visible inside the container without rebuilding the image. Rebuild only when
`requirements.txt` or the Dockerfile changes:

```bash
docker compose build
```

By default, the tidal-stream example does not open a plot window or save an
output file. If you do want to save the final plot, pass an output path:

```bash
docker compose run --rm \
  -e TIDAL_STREAM_OUTPUT=outputs/tidal_stream.png \
  tidal-stream \
  python agama_stream_sim/example_tidal_stream.py
```

Useful runtime knobs:

```bash
docker compose run --rm \
  -e TIDAL_STREAM_NBODY=2000 \
  -e TIDAL_STREAM_TEND=0.25 \
  tidal-stream \
  python agama_stream_sim/example_tidal_stream.py
```

Set `TIDAL_STREAM_PLOT=1` only when running with a display-capable Matplotlib
backend.
