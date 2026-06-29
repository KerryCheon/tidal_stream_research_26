FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TIDAL_STREAM_PLOT=0 \
    TIDAL_STREAM_OUTPUT=outputs/tidal_stream.png

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        libgsl-dev \
        libomp-dev \
        libopenblas-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install numpy==2.4.6 setuptools wheel==0.45.1 \
    && python -m pip install --no-build-isolation -r requirements.txt \
    && python -m pip check

COPY . .
RUN mkdir -p outputs

CMD ["python", "agama_stream_sim/example_tidal_stream.py"]
