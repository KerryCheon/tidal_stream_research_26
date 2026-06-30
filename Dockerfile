FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    CPLUS_INCLUDE_PATH=/usr/include/eigen3 \
    CPATH=/usr/include/eigen3 \
    TIDAL_STREAM_PLOT=0

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        libeigen3-dev \
        libgsl-dev \
        libomp-dev \
        libopenblas-dev \
        pkg-config \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install numpy==2.4.6 setuptools wheel==0.45.1 \
    && python -m pip install cvxopt cython setuptools_scm \
    && grep -v -e '^agama==' -e '^gala==' -e '^git+https://github.com/GalacticDynamics-Oxford/pyfalcon' requirements.txt > /tmp/requirements-docker.txt \
    && python -m pip install --no-build-isolation -r /tmp/requirements-docker.txt \
    && python -m pip install --no-build-isolation --config-settings --build-option=--yes agama==1.0 \
    && python -m pip install --no-deps --no-build-isolation git+https://github.com/adrn/gala.git@v1.11.0 \
    && python -m pip check

COPY . .
RUN mkdir -p outputs

CMD ["bash"]
