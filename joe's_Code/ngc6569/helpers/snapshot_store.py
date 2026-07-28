"""Flat binary storage for N-body snapshots, written one record at a time.

Each snapshot (time, pos, vel, phi) is a fixed-size record so a run can be
streamed to disk instead of held in memory, and so a reader can seek directly
to record i without loading the whole file. A JSON sidecar written at open
time records the particle count needed to interpret the file; the number of
complete records is computed from the file size, not from a count stored at
close time, so a file left behind by a crashed kernel is still readable up to
its last fully-written snapshot.
"""
import json
from pathlib import Path

import numpy as np


def _manifest_path(path):
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def _record_dtype(n_particles):
    return np.dtype([
        ("time", "f8"),
        ("pos", "f8", (n_particles, 3)),
        ("vel", "f8", (n_particles, 3)),
        ("phi", "f8", (n_particles,)),
    ])


class SnapshotBinaryWriter:
    """Append snapshots to a flat binary file, one fixed-size record at a time."""

    def __init__(self, path, n_particles):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_particles = int(n_particles)
        self._dtype = _record_dtype(self.n_particles)

        # Written immediately (not at close) so a reader can interpret whatever
        # got flushed to disk even if this process is killed mid-run.
        _manifest_path(self.path).write_text(json.dumps(
            {"n_particles": self.n_particles, "record_bytes": int(self._dtype.itemsize)},
            indent=2,
        ))
        self._file = open(self.path, "wb")

    def append(self, time, pos, vel, phi):
        record = np.zeros(1, dtype=self._dtype)
        record["time"] = time
        record["pos"] = pos
        record["vel"] = vel
        record["phi"] = phi
        record.tofile(self._file)
        self._file.flush()  # Make each snapshot durable as soon as it's written.

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SnapshotBinaryReader:
    """Lazily read snapshots back from a file written by SnapshotBinaryWriter.

    Behaves like a read-only sequence of snapshot dicts: supports len(),
    indexing (including negative indices), and iteration, without loading
    more than one snapshot into memory at a time.
    """

    def __init__(self, path):
        self.path = Path(path)
        manifest = json.loads(_manifest_path(self.path).read_text())
        self.n_particles = manifest["n_particles"]
        self._dtype = _record_dtype(self.n_particles)

        file_size = self.path.stat().st_size
        n_complete = file_size // self._dtype.itemsize  # Ignores a partial trailing record.
        self._mmap = np.memmap(self.path, dtype=self._dtype, mode="r", shape=(n_complete,))

    def __len__(self):
        return len(self._mmap)

    def __getitem__(self, i):
        record = self._mmap[i]
        return {
            "time": float(record["time"]),
            "pos": np.array(record["pos"]),
            "vel": np.array(record["vel"]),
            "phi": np.array(record["phi"]),
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
