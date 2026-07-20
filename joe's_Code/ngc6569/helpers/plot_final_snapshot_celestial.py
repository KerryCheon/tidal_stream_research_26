#!/usr/bin/env python
"""
Plot the NGC 6569 final N-body snapshot in Cartesian galactocentric coordinates,
then transform it to observable celestial coordinates and plot RA vs Dec.

Coordinate background
----------------------
Galactocentric: Cartesian frame centered on the Milky Way's center. This is the
frame the simulation is integrated and stored in (x,y,z in kpc; vx,vy,vz in km/s).

Celestial (ICRS): observer-centric spherical frame centered on the solar system
barycenter -- right ascension (RA/alpha), declination (Dec/delta), distance (r),
proper motions (pm_ra_cosdec, pm_dec), and radial velocity (v_r) are what an actual
telescope/astrometric survey measures. Converting Galactocentric -> ICRS requires
knowing the Sun's position/velocity within the Galaxy; Astropy's Galactocentric
frame encodes this (solar position, solar peculiar velocity, circular velocity at
the Sun). This repo standardizes on the Astropy v4.0 Galactocentric defaults
(same convention used in the Dev3 notebooks and stream_sim/run_nbody_mw2014.py),
so we set that explicitly here too.

Radial distance r = the heliocentric/ICRS `distance` (the natural third spherical
coordinate alongside RA/Dec) -- not the galactocentric radius |pos|, which is a
different quantity used elsewhere in the pipeline (e.g. tidal-radius diagnostics).

Run with the root .venv python (has astropy):
    /Users/kerrycheon/repos/Work/tidal_stream_research_26/.venv/bin/python plot_final_snapshot_celestial.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.coordinates as coord

HERE = Path(__file__).parent
SNAPSHOT_DIR = HERE / "output" / "dev3_original_binary_1gyr"
OUT_DIR = HERE / "output"

# NGC 6569's known present-day observables (Hughes et al. 2026 / Baumgardt & Vasiliev
# 2021), used as a sanity-check target -- same values adopted in stream_sim/run_nbody_mw2014.py.
TARGET = dict(ra=273.412, dec=-31.827, distance=10.53,
              pm_ra_cosdec=-4.125, pm_dec=-7.354, radial_velocity=-49.82)

# ----- 1) load the final snapshot --------------------------------------------
sidecar = json.loads((SNAPSHOT_DIR / "ngc6569_final_particles.json").read_text())
data = np.fromfile(SNAPSHOT_DIR / sidecar["binary_file"], dtype=sidecar["dtype"])
data = data.reshape(sidecar["shape"])
cols = {name: data[:, i] for i, name in enumerate(sidecar["columns"])}

mass = cols["mass"]
pos = np.column_stack([cols["x_gc"], cols["y_gc"], cols["z_gc"]])  # kpc
vel = np.column_stack([cols["vx_gc"], cols["vy_gc"], cols["vz_gc"]])  # km/s
print(f"loaded {pos.shape[0]} particles from {SNAPSHOT_DIR / sidecar['binary_file']}")

# ----- 2) Cartesian plots: x-y, x-z, y-z -------------------------------------
pairs = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for k, (i, j, li, lj) in enumerate(pairs):
    ax[k].scatter(pos[:, i], pos[:, j], s=3, c="tab:blue", lw=0, alpha=.3)
    ax[k].set_xlabel(f"{li} [kpc]")
    ax[k].set_ylabel(f"{lj} [kpc]")
    ax[k].set_aspect("equal")
fig.suptitle("NGC 6569 final snapshot: galactocentric Cartesian coordinates", size=14)
plt.tight_layout()
cartesian_png = OUT_DIR / "ngc6569_final_snapshot_cartesian.png"
fig.savefig(cartesian_png, dpi=130)
plt.close(fig)
print("wrote", cartesian_png)

# ----- 3) transform to celestial (ICRS) coordinates --------------------------
coord.galactocentric_frame_defaults.set('v4.0')

galactocentric = coord.SkyCoord(
    x=pos[:, 0] * u.kpc, y=pos[:, 1] * u.kpc, z=pos[:, 2] * u.kpc,
    v_x=vel[:, 0] * u.km / u.s, v_y=vel[:, 1] * u.km / u.s, v_z=vel[:, 2] * u.km / u.s,
    representation_type="cartesian", differential_type="cartesian",
    frame=coord.Galactocentric(),
)
icrs = galactocentric.transform_to(coord.ICRS())

observables = pd.DataFrame({
    "mass_Msun": mass,
    "ra_deg": icrs.ra.to_value(u.deg),
    "dec_deg": icrs.dec.to_value(u.deg),
    "distance_kpc": icrs.distance.to_value(u.kpc),
    "pm_ra_cosdec_masyr": icrs.pm_ra_cosdec.to_value(u.mas / u.yr),
    "pm_dec_masyr": icrs.pm_dec.to_value(u.mas / u.yr),
    "radial_velocity_kms": icrs.radial_velocity.to_value(u.km / u.s),
})
celestial_csv = OUT_DIR / "ngc6569_final_snapshot_celestial.csv"
observables.to_csv(celestial_csv, index=False)
print("wrote", celestial_csv)

# ----- 4) celestial plot: RA vs Dec ------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(observables["ra_deg"], observables["dec_deg"], s=3, c="tab:blue", lw=0, alpha=.3)
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Dec [deg]")
ax.invert_xaxis()  # RA increases to the left, by astronomical convention
ax.set_title("NGC 6569 final snapshot: sky position (ICRS)")
plt.tight_layout()
sky_png = OUT_DIR / "ngc6569_final_snapshot_skyplot.png"
fig.savefig(sky_png, dpi=130)
plt.close(fig)
print("wrote", sky_png)

# ----- 5) validation: compare mass-weighted mean to NGC 6569's known values --
w = observables["mass_Msun"].to_numpy()
mean = {col: np.average(observables[col], weights=w) for col in
        ["ra_deg", "dec_deg", "distance_kpc", "pm_ra_cosdec_masyr", "pm_dec_masyr", "radial_velocity_kms"]}
print("\nmass-weighted mean of the final snapshot vs. NGC 6569's known present-day values:")
print(f"{'quantity':22s}{'snapshot':>12s}{'target':>12s}")
print(f"{'RA [deg]':22s}{mean['ra_deg']:12.3f}{TARGET['ra']:12.3f}")
print(f"{'Dec [deg]':22s}{mean['dec_deg']:12.3f}{TARGET['dec']:12.3f}")
print(f"{'distance [kpc]':22s}{mean['distance_kpc']:12.3f}{TARGET['distance']:12.3f}")
print(f"{'pm_ra_cosdec [mas/yr]':22s}{mean['pm_ra_cosdec_masyr']:12.3f}{TARGET['pm_ra_cosdec']:12.3f}")
print(f"{'pm_dec [mas/yr]':22s}{mean['pm_dec_masyr']:12.3f}{TARGET['pm_dec']:12.3f}")
print(f"{'radial velocity [km/s]':22s}{mean['radial_velocity_kms']:12.3f}{TARGET['radial_velocity']:12.3f}")
