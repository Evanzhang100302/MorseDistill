"""Unified spatio-temporal Morse function.

Replaces the two separate Morse paths of MCM-DM (tile-graph spatial Morse +
per-sequence temporal Morse) with a single discrete Morse function built on a
joint (dt, lon, lat) cell complex.

Cells live in the same normalized space as the diffusion target
x = (dt_nm, lon_nm, lat_nm), so critical cells can be used directly as the
anchor set of the topology loss.

Shared by build_st_morse_features.py (offline) and the train/test scripts
(online lookup).
"""
import math
import bisect

import numpy as np


# ─── coordinate helpers ──────────────────────────────────────────────────────

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(zoom, tx, ty):
    """Center of tile (tx, ty)."""
    n = 2 ** zoom
    lon = (tx + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
    return lon, lat


def dt_bin(dt_nm, edges):
    """Index of the quantile bin containing dt_nm. edges are interior cut points."""
    return bisect.bisect_right(edges, dt_nm)


def cell_key(dt_nm, lon_raw, lat_raw, zoom, edges):
    tx, ty = lat_lon_to_tile(lat_raw, lon_raw, zoom)
    return (tx, ty, dt_bin(dt_nm, edges))


# ─── vertex weights ──────────────────────────────────────────────────────────

def initialize_st_vertex_weights(G, density, mode='density', seed=42):
    """g(v) for the discrete Morse function.

    Contract matches morse_function.initialize_vertex_weights: returns {node: float}
    where LOW g marks the features we want to come out critical.

    mode='density': g = (1 - normalized density) + eps.  Density peaks get the
        lowest g and therefore surface as critical 0-simplices (local minima of
        f), which is the spatio-temporal analogue of the KDE peaks used by the
        abandoned kde variant -- but resolved through real discrete Morse theory
        instead of a k-NN local-minimum heuristic.
    mode='degree': reproduces the original tile-graph behaviour
        (g = deg_max - deg + eps), kept for ablation.
    """
    rng = np.random.RandomState(seed)
    g = {}
    if mode == 'degree':
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 0
        for v in G.nodes():
            g[v] = max_degree - degrees[v] + rng.uniform(0, 0.5)
        return g

    vals = np.array([density[v] for v in G.nodes()], dtype=np.float64)
    lo, hi = vals.min(), vals.max()
    span = (hi - lo) if hi > lo else 1.0
    # Scale the informative part above the eps noise floor so ties broken by eps
    # cannot outrank a genuine density difference.
    for v in G.nodes():
        g[v] = (1.0 - (density[v] - lo) / span) * 10.0 + rng.uniform(0, 0.5)
    return g


# ─── ST distance ─────────────────────────────────────────────────────────────

def st_pairwise_dist(centers, w_time):
    """Pairwise distance in normalized (dt, lon, lat) space.

    centers : (N, 3) array of (dt_nm, lon_nm, lat_nm), all in [0, 1].
    w_time  : anisotropy weight on the temporal axis (w=0 -> purely spatial).

    All three axes are already unit-normalized, so no km/second unit conversion
    is needed and w_time alone controls the space/time trade-off.
    """
    scaled = centers.copy()
    scaled[:, 0] *= w_time
    diff = scaled[:, None, :] - scaled[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))
