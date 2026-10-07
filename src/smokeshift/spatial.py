"""What nearby and upwind monitors are reading right now.

Smoke travels. A plume over Minnesota today is over Michigan tomorrow, and a
model that only sees one monitor's history cannot know it is coming. These
features summarise the current state of the monitors around each site, at
several distances and in four compass quadrants, using only readings taken at
or before the issue time.
"""
import numpy as np

from . import config

EARTH_KM = 6371.0


def distances_and_bearings(lat, lon):
    """Great-circle distance (km) and initial bearing (degrees from north) between all pairs."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dla, dlo = la.T - la, lo.T - lo
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    dist = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    x = np.sin(dlo) * np.cos(la.T)
    y = np.cos(la) * np.sin(la.T) - np.sin(la) * np.cos(la.T) * np.cos(dlo)
    return dist, (np.degrees(np.arctan2(x, y)) + 360) % 360


def masked_mean(W, V):
    """Weighted mean over neighbours that have a reading; NaN if none do."""
    ok = np.isfinite(V)
    num = W @ np.where(ok, V, 0.0)
    den = W @ ok.astype(W.dtype)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan).astype("float32")


def masked_max(M, V):
    """Largest reading among each site's neighbours (boolean matrix M)."""
    filled = np.where(np.isfinite(V), V, -np.inf)
    out = np.full(V.shape, np.nan, dtype="float32")
    for i in range(M.shape[0]):
        if M[i].any():
            out[i] = filled[M[i]].max(axis=0)
    return np.where(np.isfinite(out), out, np.nan)


class Neighbours:
    """Pairwise distances and bearings between monitors, with helpers that select neighbours."""
    def __init__(self, stations):
        """Compute the distance and bearing matrices once."""
        self.dist, self.bearing = distances_and_bearings(stations["lat"].to_numpy(), stations["lon"].to_numpy())
        np.fill_diagonal(self.dist, np.inf)          # a site is not its own neighbour

    def band(self, lo_km, hi_km, sector=None):
        """Boolean matrix: j is within [lo, hi) km of i, optionally in a compass quadrant."""
        m = (self.dist >= lo_km) & (self.dist < hi_km)
        if sector is not None:
            lo_deg, hi_deg = sector
            m &= (self.bearing >= lo_deg) & (self.bearing < hi_deg)
        return m

    def idw(self, lo_km, hi_km):
        """Inverse-distance weights within a band (not normalised; masked_mean does that)."""
        return np.where(self.band(lo_km, hi_km), 1.0 / np.maximum(self.dist, 10.0), 0.0).astype("float32")


QUADRANTS = {"ne": (0, 90), "se": (90, 180), "sw": (180, 270), "nw": (270, 360)}


def features(bench, t_idx, nb=None):
    """Spatial features at each issue time: ([monitors, issues, F], names)."""
    nb = nb or Neighbours(bench.stations)
    now = bench.last_obs(t_idx)
    before = bench.last_obs(t_idx - 6)
    near, mid, ring = nb.idw(0, 150), nb.idw(0, 300), nb.idw(300, 800)
    f = {
        "nb_mean_150": masked_mean(near, now),
        "nb_mean_300": masked_mean(mid, now),
        "nb_max_300": masked_max(nb.band(0, 300), now),
        "ring_mean_800": masked_mean(ring, now),
        "ring_max_800": masked_max(nb.band(300, 800), now),
        "share_elevated_500": masked_mean(nb.band(0, 500).astype("float32"), (now > config.USG).astype("float32")
                                          + np.where(np.isfinite(now), 0, np.nan)),
    }
    f["nb_trend6_300"] = f["nb_mean_300"] - masked_mean(mid, before)
    f["ring_trend6_800"] = f["ring_mean_800"] - masked_mean(ring, before)
    for name, sector in QUADRANTS.items():
        f[f"quad_{name}_800"] = masked_mean(nb.band(100, 800, sector).astype("float32"), now)
    # Relative to the site itself: is the surrounding air dirtier than here?
    for key in ("nb_mean_300", "nb_max_300", "ring_mean_800", "ring_max_800"):
        f[f"{key}_minus_self"] = f[key] - now
    names = list(f)
    return np.stack([f[k] for k in names], axis=2).astype("float32"), names


def hourly_channels(bench, nb=None):
    """Two neighbour series at every hour, for sequence models: within 300 km, and 300-800 km."""
    nb = nb or Neighbours(bench.stations)
    return masked_mean(nb.idw(0, 300), bench.values), masked_mean(nb.idw(300, 800), bench.values)
