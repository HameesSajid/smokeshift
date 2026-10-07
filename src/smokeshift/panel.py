"""Turn the raw EPA files into one monitor-by-hour panel.

Raw: about 27 million rows, one per monitor-instrument-hour, in local and
GMT time. Output: a dense matrix (monitors x hours, GMT) holding only monitors
that reported consistently in every year, plus a table describing them.

Decisions made here, and why:
* GMT timestamps, so that "the same hour" means the same instant everywhere.
  Spatial features depend on that.
* Collocated instruments at one site are averaged.
* Readings outside a physical range are treated as missing. Small negative
  values are kept: they are instrument noise around zero, and deleting them
  would bias clean-air averages upwards.
* Nothing is imputed. A missing hour stays missing.
"""
import json

import numpy as np
import pandas as pd

from . import config

COLS = ["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude", "Date GMT", "Time GMT",
        "Sample Measurement", "State Name", "County Name"]
EPOCH = pd.Timestamp(f"{config.YEARS[0]}-01-01")
N_HOURS = int((pd.Timestamp(f"{config.YEARS[-1] + 1}-01-01") - EPOCH) / pd.Timedelta(hours=1))
NOT_CONUS = {"02", "15", "72", "78", "80", "CC", "66"}  # Alaska, Hawaii, territories, Mexico, Canada

REGIONS = {
    "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey",
                  "New York", "Pennsylvania"],
    "Midwest": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri",
                "Nebraska", "North Dakota", "South Dakota"],
    "South": ["Delaware", "District Of Columbia", "Florida", "Georgia", "Maryland", "North Carolina",
              "South Carolina", "Virginia", "West Virginia", "Alabama", "Kentucky", "Mississippi", "Tennessee",
              "Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "West": ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "California",
             "Oregon", "Washington"],
}
STATE_REGION = {s: r for r, states in REGIONS.items() for s in states}


def _read_year(year, log):
    """One year of raw rows -> (site, hour index, value) with instruments averaged."""
    path = config.RAW / "epa" / f"hourly_88101_{year}.zip"
    parts, meta, n_raw, n_bad = [], [], 0, 0
    reader = pd.read_csv(path, usecols=COLS, chunksize=2_000_000,
                         dtype={"State Code": str, "County Code": str, "Site Num": str})
    for chunk in reader:
        n_raw += len(chunk)
        chunk = chunk[~chunk["State Code"].isin(NOT_CONUS)]
        site = chunk["State Code"] + "-" + chunk["County Code"] + "-" + chunk["Site Num"]
        dcat = chunk["Date GMT"].astype("category")
        days = (pd.to_datetime(dcat.cat.categories) - EPOCH).days.to_numpy()
        tcat = chunk["Time GMT"].astype("category")
        hours = np.array([int(t[:2]) for t in tcat.cat.categories])
        hour_idx = days[dcat.cat.codes] * 24 + hours[tcat.cat.codes]
        v = chunk["Sample Measurement"].to_numpy(dtype="float32")
        ok = (v >= config.VALID_RANGE[0]) & (v <= config.VALID_RANGE[1]) & (hour_idx >= 0) & (hour_idx < N_HOURS)
        n_bad += int((~ok).sum())
        parts.append(pd.DataFrame({"site": site.to_numpy()[ok], "hour": hour_idx[ok].astype("int32"), "value": v[ok]}))
        meta.append(chunk.assign(site=site).drop_duplicates("site")[
            ["site", "Latitude", "Longitude", "State Name", "County Name"]])
    df = pd.concat(parts, ignore_index=True)
    df["site"] = df["site"].astype("category")
    hourly = df.groupby(["site", "hour"], observed=True)["value"].mean().reset_index()
    log(f"  {year}: {n_raw:,} raw rows, {hourly['site'].nunique():,} sites, {n_bad:,} out-of-range readings dropped")
    return hourly, pd.concat(meta).drop_duplicates("site"), {"year": year, "raw_rows": n_raw, "out_of_range": n_bad}


def build(log=print):
    """Read all years, pivot to a dense matrix and keep the monitors that pass the completeness rule."""
    frames, metas, stats = [], [], []
    for year in config.YEARS:
        h, m, s = _read_year(year, log)
        frames.append(h), metas.append(m), stats.append(s)
    meta = pd.concat(metas).drop_duplicates("site").set_index("site")
    sites = sorted(set().union(*[set(f["site"].astype(str)) for f in frames]))
    row = {s: i for i, s in enumerate(sites)}

    wide = np.full((len(sites), N_HOURS), np.nan, dtype="float32")
    for f in frames:
        wide[f["site"].astype(str).map(row).to_numpy(), f["hour"].to_numpy()] = f["value"].to_numpy()

    hours = pd.date_range(EPOCH, periods=N_HOURS, freq="h")
    year_of = hours.year.to_numpy()
    completeness = np.stack([np.isfinite(wide[:, year_of == y]).mean(axis=1) for y in config.YEARS], axis=1)
    keep = (completeness >= config.MIN_COMPLETENESS).all(axis=1)
    log(f"  {len(sites):,} sites in total; {int(keep.sum()):,} report at least "
        f"{config.MIN_COMPLETENESS:.0%} of hours in every year")

    values = wide[keep]
    stations = meta.loc[np.array(sites)[keep]].reset_index().rename(columns={
        "Latitude": "lat", "Longitude": "lon", "State Name": "state", "County Name": "county"})
    stations["region"] = stations["state"].map(STATE_REGION)
    for j, y in enumerate(config.YEARS):
        stations[f"complete_{y}"] = completeness[keep, j].round(4)
        stations[f"mean_{y}"] = np.nanmean(values[:, year_of == y], axis=1).round(2)
        stations[f"hours_over_usg_{y}"] = (values[:, year_of == y] > config.USG).sum(axis=1)
    summary = {
        "years": config.YEARS, "raw_rows": int(sum(s["raw_rows"] for s in stats)), "per_year": stats,
        "sites_total": len(sites), "sites_kept": int(keep.sum()), "hours": N_HOURS,
        "observations_kept": int(np.isfinite(values).sum()),
        "missing_share_kept": float(np.isnan(values).mean()),
        "negative_share": float((values < 0).sum() / np.isfinite(values).sum()),
    }
    return values, hours, stations, summary


def save(values, stations, summary):
    """Write panel.npz, stations.csv and panel_summary.json to data/processed."""
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    # Tenths of a µg/m³ as int16 with a sentinel for missing: 5x smaller than floats.
    tenths = np.where(np.isfinite(values), np.round(values * 10), -32768).astype("int16")
    np.savez_compressed(config.PROCESSED / "panel.npz", tenths=tenths, start=str(EPOCH))
    stations.to_csv(config.PROCESSED / "stations.csv", index=False)
    (config.PROCESSED / "panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load():
    """Return (values [monitors x hours, float32 with NaN], hours, stations)."""
    z = np.load(config.PROCESSED / "panel.npz")
    tenths = z["tenths"]
    values = np.where(tenths == -32768, np.nan, tenths / 10).astype("float32")
    hours = pd.date_range(str(z["start"]), periods=values.shape[1], freq="h")
    stations = pd.read_csv(config.PROCESSED / "stations.csv", dtype={"site": str})
    return values, hours, stations


def run(log=print):
    """Build and save the panel."""
    values, hours, stations, summary = build(log)
    save(values, stations, summary)
    log(f"  panel: {values.shape[0]} monitors x {values.shape[1]:,} hours, "
        f"{summary['missing_share_kept']:.1%} missing, saved to {config.PROCESSED}")
    return summary
