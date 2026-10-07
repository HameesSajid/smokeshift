"""Fetch the raw EPA files into the cache directory.

Source: US EPA Air Quality System pre-generated data files (public domain),
https://aqs.epa.gov/aqsweb/airdata/download_files.html
Parameter 88101 is PM2.5 measured by federal reference or equivalent methods.
"""
import hashlib
import json
import urllib.request

from . import config


def _sha256(path):
    """Checksum of a file, read in 1 MB pieces."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(year, log=print):
    """Download one year's EPA file into the cache, skipping it if a complete copy is already there."""
    url = config.EPA_URL.format(year=year)
    dest = config.RAW / "epa" / f"hourly_88101_{year}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "smokeshift/0.1 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        size = int(r.headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == size:
            log(f"  {dest.name}: already present ({size / 1e6:.0f} MB)")
            return dest
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            done = 0
            for chunk in iter(lambda: r.read(1 << 20), b""):
                f.write(chunk)
                done += len(chunk)
        if done != size:
            raise IOError(f"{url}: got {done} of {size} bytes")
        tmp.replace(dest)
    log(f"  {dest.name}: downloaded {size / 1e6:.0f} MB")
    return dest


STATES_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json"


def fetch_states(log=print):
    """State outlines for the dashboard map (US Census Bureau cartographic boundaries, via us-atlas)."""
    dest = config.RAW / "us-atlas" / "states-10m.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(STATES_URL, headers={"User-Agent": "smokeshift/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        log(f"  {dest.name}: downloaded {dest.stat().st_size / 1e3:.0f} kB")
    return dest


def run(log=print):
    """Fetch every raw file and write their sizes and checksums to data/processed/raw_manifest.json."""
    log(f"Downloading EPA hourly PM2.5 files to {config.RAW / 'epa'}")
    fetch_states(log)
    manifest = []
    for year in config.YEARS:
        path = fetch(year, log)
        manifest.append({"file": path.name, "url": config.EPA_URL.format(year=year),
                         "bytes": path.stat().st_size, "sha256": _sha256(path)})
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED / "raw_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
