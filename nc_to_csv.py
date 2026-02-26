#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ERA5 single-level zip (.zip containing .nc) -> tidy pandas DataFrame.

- Required vars: u10, v10, t2m, d2m, sp, blh
- Optional vars: tp (total precipitation)

Also:
- Rename valid_time -> time if needed
- Squeeze expver dimension if exists
- Unit conversions:
  t2m/d2m: K -> C
  sp: Pa -> hPa
  tp: m -> mm
- Derived:
  wind_speed, wind_dir (meteorological, degrees), rh (%)

Usage (CLI):
  python nc_to_csv.py --zip era5_single_202001.zip --out era5_202001.parquet
  python nc_to_csv.py --zip era5_single_202001.zip --out era5_202001.csv --format csv

Usage (Jupyter):
  import nc_to_csv
  df = nc_to_csv.load_zip_nc_to_df("era5_single_202001.zip")
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ---------------- helpers ----------------
def _pick_first_existing_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _maybe_squeeze_expver(ds: xr.Dataset) -> xr.Dataset:
    # ERA5 sometimes has expver dimension (e.g., expver=1/5)
    if "expver" in ds.dims:
        ds = ds.isel(expver=0, drop=True)
    return ds


def _to_celsius(k: xr.DataArray) -> xr.DataArray:
    return k - 273.15


def _pa_to_hpa(pa: xr.DataArray) -> xr.DataArray:
    return pa / 100.0


def _tp_m_to_mm(tp_m: xr.DataArray) -> xr.DataArray:
    return tp_m * 1000.0


def _rh_from_t_td_c(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    # Magnus approximation
    a = 17.625
    b = 243.04
    rh = 100.0 * np.exp((a * td_c) / (b + td_c)) / np.exp((a * t_c) / (b + t_c))
    return rh.clip(0, 100)


def _wind_speed_dir(u: xr.DataArray, v: xr.DataArray):
    # wind direction: meteorological (from which it blows)
    speed = np.sqrt(u ** 2 + v ** 2)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


# ---------------- core ----------------
def load_zip_nc_to_df(zip_path: str | Path) -> pd.DataFrame:
    """
    Load ERA5 single-level zip file and return tidy DataFrame.

    Parameters
    ----------
    zip_path : str | Path
        Path to era5_single_yyyymm.zip

    Returns
    -------
    pd.DataFrame
        Columns include time/latitude/longitude and selected variables.
        If tp not present, tp_mm column will be absent.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract first .nc from zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
            if not nc_members:
                raise ValueError(f"No .nc file found inside zip: {zip_path}")

            nc_name = nc_members[0]
            extracted_nc = tmpdir / Path(nc_name).name
            with zf.open(nc_name) as f_in, open(extracted_nc, "wb") as f_out:
                f_out.write(f_in.read())

        # Open NetCDF
        ds = xr.open_dataset(extracted_nc, engine="netcdf4")
        ds = _maybe_squeeze_expver(ds)

        # Variable map (canonical short names)
        var_map = {
            "u10": ["u10", "10m_u_component_of_wind"],
            "v10": ["v10", "10m_v_component_of_wind"],
            "t2m": ["t2m", "2m_temperature"],
            "d2m": ["d2m", "2m_dewpoint_temperature"],
            "sp":  ["sp", "surface_pressure"],
            "blh": ["blh", "boundary_layer_height"],
            "tp":  ["tp", "total_precipitation"],  # optional
        }

        required_keys = ["u10", "v10", "t2m", "d2m", "sp", "blh"]
        optional_keys = ["tp"]

        chosen: dict[str, str] = {}
        missing_required: list[tuple[str, list[str]]] = []

        for key, candidates in var_map.items():
            found = _pick_first_existing_var(ds, candidates)
            if found is None:
                if key in required_keys:
                    missing_required.append((key, candidates))
            else:
                chosen[key] = found

        if missing_required:
            msg = "\n".join([f"- {k}: tried {cands}" for k, cands in missing_required])
            raise KeyError(
                "Some required variables are missing from this NetCDF.\n"
                f"Missing:\n{msg}\n\nAvailable variables: {list(ds.variables)}"
            )

        # Keep only vars that exist
        vars_to_keep = [chosen[k] for k in required_keys]
        if "tp" in chosen:
            vars_to_keep.append(chosen["tp"])
        ds_sel = ds[vars_to_keep]

        # Coord compatibility
        # valid_time -> time
        if "valid_time" in ds_sel.coords and "time" not in ds_sel.coords:
            ds_sel = ds_sel.rename({"valid_time": "time"})

        # lat/lon -> latitude/longitude (if needed)
        rename_coords = {}
        if "lat" in ds_sel.coords and "latitude" not in ds_sel.coords:
            rename_coords["lat"] = "latitude"
        if "lon" in ds_sel.coords and "longitude" not in ds_sel.coords:
            rename_coords["lon"] = "longitude"
        if rename_coords:
            ds_sel = ds_sel.rename(rename_coords)

        # Build outputs with conversions + derived
        u = ds_sel[chosen["u10"]].rename("u10")
        v = ds_sel[chosen["v10"]].rename("v10")
        t2m_c = _to_celsius(ds_sel[chosen["t2m"]]).rename("t2m_c")
        d2m_c = _to_celsius(ds_sel[chosen["d2m"]]).rename("d2m_c")
        sp_hpa = _pa_to_hpa(ds_sel[chosen["sp"]]).rename("sp_hpa")
        blh_m = ds_sel[chosen["blh"]].rename("blh_m")

        wind_speed, wind_dir = _wind_speed_dir(u, v)
        wind_speed = wind_speed.rename("wind_speed")
        wind_dir = wind_dir.rename("wind_dir")
        rh = _rh_from_t_td_c(t2m_c, d2m_c).rename("rh")

        data_vars = {
            "u10": u,
            "v10": v,
            "t2m_c": t2m_c,
            "d2m_c": d2m_c,
            "sp_hpa": sp_hpa,
            "blh_m": blh_m,
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "rh": rh,
        }

        if "tp" in chosen:
            tp_mm = _tp_m_to_mm(ds_sel[chosen["tp"]]).rename("tp_mm")
            data_vars["tp_mm"] = tp_mm

        ds_out = xr.Dataset(data_vars)

        # To DataFrame
        df = ds_out.to_dataframe().reset_index()

        # Ensure time dtype
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=False)

        # Optional: reorder columns nicely if present
        front = [c for c in ["time", "latitude", "longitude"] if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]

        return df


# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", required=True, help="Path to era5_single_yyyymm.zip")
    p.add_argument("--out", dest="out_path", required=True, help="Output file path (.parquet or .csv)")
    p.add_argument("--format", choices=["parquet", "csv"], default=None, help="Force output format")
    p.add_argument("--csv-index", action="store_true", help="When output is csv, keep index (default: False)")
    args = p.parse_args()

    zip_path = Path(args.zip_path)
    out_path = Path(args.out_path)

    fmt = args.format
    if fmt is None:
        suf = out_path.suffix.lower()
        if suf == ".parquet":
            fmt = "parquet"
        elif suf == ".csv":
            fmt = "csv"
        else:
            fmt = "parquet"

    df = load_zip_nc_to_df(zip_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=args.csv_index)

    print(f"[OK] rows={len(df):,} cols={len(df.columns)} -> {out_path}")


if __name__ == "__main__":
    main()