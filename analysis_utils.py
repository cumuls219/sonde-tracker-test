import math
import numpy as np
import pandas as pd

RD = 287.05
CP = 1004.0
G = 9.80665
EPS = 0.622


def latlon_to_xy_km(df: pd.DataFrame, lat_col="Lat(deg)", lon_col="Lon(deg)") -> pd.DataFrame:
    out = df.copy()
    lat0 = out[lat_col].iloc[0]
    lon0 = out[lon_col].iloc[0]
    out["x_km"] = (out[lon_col] - lon0) * 111.32 * math.cos(math.radians(lat0))
    out["y_km"] = (out[lat_col] - lat0) * 111.32
    out["z_km"] = out["Alt(m)"] / 1000.0
    return out


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wspd = out["Wspd(knot)"]
    wdir = np.deg2rad(out["Wdir(deg)"])
    # meteorological direction: wind from direction; u positive east, v positive north
    out["u_kt"] = -wspd * np.sin(wdir)
    out["v_kt"] = -wspd * np.cos(wdir)
    return out


def downsample_by_seconds(df: pd.DataFrame, interval_s: int) -> pd.DataFrame:
    if interval_s <= 1:
        return df.copy()
    work = df.copy()
    work["_bucket"] = (work["time_s"] // interval_s).astype(int)
    idx = work.groupby("_bucket")["time_s"].idxmin()
    out = work.loc[idx].drop(columns=["_bucket"]).reset_index(drop=True)
    # 시작/끝은 반드시 포함
    add = pd.concat([df.iloc[[0]], out, df.iloc[[-1]]], ignore_index=True)
    add = add.drop_duplicates(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    return add


def downsample_by_altitude(df: pd.DataFrame, interval_m: int) -> pd.DataFrame:
    if interval_m <= 1:
        return df.copy()
    work = df.copy()
    work["_bucket"] = (work["Alt(m)"] // interval_m).astype(int)
    idx = work.groupby("_bucket")["Alt(m)"].idxmin()
    out = work.loc[idx].drop(columns=["_bucket"]).reset_index(drop=True)
    add = pd.concat([df.iloc[[0]], out, df.iloc[[-1]]], ignore_index=True)
    add = add.drop_duplicates(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    return add


def cloud_layers(df: pd.DataFrame, rh_threshold=85.0, spread_threshold=2.0, min_thickness_m=100.0):
    work = df.sort_values("Alt(m)").copy()
    spread = work["T(C)"] - work["Dew(deg)"]
    mask = ((work["U(%)"] >= rh_threshold) | (spread <= spread_threshold)).fillna(False).values
    layers = []
    start_i = None
    for i, ok in enumerate(mask):
        if ok and start_i is None:
            start_i = i
        if (not ok or i == len(mask) - 1) and start_i is not None:
            end_i = i if ok and i == len(mask) - 1 else i - 1
            z1 = float(work["Alt(m)"].iloc[start_i])
            z2 = float(work["Alt(m)"].iloc[end_i])
            if z2 - z1 >= min_thickness_m:
                layers.append({"base_m": z1, "top_m": z2, "thickness_m": z2-z1,
                               "base_km": z1/1000.0, "top_km": z2/1000.0})
            start_i = None
    return layers


def inversion_layers(df: pd.DataFrame, min_thickness_m=50.0, min_warming_c=0.2):
    work = df.sort_values("Alt(m)").copy().reset_index(drop=True)
    z = work["Alt(m)"].to_numpy()
    t = work["T(C)"].to_numpy()
    dz = np.diff(z)
    dt = np.diff(t)
    good = (dz > 0) & np.isfinite(dt) & np.isfinite(dz)
    inv = np.zeros_like(dt, dtype=bool)
    inv[good] = dt[good] > 0
    layers = []
    start = None
    for i, ok in enumerate(inv):
        if ok and start is None:
            start = i
        if ((not ok) or i == len(inv)-1) and start is not None:
            end = i if ok and i == len(inv)-1 else i-1
            z1 = z[start]
            z2 = z[end+1]
            warming = t[end+1] - t[start]
            if z2-z1 >= min_thickness_m and warming >= min_warming_c:
                layers.append({"base_m": float(z1), "top_m": float(z2), "thickness_m": float(z2-z1),
                               "warming_c": float(warming), "base_km": float(z1/1000), "top_km": float(z2/1000)})
            start = None
    return layers


def lcl_temperature_bolton(t_c, td_c):
    t = t_c + 273.15
    td = td_c + 273.15
    if not np.isfinite(t) or not np.isfinite(td):
        return np.nan
    return 1.0 / (1.0 / (td - 56.0) + math.log(t / td) / 800.0) + 56.0


def saturation_vapor_pressure_hpa(t_c):
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def mixing_ratio_kgkg(p_hpa, td_c):
    e = saturation_vapor_pressure_hpa(td_c)
    return EPS * e / (p_hpa - e)


def virtual_temp_k(t_c, w_kgkg):
    return (t_c + 273.15) * (1 + 0.61 * w_kgkg)


def thermo_summary(df: pd.DataFrame):
    work = df.dropna(subset=["P(hPa)", "T(C)", "Dew(deg)", "Alt(m)"]).sort_values("Alt(m)").copy()
    if len(work) < 10:
        return {}, pd.DataFrame()
    sfc = work.iloc[0]
    t0 = float(sfc["T(C)"])
    td0 = float(sfc["Dew(deg)"])
    p0 = float(sfc["P(hPa)"])
    z0 = float(sfc["Alt(m)"])
    tlcl_k = lcl_temperature_bolton(t0, td0)
    # dry adiabatic from surface to LCL
    z_lcl = z0 + max(0.0, (t0 + 273.15 - tlcl_k) / 0.0098)
    gamma_m = 0.006  # simple moist lapse rate approximation, K/m
    z = work["Alt(m)"].to_numpy(dtype=float)
    env_t_c = work["T(C)"].to_numpy(dtype=float)
    parcel_t_c = np.empty_like(env_t_c)
    below = z <= z_lcl
    parcel_t_c[below] = t0 - 0.0098 * (z[below] - z0)
    parcel_t_c[~below] = (tlcl_k - 273.15) - gamma_m * (z[~below] - z_lcl)
    diff = parcel_t_c - env_t_c
    # approximate buoyancy using delta T / Tv
    tv_env = env_t_c + 273.15
    buoy = G * diff / tv_env
    cape = 0.0
    cin = 0.0
    lfc = np.nan
    el = np.nan
    positive_seen = False
    for i in range(len(z)-1):
        dz = z[i+1] - z[i]
        if dz <= 0 or not np.isfinite(buoy[i]) or not np.isfinite(buoy[i+1]):
            continue
        b = 0.5 * (buoy[i] + buoy[i+1])
        if b > 0:
            cape += b * dz
            if not positive_seen:
                lfc = z[i]
                positive_seen = True
            el = z[i+1]
        else:
            if not positive_seen:
                cin += b * dz
    # CCL: first level where environmental T roughly equals surface mixing ratio saturation temperature
    w0 = mixing_ratio_kgkg(p0, td0)
    # compute saturation mixing ratio at env T and pressure; find first where ws <= w0
    ws = mixing_ratio_kgkg(work["P(hPa)"].to_numpy(dtype=float), env_t_c)
    ccl_candidates = work.loc[ws <= w0, "Alt(m)"]
    ccl = float(ccl_candidates.iloc[0]) if len(ccl_candidates) else np.nan
    conv_temp = np.nan
    if np.isfinite(ccl):
        # dry adiabat from surface pressure to CCL pressure. Simple fallback using lapse to CCL.
        conv_temp = float(work.loc[work["Alt(m)"].sub(ccl).abs().idxmin(), "T(C)"] + 0.0098 * (ccl - z0))
    profile = work[["Alt(m)", "P(hPa)", "T(C)", "Dew(deg)", "U(%)"]].copy()
    profile["Parcel_T(C)"] = parcel_t_c
    summary = {
        "SBCAPE 근사(J/kg)": max(0.0, cape),
        "SBCIN 근사(J/kg)": cin,
        "LCL(m)": z_lcl,
        "LFC(m)": lfc,
        "EL(m)": el,
        "CCL(m)": ccl,
        "대류온도 근사(C)": conv_temp,
    }
    return summary, profile


def low_level_jet_layers(df: pd.DataFrame, max_alt_m=3000.0, min_speed_kt=20.0, drop_threshold_kt=5.0):
    """하층제트 후보층 탐지.

    기준은 경량 진단용 경험식입니다.
    - max_alt_m 이하의 풍속 극대값
    - 극대 풍속이 min_speed_kt 이상
    - 극대값 위쪽 약 500~1500 m 범위에서 drop_threshold_kt 이상 약화

    공식 현업 판정값이 아니라, 3D/프로파일에서 빠르게 강조하기 위한 후보 탐지입니다.
    """
    required = ["Alt(m)", "Wspd(knot)"]
    if any(c not in df.columns for c in required):
        return []
    work = df.dropna(subset=required).sort_values("Alt(m)").reset_index(drop=True)
    work = work.loc[work["Alt(m)"] <= max_alt_m].copy()
    if len(work) < 8:
        return []

    z = work["Alt(m)"].to_numpy(dtype=float)
    wspd = work["Wspd(knot)"].to_numpy(dtype=float)
    layers = []

    # 작은 잡음을 줄이기 위해 rolling median 기반으로 극대값을 찾음
    smooth = pd.Series(wspd).rolling(window=5, center=True, min_periods=1).median().to_numpy()

    for i in range(1, len(work)-1):
        if not np.isfinite(smooth[i]):
            continue
        is_local_max = smooth[i] >= smooth[i-1] and smooth[i] >= smooth[i+1]
        if not is_local_max or smooth[i] < min_speed_kt:
            continue

        zi = z[i]
        above_mask = (z > zi + 300) & (z <= min(max_alt_m, zi + 1500))
        below_mask = (z >= max(0, zi - 700)) & (z < zi - 100)
        above_min = np.nanmin(smooth[above_mask]) if np.any(above_mask) else np.nan
        below_min = np.nanmin(smooth[below_mask]) if np.any(below_mask) else np.nan

        drop_above = smooth[i] - above_min if np.isfinite(above_min) else 0.0
        drop_below = smooth[i] - below_min if np.isfinite(below_min) else 0.0
        if max(drop_above, drop_below) < drop_threshold_kt:
            continue

        base_m = max(float(z[0]), float(zi - 250))
        top_m = min(float(z[-1]), float(zi + 250))
        layers.append({
            "base_m": base_m,
            "top_m": top_m,
            "core_m": float(zi),
            "base_km": base_m / 1000.0,
            "top_km": top_m / 1000.0,
            "core_km": float(zi / 1000.0),
            "core_wspd_kt": float(wspd[i]),
            "drop_above_kt": float(drop_above) if np.isfinite(drop_above) else np.nan,
            "drop_below_kt": float(drop_below) if np.isfinite(drop_below) else np.nan,
        })

    # 너무 가까운 후보는 가장 강한 것만 남김
    if not layers:
        return []
    layers = sorted(layers, key=lambda r: r["core_wspd_kt"], reverse=True)
    kept = []
    for cand in layers:
        if all(abs(cand["core_m"] - k["core_m"]) > 600 for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda r: r["core_m"])


def layer_mean_table(df: pd.DataFrame, dz_m=1000):
    work = df.copy()
    work["layer"] = (work["Alt(m)"] // dz_m).astype(int)
    rows = []
    for lyr, g in work.groupby("layer"):
        base = lyr * dz_m
        top = base + dz_m
        rows.append({
            "layer": f"{base/1000:.0f}-{top/1000:.0f} km",
            "mean_wspd_kt": g["Wspd(knot)"].mean(),
            "max_wspd_kt": g["Wspd(knot)"].max(),
            "mean_asc_m_min": g["Asc(m/m)"].mean(),
            "mean_temp_c": g["T(C)"].mean(),
            "mean_rh_pct": g["U(%)"].mean(),
            "count": len(g),
        })
    return pd.DataFrame(rows)
