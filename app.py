import math
import time
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from PIL import Image

from parser_raw import parse_upp_raw
from analysis_utils import (
    latlon_to_xy_km, add_wind_components, downsample_by_seconds, downsample_by_altitude,
    cloud_layers, inversion_layers, low_level_jet_layers, thermo_summary, layer_mean_table
)

st.set_page_config(page_title="Sonde Tracker RAW v6.8.15 Terrain Label Wind Fix", page_icon="🎈", layout="wide")

st.title("🎈 Sonde Tracker RAW v6.8.15 Terrain Label Wind Fix")
st.caption("UPP RAW 원시자료 업로드는 유지하고, 3D Tracker와 Log-P 연직축, 구름·역전층·하층제트 강조, 기상학적 바람깃 방향 보정, 하단 지도 지역명·강한 지형 음영 보강, 3D 바닥면 실제 지도/기본 격자, 상승 애니메이션을 포함한 테스트용 버전입니다.")


def metric_fmt(v, unit="", digits=1):
    if v is None or pd.isna(v):
        return "-"
    return f"{v:,.{digits}f}{unit}"


def variable_label(col):
    """원시자료 컬럼명을 업무용 표시명으로 변환."""
    labels = {
        "T(C)": "T(C) 기온",
        "U(%)": "RH(상대습도)",
        "Wspd(knot)": "Wspd(knot) 풍속",
        "Asc(m/m)": "Asc(m/m) 상승률(분당 m)",
        "Asc(m/s)": "Asc(m/s) 상승속도",
        "P(hPa)": "P(hPa) 기압",
        "Dew(deg)": "Dew(deg) 노점",
    }
    return labels.get(col, col)


def get_display_df(df, mode):
    if mode == "전체":
        return df.copy()
    if mode == "2초 간격":
        return downsample_by_seconds(df, 2)
    if mode == "5초 간격":
        return downsample_by_seconds(df, 5)
    if mode == "10초 간격":
        return downsample_by_seconds(df, 10)
    if mode == "고도 50m 간격":
        return downsample_by_altitude(df, 50)
    if mode == "고도 100m 간격":
        return downsample_by_altitude(df, 100)
    return downsample_by_seconds(df, 5)



def prepare_vertical_axis(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """3D 표시용 z축을 실제 고도 또는 Log-P 좌표로 변환.

    - 실제 고도: Alt(m)/1000, 전체 표시
    - Skew-T형 Log-P: P(hPa)만 사용하며 지상~100hPa 범위(P>=100hPa)만 표시
    """
    out = df.copy()
    if mode == "Skew-T형 Log-P":
        if "P(hPa)" not in out.columns:
            raise ValueError("Log-P 축을 사용하려면 P(hPa) 컬럼이 필요합니다.")
        out["P(hPa)"] = pd.to_numeric(out["P(hPa)"], errors="coerce")
        # Log-P 모드에서는 그림 가림을 줄이고 기상학적으로 주로 쓰는 지상~100hPa 범위만 표시.
        # 압력 좌표에서 '100hPa까지'는 P >= 100hPa 조건에 해당함.
        out = out.loc[out["P(hPa)"].notna() & (out["P(hPa)"] >= 100.0)].copy()
        if len(out) < 2:
            raise ValueError("Log-P 축 표시를 위한 100hPa 이상 자료가 부족합니다.")
        p = out["P(hPa)"]
        p0 = float(p.iloc[0]) if p.notna().any() else 1000.0
        # Pressure-only Log-P coordinate. 배율은 눈금 표시용이며 실제 계산에는 영향 없음.
        out["z_plot"] = 7.0 * np.log(p0 / p.clip(lower=1.0))
    else:
        out["z_plot"] = out["Alt(m)"] / 1000.0
    return out


def vertical_axis_title(mode: str) -> str:
    if mode == "Skew-T형 Log-P":
        return "Log-P 연직축: P(hPa)"
    return "고도(km)"


def vertical_axis_ticks(df_display: pd.DataFrame, mode: str):
    """변환된 z_plot 좌표에 실제 고도/기압 라벨을 붙이기 위한 tick 설정."""
    if "z_plot" not in df_display.columns or len(df_display) == 0:
        return None, None

    if mode == "Skew-T형 Log-P":
        p = pd.to_numeric(df_display["P(hPa)"], errors="coerce").dropna()
        if len(p) < 2:
            return None, None
        p0 = float(p.iloc[0])
        p_min, p_max = float(p.min()), float(p.max())
        standard_ticks = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
        ticks = [pt for pt in standard_ticks if p_min <= pt <= p_max]
        # 지상기압이 1000hPa보다 높으면 첫 관측 기압도 보조 눈금으로 추가.
        if p_max > 1000 and all(abs(p_max - t) > 15 for t in ticks):
            ticks = [round(p_max)] + ticks
        if 100 not in ticks and p_min <= 110:
            ticks.append(100)
        ticks = sorted(set(ticks), reverse=True)
        zvals = [7.0 * np.log(p0 / max(pt, 1.0)) for pt in ticks]
        labels = [f"{pt:.0f} hPa" for pt in ticks]
        return zvals, labels

    max_alt = float(np.nanmax(df_display["Alt(m)"]))
    if not np.isfinite(max_alt):
        return None, None
    max_km = max_alt / 1000.0
    if max_km <= 2:
        alt_ticks_km = np.arange(0, math.ceil(max_km) + 0.5, 0.5)
    elif max_km <= 8:
        alt_ticks_km = np.arange(0, math.ceil(max_km) + 1, 1)
    else:
        alt_ticks_km = np.arange(0, math.ceil(max_km / 2) * 2 + 1, 2)
    return list(alt_ticks_km), [f"{v:g} km" for v in alt_ticks_km]



def padded_range(values, pad_ratio=0.08, min_pad=0.05):
    arr = pd.to_numeric(values, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    vmin = float(arr.min())
    vmax = float(arr.max())
    span = vmax - vmin
    if span <= 0:
        span = min_pad
    pad = max(span * pad_ratio, min_pad)
    return [vmin - pad, vmax + pad]






def context_factor_from_label(label: str) -> float:
    """지도/3D 공간을 얼마나 여유 있게 보여줄지 결정."""
    return {
        "보통": 1.0,
        "넓게": 1.8,
        "아주 넓게": 2.6,
    }.get(label, 1.8)


def ns_factor_from_label(label: str) -> float:
    """남북(위도) 방향을 동서보다 더 여유 있게 보여주기 위한 배율."""
    return {
        "기본": 1.0,
        "남북 넓게": 1.7,
        "남북 아주 넓게": 2.3,
    }.get(label, 1.7)


def apply_context_to_map_profile(profile: dict, context_factor: float, ns_factor: float = 1.0) -> dict:
    """지도 영역을 넓힐 때 타일 수가 과도해지지 않도록 범위와 타일 제한을 함께 조정."""
    out = dict(profile)
    out["pad_ratio"] = float(out.get("pad_ratio", 0.12)) * context_factor
    out["min_span_deg"] = float(out.get("min_span_deg", 0.10)) * context_factor
    # 넓은 영역은 자연스럽게 줌이 낮아지므로 과도한 타일 요청을 방지
    if context_factor >= 2.5:
        out["max_tiles"] = min(int(out.get("max_tiles", 8)), 8)
    return out


def _lonlat_to_xy_km(lon, lat, lon0, lat0):
    x = (np.asarray(lon) - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (np.asarray(lat) - lat0) * 111.32
    return x, y


def _tile_num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def _global_pixel(lat_deg, lon_deg, zoom, tile_size=256):
    sin_lat = math.sin(math.radians(lat_deg))
    n = 2.0 ** zoom * tile_size
    x = (lon_deg + 180.0) / 360.0 * n
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n
    return x, y


def _auto_map_bounds(df: pd.DataFrame, pad_ratio=0.12, min_span_deg=0.10,
                     lat_pad_ratio=None, lon_pad_ratio=None,
                     min_lat_span_deg=None, min_lon_span_deg=None):
    lat = pd.to_numeric(df["Lat(deg)"], errors="coerce").dropna()
    lon = pd.to_numeric(df["Lon(deg)"], errors="coerce").dropna()
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_c = 0.5 * (lat_min + lat_max)
    lon_c = 0.5 * (lon_min + lon_max)
    lat_pad = pad_ratio if lat_pad_ratio is None else float(lat_pad_ratio)
    lon_pad = pad_ratio if lon_pad_ratio is None else float(lon_pad_ratio)
    min_lat_span = min_span_deg if min_lat_span_deg is None else float(min_lat_span_deg)
    min_lon_span = min_span_deg if min_lon_span_deg is None else float(min_lon_span_deg)
    lat_span = max(lat_max - lat_min, min_lat_span)
    lon_span = max(lon_max - lon_min, min_lon_span)
    lat_span *= (1 + 2 * lat_pad)
    lon_span *= (1 + 2 * lon_pad)
    return {
        "south": lat_c - lat_span / 2,
        "north": lat_c + lat_span / 2,
        "west": lon_c - lon_span / 2,
        "east": lon_c + lon_span / 2,
    }




def floor_z_for_display(df_display: pd.DataFrame, vertical_mode: str = "실제 고도"):
    """3D 바닥면 표시 높이를 계산한다.

    Log-P 모드에서 바닥 격자가 궤적 시작점/축면에 묻히지 않도록
    자료 최저 z보다 아주 조금 낮은 위치에 바닥면을 둔다.
    """
    z = pd.to_numeric(df_display["z_plot"], errors="coerce").dropna()
    if len(z) == 0:
        return 0.0, 0.02
    z_min = float(z.min())
    z_max = float(z.max())
    z_span = max(z_max - z_min, 1.0)
    offset = max(0.08, z_span * 0.025)
    if vertical_mode == "Skew-T형 Log-P":
        floor_z = z_min - offset
    else:
        floor_z = min(0.0, z_min) - offset
    line_z = floor_z + max(0.015, z_span * 0.004)
    return floor_z, line_z

def _choose_zoom(bounds, max_tiles=8):
    # 행정명·지역명이 보이도록 가능한 높은 줌을 선택하되, 타일 수는 제한한다.
    for z in range(16, 5, -1):
        x0, y0 = _tile_num(bounds["north"], bounds["west"], z)
        x1, y1 = _tile_num(bounds["south"], bounds["east"], z)
        nx = abs(x1 - x0) + 1
        ny = abs(y1 - y0) + 1
        if nx <= max_tiles and ny <= max_tiles:
            return z
    return 6


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_osm_static_map(bounds: dict, zoom: int, out_size: int = 1400):
    """OpenStreetMap 타일을 필요한 범위만 받아 정적 지도 1장으로 합성한다."""
    tile_size = 256
    x_min, y_min = _tile_num(bounds["north"], bounds["west"], zoom)
    x_max, y_max = _tile_num(bounds["south"], bounds["east"], zoom)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    nx, ny = x_max - x_min + 1, y_max - y_min + 1
    canvas = Image.new("RGB", (nx * tile_size, ny * tile_size), (235, 240, 242))
    headers = {"User-Agent": "sonde-3dtracker-busan/1.0"}
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            tile = Image.open(BytesIO(r.content)).convert("RGB")
            canvas.paste(tile, ((x - x_min) * tile_size, (y - y_min) * tile_size))
    px_w, py_n = _global_pixel(bounds["north"], bounds["west"], zoom, tile_size)
    px_e, py_s = _global_pixel(bounds["south"], bounds["east"], zoom, tile_size)
    crop = (
        int(round(px_w - x_min * tile_size)),
        int(round(py_n - y_min * tile_size)),
        int(round(px_e - x_min * tile_size)),
        int(round(py_s - y_min * tile_size)),
    )
    crop = tuple(max(0, v) for v in crop)
    img = canvas.crop(crop)
    # 지도를 정사각형으로 강제 리사이즈하면 바닥면에서 눌려 보이므로,
    # 원본 crop의 가로/세로 비율을 유지한 채 긴 변만 out_size에 맞춘다.
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("지도 crop 영역이 비어 있습니다.")
    scale = float(out_size) / float(max(w, h))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img)


def make_map_floor_trace(raw_df: pd.DataFrame, display_df: pd.DataFrame, vertical_mode: str,
                         out_size: int = 1400, opacity: float = 0.76, grid_cells: int = 128,
                         pad_ratio: float = 0.12, min_span_deg: float = 0.10, max_tiles: int = 8,
                         lat_pad_ratio=None, lon_pad_ratio=None, min_lat_span_deg=None, min_lon_span_deg=None):
    """실제 OSM 정적 지도 1장을 3D 바닥면 Mesh3d로 변환한다.

    Plotly 3D에는 이미지 텍스처를 직접 까는 기능이 제한적이므로,
    지도 이미지를 작은 색상 격자로 줄여 Mesh3d 바닥면처럼 표시한다.
    """
    bounds = _auto_map_bounds(raw_df, pad_ratio=pad_ratio, min_span_deg=min_span_deg,
                              lat_pad_ratio=lat_pad_ratio, lon_pad_ratio=lon_pad_ratio,
                              min_lat_span_deg=min_lat_span_deg, min_lon_span_deg=min_lon_span_deg)
    zoom = _choose_zoom(bounds, max_tiles=max_tiles)
    img = _fetch_osm_static_map(bounds, zoom=zoom, out_size=out_size)

    lat0 = float(raw_df["Lat(deg)"].iloc[0])
    lon0 = float(raw_df["Lon(deg)"].iloc[0])

    # 실제 km 단위 가로/세로 비율에 맞춰 지도 격자도 직사각형으로 생성한다.
    x_w, y_n = _lonlat_to_xy_km(bounds["west"], bounds["north"], lon0, lat0)
    x_e, y_s = _lonlat_to_xy_km(bounds["east"], bounds["south"], lon0, lat0)
    x_span = max(abs(float(x_e) - float(x_w)), 0.1)
    y_span = max(abs(float(y_n) - float(y_s)), 0.1)
    base_cells = int(max(48, min(grid_cells, 180)))
    if x_span >= y_span:
        nx_cells = base_cells
        ny_cells = max(28, int(round(base_cells * y_span / x_span)))
    else:
        ny_cells = base_cells
        nx_cells = max(28, int(round(base_cells * x_span / y_span)))
    nx_cells = int(min(nx_cells, 180))
    ny_cells = int(min(ny_cells, 180))

    small = Image.fromarray(img).resize((nx_cells, ny_cells), Image.BILINEAR)
    arr = np.asarray(small)

    # 이미지 좌표: 위쪽이 north. 격자 꼭짓점은 north->south, west->east 순서.
    lons = np.linspace(bounds["west"], bounds["east"], nx_cells + 1)
    lats = np.linspace(bounds["north"], bounds["south"], ny_cells + 1)
    xs = []
    ys = []
    zs = []
    floor_z, _ = floor_z_for_display(display_df, vertical_mode)
    for lat in lats:
        for lon in lons:
            x, y = _lonlat_to_xy_km(lon, lat, lon0, lat0)
            xs.append(float(x)); ys.append(float(y)); zs.append(floor_z)

    i_list=[]; j_list=[]; k_list=[]; facecolors=[]
    for r in range(ny_cells):
        for c in range(nx_cells):
            v00 = r*(nx_cells+1)+c
            v01 = r*(nx_cells+1)+c+1
            v10 = (r+1)*(nx_cells+1)+c
            v11 = (r+1)*(nx_cells+1)+c+1
            rr, gg, bb = arr[r, c]
            col = f"rgba({int(rr)},{int(gg)},{int(bb)},{opacity:.3f})"
            i_list += [v00, v00]
            j_list += [v10, v11]
            k_list += [v11, v01]
            facecolors += [col, col]

    trace = go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_list, j=j_list, k=k_list,
        facecolor=facecolors,
        flatshading=True,
        opacity=opacity,
        hoverinfo="skip",
        name="바닥면 실제 지도",
        showlegend=True,
    )
    return trace, bounds, zoom




def make_floor_enhancement_traces(df_display: pd.DataFrame, vertical_mode: str = "실제 고도",
                                  x_range=None, y_range=None, floor_z=None,
                                  context_factor: float = 1.0, ns_factor: float = 1.0):
    """하단 지도 보강 trace 생성.

    v6.8.15 핵심 수정:
    - 기존 3D 좌표계는 lon/lat가 아니라 발사지점 기준 x_km/y_km이므로, 지역명을 반드시 x_km/y_km로 변환해 표시한다.
    - DEM 없이도 눈에 보이도록 강한 relief surface + 등고선 contour를 바닥면 바로 위에 얹는다.
    - 별도 탭/복잡한 옵션 없이 기존 3D 바닥면에 기본 표시한다.
    """
    traces = []
    if df_display is None or len(df_display) == 0:
        return traces
    required = ["x_km", "y_km", "Lat(deg)", "Lon(deg)"]
    if any(c not in df_display.columns for c in required):
        return traces

    if x_range is None:
        x_range = padded_range(df_display["x_km"], pad_ratio=0.28 * max(1.0, float(context_factor)), min_pad=3.0)
    if y_range is None:
        y_range = padded_range(df_display["y_km"], pad_ratio=0.28 * max(1.0, float(context_factor)) * max(1.0, float(ns_factor)), min_pad=3.0)
    if x_range is None or y_range is None:
        return traces
    if floor_z is None:
        floor_z, _ = floor_z_for_display(df_display, vertical_mode)

    # 1) 눈에 보이는 하단 relief surface. 실제 DEM은 아니지만, 산지/해안 지형감처럼 보이도록 강하게 표시.
    nx, ny = 46, 46
    xs = np.linspace(float(x_range[0]), float(x_range[1]), nx)
    ys = np.linspace(float(y_range[0]), float(y_range[1]), ny)
    X, Y = np.meshgrid(xs, ys)
    xspan = max(float(x_range[1] - x_range[0]), 0.1)
    yspan = max(float(y_range[1] - y_range[0]), 0.1)
    Xn = (X - float(x_range[0])) / xspan
    Yn = (Y - float(y_range[0])) / yspan

    # 여러 개의 가우시안 봉우리 + 파동을 합성해 입체감을 명확히 함
    relief = (
        0.95 * np.exp(-(((Xn - 0.25) / 0.18) ** 2 + ((Yn - 0.72) / 0.20) ** 2)) +
        0.75 * np.exp(-(((Xn - 0.68) / 0.20) ** 2 + ((Yn - 0.62) / 0.16) ** 2)) +
        0.55 * np.exp(-(((Xn - 0.52) / 0.16) ** 2 + ((Yn - 0.30) / 0.20) ** 2)) +
        0.18 * np.sin(Xn * np.pi * 5.0) * np.cos(Yn * np.pi * 4.0)
    )
    relief = relief - np.nanmin(relief)
    relief = relief / max(float(np.nanmax(relief)), 1e-6)

    # 실제 고도 모드에서는 180m 정도의 과장 지형. Log-P 모드에서도 바닥 위에 보이도록 같은 z_plot 단위 사용.
    relief_height = 0.18 if vertical_mode == "실제 고도" else 0.11
    Z = float(floor_z) + 0.018 + relief * relief_height

    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=relief,
        colorscale=[[0.0, "rgb(214,232,205)"], [0.42, "rgb(177,205,151)"], [0.70, "rgb(139,172,112)"], [1.0, "rgb(104,134,88)"]],
        opacity=0.58,
        showscale=False,
        name="하단 지형 음영",
        hoverinfo="skip",
        contours={"z": {"show": True, "start": float(floor_z) + 0.035, "end": float(floor_z) + relief_height, "size": relief_height / 5.0, "color": "rgba(80,95,70,0.55)", "width": 1}},
        showlegend=True,
    ))

    # 2) 지역명. lon/lat를 현재 3D 좌표계 x_km/y_km로 변환해야 보인다.
    lat0 = float(df_display["Lat(deg)"].iloc[0])
    lon0 = float(df_display["Lon(deg)"].iloc[0])
    places = [
        ("부산", 35.1796, 129.0756), ("김해", 35.2285, 128.8894), ("양산", 35.3350, 129.0370),
        ("울산", 35.5384, 129.3114), ("창원", 35.2279, 128.6811), ("밀양", 35.5038, 128.7466),
        ("거제", 34.8806, 128.6210), ("통영", 34.8544, 128.4332), ("진주", 35.1800, 128.1076),
        ("사천", 35.0038, 128.0642), ("창녕", 35.5446, 128.4922), ("합천", 35.5666, 128.1658),
        ("거창", 35.6867, 127.9095), ("함안", 35.2725, 128.4066), ("기장", 35.2447, 129.2223),
    ]
    label_x, label_y, label_z, label_text = [], [], [], []
    # 축 범위 밖 지명은 제외하되, 살짝 여유를 둔다.
    margin_x = xspan * 0.08
    margin_y = yspan * 0.08
    for name, lat, lon in places:
        x, y = _lonlat_to_xy_km(lon, lat, lon0, lat0)
        x = float(x); y = float(y)
        if (float(x_range[0]) - margin_x <= x <= float(x_range[1]) + margin_x and
            float(y_range[0]) - margin_y <= y <= float(y_range[1]) + margin_y):
            label_x.append(x); label_y.append(y); label_z.append(float(floor_z) + relief_height + 0.10); label_text.append(name)

    # 최소한 발사지점명은 항상 보이도록 추가
    label_x.append(0.0); label_y.append(0.0); label_z.append(float(floor_z) + relief_height + 0.14); label_text.append("발사점")

    traces.append(go.Scatter3d(
        x=label_x, y=label_y, z=label_z,
        mode="markers+text",
        marker=dict(size=4.5, color="rgba(20,20,20,0.88)", symbol="circle"),
        text=label_text,
        textposition="top center",
        textfont=dict(size=15, color="rgba(10,20,30,1)"),
        name="지역명",
        hoverinfo="skip",
        showlegend=True,
    ))
    return traces

def make_floor_grid_traces(df_display: pd.DataFrame, vertical_mode: str = "실제 고도",
                           grid_count: int = 8, show_origin_cross: bool = True,
                           show_sky_walls: bool = False, context_factor: float = 1.0, ns_factor: float = 1.0):
    """3D 바닥면·옆면 격자 표시.

    v6.6:
    - 예시 모식도처럼 바닥 격자를 얇지만 눈에 들어오도록 보정
    - 지도 표시 옵션과 연동되는 연한 하늘색 측면/후면 패널 추가
    - 연직면 격자를 x/y/z 공간에 직접 그려 Plotly 배경색에 의존하지 않도록 함
    """
    if df_display is None or len(df_display) == 0:
        return []

    context_factor = max(1.0, float(context_factor))
    ns_factor = max(1.0, float(ns_factor))
    x_range = padded_range(df_display["x_km"], pad_ratio=0.24 * context_factor, min_pad=1.0 * context_factor)
    y_range = padded_range(df_display["y_km"], pad_ratio=0.24 * context_factor * ns_factor, min_pad=1.0 * context_factor * ns_factor)
    z_range = padded_range(df_display["z_plot"], pad_ratio=0.04, min_pad=0.25)
    if x_range is None or y_range is None or z_range is None:
        return []

    floor_z, line_z = floor_z_for_display(df_display, vertical_mode)
    zmin = floor_z
    zmax = max(float(z_range[1]), float(df_display["z_plot"].max()))

    grid_count = int(max(4, min(grid_count, 14)))
    xs = np.linspace(x_range[0], x_range[1], grid_count + 1)
    ys = np.linspace(y_range[0], y_range[1], grid_count + 1)
    zline_count = 7 if vertical_mode == "실제 고도" else 8
    zs = np.linspace(zmin, zmax, zline_count + 1)

    traces = []

    # 지도와 같이 표시되는 하늘색 연직면: Plotly scene 배경색이 보이지 않는 환경에서도 직접 보이게 함
    if show_sky_walls:
        sky_color = "rgba(210,235,250,0.30)"
        # 뒤쪽 y면
        traces.append(go.Mesh3d(
            x=[x_range[0], x_range[1], x_range[1], x_range[0]],
            y=[y_range[1], y_range[1], y_range[1], y_range[1]],
            z=[zmin, zmin, zmax, zmax],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=sky_color, opacity=0.30, name="하늘색 배경면", hoverinfo="skip", showlegend=False
        ))
        # 왼쪽 x면
        traces.append(go.Mesh3d(
            x=[x_range[0], x_range[0], x_range[0], x_range[0]],
            y=[y_range[0], y_range[1], y_range[1], y_range[0]],
            z=[zmin, zmin, zmax, zmax],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=sky_color, opacity=0.24, name="하늘색 배경면", hoverinfo="skip", showlegend=False
        ))

    # 기본 바닥판: 지도 실패/해제 시에도 바닥이 보이도록 매우 연하게 유지
    traces.append(go.Mesh3d(
        x=[x_range[0], x_range[1], x_range[1], x_range[0]],
        y=[y_range[0], y_range[0], y_range[1], y_range[1]],
        z=[floor_z, floor_z, floor_z, floor_z],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="rgba(230,238,235,0.12)", opacity=0.12,
        name="바닥면 기준판", hoverinfo="skip", showlegend=False,
    ))

    floor_grid_color = "rgba(58,82,94,0.46)"
    wall_grid_color = "rgba(82,120,145,0.34)"
    frame_color = "rgba(42,62,74,0.62)"

    # 바닥면 격자: 너무 굵지 않게, 그러나 배경에 묻히지 않도록
    for x in xs:
        traces.append(go.Scatter3d(
            x=[x, x], y=[y_range[0], y_range[1]], z=[line_z, line_z],
            mode="lines", line=dict(width=1.25, color=floor_grid_color),
            hoverinfo="skip", showlegend=False
        ))
    for y in ys:
        traces.append(go.Scatter3d(
            x=[x_range[0], x_range[1]], y=[y, y], z=[line_z, line_z],
            mode="lines", line=dict(width=1.25, color=floor_grid_color),
            hoverinfo="skip", showlegend=False
        ))

    # 옆면/후면 연직 격자: 예시 모식도처럼 3D 박스감을 만듦
    if show_sky_walls:
        yb = y_range[1]
        xb = x_range[0]
        for x in xs:
            traces.append(go.Scatter3d(
                x=[x, x], y=[yb, yb], z=[zmin, zmax],
                mode="lines", line=dict(width=0.9, color=wall_grid_color),
                hoverinfo="skip", showlegend=False
            ))
        for z in zs:
            traces.append(go.Scatter3d(
                x=[x_range[0], x_range[1]], y=[yb, yb], z=[z, z],
                mode="lines", line=dict(width=0.9, color=wall_grid_color),
                hoverinfo="skip", showlegend=False
            ))
        for y in ys:
            traces.append(go.Scatter3d(
                x=[xb, xb], y=[y, y], z=[zmin, zmax],
                mode="lines", line=dict(width=0.9, color=wall_grid_color),
                hoverinfo="skip", showlegend=False
            ))
        for z in zs:
            traces.append(go.Scatter3d(
                x=[xb, xb], y=[y_range[0], y_range[1]], z=[z, z],
                mode="lines", line=dict(width=0.9, color=wall_grid_color),
                hoverinfo="skip", showlegend=False
            ))

    # 외곽 프레임
    bx = [x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]]
    by = [y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]]
    traces.append(go.Scatter3d(
        x=bx, y=by, z=[line_z] * 5,
        mode="lines", line=dict(width=2.0, color=frame_color),
        name="바닥면 격자", hoverinfo="skip", showlegend=True
    ))

    if show_origin_cross:
        traces.append(go.Scatter3d(
            x=[0, 0], y=[y_range[0], y_range[1]], z=[line_z, line_z],
            mode="lines", line=dict(width=1.8, color="rgba(38,68,82,0.56)"),
            name="남북 기준선", hoverinfo="skip", showlegend=False
        ))
        traces.append(go.Scatter3d(
            x=[x_range[0], x_range[1]], y=[0, 0], z=[line_z, line_z],
            mode="lines", line=dict(width=1.8, color="rgba(38,68,82,0.56)"),
            name="동서 기준선", hoverinfo="skip", showlegend=False
        ))
    # 하단 지도 보강: 강한 지형 음영과 지역명은 기본 격자 trace 묶음에 포함해
    # 3D Tracker와 상승 애니메이션에 모두 동일하게 반영한다.
    traces.extend(make_floor_enhancement_traces(
        df_display,
        vertical_mode=vertical_mode,
        x_range=x_range,
        y_range=y_range,
        floor_z=floor_z,
        context_factor=context_factor,
        ns_factor=ns_factor,
    ))
    return traces

def wind_panel_sample(df: pd.DataFrame, interval_m: int = 500) -> pd.DataFrame:
    """오른쪽 바람 패널용 경량 샘플링.

    원본 전체를 그리면 복잡하므로 고도 간격 기준으로 대표값만 사용한다.
    실제 고도/Log-P 모드 모두 동일하게 Alt(m) 간격으로 샘플링한다.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    work["Alt(m)"] = pd.to_numeric(work["Alt(m)"], errors="coerce")
    work = work.dropna(subset=["Alt(m)", "Wspd(knot)", "Wdir(deg)", "u_kt", "v_kt", "z_plot"]).copy()
    if len(work) == 0:
        return work
    interval_m = max(100, int(interval_m))
    work["_alt_bin"] = (work["Alt(m)"] / interval_m).round().astype(int)
    idx = work.groupby("_alt_bin")["Alt(m)"].idxmin()
    out = work.loc[idx].sort_values("Alt(m)").reset_index(drop=True)
    # 너무 많으면 추가로 줄여서 렌더링 안정성 확보
    if len(out) > 60:
        out = out.iloc[::max(1, int(np.ceil(len(out) / 60)))].reset_index(drop=True)
    return out





def _wind_barb_segments(wdf: pd.DataFrame, df_display: pd.DataFrame):
    """기상청식 2D 연직 바람깃 좌표 생성.

    v6.7:
    - 모든 바람깃의 기준점을 하나의 연직 기준선(x=0)에 고정
    - 각 고도/기압층에 짧은 wind barb만 붙임
    - 패널을 가로로 가득 채우는 선을 없애고, 예시 자료처럼 세로축 옆에 촘촘히 배치
    - 풍향은 2D 패널 안에서 작게만 기울여 가시성을 우선
    """
    if wdf is None or len(wdf) == 0:
        return [], [], []

    zvals = pd.to_numeric(wdf["z_plot"], errors="coerce").dropna().sort_values().to_numpy()
    if len(zvals) >= 2:
        gaps = np.diff(zvals)
        gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
        dy_unit = float(np.nanmedian(gaps)) if len(gaps) else 0.15
    else:
        y_span = float(np.nanmax(df_display["z_plot"]) - np.nanmin(df_display["z_plot"]))
        dy_unit = max(y_span * 0.035, 0.12)
    dy_unit = max(dy_unit, 0.09)

    # 화면 좌표계 기준. x는 좌우, y는 고도/압력축.
    base_x = 0.0
    staff_len = 0.34       # 짧게: 패널 전체를 가로지르지 않음
    barb_len = 0.15
    flag_len = 0.20
    spacing = 0.038
    max_vertical_tilt = dy_unit * 0.32

    xs, ys, hover_points = [], [], []
    for _, r in wdf.iterrows():
        spd = float(r["Wspd(knot)"])
        wdir = float(r["Wdir(deg)"])
        z0 = float(r["z_plot"])

        # 기상학적 풍향: 바람이 불어오는 방향.
        # 바람깃은 "불어오는 쪽"을 향해 보여야 하므로, 실제 이동성분(-sin/-cos)이 아니라
        # 풍향 자체의 방향성(+sin/+cos)을 사용한다.
        # 예: 270° 서풍 → 서쪽 방향, 0° 북풍 → 북쪽 방향.
        rad = np.deg2rad(wdir)
        dx_raw = np.sin(rad)
        dy_raw = np.cos(rad)
        if abs(dx_raw) < 0.28:
            dx_raw = 0.28 if dx_raw >= 0 else -0.28
        x1 = base_x + staff_len * np.sign(dx_raw)
        y1 = z0 + max_vertical_tilt * dy_raw

        # 줄기
        xs += [base_x, x1, None]
        ys += [z0, y1, None]

        # 줄기 단위벡터와 법선벡터
        vx, vy = x1 - base_x, y1 - z0
        norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
        ux, uy = vx / norm, vy / norm
        # 깃은 줄기 한쪽에 통일되게 붙여 가시성 확보
        nx, ny = -uy, ux
        if ny < 0:
            nx, ny = -nx, -ny

        speed5 = int(round(spd / 5.0) * 5)
        n50 = speed5 // 50
        rem = speed5 % 50
        n10 = rem // 10
        n5 = 1 if (rem % 10) >= 5 else 0

        cursor = 0.035
        # 50 kt 삼각 깃
        for _ in range(n50):
            bx = x1 - ux * cursor
            by = y1 - uy * cursor
            p1x = bx - ux * spacing * 1.25
            p1y = by - uy * spacing * 1.25
            p2x = bx + nx * flag_len
            p2y = by + ny * flag_len
            xs += [bx, p2x, p1x, bx, None]
            ys += [by, p2y, p1y, by, None]
            cursor += spacing * 2.0
        # 10 kt 긴 깃
        for _ in range(n10):
            bx = x1 - ux * cursor
            by = y1 - uy * cursor
            xs += [bx, bx + nx * barb_len, None]
            ys += [by, by + ny * barb_len, None]
            cursor += spacing
        # 5 kt 짧은 깃
        if n5:
            bx = x1 - ux * cursor
            by = y1 - uy * cursor
            xs += [bx, bx + nx * barb_len * 0.58, None]
            ys += [by, by + ny * barb_len * 0.58, None]

        hover_points.append((base_x, z0, f"고도: {r['Alt(m)']:.0f} m<br>기압: {r['P(hPa)']:.0f} hPa<br>풍향/풍속: {wdir:.0f}° / {spd:.1f} kt"))
    return xs, ys, hover_points

def make_wind_barb_panel(df_display: pd.DataFrame, vertical_mode: str = "실제 고도", interval_m: int = 500,
                         clouds=None, inversions=None, llj_layers=None,
                         show_cloud=False, show_inversion=False, show_llj=False) -> go.Figure:
    """3D Tracker 오른쪽에 붙이는 기상청식 연직 바람깃 패널."""
    wdf = wind_panel_sample(df_display, interval_m=interval_m)
    fig = go.Figure()

    if wdf is None or len(wdf) == 0:
        fig.update_layout(title="연직 바람깃", height=900)
        return fig

    def add_layer_rect(layers, color, opacity, label):
        if not layers:
            return
        for lyr in layers:
            layer_df = df_display[(df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"])]
            if len(layer_df) < 1:
                continue
            y0 = float(layer_df["z_plot"].min())
            y1 = float(layer_df["z_plot"].max())
            fig.add_hrect(y0=y0, y1=y1, x0=-1.05, x1=1.05, fillcolor=color, opacity=opacity,
                          line_width=0, annotation_text=label, annotation_position="top left")

    if show_cloud:
        add_layer_rect(clouds, "rgba(165,165,165,0.20)", 0.20, "cloud")
    if show_inversion:
        add_layer_rect(inversions, "rgba(142,68,173,0.16)", 0.16, "inv")
    if show_llj:
        add_layer_rect(llj_layers, "rgba(230,126,34,0.16)", 0.16, "LLJ")

    xs, ys, hover_points = _wind_barb_segments(wdf, df_display)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(width=2.0, color="rgba(0,70,210,1.0)"),
        hoverinfo="skip", showlegend=False
    ))
    if hover_points:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in hover_points], y=[p[1] for p in hover_points],
            mode="markers", marker=dict(size=3, color="rgba(0,70,210,0.45)"),
            hovertext=[p[2] for p in hover_points], hoverinfo="text", showlegend=False
        ))

    tick_vals, tick_text = vertical_axis_ticks(df_display, vertical_mode)
    yaxis_cfg = dict(title=dict(text=vertical_axis_title(vertical_mode), font=dict(color="rgba(20,35,50,0.95)")), showgrid=True, gridcolor="rgba(120,150,175,0.38)", zeroline=False, tickfont=dict(color="rgba(20,35,50,0.95)"))
    if tick_vals is not None and tick_text is not None:
        yaxis_cfg.update(dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text))

    # 연직 바람깃 패널의 세로 비율을 조금 더 압축해 가시성을 개선
    panel_height = 940 if vertical_mode == "Skew-T형 Log-P" else 660
    fig.update_layout(
        title=dict(text="연직 바람깃", font=dict(color="rgba(15,30,45,1)", size=17)),
        xaxis=dict(
            title=dict(text="연직 일직선 기준 바람깃", font=dict(color="rgba(20,35,50,0.95)")),
            range=[-0.68, 0.68], showgrid=False, showticklabels=False,
            zeroline=True, zerolinecolor="rgba(25,45,70,0.78)", zerolinewidth=2
        ),
        yaxis=yaxis_cfg,
        height=panel_height,
        margin=dict(l=12, r=12, t=52, b=42),
        showlegend=False,
        plot_bgcolor="rgba(255,255,255,0.98)",
        paper_bgcolor="rgba(255,255,255,0.98)",
        annotations=[dict(text="기상청식 바람깃<br>5·10·50 kt", x=0.5, y=1.045, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color="rgba(20,35,50,0.90)"))]
    )
    return fig

def make_cloud_strip_panel(df_display: pd.DataFrame, vertical_mode: str = "실제 고도", clouds=None) -> go.Figure:
    """3D Tracker 옆에 붙이는 경량 구름 영역 패널.

    - y축은 3D Tracker의 연직축과 같은 z_plot 사용
    - x축은 의미 없는 고정 폭의 좁은 구름 띠
    - cloud_layers 결과를 배경 음영으로만 표시해 매우 가볍게 유지
    """
    fig = go.Figure()
    if df_display is None or len(df_display) == 0:
        fig.update_layout(title="구름 영역", height=900)
        return fig

    tick_vals, tick_text = vertical_axis_ticks(df_display, vertical_mode)
    yaxis_cfg = dict(title=vertical_axis_title(vertical_mode), showgrid=True, zeroline=False)
    if tick_vals is not None and tick_text is not None:
        yaxis_cfg.update(dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text))

    fig.add_trace(go.Scatter(x=[0.5], y=[df_display["z_plot"].min()], mode="markers", marker=dict(size=0.1, opacity=0), showlegend=False, hoverinfo="skip"))

    if clouds:
        for lyr in clouds:
            layer_df = df_display[(df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"])]
            if len(layer_df) < 1:
                continue
            y0 = float(layer_df["z_plot"].min())
            y1 = float(layer_df["z_plot"].max())
            fig.add_hrect(
                y0=y0, y1=y1, x0=0.08, x1=0.92,
                fillcolor="rgba(165,165,165,0.38)", opacity=0.38, line_width=0,
                annotation_text="cloud", annotation_position="top left"
            )

    # 연직 바람깃 패널의 세로 비율을 조금 더 압축해 가시성을 개선
    panel_height = 940 if vertical_mode == "Skew-T형 Log-P" else 660
    fig.update_layout(
        title="구름 영역",
        xaxis=dict(title="", range=[0,1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=yaxis_cfg,
        height=panel_height,
        margin=dict(l=10, r=10, t=45, b=35),
        showlegend=False,
        annotations=[dict(text="구름 가능층<br>배경 음영", x=0.5, y=1.04, xref="paper", yref="paper", showarrow=False, font=dict(size=10, color="gray"))]
    )
    return fig

def sonde_symbol_geometry(row: pd.Series, floor_z: float, z_span: float):
    """현재 존데 위치를 보기 좋은 풍선+존데 상징으로 그리기 위한 기하값."""
    x = float(row["x_km"])
    y = float(row["y_km"])
    z = float(row["z_plot"])
    tether = max(0.28, z_span * 0.045)
    sonde_z = max(floor_z + 0.10, z - tether)
    # 작은 라디오존데 박스/안테나/센서 위치를 위한 오프셋
    dx = 0.10
    dy = 0.03
    ant_h = max(0.12, z_span * 0.015)
    return {
        "x": x, "y": y, "z": z,
        "sonde_z": sonde_z,
        "box_x": x, "box_y": y,
        "sensor_x": x - dx, "sensor_y": y - dy,
        "antenna_x": x + dx, "antenna_y": y + dy,
        "ant_top_z": sonde_z + ant_h,
    }


def make_hover(df, simple=True):
    if simple:
        return [
            f"시간: {r['Time(min:sec)']}<br>고도: {r['Alt(m)']:.1f} m<br>기온: {r['T(C)']:.1f}℃<br>RH(상대습도): {r['U(%)']:.1f}%<br>풍속: {r['Wspd(knot)']:.1f} kt<br>상승률: {r['Asc(m/m)']:.1f} m/min"
            for _, r in df.iterrows()
        ]
    return [
        f"시간: {r['Time(min:sec)']}<br>기압: {r['P(hPa)']:.1f} hPa<br>기온: {r['T(C)']:.1f}℃<br>노점: {r['Dew(deg)']:.1f}℃<br>RH(상대습도): {r['U(%)']:.1f}%<br>풍향/풍속: {r['Wdir(deg)']:.0f}° / {r['Wspd(knot)']:.1f} kt<br>위도: {r['Lat(deg)']:.5f}<br>경도: {r['Lon(deg)']:.5f}<br>고도: {r['Alt(m)']:.1f} m<br>상승률: {r['Asc(m/m)']:.1f} m/min = {r['Asc(m/s)']:.2f} m/s"
        for _, r in df.iterrows()
    ]



def add_direction_guide(fig, df_display):
    """3D 공간에서 동·서·남·북 방향 기준선을 표시."""
    if df_display is None or len(df_display) == 0:
        return fig
    max_extent = float(np.nanmax(np.sqrt(df_display["x_km"] ** 2 + df_display["y_km"] ** 2)))
    if not np.isfinite(max_extent) or max_extent <= 0:
        max_extent = 1.0
    guide_len = max(1.0, max_extent * 0.18)
    z0 = float(np.nanmin(df_display["z_plot"]))

    directions = [
        ("동(E)", guide_len, 0.0),
        ("서(W)", -guide_len, 0.0),
        ("북(N)", 0.0, guide_len),
        ("남(S)", 0.0, -guide_len),
    ]

    for label, x, y in directions:
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[z0, z0],
            mode="lines",
            line=dict(width=3, color="rgba(120,120,120,0.55)"),
            name=label,
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z0],
            mode="text",
            text=[label],
            textposition="middle center",
            textfont=dict(size=14, color="rgba(55,55,55,0.95)"),
            name=label,
            hoverinfo="skip",
            showlegend=False,
        ))
    return fig


def find_isotherm_crossings(df_display: pd.DataFrame, values=(-0.0, -10.0, -20.0)):
    """기온이 지정값(℃)을 지나는 z_plot/고도 위치를 선형 보간으로 찾는다."""
    if df_display is None or len(df_display) < 2 or "T(C)" not in df_display.columns:
        return []
    work = df_display.dropna(subset=["T(C)", "z_plot", "Alt(m)"]).copy()
    if len(work) < 2:
        return []
    work = work.sort_values("Alt(m)").reset_index(drop=True)
    out = []
    for target in values:
        tlabel = int(target)
        found = []
        for i in range(len(work) - 1):
            r0 = work.iloc[i]
            r1 = work.iloc[i + 1]
            t0 = float(r0["T(C)"])
            t1 = float(r1["T(C)"])
            if not (np.isfinite(t0) and np.isfinite(t1)):
                continue
            # exact hit
            if abs(t0 - target) < 1e-9:
                frac = 0.0
            elif (t0 - target) * (t1 - target) <= 0 and t0 != t1:
                frac = (target - t0) / (t1 - t0)
            else:
                continue
            frac = float(np.clip(frac, 0.0, 1.0))
            alt = float(r0["Alt(m)"] + frac * (r1["Alt(m)"] - r0["Alt(m)"]))
            z = float(r0["z_plot"] + frac * (r1["z_plot"] - r0["z_plot"]))
            p = np.nan
            if "P(hPa)" in work.columns:
                try:
                    p = float(r0["P(hPa)"] + frac * (r1["P(hPa)"] - r0["P(hPa)"]))
                except Exception:
                    p = np.nan
            # 너무 가까운 중복 교차 제거
            if not any(abs(alt - prev["alt_m"]) < 25 for prev in found):
                found.append({"temp_c": float(target), "label": f"{tlabel}℃", "z": z, "alt_m": alt, "p_hpa": p})
        out.extend(found)
    return out


def add_isotherm_surfaces(fig: go.Figure, crossings, x_range, y_range, opacity=0.22):
    """0/-10/-20℃ 고도를 3D 공간의 반투명 수평면으로 추가한다."""
    if not crossings or x_range is None or y_range is None:
        return fig
    color_map = {
        0.0: "rgba(255,70,70,1.0)",
        -0.0: "rgba(255,70,70,1.0)",
        -10.0: "rgba(70,145,255,1.0)",
        -20.0: "rgba(90,70,220,1.0)",
    }
    # surface는 colorscale을 요구하므로 값별 단색 scale로 처리
    for c in crossings:
        z = float(c["z"])
        temp = float(c["temp_c"])
        color = color_map.get(temp, "rgba(80,80,80,1.0)")
        xs = np.linspace(float(x_range[0]), float(x_range[1]), 2)
        ys = np.linspace(float(y_range[0]), float(y_range[1]), 2)
        X, Y = np.meshgrid(xs, ys)
        Z = np.full_like(X, z, dtype=float)
        label = c.get("label", f"{temp:g}℃")
        p_txt = "" if not np.isfinite(c.get("p_hpa", np.nan)) else f"<br>기압: {c['p_hpa']:.0f} hPa"
        hover = f"{label} 등온면<br>고도: {c['alt_m']:.0f} m{p_txt}"
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, color], [1, color]],
            opacity=float(opacity),
            showscale=False,
            name=f"{label} 등온면",
            hovertemplate=hover + "<extra></extra>",
        ))
        # 우측 상단에 라벨을 얹어 회전해도 대략 어느 면인지 확인
        fig.add_trace(go.Scatter3d(
            x=[float(x_range[1])], y=[float(y_range[1])], z=[z],
            mode="text",
            text=[f"{label}면"],
            textposition="middle right",
            textfont=dict(size=13, color=color),
            hoverinfo="skip",
            showlegend=False,
            name=f"{label} 라벨",
        ))
    return fig


def make_3d_fig(df_display, color_col, simple_hover=True, cloud=None, inversions=None, llj_layers=None, show_cloud=False, cloud_opacity=0.35, show_inversion=False, show_llj=False, show_direction=True, vertical_mode="실제 고도", map_floor_trace=None, floor_grid_traces=None, anim_row=None, anim_trail_df=None, show_sonde_icon=True, scene_context_factor: float = 1.0, ns_context_factor: float = 1.0, show_isotherm_surfaces: bool = False, isotherm_opacity: float = 0.22):
    fig = go.Figure()
    if map_floor_trace is not None:
        fig.add_trace(map_floor_trace)
    if floor_grid_traces:
        for tr in floor_grid_traces:
            fig.add_trace(tr)
    fig.add_trace(go.Scatter3d(
        x=df_display["x_km"], y=df_display["y_km"], z=df_display["z_plot"],
        mode="lines", line=dict(width=4, color="rgba(80,80,80,0.42)"),
        name="궤적선", hoverinfo="skip"
    ))
    # 구름 가능층: 관측요소를 가리지 않도록 먼저, 큰 반투명 halo로 그림
    # trace opacity와 marker opacity를 동시에 사용해 채도 변화가 아니라 실제 투명도 변화가 되도록 보정.
    if show_cloud and cloud:
        masks = []
        for lyr in cloud:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            cdf = df_display.loc[mask]
            if len(cdf):
                fig.add_trace(go.Scatter3d(
                    x=cdf["x_km"], y=cdf["y_km"], z=cdf["z_plot"],
                    mode="markers",
                    marker=dict(
                        size=12,
                        symbol="circle",
                        color="rgba(180,180,180,1.0)",
                        opacity=float(cloud_opacity),
                        line=dict(color=f"rgba(100,100,100,{max(0.05, cloud_opacity * 0.70):.2f})", width=1)
                    ),
                    opacity=float(cloud_opacity),
                    name="구름 가능층", hoverinfo="skip", showlegend=True
                ))

    fig.add_trace(go.Scatter3d(
        x=df_display["x_km"], y=df_display["y_km"], z=df_display["z_plot"],
        mode="markers",
        marker=dict(size=4, color=df_display[color_col], colorscale="RdBu_r", showscale=True,
                    colorbar=dict(title=variable_label(color_col), len=0.56, y=0.50, x=1.015, thickness=14), opacity=0.90),
        text=make_hover(df_display, simple=simple_hover), hoverinfo="text", name=variable_label(color_col)
    ))

    # 역전층 강조: 선택 시 해당 고도 구간의 관측점을 보라색 open marker로 표시
    if show_inversion and inversions:
        masks = []
        for lyr in inversions:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            idf = df_display.loc[mask]
            if len(idf):
                fig.add_trace(go.Scatter3d(
                    x=idf["x_km"], y=idf["y_km"], z=idf["z_plot"],
                    mode="markers",
                    marker=dict(size=7, symbol="diamond-open", color="rgba(142,68,173,0.78)", line=dict(color="rgba(142,68,173,0.72)", width=2)),
                    name="역전층 강조", hoverinfo="skip"
                ))

    # 하층제트 강조: 선택 시 LLJ 후보 코어 주변을 주황색 open marker로 표시
    if show_llj and llj_layers:
        masks = []
        for lyr in llj_layers:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            jdf = df_display.loc[mask]
            if len(jdf):
                fig.add_trace(go.Scatter3d(
                    x=jdf["x_km"], y=jdf["y_km"], z=jdf["z_plot"],
                    mode="markers",
                    marker=dict(size=8, symbol="square-open", color="rgba(230,126,34,0.82)", line=dict(color="rgba(230,126,34,0.78)", width=2)),
                    name="하층제트 강조", hoverinfo="skip"
                ))
    # start/end/max height only
    points = [
        (df_display.iloc[0], "START", "green", "circle"),
        (df_display.iloc[-1], "END", "black", "x"),
        (df_display.loc[df_display["Alt(m)"].idxmax()], "MAX", "purple", "diamond"),
    ]
    for row, label, color, symbol in points:
        fig.add_trace(go.Scatter3d(
            x=[row["x_km"]], y=[row["y_km"]], z=[row["z_plot"]],
            mode="markers+text", marker=dict(size=8, color=color, symbol=symbol),
            text=[label], textposition="top center", name=label
        ))
    # 클릭형 상승 이동 표시: Plotly frame 애니메이션 대신 현재 시각의 누적 궤적과 존데 위치를 덧그려
    # 사용자가 잡아둔 3D 시점이 불필요하게 초기화되는 문제를 줄인다.
    if anim_trail_df is not None and len(anim_trail_df) > 1:
        fig.add_trace(go.Scatter3d(
            x=anim_trail_df["x_km"], y=anim_trail_df["y_km"], z=anim_trail_df["z_plot"],
            mode="lines",
            line=dict(width=6, color="rgba(230,45,35,0.86)"),
            name="현재까지 이동 경로", hoverinfo="skip", showlegend=True
        ))
    if anim_row is not None:
        try:
            floor_z_now = float(df_display["z_plot"].min())
            extra_floor_z = []
            for tr in ([map_floor_trace] if map_floor_trace is not None else []) + (floor_grid_traces or []):
                try:
                    extra_floor_z.extend([float(v) for v in list(tr.z) if v is not None and np.isfinite(float(v))])
                except Exception:
                    pass
            if extra_floor_z:
                floor_z_now = min(extra_floor_z)
            hover_now = f"시간: {anim_row['Time(min:sec)']}<br>고도: {anim_row['Alt(m)']:.0f} m<br>기온: {anim_row['T(C)']:.1f}℃<br>RH: {anim_row['U(%)']:.0f}%<br>풍향/풍속: {anim_row['Wdir(deg)']:.0f}° / {anim_row['Wspd(knot)']:.1f} kt"
            fig.add_trace(go.Scatter3d(
                x=[anim_row["x_km"], anim_row["x_km"]],
                y=[anim_row["y_km"], anim_row["y_km"]],
                z=[floor_z_now, anim_row["z_plot"]],
                mode="lines", line=dict(width=4, color="rgba(230,45,35,0.55)"),
                name="현재 위치 투영선", hoverinfo="skip", showlegend=False
            ))
            if show_sonde_icon:
                fig.add_trace(go.Scatter3d(
                    x=[anim_row["x_km"]], y=[anim_row["y_km"]], z=[anim_row["z_plot"]],
                    mode="markers+text",
                    marker=dict(size=9, color="rgba(230,45,35,0.95)", symbol="circle", line=dict(color="white", width=1)),
                    text=["🎈"], textposition="top center", textfont=dict(size=22),
                    hovertext=[hover_now], hoverinfo="text", name="현재 존데 위치", showlegend=True
                ))
            else:
                fig.add_trace(go.Scatter3d(
                    x=[anim_row["x_km"]], y=[anim_row["y_km"]], z=[anim_row["z_plot"]],
                    mode="markers+text", marker=dict(size=11, color="rgba(230,45,35,0.95)", symbol="circle"),
                    text=["NOW"], textposition="top center", hovertext=[hover_now], hoverinfo="text",
                    name="현재 존데 위치", showlegend=True
                ))
        except Exception:
            pass

    if show_direction:
        add_direction_guide(fig, df_display)

    tick_vals, tick_text = vertical_axis_ticks(df_display, vertical_mode)
    zaxis_cfg = dict(title=vertical_axis_title(vertical_mode))
    if tick_vals is not None and tick_text is not None:
        zaxis_cfg.update(dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text))

    # 확대/회전 시 하단 눈금과 범례가 잘리지 않도록 장면 영역과 여백을 분리.
    scene_context_factor = max(1.0, float(scene_context_factor))
    ns_context_factor = max(1.0, float(ns_context_factor))
    x_range = padded_range(df_display["x_km"], pad_ratio=0.10 * scene_context_factor, min_pad=0.5 * scene_context_factor)
    y_range = padded_range(df_display["y_km"], pad_ratio=0.10 * scene_context_factor * ns_context_factor, min_pad=0.5 * scene_context_factor * ns_context_factor)
    z_range = padded_range(df_display["z_plot"], pad_ratio=0.08, min_pad=0.25)
    # 바닥면 지도/격자가 자료 범위보다 넓을 수 있으므로 x/y/z축 범위에 반드시 포함
    extra_x, extra_y, extra_z = [], [], []
    for tr in ([map_floor_trace] if map_floor_trace is not None else []) + (floor_grid_traces or []):
        try:
            extra_x.extend([float(v) for v in list(tr.x) if v is not None and np.isfinite(float(v))])
            extra_y.extend([float(v) for v in list(tr.y) if v is not None and np.isfinite(float(v))])
            extra_z.extend([float(v) for v in list(tr.z) if v is not None and np.isfinite(float(v))])
        except Exception:
            pass
    if x_range is not None and extra_x:
        x_range = [min(x_range[0], min(extra_x)), max(x_range[1], max(extra_x))]
    if y_range is not None and extra_y:
        y_range = [min(y_range[0], min(extra_y)), max(y_range[1], max(extra_y))]
    if z_range is not None and extra_z:
        z_range[0] = min(z_range[0], min(extra_z) - 0.03)
    zaxis_cfg.update(dict(range=z_range))

    if show_isotherm_surfaces:
        crossings = find_isotherm_crossings(df_display, values=(0.0, -10.0, -20.0))
        add_isotherm_surfaces(fig, crossings, x_range, y_range, opacity=isotherm_opacity)

    is_logp = vertical_mode == "Skew-T형 Log-P"
    x_span = max((x_range[1] - x_range[0]) if x_range else 1.0, 0.1)
    y_span = max((y_range[1] - y_range[0]) if y_range else 1.0, 0.1)
    hmax = max(x_span, y_span)
    aspect_x = max(0.25, x_span / hmax)
    aspect_y = max(0.25, y_span / hmax)

    scene_cfg = dict(
        xaxis=dict(title="동서 이동거리(km, +동쪽)", range=x_range, showbackground=True, backgroundcolor="rgba(216,238,252,0.92)", showgrid=True, gridcolor="rgba(105,145,170,0.48)", gridwidth=1, zerolinecolor="rgba(75,115,140,0.56)"),
        yaxis=dict(title="남북 이동거리(km, +북쪽)", range=y_range, showbackground=True, backgroundcolor="rgba(216,238,252,0.92)", showgrid=True, gridcolor="rgba(105,145,170,0.48)", gridwidth=1, zerolinecolor="rgba(75,115,140,0.56)"),
        zaxis={**zaxis_cfg, "showbackground": True, "backgroundcolor":"rgba(220,242,255,0.92)", "showgrid": True, "gridcolor":"rgba(105,145,170,0.52)", "gridwidth": 1, "zerolinecolor":"rgba(75,115,140,0.56)"},
        # 3D 장면 자체는 크게 쓰되, Log-P 모드에서는 아래쪽 압력축 라벨이 잘리지 않도록
        # 하단 여백을 더 확보한다.
        domain=dict(x=[0.02, 0.91] if is_logp else [0.02, 0.92],
                    # Log-P 모드에서 세로 표출영역을 더 넓혀 상·하단 잘림을 줄임
                    y=[0.045, 0.975] if is_logp else [0.08, 0.97]),
    )
    if is_logp:
        scene_cfg.update(dict(
            aspectmode="manual",
            # x/y 비율을 실제 km 범위와 맞춰 Log-P 모드에서도 바닥면 지도가 눌리지 않게 함
            aspectratio=dict(x=aspect_x, y=aspect_y, z=2.45),
            camera=dict(eye=dict(x=1.65, y=1.85, z=1.05), center=dict(x=0, y=0, z=-0.10)),
        ))
    else:
        scene_cfg.update(dict(aspectmode="data"))

    if is_logp:
        fig_height = 1380
        fig_margin = dict(l=20, r=150, t=55, b=110)
        legend_cfg = dict(
            orientation="h",
            yanchor="bottom", y=0.015,
            xanchor="left", x=0.055,
            bgcolor="rgba(255,255,255,0.78)",
            bordercolor="rgba(210,210,210,0.65)", borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        )
    else:
        fig_height = 780
        fig_margin = dict(l=15, r=120, t=60, b=95)
        legend_cfg = dict(
            orientation="h",
            yanchor="bottom", y=0.02,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(210,210,210,0.6)", borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        )

    fig.update_layout(
        title=dict(text="GPS 기반 3D Sonde Tracker", x=0.5, y=0.985),
        scene=scene_cfg,
        uirevision="sonde-main-3d",
        paper_bgcolor="rgba(250,253,255,1)",
        plot_bgcolor="rgba(235,247,255,1)",
        height=fig_height,
        margin=fig_margin,
        legend=legend_cfg
    )
    return fig


def make_profile_fig(df, clouds=None, inversions=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["T(C)"], y=df["Alt(m)"]/1000, mode="lines", name="기온"))
    fig.add_trace(go.Scatter(x=df["Dew(deg)"], y=df["Alt(m)"]/1000, mode="lines", name="노점"))
    if "Parcel_T(C)" in df.columns:
        fig.add_trace(go.Scatter(x=df["Parcel_T(C)"], y=df["Alt(m)"]/1000, mode="lines", name="상승기온(근사)"))
    if clouds:
        for lyr in clouds:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], fillcolor="rgba(170,170,170,0.22)", opacity=0.22, line_width=0, annotation_text="구름 가능층")
    if inversions:
        for lyr in inversions:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], opacity=0.12, line_width=0, annotation_text="inv")
    fig.update_layout(title="기온·노점 프로파일", xaxis_title="℃", yaxis_title="고도(km)", height=620, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_rh_fig(df, clouds=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["U(%)"], y=df["Alt(m)"]/1000, mode="lines", name="RH(상대습도)"))
    if clouds:
        for lyr in clouds:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], fillcolor="rgba(170,170,170,0.22)", opacity=0.22, line_width=0)
    fig.update_layout(title="상대습도 프로파일", xaxis_title="RH(상대습도, %)", yaxis_title="고도(km)", height=620, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_wind_profile(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Wspd(knot)"], y=df["Alt(m)"]/1000, mode="lines", name="풍속"))
    fig.update_layout(title="고도별 풍속", xaxis_title="kt", yaxis_title="고도(km)", height=560, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_asc_profile(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Asc(m/m)"], y=df["Alt(m)"]/1000, mode="lines", name="Asc(m/m) 상승률"))
    fig.update_layout(title="고도별 상승률", xaxis_title="Asc(m/m) = m/min, 60으로 나누면 m/s", yaxis_title="고도(km)", height=560, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_hodograph(df, display_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df["u_kt"], y=display_df["v_kt"], mode="lines+markers",
        marker=dict(size=5, color=display_df["Alt(m)"], colorscale="Viridis", colorbar=dict(title="Alt(m)")),
        text=[f"{r['Alt(m)']:.0f} m<br>{r['Wspd(knot)']:.1f} kt / {r['Wdir(deg)']:.0f}°" for _, r in display_df.iterrows()],
        hoverinfo="text", name="hodograph"
    ))
    max_abs = np.nanmax(np.abs(pd.concat([df["u_kt"], df["v_kt"]])))
    max_abs = max(10, math.ceil(max_abs/10)*10)
    fig.update_layout(title="호도그래프", xaxis_title="u(kt)", yaxis_title="v(kt)", height=650,
                      xaxis=dict(range=[-max_abs,max_abs], zeroline=True), yaxis=dict(range=[-max_abs,max_abs], zeroline=True, scaleanchor="x", scaleratio=1),
                      margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_sonde_animation_fig(anim_df: pd.DataFrame, vertical_mode: str = "실제 고도", map_floor_trace=None, floor_grid_traces=None, scene_context_factor: float = 1.0, ns_context_factor: float = 1.0, show_isotherm_surfaces: bool = False, isotherm_opacity: float = 0.22):
    """시간 순서에 따라 존데 현재 위치 점이 이동하는 3D 애니메이션."""
    fig = go.Figure()
    if map_floor_trace is not None:
        fig.add_trace(map_floor_trace)
    if floor_grid_traces:
        for tr in floor_grid_traces:
            fig.add_trace(tr)

    if anim_df is None or len(anim_df) < 2:
        fig.update_layout(title="상승 애니메이션", height=760)
        return fig

    floor_z, line_z = floor_z_for_display(anim_df, vertical_mode)
    z_span = max(float(anim_df["z_plot"].max() - anim_df["z_plot"].min()), 1.0)

    fig.add_trace(go.Scatter3d(
        x=anim_df["x_km"], y=anim_df["y_km"], z=anim_df["z_plot"],
        mode="lines", line=dict(width=3, color="rgba(120,120,120,0.50)"),
        name="전체 경로", hoverinfo="skip"
    ))
    first = anim_df.iloc[0]
    geom = sonde_symbol_geometry(first, floor_z, z_span)
    cur_hover = f"시간: {first['Time(min:sec)']}<br>고도: {first['Alt(m)']:.0f} m<br>기온: {first['T(C)']:.1f}℃<br>RH: {first['U(%)']:.0f}%<br>풍향/풍속: {first['Wdir(deg)']:.0f}° / {first['Wspd(knot)']:.1f} kt"

    fig.add_trace(go.Scatter3d(
        x=[first["x_km"]], y=[first["y_km"]], z=[first["z_plot"]],
        mode="lines", line=dict(width=6, color="rgba(225,35,35,0.92)"),
        name="현재까지 이동 경로", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["x"]], y=[geom["y"]], z=[geom["z"]],
        mode="markers", marker=dict(size=12, color="rgba(247,246,240,0.98)", symbol="circle", line=dict(color="rgba(150,150,150,1)", width=2.5)),
        name="존데 풍선", hovertext=[cur_hover], hoverinfo="text", showlegend=True
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["x"], geom["x"]], y=[geom["y"], geom["y"]], z=[geom["sonde_z"], geom["z"]],
        mode="lines", line=dict(width=2.5, color="rgba(110,110,110,0.90)"),
        name="테더", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["box_x"]], y=[geom["box_y"]], z=[geom["sonde_z"]],
        mode="markers+text", marker=dict(size=8, color="white", symbol="square", line=dict(color="rgba(120,120,120,1)", width=2)),
        text=["RS"], textposition="middle center", textfont=dict(size=8, color="rgba(60,100,165,1)"),
        name="존데 본체", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["sensor_x"]], y=[geom["sensor_y"]], z=[geom["sonde_z"]],
        mode="markers", marker=dict(size=4, color="rgba(245,245,245,1)", symbol="circle", line=dict(color="rgba(160,160,160,1)", width=1)),
        name="센서", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["antenna_x"], geom["antenna_x"]], y=[geom["antenna_y"], geom["antenna_y"]], z=[geom["sonde_z"], geom["ant_top_z"]],
        mode="lines", line=dict(width=3, color="rgba(40,40,40,0.95)"),
        name="안테나", hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[geom["x"], geom["x"]], y=[geom["y"], geom["y"]], z=[floor_z, geom["sonde_z"]],
        mode="lines", line=dict(width=2, color="rgba(220,40,40,0.38)"),
        name="현재 위치 수직투영", hoverinfo="skip"
    ))
    idx_trail = len(fig.data) - 7
    idx_balloon = len(fig.data) - 6
    idx_tether = len(fig.data) - 5
    idx_sonde = len(fig.data) - 4
    idx_sensor = len(fig.data) - 3
    idx_antenna = len(fig.data) - 2
    idx_proj = len(fig.data) - 1

    frames = []
    steps = []
    for i, (_, r) in enumerate(anim_df.iterrows()):
        hover = f"시간: {r['Time(min:sec)']}<br>고도: {r['Alt(m)']:.0f} m<br>기온: {r['T(C)']:.1f}℃<br>RH: {r['U(%)']:.0f}%<br>풍향/풍속: {r['Wdir(deg)']:.0f}° / {r['Wspd(knot)']:.1f} kt"
        title = f"상승 애니메이션 | {r['Time(min:sec)']} | {r['Alt(m)']:.0f} m"
        trail = anim_df.iloc[:i+1]
        g = sonde_symbol_geometry(r, floor_z, z_span)
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(x=trail["x_km"], y=trail["y_km"], z=trail["z_plot"], mode="lines", line=dict(width=6, color="rgba(225,35,35,0.92)"), hoverinfo="skip"),
                go.Scatter3d(x=[g["x"]], y=[g["y"]], z=[g["z"]], mode="markers", marker=dict(size=12, color="rgba(247,246,240,0.98)", symbol="circle", line=dict(color="rgba(150,150,150,1)", width=2.5)), hovertext=[hover], hoverinfo="text"),
                go.Scatter3d(x=[g["x"], g["x"]], y=[g["y"], g["y"]], z=[g["sonde_z"], g["z"]], mode="lines", line=dict(width=2.5, color="rgba(110,110,110,0.90)"), hoverinfo="skip"),
                go.Scatter3d(x=[g["box_x"]], y=[g["box_y"]], z=[g["sonde_z"]], mode="markers+text", marker=dict(size=8, color="white", symbol="square", line=dict(color="rgba(120,120,120,1)", width=2)), text=["RS"], textposition="middle center", textfont=dict(size=8, color="rgba(60,100,165,1)"), hoverinfo="skip"),
                go.Scatter3d(x=[g["sensor_x"]], y=[g["sensor_y"]], z=[g["sonde_z"]], mode="markers", marker=dict(size=4, color="rgba(245,245,245,1)", symbol="circle", line=dict(color="rgba(160,160,160,1)", width=1)), hoverinfo="skip"),
                go.Scatter3d(x=[g["antenna_x"], g["antenna_x"]], y=[g["antenna_y"], g["antenna_y"]], z=[g["sonde_z"], g["ant_top_z"]], mode="lines", line=dict(width=3, color="rgba(40,40,40,0.95)"), hoverinfo="skip"),
                go.Scatter3d(x=[g["x"], g["x"]], y=[g["y"], g["y"]], z=[floor_z, g["sonde_z"]], mode="lines", line=dict(width=2, color="rgba(220,40,40,0.38)"), hoverinfo="skip"),
            ],
            traces=[idx_trail, idx_balloon, idx_tether, idx_sonde, idx_sensor, idx_antenna, idx_proj]
        ))
        if i % max(1, len(anim_df)//12) == 0 or i == len(anim_df)-1:
            steps.append(dict(method="animate", args=[[str(i)], {"mode":"immediate", "frame":{"duration":0, "redraw":True}, "transition":{"duration":0}}], label=str(r["Time(min:sec)"])))

    fig.frames = frames

    tick_vals, tick_text = vertical_axis_ticks(anim_df, vertical_mode)
    zaxis_cfg = dict(title=vertical_axis_title(vertical_mode))
    if tick_vals is not None and tick_text is not None:
        zaxis_cfg.update(dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text))

    scene_context_factor = max(1.0, float(scene_context_factor))
    ns_context_factor = max(1.0, float(ns_context_factor))
    x_range = padded_range(anim_df["x_km"], pad_ratio=0.18 * scene_context_factor, min_pad=0.8 * scene_context_factor)
    y_range = padded_range(anim_df["y_km"], pad_ratio=0.18 * scene_context_factor * ns_context_factor, min_pad=0.8 * scene_context_factor * ns_context_factor)
    z_range = padded_range(anim_df["z_plot"], pad_ratio=0.08, min_pad=0.25)
    extra_x, extra_y, extra_z = [], [], []
    for tr in ([map_floor_trace] if map_floor_trace is not None else []) + (floor_grid_traces or []):
        try:
            extra_x.extend([float(v) for v in list(tr.x) if v is not None and np.isfinite(float(v))])
            extra_y.extend([float(v) for v in list(tr.y) if v is not None and np.isfinite(float(v))])
            extra_z.extend([float(v) for v in list(tr.z) if v is not None and np.isfinite(float(v))])
        except Exception:
            pass
    if x_range is not None and extra_x:
        x_range = [min(x_range[0], min(extra_x)), max(x_range[1], max(extra_x))]
    if y_range is not None and extra_y:
        y_range = [min(y_range[0], min(extra_y)), max(y_range[1], max(extra_y))]
    if z_range is not None and extra_z:
        z_range = [min(z_range[0], min(extra_z)-0.03), max(z_range[1], max(extra_z))]
    if show_isotherm_surfaces:
        crossings = find_isotherm_crossings(anim_df, values=(0.0, -10.0, -20.0))
        add_isotherm_surfaces(fig, crossings, x_range, y_range, opacity=isotherm_opacity)
    zaxis_cfg.update(dict(range=z_range))

    is_logp = vertical_mode == "Skew-T형 Log-P"
    x_span = max((x_range[1]-x_range[0]) if x_range else 1.0, 0.1)
    y_span = max((y_range[1]-y_range[0]) if y_range else 1.0, 0.1)
    hmax = max(x_span, y_span)
    aspect_x = max(0.25, x_span / hmax)
    aspect_y = max(0.25, y_span / hmax)

    scene_anim = dict(
        xaxis=dict(title="동서 이동거리(km)", range=x_range, showbackground=True, backgroundcolor="rgba(216,238,252,0.92)", showgrid=True, gridcolor="rgba(105,145,170,0.48)"),
        yaxis=dict(title="남북 이동거리(km)", range=y_range, showbackground=True, backgroundcolor="rgba(216,238,252,0.92)", showgrid=True, gridcolor="rgba(105,145,170,0.48)"),
        zaxis={**zaxis_cfg, "showbackground": True, "backgroundcolor":"rgba(220,242,255,0.92)", "showgrid": True, "gridcolor":"rgba(105,145,170,0.52)"},
        domain=dict(x=[0.02, 0.92], y=[0.045, 0.975] if is_logp else [0.08, 0.97]),
        uirevision="sonde-animation-camera",
    )
    if is_logp:
        scene_anim.update(dict(
            aspectmode="manual",
            aspectratio=dict(x=aspect_x, y=aspect_y, z=2.25),
            camera=dict(eye=dict(x=1.65, y=1.85, z=1.05), center=dict(x=0, y=0, z=-0.10)),
        ))
    else:
        scene_anim.update(dict(aspectmode="data"))

    fig.update_layout(
        title=dict(text=f"상승 애니메이션 | {first['Time(min:sec)']} | {first['Alt(m)']:.0f} m"),
        scene=scene_anim,
        height=1120 if is_logp else 760,
        margin=dict(l=20, r=110, t=60, b=105),
        updatemenus=[dict(type="buttons", direction="left", x=0.05, y=0.02, xanchor="left", yanchor="bottom", buttons=[
            dict(label="▶ 재생", method="animate", args=[None, {"frame":{"duration":320, "redraw":True}, "fromcurrent":True, "transition":{"duration":0}, "mode":"immediate"}]),
            dict(label="⏸ 정지", method="animate", args=[[None], {"mode":"immediate", "frame":{"duration":0, "redraw":True}, "transition":{"duration":0}}]),
        ])],
        sliders=[dict(active=0, x=0.18, y=0.02, len=0.72, currentvalue=dict(prefix="시간 "), steps=steps)],
        showlegend=True,
        paper_bgcolor="rgba(250,253,255,1)",
        uirevision="sonde-animation-camera",
    )
    return fig


with st.sidebar:
    st.header("1. 원시자료 업로드")
    uploaded = st.file_uploader("UPP RAW TXT 파일", type=["txt", "dat", "csv", "log"])
    st.caption("업로드 방식은 v1/v2와 동일합니다. 내부에서만 표시 자료를 줄입니다.")
    st.divider()
    st.header("2. 경량 표시 설정")
    display_mode = st.selectbox("3D 표시 간격", ["5초 간격", "10초 간격", "2초 간격", "고도 50m 간격", "고도 100m 간격", "전체"], index=0)
    vertical_mode = st.selectbox("3D 연직축 표현", ["실제 고도", "Skew-T형 Log-P"], index=0, help="기본은 실제 고도입니다. Skew-T형 Log-P는 P(hPa)만 사용하며 지상~100hPa 범위만 표시하고, 이때만 연직축을 길게 표현합니다.")
    color_col = st.selectbox("3D 색상 변수", ["T(C)", "U(%)", "Wspd(knot)", "Asc(m/m)"], index=0, format_func=variable_label)
    simple_hover = st.checkbox("간단 hover 사용", value=True)
    show_direction_guide = st.checkbox("동·서·남·북 방위 표시", value=True)
    st.divider()
    st.header("3. 구름 가능층·강조 표시")
    show_cloud_3d = st.checkbox("3D에서 구름 가능층 강조", value=True)
    cloud_opacity = st.slider("구름 강조 투명도", 0.10, 0.90, 0.35, 0.05)
    rh_th = st.slider("구름 판단 RH(상대습도) 기준(%)", 70, 100, 85, 1)
    spread_th = st.slider("구름 판단 T-Td 기준(℃)", 0.5, 5.0, 2.0, 0.5)
    min_cloud_thick = st.slider("구름 최소 층 두께(m)", 20, 300, 100, 10)
    st.divider()

    st.header("4. 강조층·바람 표시")
    show_inversion_3d = st.checkbox("3D에서 역전층 강조", value=False)
    show_llj_3d = st.checkbox("3D에서 하층제트 강조", value=False)
    show_isotherm_surfaces = st.checkbox("3D에서 0/-10/-20℃ 등온면 표시", value=False, help="기온이 0℃, -10℃, -20℃를 지나는 고도를 반투명 수평면으로 표시합니다.")
    isotherm_opacity = st.slider("등온면 투명도", 0.08, 0.45, 0.22, 0.02)
    show_wind_panel = st.checkbox("바람깃 패널 표시", value=True)
    wind_panel_interval = st.selectbox("바람깃 표시 간격", [250, 500, 1000], index=1, format_func=lambda x: f"{x} m 간격")
    st.divider()

    st.header("5. 바닥면 지도·격자")
    show_floor_grid = st.checkbox("바닥면 기본 격자 표시", value=True, help="지도 사용 여부와 관계없이 바닥면 격자를 표시합니다. 지도 표시가 켜져 있으면 하늘색 옆면과 연직 격자도 함께 표시됩니다.")
    floor_grid_count = st.selectbox("바닥면 격자 밀도", [6, 8, 10, 12], index=1, format_func=lambda x: f"{x}분할")
    show_map_floor = st.checkbox("3D 바닥면 실제 지도 표시", value=True, help="외부망에서 OpenStreetMap 정적 지도 1장을 받아 3D 바닥면에 표시합니다. 실패해도 기본 격자는 표시됩니다.")
    map_detail = st.selectbox("지도 선명도", ["표준", "지역명 선명", "고해상도"], index=1, help="기본은 지역명 선명입니다. 고해상도는 필요할 때만 사용하세요.")
    map_context = st.selectbox("지도/공간 여유 범위", ["보통", "넓게", "아주 넓게"], index=1, help="궤적 주변만 딱 자르지 않고, 실제 지명과 위치를 가늠할 수 있도록 지도와 3D 공간을 넓게 잡습니다.")
    ns_context = st.selectbox("남북(위도) 방향 추가 확대", ["기본", "남북 넓게", "남북 아주 넓게"], index=1, help="존데 궤적은 동서보다 남북 위치감이 중요할 수 있어, 위도 방향의 지도/공간 범위를 별도로 더 넓힙니다.")
    auto_light_logp_map = st.checkbox("Log-P 모드에서 지도 자동 경량화", value=True, help="Skew-T형 Log-P에서는 화면이 길고 무거워질 수 있어 고해상도 지도/표면 품질을 자동으로 한 단계 낮춥니다.")
    map_opacity = st.slider("지도 투명도", 0.30, 0.95, 0.78, 0.05)
    map_grid_cells = st.selectbox("지도 표면 품질/무게", [80, 112, 144, 168], index=1, format_func=lambda x: f"기준 {x}셀")
    st.divider()

    st.header("6. 상승 이동 표시")
    anim_seconds = st.selectbox("상승 이동 표시 간격", [10, 20, 30, 60], index=2, format_func=lambda x: f"{x}초 간격")
    anim_max_frames = st.slider("상승 애니메이션 최대 프레임", 30, 100, 60, 10, help="애니메이션은 Plotly 내부 프레임으로 재생합니다. 기본 60프레임이 가장 안정적입니다.")
    st.divider()

    st.header("7. 하층제트 탐지")
    llj_max_alt = st.slider("LLJ 탐지 상한고도(m)", 1000, 5000, 3000, 500)
    llj_min_speed = st.slider("LLJ 최소 풍속(kt)", 10, 50, 20, 1)
    llj_drop = st.slider("LLJ 풍속 감소 기준(kt)", 2, 20, 5, 1)

if uploaded is None:
    st.info("왼쪽에서 UPP RAW 원시자료 TXT 파일을 업로드하세요.")
    st.markdown("""
    **v4 Log-P Axis 기본값**
    - 3D는 5초 간격 자료만 표시
    - 원본 전체 자료는 계산과 CSV 저장에 사용
    - 연직축은 실제 고도 또는 Skew-T형 Log-P 중 선택
    - Log-P 선택 시 P(hPa)만 이용하고 지상~100hPa 범위만 표시
    - 구름 가능층은 3D에서 직접 강조하고 투명도 조절 가능
    - 역전층/하층제트는 필요 시 3D에서 강조
    - 오른쪽 바람깃 패널은 250~1000m 간격 대표값만 표시해 가볍게 유지
    - 바닥면 기본 격자는 항상 표시 가능
    - 지도 표시가 켜지면 하늘색 옆면과 연직 격자도 함께 표시
    - 외부망에서 OpenStreetMap 정적 지도 1장을 3D 바닥면에 표시 가능
    - 지도 선명도 선택, Log-P 지도 자동 경량화, 경량 Plotly 상승 애니메이션 탭 제공
    - 열역학 분석은 해당 탭에서 버튼을 눌렀을 때 계산
    """)
    st.stop()

try:
    raw_df, info = parse_upp_raw(uploaded)
    raw_df = latlon_to_xy_km(raw_df)
    raw_df = add_wind_components(raw_df)
    if "Asc(m/m)" in raw_df.columns:
        raw_df["Asc(m/s)"] = raw_df["Asc(m/m)"] / 60.0
except Exception as e:
    st.error("원시자료 파싱 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

display_df = get_display_df(raw_df, display_mode)
display_df = prepare_vertical_axis(display_df, vertical_mode)
map_context_factor = context_factor_from_label(map_context)
ns_context_factor = ns_factor_from_label(ns_context)

clouds = cloud_layers(raw_df, rh_threshold=rh_th, spread_threshold=spread_th, min_thickness_m=min_cloud_thick)
inversions = inversion_layers(raw_df)
llj_layers = low_level_jet_layers(raw_df, max_alt_m=llj_max_alt, min_speed_kt=llj_min_speed, drop_threshold_kt=llj_drop)

map_floor_trace = None
floor_grid_traces = make_floor_grid_traces(display_df, vertical_mode=vertical_mode, grid_count=floor_grid_count, show_sky_walls=show_map_floor, context_factor=map_context_factor, ns_factor=ns_context_factor) if show_floor_grid else []
map_info_text = ""
map_profiles = {
    # 기본값 자체를 이전보다 넓게 조정: 관측 궤적만 보이지 않고 주변 지명까지 보이도록 함
    "표준": dict(out_size=1100, pad_ratio=0.22, min_span_deg=0.22, max_tiles=6),
    "지역명 선명": dict(out_size=1500, pad_ratio=0.20, min_span_deg=0.18, max_tiles=8),
    "고해상도": dict(out_size=1800, pad_ratio=0.18, min_span_deg=0.15, max_tiles=10),
}
map_detail_effective = map_detail
map_grid_cells_effective = map_grid_cells
if vertical_mode == "Skew-T형 Log-P" and auto_light_logp_map:
    # Log-P 모드는 3D 화면 자체가 길어지므로 지도는 자동으로 한 단계 가볍게 유지
    if map_detail_effective == "고해상도":
        map_detail_effective = "지역명 선명"
    if map_grid_cells_effective > 112:
        map_grid_cells_effective = 112
map_profile = apply_context_to_map_profile(map_profiles.get(map_detail_effective, map_profiles["지역명 선명"]), map_context_factor, ns_context_factor)
if show_map_floor:
    try:
        map_floor_trace, map_bounds, map_zoom = make_map_floor_trace(
            raw_df, display_df, vertical_mode=vertical_mode,
            opacity=map_opacity, grid_cells=map_grid_cells_effective, **map_profile
        )
        suffix = ""
        if map_detail_effective != map_detail or map_grid_cells_effective != map_grid_cells:
            suffix = " (Log-P 자동 경량화 적용)"
        map_info_text = f"바닥면 지도: OpenStreetMap 정적 타일 기반, zoom {map_zoom}, 선명도 {map_detail_effective}, 공간 {map_context}, 남북 {ns_context}, 표면 {map_grid_cells_effective}셀{suffix}, © OpenStreetMap contributors"
    except Exception as e:
        st.warning("바닥면 실제 지도를 불러오지 못했습니다. 진한 기본 바닥면 격자로 3D Tracker를 표시합니다.")
        st.caption(f"지도 오류: {e}")
        map_floor_trace = None

# Header metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("원본 행 수", f"{len(raw_df):,}")
with c2: st.metric("3D 표시 행 수", f"{len(display_df):,}")
with c3: st.metric("최대 고도", metric_fmt(raw_df["Alt(m)"].max(), " m", 1))
with c4: st.metric("최저 기온", metric_fmt(raw_df["T(C)"].min(), "℃", 1))
with c5: st.metric("구름 가능층", f"{len(clouds)}개")
with c6: st.metric("LLJ 후보", f"{len(llj_layers)}개")

if info:
    st.caption(f"Station {info.get('station_no')} | Lat {info.get('latitude')} | Lon {info.get('longitude')} | Alt {info.get('altitude_m')} m | Probe {info.get('probe_no')}")
st.caption("※ 원시자료의 Asc(m/m)는 meter/minute, 즉 분당 상승률(m/min)로 표시합니다. m/s 환산값은 Asc(m/m) ÷ 60입니다.")
if map_info_text:
    st.caption(map_info_text)


def animation_controls(anim_preview_df: pd.DataFrame, label_prefix: str = "sonde"):
    """클릭형 상승 이동 컨트롤.

    자동재생은 Streamlit 재실행이 반복되어 느릴 수 있으므로 기본 제거.
    사용자는 처음/이전/다음/끝 버튼과 슬라이더로 원하는 시각을 직접 확인한다.
    """
    max_idx = max(0, len(anim_preview_df) - 1)
    idx_key = f"{label_prefix}_anim_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    st.session_state[idx_key] = int(max(0, min(max_idx, st.session_state[idx_key])))

    st.markdown("<div style='font-weight:700; color:#18324a; margin:0.15rem 0 0.35rem 0;'>시간 이동</div>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if st.button("⏮ 처음", use_container_width=True, key=f"{label_prefix}_first_btn"):
            st.session_state[idx_key] = 0
    with r1c2:
        if st.button("끝 ⏭", use_container_width=True, key=f"{label_prefix}_last_btn"):
            st.session_state[idx_key] = max_idx

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if st.button("◀ 이전", use_container_width=True, key=f"{label_prefix}_prev_btn"):
            st.session_state[idx_key] = max(0, int(st.session_state[idx_key]) - 1)
    with r2c2:
        if st.button("다음 ▶", use_container_width=True, key=f"{label_prefix}_next_btn"):
            st.session_state[idx_key] = min(max_idx, int(st.session_state[idx_key]) + 1)

    anim_idx = st.slider(
        "시간 선택", 0, max_idx, int(st.session_state[idx_key]), 1,
        help="슬라이더 또는 이동 버튼으로 해당 시각의 존데 위치와 누적 이동 경로를 표시합니다.",
        key=f"{label_prefix}_time_slider"
    )
    st.session_state[idx_key] = int(anim_idx)

    if max_idx > 0:
        pct = int(round(100 * st.session_state[idx_key] / max_idx))
        st.progress(pct / 100, text=f"진행률 {pct}%")

    return int(st.session_state[idx_key]), idx_key, None, max_idx


def advance_animation_after_render(play_key: str, idx_key: str, max_idx: int, delay_s: float = 0.22):
    """자동재생 제거: 클릭형 이동만 사용."""
    return


tab1, tab_anim, tab2, tab3, tab4 = st.tabs(["3D Tracker", "상승 애니메이션", "열역학", "바람·호도그래프", "자료/다운로드"])

with tab1:
    st.subheader("3D Tracker")
    st.caption("계산은 원본 전체 자료를 쓰고, 3D 표시만 선택 간격으로 줄입니다. Log-P 선택 시 P(hPa)만 사용하며 지상~100hPa 범위(P≥100hPa)만 표시합니다. 상승 애니메이션은 별도 탭에서 재생합니다.")

    if show_wind_panel:
        main_col, wind_col = st.columns([6.2, 1.05])
        with wind_col:
            wind_fig = make_wind_barb_panel(
                display_df, vertical_mode=vertical_mode, interval_m=wind_panel_interval,
                clouds=clouds, inversions=inversions, llj_layers=llj_layers,
                show_cloud=False, show_inversion=show_inversion_3d, show_llj=show_llj_3d
            )
            st.plotly_chart(wind_fig, use_container_width=True)

        fig = make_3d_fig(display_df, color_col=color_col, simple_hover=simple_hover, cloud=clouds, inversions=inversions, llj_layers=llj_layers, show_cloud=show_cloud_3d, cloud_opacity=cloud_opacity, show_inversion=show_inversion_3d, show_llj=show_llj_3d, show_direction=show_direction_guide, vertical_mode=vertical_mode, map_floor_trace=map_floor_trace, floor_grid_traces=floor_grid_traces, anim_row=None, anim_trail_df=None, show_sonde_icon=False, scene_context_factor=map_context_factor, ns_context_factor=ns_context_factor, show_isotherm_surfaces=show_isotherm_surfaces, isotherm_opacity=isotherm_opacity)
        with main_col:
            st.plotly_chart(fig, use_container_width=True)
    else:
        fig = make_3d_fig(display_df, color_col=color_col, simple_hover=simple_hover, cloud=clouds, inversions=inversions, llj_layers=llj_layers, show_cloud=show_cloud_3d, cloud_opacity=cloud_opacity, show_inversion=show_inversion_3d, show_llj=show_llj_3d, show_direction=show_direction_guide, vertical_mode=vertical_mode, map_floor_trace=map_floor_trace, floor_grid_traces=floor_grid_traces, anim_row=None, anim_trail_df=None, show_sonde_icon=False, scene_context_factor=map_context_factor, ns_context_factor=ns_context_factor, show_isotherm_surfaces=show_isotherm_surfaces, isotherm_opacity=isotherm_opacity)
        st.plotly_chart(fig, use_container_width=True)

    html = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button("표시 중인 3D HTML 다운로드", html, "sonde_tracker_3d_light.html", "text/html")

with tab_anim:
    st.subheader("상승 애니메이션")
    st.caption("메인 3D Tracker와 분리된 전용 탭에서 Plotly 내부 프레임으로 현재 존데 위치와 누적 이동 경로를 재생합니다. 속도를 위해 구름·바람깃·강조층은 제외하고, 지도/격자·전체 경로·누적 trajectory·현재 존데(아이보리 풍선 + RS 존데 본체 상징)만 표시합니다.")
    try:
        anim_base = downsample_by_seconds(raw_df, anim_seconds)
        if len(anim_base) > anim_max_frames:
            step = max(1, int(np.ceil(len(anim_base) / anim_max_frames)))
            anim_base = anim_base.iloc[::step].reset_index(drop=True)
        anim_df = prepare_vertical_axis(anim_base, vertical_mode)
        st.caption(f"애니메이션 프레임: {len(anim_df)}개 / 표시 간격: {anim_seconds}초 기준 / 지도는 현재 설정과 공간 여유 범위를 사용하되 Log-P에서는 자동 경량화가 적용됩니다.")
        anim_fig = make_sonde_animation_fig(anim_df, vertical_mode=vertical_mode, map_floor_trace=map_floor_trace, floor_grid_traces=floor_grid_traces, scene_context_factor=map_context_factor, ns_context_factor=ns_context_factor, show_isotherm_surfaces=show_isotherm_surfaces, isotherm_opacity=isotherm_opacity)
        st.plotly_chart(anim_fig, use_container_width=True)
    except Exception as e:
        st.warning("상승 애니메이션을 생성하지 못했습니다.")
        st.exception(e)


with tab2:
    st.subheader("열역학·기온 관련 진단")
    st.caption("빠른 로딩을 위해 CAPE/CIN 계산은 버튼을 눌렀을 때만 실행합니다.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_profile_fig(raw_df, clouds=clouds, inversions=inversions), use_container_width=True)
    with right:
        st.plotly_chart(make_rh_fig(raw_df, clouds=clouds), use_container_width=True)
    st.markdown("#### 자동 탐지 결과")
    a, b = st.columns(2)
    with a:
        st.write("구름 가능층")
        st.dataframe(pd.DataFrame(clouds), use_container_width=True, hide_index=True)
    with b:
        st.write("역전층")
        st.dataframe(pd.DataFrame(inversions), use_container_width=True, hide_index=True)
    if st.button("CAPE/CIN/LCL/CCL/대류온도 계산 실행", type="primary"):
        summary, parcel_profile = thermo_summary(raw_df)
        cols = st.columns(4)
        for i, (k, v) in enumerate(summary.items()):
            with cols[i % 4]:
                unit = "" if "J/kg" in k else (" m" if "(m)" in k else "℃")
                digits = 0 if "J/kg" in k or "(m)" in k else 1
                st.metric(k, metric_fmt(v, unit, digits))
        st.plotly_chart(make_profile_fig(parcel_profile, clouds=clouds, inversions=inversions), use_container_width=True)
        st.info("CAPE/CIN/대류온도는 경량 자체식 기반 근사값입니다. 공식 현업 산출용 정밀값은 MetPy 등으로 후속 고도화가 필요합니다.")

with tab3:
    st.subheader("바람·상승률·호도그래프")
    st.caption("Asc(m/m)는 원시자료 표기 그대로 유지하되, 단위 의미는 m/min(분당 m)입니다. 예: 300 m/min ≈ 5.0 m/s")
    if llj_layers:
        st.markdown("#### 하층제트 후보")
        st.dataframe(pd.DataFrame(llj_layers), use_container_width=True, hide_index=True)
    else:
        st.info("현재 기준에서 하층제트 후보가 탐지되지 않았습니다. 필요하면 사이드바의 LLJ 기준값을 조정하세요.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_wind_profile(raw_df), use_container_width=True)
    with c2:
        st.plotly_chart(make_asc_profile(raw_df), use_container_width=True)
    st.plotly_chart(make_hodograph(raw_df, display_df), use_container_width=True)
    st.markdown("#### 1 km 층별 평균")
    st.dataframe(layer_mean_table(raw_df), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("자료 확인 및 다운로드")
    st.caption("화면에는 일부만 보여주고, 전체 자료는 다운로드로 제공합니다.")
    st.markdown("#### 원본 미리보기 50행")
    st.dataframe(raw_df.head(50), use_container_width=True)
    st.markdown("#### 표시용 자료 미리보기 50행")
    st.dataframe(display_df.head(50), use_container_width=True)
    st.download_button("원본 변환 CSV 다운로드", raw_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "sonde_raw_converted_full.csv", "text/csv")
    st.download_button("표시용 경량 CSV 다운로드", display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "sonde_display_light.csv", "text/csv")
