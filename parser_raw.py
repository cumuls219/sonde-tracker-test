import re
import io
from typing import Tuple, Dict, Optional
import pandas as pd


def _decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8", "cp949", "euc-kr", "latin1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_station_info(text: str) -> Dict[str, Optional[float]]:
    info = {"station_no": None, "longitude": None, "latitude": None, "altitude_m": None, "date": None, "probe_no": None}
    m = re.search(r"Date:\s*(.+?)(?:\t|$)", text)
    if m:
        info["date"] = m.group(1).strip()
    m = re.search(r"Number of\s+probe:\s*([^\n\r]+)", text)
    if m:
        info["probe_no"] = m.group(1).strip()
    patterns = {
        "station_no": r"Station\s+No\s*:\s*([0-9]+)",
        "longitude": r"Longitude\s*:\s*([-+]?\d+(?:\.\d+)?)",
        "latitude": r"Latitude\s*:\s*([-+]?\d+(?:\.\d+)?)",
        "altitude_m": r"Altitude\(m\)\s*:\s*([-+]?\d+(?:\.\d+)?)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            info[key] = int(val) if key == "station_no" else float(val)
    return info


def parse_time_to_seconds(s: str) -> Optional[int]:
    if pd.isna(s):
        return None
    s = str(s).strip()
    m = re.match(r"^(\d+):(\d{2})$", s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_upp_raw(uploaded_file_or_path) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """Parse UPP RAW text containing 'Sounding Data'.

    Accepts a Streamlit uploaded file object, bytes, or a local path.
    Returns dataframe and station metadata.
    """
    if isinstance(uploaded_file_or_path, (str, bytes)):
        if isinstance(uploaded_file_or_path, str):
            with open(uploaded_file_or_path, "rb") as f:
                raw = f.read()
        else:
            raw = uploaded_file_or_path
    else:
        raw = uploaded_file_or_path.read()

    text = _decode_bytes(raw)
    info = _parse_station_info(text)

    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Time(min:sec)" in line and "P(hPa)" in line and "Lon(deg)" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Sounding Data 헤더를 찾지 못했습니다. UPP RAW 원시자료인지 확인하세요.")

    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not re.match(r"^\d{3}:\d{2}\s+", stripped):
            # 관측행 형식이 아니면 종료하지 않고 건너뜀. 일부 파일에는 후단 메시지가 있을 수 있음.
            continue
        data_lines.append(stripped)

    if not data_lines:
        raise ValueError("관측자료 행을 찾지 못했습니다.")

    cols = [
        "Time(min:sec)", "P(hPa)", "T(C)", "U(%)", "Wspd(knot)", "Wdir(deg)",
        "Lon(deg)", "Lat(deg)", "Alt(m)", "Geo(gpm)", "Dew(deg)", "Asc(m/m)"
    ]
    rows = []
    for line in data_lines:
        parts = re.split(r"\s+", line)
        if len(parts) < 11:
            continue
        # Asc 값이 누락된 마지막 줄 등 대응
        if len(parts) == 11:
            parts.append("")
        row = parts[:12]
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)
    df["time_s"] = df["Time(min:sec)"].apply(parse_time_to_seconds)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_s", "Lon(deg)", "Lat(deg)", "Alt(m)"]).reset_index(drop=True)
    df["time_min"] = df["time_s"] / 60.0
    return df, info
