from __future__ import annotations

import zipfile
from pathlib import Path


def unzip_all(zips_dir: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(zips_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files found in: {zips_dir}")

    for z in zips:
        target_folder = extract_dir / z.stem
        if target_folder.exists() and any(target_folder.rglob("*.csv")):
            print(f"Already extracted: {z.name}")
            continue

        print(f"Extracting: {z.name} -> {target_folder}")
        target_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(z, "r") as zip_ref:
            # Extract selectively 
            for info in zip_ref.infolist():
                p = Path(info.filename)

                #macOS junk
                if "__MACOSX" in p.parts:
                    continue
                if p.name.startswith("._"):
                    continue

                zip_ref.extract(info, target_folder)


def pick_csv(extract_dir: Path) -> Path:
    # Get all CSVs, excluding macOS junk and "._" files
    csvs = []
    for p in extract_dir.rglob("*.csv"):
        if "__MACOSX" in p.parts:
            continue
        if p.name.startswith("._"):
            continue
        # skip tiny files (junks oftern)
        if p.stat().st_size < 200:
            continue
        csvs.append(p)

    if not csvs:
        raise FileNotFoundError(f"No valid CSV files found under: {extract_dir}")

    # Prefer RAW IMU time-series for step counting
    def score(p: Path) -> int:
        name = p.name.lower()
        parts = [x.lower() for x in p.parts]

        s = 0

        #  raw folder
        if "raw" in parts:
            s += 50
        # interim is often still time-series
        if "interim" in parts:
            s += 20

        #  foot/ankle signals
        for kw in ["foot", "ankle", "left", "right", "imu"]:
            if kw in name:
                s += 5

        # Penalize aggregate tables 
        for bad in ["aggregate", "params", "parameter", "summary"]:
            if bad in name:
                s -= 40

        # is better to prefer bigger files (when time-series usually larger)
        s += min(30, int(p.stat().st_size / 200_000))  # +0..+30 approx

        return s

    csvs.sort(key=score, reverse=True)
    return csvs[0]