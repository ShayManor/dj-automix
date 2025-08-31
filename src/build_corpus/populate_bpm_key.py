#!/usr/bin/env python3
"""
Fill BPM and Key columns for a CSV of music files.

Input CSV schema (comma-separated):
SourceFile,Artist,Title,Album,Track,Discnumber,Genre,Year,Bpm,Key,FileName,Directory

- SourceFile examples appear to use ';' as path separators (e.g., "./Artist;Album;01;Track.mp3").
- Directory holds the base path (e.g., "/Volumes/Elements 1").
- FileName is the basename (e.g., "Artist;Album;01;Track.mp3").

This script:
- Resolves the real audio path from (Directory, SourceFile) or (Directory, FileName).
- Estimates BPM (tempo) and musical key (e.g., "C# minor") using librosa.
- Only fills empty/missing Bpm/Key unless --force is used.
- Parallelizes processing for speed.

Dependencies:
  pip install pandas numpy librosa soundfile audioread tqdm
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa

# ---- Key detection utilities (Krumhansl-Schmuckler template matching) ---- #

PITCHES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl & Kessler (1982) key profiles (normalized later)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)

def rotate(arr, n):
    return np.roll(arr, n)

def chroma_key(chroma_mean: np.ndarray) -> Tuple[str, str]:
    """Return (tonic, mode) like ('C#', 'minor') given a 12-dim mean chroma."""
    if chroma_mean.ndim != 1 or chroma_mean.shape[0] != 12:
        raise ValueError("chroma_mean must be shape (12,)")

    # Normalize inputs
    x = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    maj = KK_MAJOR / np.linalg.norm(KK_MAJOR)
    min_ = KK_MINOR / np.linalg.norm(KK_MINOR)

    best_score = -np.inf
    best = (0, "major")
    for i in range(12):
        s_maj = float(np.dot(x, rotate(maj, i)))
        s_min = float(np.dot(x, rotate(min_, i)))
        if s_maj > best_score:
            best_score = s_maj
            best = (i, "major")
        if s_min > best_score:
            best_score = s_min
            best = (i, "minor")

    tonic = PITCHES_SHARP[best[0] % 12]
    return tonic, best[1]

# ---- Audio analysis ---- #

def estimate_bpm_and_key(
    path: Path,
    sr: int = 22050,
    duration: float = 90.0,
    offset: float = 15.0,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Load a slice of the audio and estimate BPM and Key.
    - duration/offset chosen to skip cold intros and keep runtime down.
    """
    try:
        y, sr = librosa.load(path.as_posix(), sr=sr, mono=True, offset=offset, duration=duration)
        if y.size < sr * 5:  # too little signal
            # Try whole file short fallback
            y, sr = librosa.load(path.as_posix(), sr=sr, mono=True)

        # Simple pre-emphasis / HP to reduce DC & rumble
        y = librosa.effects.preemphasis(y)

        # BPM (tempo)
        # Use beat tracker; aggregate mean tempo (librosa returns array)
        tempo = _tempo(y, sr)
        if math.isnan(tempo) or tempo <= 0:
            tempo = None
        else:
            # Normalize to typical dance range heuristic (unfold double/half-time)
            # Bring into 70–180 BPM window
            while tempo < 70:
                tempo *= 2
            while tempo > 180:
                tempo /= 2
            tempo = round(float(tempo), 2)

        # Key via mean CQT chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        tonic, mode = chroma_key(chroma_mean)
        key_str = f"{tonic} {mode}"

        return tempo, key_str
    except Exception as e:
        # Print once but don't crash the batch
        print(f"[WARN] Failed to analyze {path}: {e}", file=sys.stderr)
        return None, None

def analyze_item(idx: int, path_str: Optional[str], sr: int, duration: float, offset: float) -> Tuple[int, Optional[float], Optional[str]]:
    if not path_str:
        return idx, None, None
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        return idx, None, None
    tempo, key = estimate_bpm_and_key(path, sr=sr, duration=duration, offset=offset)
    return idx, tempo, key
def _tempo(y, sr):
    """Version-safe tempo wrapper using a callable aggregate."""
    try:
        # librosa >= 0.10
        from librosa.feature.rhythm import tempo as lr_tempo
        t = lr_tempo(y=y, sr=sr, aggregate=np.mean)
    except Exception:
        # librosa < 0.10
        t = librosa.beat.tempo(y=y, sr=sr, aggregate=np.mean)
    # librosa returns an array; take float
    return float(np.asarray(t).squeeze())

# ---- Path resolution ---- #

def semicolon_path_to_real(base_dir: Path, token_path: str) -> Path:
    """
    Convert a semicolon-separated pseudo-path (./A;B;C.mp3) to a real path under base_dir.
    - Strips leading './'
    - Replaces ';' with '/'
    """
    tp = token_path.strip()
    if tp.startswith("./"):
        tp = tp[2:]
    tp = tp.replace(";", os.sep)
    return (base_dir / tp).resolve()

def _strip_dot_slash(s: str) -> str:
    s = s.strip()
    return s[2:] if s.startswith("./") else s

def find_audio_path(row) -> Optional[Path]:
    directory = str(row.get("Directory", "") or "").strip()
    source_file = str(row.get("SourceFile", "") or "").strip()
    file_name   = str(row.get("FileName", "")   or "").strip()

    if not directory:
        return None

    base = Path(directory)

    # candidates in priority order; DO NOT replace ';'
    cand_strings = []
    if source_file:
        sf = _strip_dot_slash(source_file)
        cand_strings += [sf, os.path.basename(sf)]
    if file_name:
        fn = _strip_dot_slash(file_name)
        cand_strings += [fn, os.path.basename(fn)]

    # de-dup while preserving order
    seen = set()
    cands = []
    for s in cand_strings:
        if s and s not in seen:
            seen.add(s)
            cands.append((base / s).resolve())

    for c in cands:
        if c.exists() and c.is_file():
            return c

    # fallback: search by basename
    basename = None
    if file_name:
        basename = os.path.basename(file_name)
    elif source_file:
        basename = os.path.basename(source_file)

    if basename:
        try:
            for p in base.rglob(basename):
                if p.is_file():
                    return p.resolve()
        except Exception:
            pass

    return None

# ---- Main ---- #

def main():
    ap = argparse.ArgumentParser(description="Fill BPM and Key in a music CSV.")
    ap.add_argument("-o", "--output-csv", help="Path to write updated CSV. Default: overwrite input.", default="tags_updated.csv")
    ap.add_argument("--force", action="store_true", help="Recompute BPM/Key even if already present.")
    ap.add_argument("--duration", type=float, default=90.0, help="Analysis duration in seconds (default: 90).")
    ap.add_argument("--offset", type=float, default=15.0, help="Start offset in seconds (default: 15).")
    ap.add_argument("--sr", type=int, default=22050, help="Target sample rate (default: 22050).")
    ap.add_argument("--workers", type=int, default=10, help="Parallel workers (0/1 = single-thread).")
    args = ap.parse_args()

    df = pd.read_csv("tags.csv")

    # Normalize column names just in case
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Ensure columns exist
    if "Bpm" not in df.columns:
        df["Bpm"] = np.nan
    if "Key" not in df.columns:
        df["Key"] = ""

    # Row indices to process
    to_process = []
    for idx, row in df.iterrows():
        bpm_empty = (args.force or pd.isna(row.get("Bpm")) or str(row.get("Bpm")).strip() in ("", "nan", "None", "0"))
        key_empty = (args.force or not str(row.get("Key", "")).strip())
        if bpm_empty or key_empty:
            to_process.append(idx)

    if not to_process:
        print("Nothing to do. All rows have BPM and Key (use --force to recompute).")
        return
    print("Processing task")
    tasks = []
    for idx in to_process:
        row = df.loc[idx]
        audio_path = find_audio_path(row)
        tasks.append((idx, audio_path))
        print(f"Finished {idx}")

    # Worker function
    def work(item):
        idx, path = item
        if path is None:
            return idx, None, None
        tempo, key = estimate_bpm_and_key(path, sr=args.sr, duration=args.duration, offset=args.offset)
        return idx, tempo, key

    results = []
    if args.workers and args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(analyze_item, idx, p, args.sr, args.duration, args.offset) for (idx, p) in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Analyzing"):
                results.append(f.result())
    else:
        for (idx, p) in tqdm(tasks, desc="Analyzing", total=len(tasks)):
            results.append(analyze_item(idx, p, args.sr, args.duration, args.offset))
    print("Applying results")
    # Apply results
    for idx, tempo, key in results:
        if tempo is not None:
            df.at[idx, "Bpm"] = tempo
        if key is not None:
            df.at[idx, "Key"] = key

    out = args.output_csv or args.input_csv
    df.to_csv(out, index=False)
    print(f"Done. Wrote {out}")

if __name__ == "__main__":
    main()
