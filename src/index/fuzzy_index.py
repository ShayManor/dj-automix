#!/usr/bin/env python3
import argparse, os, re, json, sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

# Use orjson if installed (faster); fall back to stdlib json
try:
    import orjson as _json
    def jloads(b): return _json.loads(b)
    def jdumps(o): return _json.dumps(o)
except Exception:
    def jloads(b): return json.loads(b)
    def jdumps(o): return json.dumps(o).encode()

from rapidfuzz import process, fuzz

FIELDS = ["title","artist","album","key","bpm","path"]

# ---------- Normalization ----------
def norm(s: str) -> str:
    if s is None: return ""
    s = str(s)
    s = s.lower()
    # collapse whitespace, strip punctuation commonly found in song titles
    s = re.sub(r"[^\w\s+#/.-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_int(x) -> Optional[int]:
    try: return int(str(x).strip())
    except Exception: return None

@dataclass
class Song:
    title: str = ""
    artist: str = ""
    album: str = ""
    key: str = ""
    bpm: Any = ""
    path: str = ""

    def normalized(self) -> Dict[str,str]:
        return {
            "title_n": norm(self.title),
            "artist_n": norm(self.artist),
            "album_n": norm(self.album),
            "key_n": norm(self.key).replace(" major","").replace(" minor","m"),
            "bpm_n": str(to_int(self.bpm) or ""),
            "path_n": norm(self.path),
        }

# ---------- Index ----------
class SongIndex:
    def __init__(self, items: List[Song]):
        self.items: List[Song] = items
        # Precompute normalized fields and a combined searchable string
        self.norms: List[Dict[str,str]] = []
        self.search_texts: List[str] = []
        for s in items:
            n = s.normalized()
            self.norms.append(n)
            # Combined text for coarse candidate gen; keep small to save RAM
            self.search_texts.append(" | ".join([n["title_n"], n["artist_n"], n["album_n"]]))

    # Weighted fuzzy score across fields (like Fuse field weights)
    def score(self, q: str, idx: int) -> float:
        n = self.norms[idx]
        w_title  = 0.55
        w_artist = 0.4
        w_album  = 0.05
        # RapidFuzz scores are 0..100
        s_title  = fuzz.WRatio(q, n["title_n"])   if n["title_n"]  else 0
        s_artist = fuzz.WRatio(q, n["artist_n"])  if n["artist_n"] else 0
        s_album  = fuzz.WRatio(q, n["album_n"])   if n["album_n"]  else 0
        return w_title*s_title + w_artist*s_artist + w_album*s_album

    # Simple query language: field:value, bpm:lo..hi, quoted phrases
    def parse_query(self, query: str):
        filters = {}
        # bpm range e.g. bpm:120..130
        m = re.search(r"\bbpm\s*:\s*(\d+)\s*\.\.\s*(\d+)", query, flags=re.I)
        if m:
            filters["bpm_range"] = (int(m.group(1)), int(m.group(2)))
            query = re.sub(r"\bbpm\s*:\s*\d+\s*\.\.\s*\d+", "", query, flags=re.I)

        # field:value pairs (artist:, title:, album:, key:, bpm:)
        pairs = re.findall(r"\b(title|artist|album|key|bpm)\s*:\s*([^\s][^']*|'[^']*')", query, flags=re.I)
        for f, v in pairs:
            v = v.strip()
            if v.startswith("'") and v.endswith("'"): v = v[1:-1]
            filters[f.lower()] = norm(v) if f.lower() != "bpm" else v
        if pairs:
            # remove them from free-text
            query = re.sub(r"\b(title|artist|album|key|bpm)\s*:\s*([^\s][^']*|'[^']*')", "", query, flags=re.I)

        # exact phrase: 'once in a lifetime'
        phrase = None
        pm = re.search(r"'([^']+)'", query)
        if pm: phrase = norm(pm.group(1)); query = re.sub(r"'[^']+'", "", query)

        query = norm(query)
        return query, phrase, filters

    def filter_candidates(self, phrase: Optional[str], filters: Dict[str,Any]) -> List[int]:
        cands = []
        for i, n in enumerate(self.norms):
            ok = True
            # phrase must be in title or artist
            if phrase:
                if phrase not in n["title_n"] and phrase not in n["artist_n"]:
                    ok = False
            # field filters
            if ok and "artist" in filters and filters["artist"] not in n["artist_n"]:
                ok = False
            if ok and "title" in filters and filters["title"] not in n["title_n"]:
                ok = False
            if ok and "album" in filters and filters["album"] not in n["album_n"]:
                ok = False
            if ok and "key" in filters:
                # allow “am”, “a minor”, “A minor” → we normalized to short (e.g., am, c#, etc.)
                keyq = filters["key"].replace(" major","").replace(" minor","m")
                if keyq not in n["key_n"]:
                    ok = False
            if ok and "bpm" in filters:
                if str(filters["bpm"]) != n["bpm_n"]:
                    ok = False
            if ok and "bpm_range" in filters:
                lo, hi = filters["bpm_range"]
                bpm_i = to_int(n["bpm_n"])
                if bpm_i is None or bpm_i < lo or bpm_i > hi:
                    ok = False
            if ok:
                cands.append(i)
        return cands

    def search(self, query: str, limit: int = 10, threshold: float = 60.0) -> List[Tuple[Song, float]]:
        q_free, phrase, filters = self.parse_query(query)

        # Step 1: coarse candidate gen (if free text present)
        if q_free:
            # RapidFuzz extract gives (text, score, index)
            # We just use it to shortlist indices, then rescore with field weights
            ex = process.extract(q_free, self.search_texts, scorer=fuzz.WRatio, limit=max(100, limit*10))
            cand_idx = [idx for _, _, idx in ex]
        else:
            cand_idx = list(range(len(self.items)))

        # Step 2: apply filters (phrase, field filters, bpm)
        cand_idx = [i for i in cand_idx if i in set(self.filter_candidates(phrase, filters))]

        # Step 3: weighted rescore and sort
        scored = []
        for i in cand_idx:
            s = self.score(q_free or self.norms[i]["title_n"], i)
            if s >= threshold:
                scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for i, s in scored[:limit]:
            out.append((self.items[i], round(s, 2)))
        return out

    # -------- persist/load --------
    def to_bytes(self) -> bytes:
        payload = {
            "items": [asdict(s) for s in self.items],
            "norms": self.norms,
            "search_texts": self.search_texts,
        }
        return jdumps(payload)

    @staticmethod
    def from_bytes(b: bytes) -> "SongIndex":
        p = jloads(b)
        items = [Song(**d) for d in p["items"]]
        idx = SongIndex(items)
        # load precomputed to avoid recompute
        idx.norms = p["norms"]
        idx.search_texts = p["search_texts"]
        return idx

# ---------- IO helpers ----------
def load_songs(path: str) -> List[Song]:
    if path.lower().endswith(".json"):
        with open(path, "rb") as f:
            arr = jloads(f.read())
    elif path.lower().endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        arr = df.to_dict(orient="records")
    else:
        raise SystemExit("Provide .json or .csv of tracks.")
    songs = []
    for r in arr:
        songs.append(Song(
            title = r.get("title") or r.get("Title") or "",
            artist= r.get("artist") or r.get("Artist") or "",
            album = r.get("album") or r.get("Album") or "",
            key   = r.get("key")   or r.get("Key")   or "",
            bpm   = r.get("bpm")   or r.get("BPM")   or "",
            path  = r.get("path")  or r.get("SourceFile") or r.get("filename") or ""
        ))
    return songs

# ---------- CLI ----------
def cmd_build(args):
    items = load_songs(args.input)
    idx = SongIndex(items)
    with open(args.output, "wb") as f:
        f.write(idx.to_bytes())
    print(f"Index built: {args.output}  ({len(items)} tracks)")

def cmd_search(args):
    with open(args.index, "rb") as f:
        idx = SongIndex.from_bytes(f.read())
    res = idx.search(args.query, limit=args.limit, threshold=args.threshold)
    for song, score in res:
        bpm = str(song.bpm)
        key = song.key
        print(f"{score:6.2f}  {song.artist} - {song.title}  [{key} • {bpm}]  {song.path}")

def main():
    ap = argparse.ArgumentParser(description="Fuzzy song search (Fuse.js-like) in Python.")
    sub = ap.add_subparsers(required=True)

    ap_b = sub.add_parser("build", help="Build index from CSV/JSON.")
    ap_b.add_argument("-i","--input", required=True, help="CSV or JSON of tracks.")
    ap_b.add_argument("-o","--output", default="songs.index", help="Output index file.")
    ap_b.set_defaults(func=cmd_build)

    ap_s = sub.add_parser("search", help="Search the index.")
    ap_s.add_argument("-x","--index", required=True, help="Index file from 'build'.")
    ap_s.add_argument("query", help="Query. Supports field:term, bpm:120..130, 'exact phrase'")
    ap_s.add_argument("-k","--limit", type=int, default=10)
    ap_s.add_argument("-t","--threshold", type=float, default=60.0, help="Score threshold 0..100")
    ap_s.set_defaults(func=cmd_search)

    args = ap.parse_args()
    args.func(args)

def search_index(query: str, limit: int = 10, threshold: float = 60.0, as_dict: bool = False, index_path: str="songs.index"):
    """
    Convenience search helper.
    - query: free text and/or filters (e.g., "artist:twice bpm:120..130 'once in a lifetime'")
    - limit: max results
    - threshold: 0..100 fuzzy score cutoff
    - as_dict: return list of dicts instead of (Song, score) tuples
    - index_path: path to songs.index produced by `build`

    Returns:
      [] on no matches, else:
        - as_dict=False: list[(Song, score)]
        - as_dict=True:  list[{"score": float, "title": ..., "artist": ..., "album": ..., "key": ..., "bpm": ..., "path": ...}]
    """
    if not isinstance(query, str) or not query.strip():
        return []

    # Robust file read
    try:
        with open(index_path, "rb") as f:
            idx = SongIndex.from_bytes(f.read())
    except FileNotFoundError:
        raise FileNotFoundError(f"Index not found: {index_path}. Build it with: songsearch.py build -i songs.json -o {index_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load index '{index_path}': {e}")

    try:
        results = idx.search(query.strip(), limit=max(1, int(limit)), threshold=float(threshold))
    except Exception as e:
        # Catch any parsing/scoring edge cases
        raise RuntimeError(f"Search failed for query '{query}': {e}")

    if not results:
        return []

    if not as_dict:
        return results

    out = []
    for song, score in results:
        out.append({
            "score": float(score),
            "title": song.title,
            "artist": song.artist,
            "album": song.album,
            "key": song.key,
            "bpm": song.bpm,
            "path": song.path,
        })
    return out

if __name__ == "__main__":
    main()
