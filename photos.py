"""The real photo count, read from the lot data already in the database.

    python photos.py inspect          # step 1: find where the photos are, show a sample
    python photos.py backfill         # step 2: store the count for every lot
    python photos.py backfill PATH    #         … from a different path

Why: photo_count is broken. The parser counts every Catawiki image on the page
(site chrome, related lots) and caps at 12, so a third of lots sit at exactly
12 and the model learns nothing from it.

Every lot row keeps the page's lot object in lots.raw. The photo list is in
there, but its key name is not known for sure, so this finds it rather than
guessing:

* inspect walks raw for every list whose entries look like images (URLs, or
  objects holding an image URL), reports each path with how often it occurs
  and its typical length, and prints a few lots so you can open them and count
  the photos by hand. The right path is the one whose count matches.
* backfill stores that count in a new column, lots.photo_n, for every lot where
  it is still empty. Run it again at any time; it only fills gaps.

Nothing is fetched from Catawiki. Nothing is changed by `inspect`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np

import db

IMG_RE = re.compile(r"\.(jpe?g|png|webp)(\?|$)|/images?/|assets\.catawiki|cloudfront", re.I)
NAME_HINT = re.compile(r"image|photo|picture|gallery|media|thumb", re.I)

# Found by `inspect` on 26 Sep: 5–47 per lot, median 11. The runner-up,
# seo.ldSchema.image, is always 2 — an SEO preview, not the gallery.
PHOTO_PATH = "lotDetailsData.images"


def _is_image(v) -> bool:
    if isinstance(v, str):
        return bool(IMG_RE.search(v))
    if isinstance(v, dict):
        return any(isinstance(x, str) and IMG_RE.search(x) for x in v.values()) \
            or any(_is_image(x) for x in v.values() if isinstance(x, dict))
    return False


def image_lists(node, path: str = ""):
    """Yield (path, count) for every list of image-like entries in the JSON."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from image_lists(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        if node and sum(_is_image(x) for x in node) >= max(1, len(node) // 2):
            yield path, len(node)
        for x in node:
            yield from image_lists(x, f"{path}[]")


def count_at(raw, path: str) -> int | None:
    """Length of the list at a dotted path like 'images' or 'lot.media.photos'."""
    node = raw
    for part in path.split("."):
        if part.endswith("[]"):          # lists inside lists are not photo galleries
            return None
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return len(node) if isinstance(node, list) else None


def _raw(r):
    raw = r["raw"]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def inspect(n_sample: int = 6) -> None:
    with db.connect() as conn:
        rows = conn.execute(
            "select lot_id, url, photo_count, raw from lots "
            "where raw is not null order by last_seen_at desc limit 400"
        ).fetchall()
    if not rows:
        sys.exit("no lots with raw data")

    seen = defaultdict(list)
    empty = 0
    for r in rows:
        raw = _raw(r)
        found = dict(image_lists(raw)) if raw else {}
        if not found:
            empty += 1
        for p, c in found.items():
            seen[p].append(c)

    print(f"{len(rows)} recent lots checked, {empty} with no image list in raw\n")
    if not seen:
        print("No image lists found. The photos are not in lots.raw, so the parser")
        print("itself has to be fixed — paste this output and I will do that instead.")
        return

    print(f"{'path':<50}{'lots':>6}{'median':>8}{'min':>5}{'max':>5}")
    # A gallery covers nearly every lot AND varies in length from lot to lot;
    # a list that is always the same size is page furniture (SEO, badges).
    def score(kv):
        p, counts = kv
        varies = max(counts) > min(counts)
        return (-varies, -len(counts), -bool(NAME_HINT.search(p)), -np.median(counts))
    ranked = sorted(seen.items(), key=score)
    for p, counts in ranked[:10]:
        print(f"{p[:49]:<50}{len(counts):>6}{np.median(counts):>8.0f}"
              f"{min(counts):>5}{max(counts):>5}")

    best = next((p for p, _ in ranked if "[]" not in p), None)
    if best is None:
        print("\nEvery image list sits inside another list — paste this output.")
        return
    print(f"\nMost likely: {best}")
    print("Open two or three of these lots and count the photos in the gallery.")
    cmd = "python photos.py backfill" + ("" if best == PHOTO_PATH else f" {best}")
    print(f"If the numbers match, run:  {cmd}\n")
    print(f"   {'counted':>7}{'old':>5}  lot")
    for r in rows[:n_sample]:
        raw = _raw(r)
        print(f"   {str(count_at(raw, best) if raw else None):>7}{r['photo_count']:>5}  {r['url']}")


def has_column(conn) -> bool:
    return bool(conn.execute(
        "select 1 from information_schema.columns "
        "where table_name = 'lots' and column_name = 'photo_n'").fetchall())


def fill_new(path: str = PHOTO_PATH) -> None:
    """Fill photo_n for newly discovered lots. Does nothing until the first
    manual backfill has created the column — so nothing happens before you
    have checked the counts by hand."""
    with db.connect() as conn:
        if not has_column(conn):
            return
        rows = conn.execute("select lot_id, raw from lots "
                            "where photo_n is null and raw is not null").fetchall()
        for r in rows:
            raw = _raw(r)
            n = count_at(raw, path) if raw else None
            if n is not None:
                conn.execute("update lots set photo_n = %s where lot_id = %s", (n, r["lot_id"]))
        conn.commit()


def backfill(path: str = PHOTO_PATH) -> None:
    with db.connect() as conn:
        conn.execute("alter table lots add column if not exists photo_n int")
        conn.commit()
        rows = conn.execute(
            "select lot_id, raw from lots where photo_n is null and raw is not null"
        ).fetchall()
        done = missing = 0
        for r in rows:
            raw = _raw(r)
            n = count_at(raw, path) if raw else None
            if n is None:
                missing += 1
                continue
            conn.execute("update lots set photo_n = %s where lot_id = %s", (n, r["lot_id"]))
            done += 1
            if done % 500 == 0:
                conn.commit()
        conn.commit()
    print(f"photo_n stored for {done} lots; {missing} had no list at '{path}'")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inspect":
        inspect()
    elif cmd == "backfill":
        backfill(sys.argv[2] if len(sys.argv) > 2 else PHOTO_PATH)
    else:
        sys.exit(__doc__)
