"""When does the Mac need to be awake?

    python when.py          # the next 7 days
    python when.py 3        # the next 3 days

Lists upcoming closes of vintage-Omega lots, day by day and hour by hour, and
suggests an awake window for each day. The window starts AWAKE_BEFORE hours
before the first busy hour — the monitor needs a bid from 1–5 hours before the
close, and the first alert goes out 3 hours before — and ends after the last
busy hour.

Only lots already discovered are counted, so days further ahead fill up as
the discover job finds more.
"""

from __future__ import annotations

import sys

import pandas as pd

import db
import model
from config import cfg

AWAKE_BEFORE = 5      # hours before the first busy hour
BUSY = 3              # an hour with at least this many target lots is "busy"

QUERY = """
select l.lot_id, l.brand, l.gender, l.watch_year, l.watch_period, l.close_time
from lots l left join lot_results r using (lot_id)
where r.lot_id is null
  and l.close_time > now()
  and l.close_time < now() + make_interval(days => %s)
"""


def main(days: int) -> None:
    with db.connect() as conn:
        rows = conn.execute(QUERY, (days,)).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        print("No upcoming lots in the database. Is discover running?")
        return
    df["watch_year"] = pd.to_numeric(df["watch_year"], errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df = df[model.target_mask(df)]
    if df.empty:
        print("No upcoming vintage-Omega lots yet.")
        return

    local = df["close_time"].dt.tz_convert(cfg.timezone)
    df = df.assign(day=local.dt.date, hour=local.dt.hour)

    print(f"Upcoming vintage-Omega closes, next {days} days ({len(df)} lots)\n")
    for day, g in df.groupby("day"):
        counts = g["hour"].value_counts().sort_index()
        busy = counts[counts >= BUSY]
        label = pd.Timestamp(day).strftime("%a %d %b")
        hours = "  ".join(f"{h:02d}h:{n}" for h, n in counts.items())
        print(f"{label}  {len(g):>3} lots   {hours}")
        if len(busy):
            start = max(0, int(busy.index.min()) - AWAKE_BEFORE)
            end = min(24, int(busy.index.max()) + 1)
            print(f"{'':<11}→ keep the Mac awake {start:02d}:00–{end:02d}:00")
        print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
