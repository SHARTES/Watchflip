"""What the agent knows, and how well each part is working. Read-only.

    python status.py

Sections:
  A. auction data       — lots, results, how far back, how much is your target
  B. coverage           — per day: did we see each lot's bid in its last hours?
  C. training labels    — what the price model actually learns from
  D. eBay comparables   — live and disappeared listings, references covered
  E. lot details        — material and photo counts (feed the valuation and model)
  F. live alerts        — what was sent, and how alerted lots really closed
  G. operations         — job runs and errors, last 7 days
  H. your trades

Each section runs on its own; one failing does not stop the rest.
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

import db
import model


def q(sql, params=None) -> pd.DataFrame:
    with db.connect() as conn:
        return pd.DataFrame(conn.execute(sql, params).fetchall())


def line(label, value, note=""):
    print(f"   {label:<46}{value:>10}  {note}")


def section(title):
    def wrap(fn):
        def run():
            print(f"\n{title}")
            try:
                fn()
            except Exception as e:
                print(f"   (could not run: {type(e).__name__}: {str(e).splitlines()[0][:120]})")
        return run
    return wrap


def _all_lots() -> pd.DataFrame:
    df = q("""
        select l.lot_id, l.brand, l.gender, l.watch_year, l.watch_period, l.close_time,
               l.reference_number, l.estimate_low, r.final_price, r.sold, r.lot_id as has_result
        from lots l left join lot_results r using (lot_id)""")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["watch_year"] = pd.to_numeric(df["watch_year"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["target"] = model.target_mask(df)
    return df


LOTS = None


def lots():
    global LOTS
    if LOTS is None:
        LOTS = _all_lots()
    return LOTS


@section("A. AUCTION DATA")
def auction():
    df = lots()
    now = pd.Timestamp.now(tz="UTC")
    t = df[df["target"]]
    line("lots collected, all watches", f"{len(df):,}")
    line("  with a final result", f"{df['has_result'].notna().sum():,}")
    line("  sold (training labels)", f"{(df['sold'] == True).sum():,}")
    line("  first / last close", f"{df['close_time'].min():%d %b}",
         f"→ {df['close_time'].max():%d %b %Y}")
    line("vintage-Omega target lots", f"{len(t):,}")
    line("  sold", f"{(t['sold'] == True).sum():,}")
    line("  unsold (no bid met reserve)", f"{(t['sold'] == False).sum():,}")
    line("  still open", f"{(t['has_result'].isna() & (t['close_time'] > now)).sum():,}")
    line("  closed but no result yet", f"{(t['has_result'].isna() & (t['close_time'] <= now)).sum():,}",
         "sweeper backlog or gave up")
    s = t[t["sold"] == True].set_index("close_time").sort_index()
    if len(s):
        wk = s["lot_id"].resample("W").count().tail(6)
        print("   target lots sold per week: " +
              "  ".join(f"{d:%d %b}:{n}" for d, n in wk.items()))


@section("B. COVERAGE — was each lot seen 1–5 h before close? (last 14 days)")
def coverage():
    d = q("""
        select date(l.close_time at time zone 'Europe/Vienna') as day, l.lot_id,
               exists (select 1 from bid_snapshots s where s.lot_id = l.lot_id
                       and s.minutes_to_close between 60 and 300) as seen_late,
               r.lot_id is not null as has_result
        from lots l left join lot_results r using (lot_id)
        where l.close_time between now() - interval '14 days' and now()""")
    if d.empty:
        print("   no lots closed in the last 14 days")
        return
    g = d.groupby("day").agg(lots=("lot_id", "count"), result=("has_result", "mean"),
                             late=("seen_late", "mean"))
    print(f"   {'day':<12}{'lots':>6}{'result':>9}{'seen late':>11}")
    for day, r in g.iterrows():
        flag = "  ← low" if r["late"] < 0.6 else ""
        print(f"   {day!s:<12}{int(r['lots']):>6}{r['result']:>9.0%}{r['late']:>11.0%}{flag}")
    print("   'seen late' below ~60% = the Mac slept or the monitor fell behind that day.")


@section("C. TRAINING LABELS — what the price model learns from")
def labels():
    df = model.load()
    t = df[model.target_mask(df)]
    f = model.featurise(df)
    active = f["late_bid"].notna()
    line("sold lots, all watches", f"{len(df):,}")
    line("  with an active bid seen 1–5 h before close", f"{int(active.sum()):,}",
         f"{active.mean():.0%}")
    line("sold target lots", f"{len(t):,}")
    tm = model.target_mask(df)
    line("  with an active late bid", f"{int(active[tm].sum()):,}", f"{active[tm].mean():.0%}")
    line("  with a reference number", f"{int(t['reference_number'].notna().sum()):,}")
    line("  with a Catawiki estimate", f"{int(t['estimate_low'].notna().sum()):,}")
    k = t["reference_number"].map(model.ref_key).dropna()
    vc = k.value_counts()
    line("  distinct references", f"{len(vc):,}")
    line("  references sold 2+ times", f"{int((vc >= 2).sum()):,}", "usable as Catawiki evidence")


@section("D. EBAY COMPARABLES")
def ebay():
    import asks
    c = q("select query_reference, price, currency, title, disappeared_at from comps")
    line("listings stored", f"{len(c):,}")
    line("  live", f"{c['disappeared_at'].isna().sum():,}")
    line("  disappeared (likely sold or withdrawn)", f"{c['disappeared_at'].notna().sum():,}")
    by_ref = asks.load()
    usable = sum(1 for g in by_ref.values()
                 if (v := asks.value(g, "steel")) is not None and v[1] >= 3)
    line("references with live listings", f"{len(by_ref):,}")
    line("  with 3+ distinct clean listings (valuable)", f"{usable:,}")
    df = lots()
    t = df[df["target"]]
    keys = t["reference_number"].map(model.ref_key)
    line("share of target lots whose reference has eBay value",
         f"{keys.map(lambda k: k in by_ref and (asks.value(by_ref[k], 'steel') or (0, 0))[1] >= 3).mean():.0%}")
    qq = q("select count(*) as n, max(last_run_at) as last from comp_queries")
    if not qq.empty:
        line("references queried on eBay", f"{int(qq['n'][0]):,}", f"last run {qq['last'][0]}")


@section("E. LOT DETAILS — material and photos")
def details():
    d = q("""select l.lot_id, l.title, l.description, l.brand, l.gender, l.watch_year,
                    l.watch_period,
                    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Case material").value') #>> '{}' as case_material
             from lots l""")
    d["watch_year"] = pd.to_numeric(d["watch_year"], errors="coerce")
    t = d[model.target_mask(d)]
    mat = t.apply(model.lot_material, axis=1)
    from_field = t["case_material"].map(model.case_material).notna()
    print(f"   case material for target lots ({from_field.mean():.0%} from Catawiki's own field):")
    for k, v in mat.value_counts().items():
        print(f"      {k:<10}{v:>6}  ({v / len(t):.0%})")
    p = q("select count(*) filter (where photo_n is not null) as filled, count(*) as n from lots")
    line("lots with a real photo count", f"{int(p['filled'][0]):,}", f"of {int(p['n'][0]):,}")
    r = q("""select jsonb_path_query_first(raw, '$.lotDetailsData.specifications[*] ? (@.name == "Repainted dial").value') #>> '{}' as v,
                    count(*) as n from lots group by 1 order by 2 desc""")
    print("   'Repainted dial' field: " + "  ".join(f"{v or 'missing'}: {int(n):,}" for v, n in zip(r["v"], r["n"])))
    try:
        d = q("""select count(*) as n, count(distinct lot_id) as lots,
                        max(observed_at at time zone 'Europe/Vienna') as last from demand_snapshots""")
        line("watcher/bidder snapshots recorded", f"{int(d['n'][0]):,}",
             f"{int(d['lots'][0]):,} lots · last {d['last'][0]}")
    except Exception:
        print("   watcher/bidder snapshots: none yet (start after the new poller.py runs)")


@section("F. LIVE ALERTS — what the shortlist sent, and what happened")
def alerts():
    a = q("""
        select a.lot_id, a.sent_at, a.max_bid, a.p_win, a.exp_profit, a.reminded_at,
               r.final_price, r.sold
        from shortlist_alerts a left join lot_results r using (lot_id)""")
    if a.empty:
        print("   no alerts sent yet")
        return
    for c in ("max_bid", "p_win", "final_price"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["sent_at"] = pd.to_datetime(a["sent_at"], utc=True)
    line("alerts sent", f"{len(a):,}", f"first {a['sent_at'].min():%d %b}")
    line("  last 7 days", f"{(a['sent_at'] > pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)).sum():,}")
    if "reminded_at" in a:
        line("  got a last-call reminder", f"{a['reminded_at'].notna().sum():,}")
    done = a[a["final_price"].notna()]
    if len(done):
        under = done["final_price"] <= done["max_bid"]
        line("alerted lots closed so far", f"{len(done):,}")
        line("  closed at or under the max bid", f"{int(under.sum()):,}", f"{under.mean():.0%}")
        line("  the alerts predicted on average", f"{done['p_win'].mean():.0%}",
             "← close to the line above = honest chances")
        print("   (alerts sent before the 0-bid fixes on 27 Sep overstate the chance;")
        print("    compare only recent ones once there are 20+)")


@section("G. OPERATIONS — last 7 days")
def ops():
    f = q("""select job, count(*) as runs, sum(errors) as errors,
                    max(ran_at at time zone 'Europe/Vienna') as last
             from fetch_log where ran_at > now() - interval '7 days' group by job order by job""")
    if f.empty:
        print("   no runs logged in the last 7 days — is the daemon running?")
        return
    expected = {"monitor": 96 * 7, "discover": 12 * 7, "sweeper": 48 * 7}
    for _, r in f.iterrows():
        exp = expected.get(r["job"])
        share = f"{r['runs'] / exp:.0%} of schedule" if exp else ""
        line(f"{r['job']}", f"{int(r['runs']):,} runs", f"{int(r['errors'] or 0)} errors · {share} · last {r['last']:%a %H:%M}")


@section("H. YOUR TRADES")
def trades():
    try:
        t = q("select lot_id, paid_total, sold_price from trades")
    except Exception:
        print("   none recorded yet — python trade.py buy …")
        return
    if t.empty:
        print("   none recorded — python trade.py buy …")
        return
    line("bought", f"{len(t)}")
    line("  sold", f"{t['sold_price'].notna().sum()}")
    print("   details: python trade.py show")


if __name__ == "__main__":
    for fn in (auction, coverage, labels, ebay, details, alerts, ops, trades):
        fn()
