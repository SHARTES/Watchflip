"""Phase 1 — does the listing data predict the closing price?

    python model.py audit     # what features do we actually have
    python model.py baseline  # the numbers a model has to beat
    python model.py train     # fit, and compare old features against new
    python model.py calibrate # make the p20/p80 band honest, per regime

Design notes, because they decide whether the result means anything:

* The split is by TIME, not random. Lots closing the same evening share an
  auction, a bidder pool and a cohort of watches. A random split puts siblings
  on both sides and flatters the score.

* The target is log(price). Prices are heavily right-skewed, so squared error
  on raw euros would let a handful of expensive lots dominate the fit.

* Only sold lots. Unsold ones have no price to predict.

* Error is median absolute percentage error — what you would feel when bidding,
  and the median resists the one wild miss.

What v2 adds, and why
---------------------

The v1 model leaned 75% on Catawiki's expert estimate, which is missing on a
third of lots. When it was missing, error jumped from 19% to 33%. So v2 adds
features that carry value information directly:

* Case material from the text. Solid gold sells for roughly twice steel, and
  the Genève line runs from EUR 96 to EUR 1,600 largely on this. Plated is
  checked before gold, because "18k gold plated" contains "18k".
* Condition words: serviced, box, papers, defect, chronometer.
* Late bid — the current bid 1 to 5 hours before close. Legitimate for the
  real use case, since you decide shortly before the end and can see it.
* Reference price history, leave-one-out and smoothed toward the global mean,
  so a reference sold once does not dominate.
"""

from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pandas as pd

import db

warnings.filterwarnings("ignore")

TEST_FRACTION = 0.25

CONDITION_ORDER = {
    "new": 5,
    "unworn - no signs of wear": 5,
    "very good - minor signs of wear": 4,
    "good - visible signs of wear": 3,
    "fair - major signs of wear": 2,
    "poor - significant signs of wear": 1,
}

PERIOD_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Order matters: plated first, because "18k gold plated" would otherwise read
# as solid gold.
PLATED_RE = re.compile(
    r"gold.?plated|goldplated|plaqu|gold.?capped|\bcapped\b|micron|doublé|"
    r"vergoldet|gold.?filled|rolled gold|placcat", re.I)
GOLD_RE = re.compile(
    r"\b(18|14|9)\s?(k|kt|ct|carat|karat)\b|solid gold|massiv\w*\s?gold|"
    r"oro massiccio|or massif|\b750\s?(gold|/1000)|gold\s?750", re.I)
STEEL_RE = re.compile(
    r"stainless steel|\bsteel\b|edelstahl|\bacier\b|acciaio|\binox", re.I)

FLAG_PATTERNS = {
    "serviced":    r"\bserviced\b|revision|revisioniert|révis|revisionat",
    "has_box":     r"\bbox\b|scatola|boîte|schachtel",
    "has_papers":  r"papers|certificat|papiere|garantie|warranty",
    "defect":      r"not running|not working|defect|for parts|spares|non funziona|"
                   r"defekt|ne fonctionne|needs service|needs repair",
    "chronometer": r"chronometer",
}
FLAG_RES = {k: re.compile(v, re.I) for k, v in FLAG_PATTERNS.items()}

# The v1 feature set, kept so every run shows what the new features bought.
V1_COLS = [
    "year", "condition", "diameter", "est_mid", "est_spread", "photos",
    "title_len", "desc_len", "has_ref", "bid_count", "snapshots", "bid_24h",
    "no_reserve", "close_hour", "close_dow", "brand", "model", "movement",
    "country",
]


# --------------------------------------------------------------------- data

QUERY = """
select
    l.lot_id, l.title, l.description, l.brand, l.watch_model, l.gender,
    l.reference_number, l.watch_period, l.watch_year, l.watch_condition,
    l.movement, l.case_diameter_mm, l.photo_count, l.seller_country, l.seller_name,
    l.estimate_low, l.estimate_high, l.close_time,
    r.final_price, r.bid_count,
    (select count(*) from bid_snapshots s where s.lot_id = l.lot_id) as snapshots,
    (select s.current_bid from bid_snapshots s
     where s.lot_id = l.lot_id and s.minutes_to_close between 1200 and 1680
     order by s.minutes_to_close limit 1) as bid_24h,
    (select s.current_bid from bid_snapshots s
     where s.lot_id = l.lot_id and s.minutes_to_close between 60 and 300
     order by s.minutes_to_close limit 1) as late_bid
from lots l
join lot_results r using (lot_id)
where r.sold and r.final_price is not null and r.final_price > 0
order by l.close_time
"""


def with_photos(conn, query: str) -> str:
    """Use the real photo count (lots.photo_n, see photos.py) once it exists.

    Falls back to the old, broken count for the few lots without one, and
    everywhere until the first backfill has been run.
    """
    has = conn.execute("select 1 from information_schema.columns "
                       "where table_name = 'lots' and column_name = 'photo_n'").fetchall()
    if not has:
        return query
    return query.replace("l.photo_count,", "coalesce(l.photo_n, l.photo_count) as photo_count,")


def load() -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(with_photos(conn, QUERY)).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no sold lots with prices — nothing to model")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    # psycopg returns numeric columns as Decimal, which numpy cannot mix with
    # floats. Cast once here rather than at every use site.
    for col in ("final_price", "estimate_low", "estimate_high", "case_diameter_mm",
                "bid_24h", "late_bid", "bid_count", "photo_count", "watch_year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def period_midpoint(period) -> float | None:
    if not isinstance(period, str):
        return None
    years = [int(y) for y in PERIOD_RE.findall(period)]
    return sum(years) / len(years) if years else None


def material(text: str) -> str:
    if PLATED_RE.search(text):
        return "plated"
    if GOLD_RE.search(text):
        return "gold"
    if STEEL_RE.search(text):
        return "steel"
    return "unknown"


DOTTED_REF_RE = re.compile(r"(?<!\d)(\d{3})\.(\d{3,4})(?!\d)")


def ref_key(ref) -> str | None:
    """Normalise so '166.041' and '166041' count as the same reference.

    Sellers often glue the calibre on — '191.0101–CAL.1365', 'CAL.1330REF.191.032'.
    When a ddd.dddd pattern is present it is the Omega reference, so it wins;
    otherwise fall back to the whole string with punctuation stripped.
    """
    if not isinstance(ref, str):
        return None
    m = DOTTED_REF_RE.search(ref)
    if m:
        return m.group(1) + m.group(2)
    k = re.sub(r"[^0-9A-Za-z]", "", ref).upper()
    return k if len(k) >= 4 else None


def target_mask(df: pd.DataFrame) -> np.ndarray:
    """Vintage men's and unisex Omega — the watches actually being bought."""
    era = (df["watch_year"].between(1950, 1989)
           | df["watch_period"].astype(str).str.match(r"^(1950|1960|1970|1980)"))
    gender = df["gender"].isin(["men", "unisex"]) if "gender" in df else True
    brand = df["brand"].astype(str).str.lower() == "omega"
    return (brand & era & gender).to_numpy()


def featurise(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    f["year"] = df["watch_year"].fillna(df["watch_period"].map(period_midpoint))
    f["condition"] = df["watch_condition"].astype(str).str.lower().map(CONDITION_ORDER)
    f["diameter"] = df["case_diameter_mm"]

    lo, hi = df["estimate_low"], df["estimate_high"]
    f["est_mid"] = (lo + hi) / 2
    f["est_spread"] = (hi - lo) / ((lo + hi) / 2)

    # photo_count is unreliable (parser counts site chrome, caps at 12). Kept
    # so its importance can be read; do not trust it until the parser is fixed.
    f["photos"] = df["photo_count"]
    f["title_len"] = df["title"].fillna("").str.len()
    f["desc_len"] = df["description"].fillna("").str.len()
    f["has_ref"] = df["reference_number"].notna().astype(int)

    f["bid_count"] = df["bid_count"]
    f["snapshots"] = df["snapshots"]
    # A bid of 0 means "no bids yet", not "worth nothing". Treated as missing,
    # so the lot falls into the no-late-bid regime (listing-based, wider band)
    # instead of being predicted to close near zero.
    f["bid_24h"] = df["bid_24h"].where(df["bid_24h"] > 0)
    f["no_reserve"] = df["title"].fillna("").str.lower() \
                        .str.contains("no reserve").astype(int)

    local = df["close_time"].dt.tz_convert("Europe/Vienna")
    f["close_hour"] = local.dt.hour
    f["close_dow"] = local.dt.dayofweek

    for col, name in (("brand", "brand"), ("watch_model", "model"),
                      ("movement", "movement"), ("seller_country", "country")):
        f[name] = df[col].astype("category").cat.codes

    # ---- v2 additions ----------------------------------------------------
    text = (df["title"].fillna("") + " " + df["description"].fillna(""))
    mat = text.map(material)
    f["mat_gold"] = (mat == "gold").astype(int)
    f["mat_plated"] = (mat == "plated").astype(int)
    f["mat_steel"] = (mat == "steel").astype(int)
    f["mat_unknown"] = (mat == "unknown").astype(int)

    for name, rx in FLAG_RES.items():
        f[name] = text.map(lambda t, rx=rx: int(bool(rx.search(t))))

    f["late_bid"] = df["late_bid"].where(df["late_bid"] > 0)

    return f


def add_ref_encoding(X_tr, X_te, tr, te, m: float = 3.0):
    """Smoothed mean log-price of other sales of the same reference.

    Leave-one-out in training so a row never sees its own price. Test rows see
    only training sales. m controls how hard a thin reference is pulled toward
    the global mean: one prior sale moves the estimate only a quarter of the way.
    """
    y = np.log(tr["final_price"].astype(float))
    prior = float(y.mean())
    k_tr = tr["reference_number"].map(ref_key)
    k_te = te["reference_number"].map(ref_key)

    stats = (pd.DataFrame({"k": k_tr, "y": y}).dropna(subset=["k"])
             .groupby("k")["y"].agg(["sum", "count"]))

    s, c = k_tr.map(stats["sum"]), k_tr.map(stats["count"])
    others = c - 1
    loo = (s - y + m * prior) / (others + m)

    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr["ref_enc"] = np.where(others > 0, loo, np.nan)
    X_tr["ref_n"] = others.fillna(0).to_numpy()

    s_te, c_te = k_te.map(stats["sum"]), k_te.map(stats["count"])
    X_te["ref_enc"] = ((s_te + m * prior) / (c_te + m)).to_numpy()
    X_te["ref_n"] = c_te.fillna(0).to_numpy()
    return X_tr, X_te


def add_seller_encoding(X_tr, X_te, tr, te, m: float = 5.0):
    """How a Catawiki seller's lots price, relative to what they are.

    Same leave-one-out, smoothed scheme as the reference encoding, but on the
    RESIDUAL: log price minus the median log price of that model line, so a
    seller who only lists expensive lines does not look like a premium seller.
    What is left is presentation, trust and audience — the +51% seen on Second
    Vintage lots. m = 5: a seller needs several sales before it moves much.
    """
    y = np.log(tr["final_price"].astype(float)).to_numpy()
    line = tr["watch_model"].astype(str).str.lower()
    line_med = pd.Series(y, index=tr.index).groupby(line).median()
    resid = y - line.map(line_med).to_numpy()
    s_tr = tr["seller_name"].fillna("").astype(str).str.strip().str.lower()
    s_te = te["seller_name"].fillna("").astype(str).str.strip().str.lower()

    stats = (pd.DataFrame({"s": s_tr.to_numpy(), "r": resid})
             .query("s != ''").groupby("s")["r"].agg(["sum", "count"]))
    s, c = s_tr.map(stats["sum"]).to_numpy(), s_tr.map(stats["count"]).to_numpy()
    others = c - 1
    X_tr, X_te = X_tr.copy(), X_te.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        X_tr["seller_enc"] = np.where(others > 0, (s - resid) / (others + m), np.nan)
    X_tr["seller_n"] = np.nan_to_num(others, nan=0.0)
    s2, c2 = s_te.map(stats["sum"]).to_numpy(), s_te.map(stats["count"]).to_numpy()
    X_te["seller_enc"] = s2 / (c2 + m)
    X_te["seller_n"] = np.nan_to_num(c2, nan=0.0)
    return X_tr, X_te


def split(df: pd.DataFrame):
    cut = int(len(df) * (1 - TEST_FRACTION))
    return df.iloc[:cut], df.iloc[cut:]


def mape(pred, actual) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
    if ok.sum() == 0:
        return float("nan")
    return float(np.median(np.abs(pred[ok] - actual[ok]) / actual[ok]) * 100)


def fit(X, y, q=0.5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q,
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=15, l2_regularization=1.0, random_state=0,
    ).fit(X, y)


# -------------------------------------------------------------------- steps

def audit() -> None:
    df = load()
    f = featurise(df)
    print(f"{len(df):,} sold lots, {df['close_time'].min().date()} "
          f"→ {df['close_time'].max().date()}\n")
    print("feature availability:")
    for col in f.columns:
        filled = f[col].notna().sum()
        print(f"  {col:<14} {filled:>6,} / {len(f):,}  ({filled/len(f)*100:>5.1f}%)")

    tm = target_mask(df)
    mat = (df["title"].fillna("") + " " + df["description"].fillna("")).map(material)
    print("\nmaterial on target lots:")
    for k, v in mat[tm].value_counts().items():
        print(f"  {k:<8} {v:>5}  median €{df.loc[tm & (mat == k).to_numpy(), 'final_price'].median():.0f}")


def baseline() -> None:
    df = load()
    f = featurise(df)
    tr, te = split(df)
    print(f"train {len(tr):,}  test {len(te):,}  (test from {te['close_time'].min().date()})\n")
    gm = tr["final_price"].median()
    print(f"  global median ({gm:.0f})           {mape([gm]*len(te), te['final_price']):>6.1f}%")
    pred = te["watch_model"].map(tr.groupby("watch_model")["final_price"].median()).fillna(gm)
    print(f"  median per model line             {mape(pred, te['final_price']):>6.1f}%")
    est = f.loc[te.index, "est_mid"]
    ok = est.notna()
    print(f"  Catawiki expert estimate          {mape(est[ok], te['final_price'][ok]):>6.1f}%")


def train() -> None:
    df = load()
    f = featurise(df)
    tr, te = split(df)
    X_tr, X_te = f.loc[tr.index], f.loc[te.index]
    X_tr, X_te = add_ref_encoding(X_tr, X_te, tr, te)
    y_tr = np.log(tr["final_price"].to_numpy(float))
    actual = te["final_price"].to_numpy(float)
    tm = target_mask(te)

    gm = tr["final_price"].median()
    base = te["watch_model"].map(
        tr.groupby("watch_model")["final_price"].median()).fillna(gm).to_numpy(float)

    old = np.exp(fit(X_tr[V1_COLS], y_tr).predict(X_te[V1_COLS]))
    new = np.exp(fit(X_tr, y_tr).predict(X_te))

    print(f"train {len(tr):,}  test {len(te):,}  "
          f"features {len(V1_COLS)} → {X_tr.shape[1]}  "
          f"vintage-Omega test lots {tm.sum()}\n")

    print(f"{'':<24}{'all':>9}{'vintage Omega':>16}")
    print(f"{'baseline (line median)':<24}{mape(base, actual):>8.1f}%"
          f"{mape(base[tm], actual[tm]):>15.1f}%")
    print(f"{'v1 features':<24}{mape(old, actual):>8.1f}%"
          f"{mape(old[tm], actual[tm]):>15.1f}%")
    print(f"{'v2 features':<24}{mape(new, actual):>8.1f}%"
          f"{mape(new[tm], actual[tm]):>15.1f}%")

    has_est = X_te["est_mid"].notna().to_numpy()
    print(f"\nv2 with estimate    ({has_est.sum():>4}): {mape(new[has_est], actual[has_est]):.1f}%")
    print(f"v2 without estimate ({(~has_est).sum():>4}): {mape(new[~has_est], actual[~has_est]):.1f}%")
    late = X_te["late_bid"].notna().to_numpy()
    print(f"v2 with late bid    ({late.sum():>4}): {mape(new[late], actual[late]):.1f}%")
    print(f"v2 without late bid ({(~late).sum():>4}): {mape(new[~late], actual[~late]):.1f}%")

    p20 = np.exp(fit(X_tr, y_tr, 0.2).predict(X_te))
    p80 = np.exp(fit(X_tr, y_tr, 0.8).predict(X_te))
    below = (actual < p20).mean() * 100
    above = (actual > p80).mean() * 100
    print(f"\ncalibration — below p20: {below:.0f}% (want 20), "
          f"inside: {100-below-above:.0f}% (want 60), above p80: {above:.0f}% (want 20)")

    from sklearn.inspection import permutation_importance
    m = fit(X_tr, y_tr)
    imp = permutation_importance(m, X_te, np.log(actual), n_repeats=6, random_state=0)
    print("\nwhat v2 actually uses:")
    for i in np.argsort(imp.importances_mean)[::-1][:12]:
        print(f"  {X_tr.columns[i]:<14} {imp.importances_mean[i]:>7.4f}")


# ----------------------------------------------------------------- calibration
#
# The quantile models produce a p20-p80 band that is too narrow: on held-out
# data 54% of prices landed outside a band meant to hold 40%. Any bid ceiling
# built on p20 would therefore be overconfident.
#
# The fix is conformal calibration. Hold back the most recent slice of the
# training period, measure how far real prices actually fell outside the band
# there, and move each edge by exactly that amount. It is a correction measured
# from data, not a guess, and it comes with a finite-sample guarantee on the
# held-out coverage.
#
# It is done separately for the two regimes, because they differ threefold in
# accuracy (about 13% with a late bid, 32% without). One shared correction
# would leave late-bid bands too wide and no-late-bid bands still too narrow.

CAL_FRACTION = 0.20   # share of the training period held back to calibrate on


def regime(X: pd.DataFrame) -> np.ndarray:
    return np.where(X["late_bid"].notna(), "late_bid", "no_late_bid")


def conformal_offset(scores: np.ndarray, level: float = 0.8) -> float:
    """Smallest shift leaving at most (1 - level) of held-out scores above it.

    Uses the finite-sample rank ceil((n + 1) * level), which is what makes the
    coverage guarantee hold on a small calibration set rather than only in the
    limit.
    """
    n = len(scores)
    if n == 0:
        return 0.0
    k = min(int(np.ceil((n + 1) * level)), n)
    return float(np.sort(scores)[k - 1])


class CalibratedModel:
    """p20 / p50 / p80 with band edges corrected per regime, in log space."""

    def __init__(self, models, adj, columns):
        self.models, self.adj, self.columns = models, adj, columns

    def raw(self, X: pd.DataFrame):
        X = X[self.columns]
        return (self.models[0.2].predict(X), self.models[0.5].predict(X),
                self.models[0.8].predict(X))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        lo, mid, hi = self.raw(X)
        reg = regime(X)
        a_lo = np.array([self.adj.get(r, (0.0, 0.0))[0] for r in reg])
        a_hi = np.array([self.adj.get(r, (0.0, 0.0))[1] for r in reg])
        # A negative offset narrows the band; keep it from crossing the median.
        lo = np.minimum(lo - a_lo, mid)
        hi = np.maximum(hi + a_hi, mid)
        return pd.DataFrame({"p20": np.exp(lo), "p50": np.exp(mid),
                             "p80": np.exp(hi)}, index=X.index)


def fit_calibrated(fit_df, cal_df, X_fit, X_cal):
    cols = list(X_fit.columns)
    y_fit = np.log(fit_df["final_price"].to_numpy(float))
    y_cal = np.log(cal_df["final_price"].to_numpy(float))
    models = {q: fit(X_fit, y_fit, q) for q in (0.2, 0.5, 0.8)}

    lo = models[0.2].predict(X_cal[cols])
    hi = models[0.8].predict(X_cal[cols])
    reg = regime(X_cal)

    adj, counts = {}, {}
    for r in np.unique(reg):
        m = reg == r
        # Positive score = the price landed outside the band on that side.
        adj[r] = (conformal_offset(lo[m] - y_cal[m]),
                  conformal_offset(y_cal[m] - hi[m]))
        counts[r] = int(m.sum())
    return CalibratedModel(models, adj, cols), counts


def _row(label, lo, mid, hi, y):
    if len(y) == 0:
        print(f"  {label:<28}   (no lots)")
        return
    below = (y < lo).mean() * 100
    above = (y > hi).mean() * 100
    width = np.median((hi - lo) / mid) * 100
    print(f"  {label:<28}{below:>6.0f}%{100 - below - above:>9.0f}%"
          f"{above:>9.0f}%{width:>9.0f}%{mape(mid, y):>9.1f}%   n={len(y)}")


def calibrate() -> None:
    df = load()
    f = featurise(df)
    tr, te = split(df)
    cut = int(len(tr) * (1 - CAL_FRACTION))
    fit_df, cal_df = tr.iloc[:cut], tr.iloc[cut:]

    X_fit, X_cal = add_ref_encoding(f.loc[fit_df.index], f.loc[cal_df.index],
                                    fit_df, cal_df)
    _, X_te = add_ref_encoding(f.loc[fit_df.index], f.loc[te.index], fit_df, te)

    model, counts = fit_calibrated(fit_df, cal_df, X_fit, X_cal)

    print(f"fit {len(fit_df):,}  calibration {len(cal_df):,}  test {len(te):,}  "
          f"(test from {te['close_time'].min().date()})\n")

    print("edge corrections, log space (+ widens, − narrows):")
    for r, (a_lo, a_hi) in model.adj.items():
        print(f"  {r:<12} lower {a_lo:+.3f}  upper {a_hi:+.3f}   "
              f"(calibrated on {counts[r]} lots)")

    y = te["final_price"].to_numpy(float)
    lo, mid, hi = (np.exp(v) for v in model.raw(X_te))
    cal = model.predict(X_te)
    clo, cmid, chi = (cal[c].to_numpy() for c in ("p20", "p50", "p80"))

    tm = target_mask(te)
    reg = regime(X_te)

    print(f"\n  {'':<28}{'below':>7}{'inside':>10}{'above':>9}"
          f"{'width':>10}{'p50 err':>10}")
    print(f"  {'target':<28}{'20%':>7}{'60%':>10}{'20%':>9}")
    _row("raw, all", lo, mid, hi, y)
    _row("calibrated, all", clo, cmid, chi, y)
    print()
    _row("raw, vintage Omega", lo[tm], mid[tm], hi[tm], y[tm])
    _row("calibrated, vintage Omega", clo[tm], cmid[tm], chi[tm], y[tm])
    print()
    for r in ("late_bid", "no_late_bid"):
        m = reg == r
        _row(f"calibrated, {r}", clo[m], cmid[m], chi[m], y[m])

    print("\nWidth is the price of honesty: a calibrated band is wider, because the\n"
          "narrow one was claiming a confidence the model did not have.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "audit"
    {"audit": audit, "baseline": baseline, "train": train,
     "calibrate": calibrate}.get(
        cmd, lambda: sys.exit(__doc__))()
