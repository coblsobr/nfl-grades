"""
Calibrate position weights from free data only: which first-half stats best predict how a player
performs in the second half of the season? (2023-2025, first half = Weeks 1-9, second half = Weeks 10-18)

Why this approach: a grade is supposed to measure skill. Skill carries over; luck doesn't.
So the stats that predict a player's future results deserve the weight, and noisy stats that
don't carry over get less. No PFF data is used anywhere in fitting.

Writes weights.json (read automatically by grades.py) and output/calibration_report.csv.
"""
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.argv = [sys.argv[0]]
import grades

SEASONS = [2023, 2024, 2025]
HALVES = [(1, 9), (10, 18)]

# What "doing the job" means at each position, measured on the second half (higher = better).
# These are the targets the first-half stats must predict.
OUTCOMES = {
    "QB":   "EPA per play",
    "RB":   "EPA per touch (runs + targets)",
    "WR":   "receiving EPA per route",
    "TE":   "receiving EPA per route",
    "EDGE": "65% pressure rate + 35% run-stop rate",
    "IDL":  "50% pressure rate + 50% run-stop rate",
    "LB":   "50% run-stop rate + 50% fewer coverage yards per snap",
    "CB":   "fewer coverage yards allowed per snap",
    "S":    "60% fewer coverage yards per snap + 40% run-stop rate",
}


def zs(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def subset(d, lo, hi, season):
    return grades.subset_weeks(d, lo, hi, season)


def _old_subset(d, lo, hi, season):
    out = dict(d)
    for k in ["pbp", "ftn", "snaps", "pfr_def", "pfr_pass", "pfr_rush", "pfr_rec"]:
        out[k] = d[k][(d[k]["week"] >= lo) & (d[k]["week"] <= hi)]
    # rebuild Next Gen Stats for just these weeks from the weekly rows (week 0 = full season would leak)
    ngs = {}
    for st, idc, wcol, cols in [
        ("rushing", "player_gsis_id", "rush_attempts", ["rush_yards_over_expected_per_att"]),
        ("passing", "team_abbr", "attempts", ["avg_time_to_throw"]),
        ("receiving", "player_gsis_id", "targets",
         ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"]),
    ]:
        n = pd.read_parquet(grades.CACHE / f"ngs_{st}_{season}.parquet")
        n = n[(n["week"] >= lo) & (n["week"] <= hi)].dropna(subset=cols)
        agg = {c: (n[c] * n[wcol]).groupby(n[idc]).sum() / n[wcol].groupby(n[idc]).sum() for c in cols}
        agg[wcol] = n[wcol].groupby(n[idc]).sum()
        if st == "receiving":
            agg["receptions"] = n["receptions"].groupby(n[idc]).sum()
        a = pd.DataFrame(agg).reset_index()
        if st == "passing":
            a = a.rename(columns={"attempts": "attempts"})
        ngs[st] = a
    out["ngs"] = ngs
    return out


def half_tables(d):
    off, de, tg = grades.team_context(d)
    routes = grades.est_routes(d, off)
    rec = grades.receiving_table(d, routes)
    res = {"QB": grades.grade_qb(d, tg), "RB": grades.grade_rb(d, rec, tg),
           "WR": grades.grade_receivers(rec, "WR", 18, tg, grades.ngs_receiving(d)),
           "TE": grades.grade_receivers(rec, "TE", 12, tg, grades.ngs_receiving(d))}
    res.update(grades.grade_defense(d, de, tg))
    # RB outcome needs EPA per touch from play-by-play
    p = d["pbp"]
    ru = p[(p["rush"] == 1) & p["rusher_id"].notna() & p["epa"].notna()]
    tg_ = p[(p["pass_attempt"] == 1) & (p["sack"] == 0) & (p["interception"] == 0) & p["receiver_player_id"].notna()]
    e = pd.concat([ru[["rusher_id", "epa"]].rename(columns={"rusher_id": "gsis_id"}),
                   tg_[["receiver_player_id", "epa"]].rename(columns={"receiver_player_id": "gsis_id"})])
    res["_rb_epa_touch"] = e.groupby("gsis_id")["epa"].mean()
    return res


def outcome(pos, t, extra):
    if pos == "QB":
        return t["epa"] / t["plays"]
    if pos == "RB":
        return t["gsis_id"].map(extra)
    if pos in ("WR", "TE"):
        return t["epa"] / t["routes"].clip(lower=1)
    pr = t["def_pressures"] / t["pass_snaps"].clip(lower=1)
    rs = t["run_stops"] / t["run_snaps"].clip(lower=1)
    cy = -t["def_yards_allowed"] / t["pass_snaps"].clip(lower=1)
    return {"EDGE": .65 * zs(pr) + .35 * zs(rs), "IDL": .5 * zs(pr) + .5 * zs(rs),
            "LB": .5 * zs(rs) + .5 * zs(cy), "CB": cy, "S": .6 * zs(cy) + .4 * zs(rs)}[pos]


def composite(X, w):
    return X @ w


def main():
    rows = {pos: [] for pos in OUTCOMES}
    for season in SEASONS:
        d = grades.load_all(season, 18)
        first = half_tables(subset(d, *HALVES[0], season))
        second = half_tables(subset(d, *HALVES[1], season))
        for pos in OUTCOMES:
            spec = grades.DEFAULT_WEIGHTS[pos]
            a = first[pos].drop_duplicates("gsis_id").set_index("gsis_id")
            b = second[pos].drop_duplicates("gsis_id").set_index("gsis_id")
            y = outcome(pos, b.reset_index(), second["_rb_epa_touch"]).set_axis(b.index)
            common = a.index.intersection(y.dropna().index)
            X = pd.DataFrame({m: sign * zs(a.loc[common, m]) for m, _, sign in spec}).fillna(0)
            X["season"] = season
            X["y"] = zs(y.loc[common]).values
            rows[pos].append(X)
        print(f"loaded {season}")

    weights, report, options = {}, [], {}
    for pos, parts in rows.items():
        df = pd.concat(parts)
        spec = grades.DEFAULT_WEIGHTS[pos]
        mets = [m for m, _, _ in spec]
        old = np.array([w for _, w, _ in spec])

        def fit(train):
            w, _ = nnls(train[mets].values, train["y"].values)
            w = np.where(w / max(w.sum(), 1e-9) < .03, 0, w)
            return w / w.sum() if w.sum() > 0 else old

        # leave-one-season-out: fit on two seasons, test on the third
        scores = {"old": [], "new": [], "blend": []}
        for s in SEASONS:
            tr, te = df[df["season"] != s], df[df["season"] == s]
            wn = fit(tr)
            for k, w in [("old", old), ("new", wn), ("blend", .5 * old + .5 * wn)]:
                scores[k].append(np.corrcoef(composite(te[mets].values, w), te["y"])[0, 1])
        wn = fit(df)
        cands = {"old": old, "new": wn, "blend": .5 * old + .5 * wn}
        r = {k: float(np.mean(v)) for k, v in scores.items()}
        options[pos] = {k: dict(r=r[k], w=[(m, round(float(w), 3), sg) for (m, _, sg), w in zip(spec, cands[k]) if w > 0.005])
                        for k in cands}
        best = max(r, key=r.get)
        final = cands[best]
        weights[pos] = [(m, round(float(w), 3), sg) for (m, _, sg), w in zip(spec, final) if w > 0.005]
        report.append(dict(pos=pos, n=len(df), outcome=OUTCOMES[pos],
                           r_old=round(np.mean(scores["old"]), 3), r_new=round(np.mean(scores["new"]), 3),
                           r_blend=round(np.mean(scores["blend"]), 3), chosen=best,
                           weights=", ".join(f"{m} {w:.0%}" for m, w, _ in weights[pos])))
        print(f"{pos:4s} n={len(df):4d}  predict-2nd-half r: old {np.mean(scores['old']):.3f} | "
              f"fitted {np.mean(scores['new']):.3f} | blend {np.mean(scores['blend']):.3f}  -> {best}")
        print("      ", weights[pos])

    # OL: no clean second-half outcome in free data, so weights follow the 2025 ridge validation instead:
    # matchup pass-pro only agreed with the per-play model for tackles, so interior linemen lean on
    # the team unit, run results and penalties.
    weights["OT"] = grades.DEFAULT_WEIGHTS["OT"]
    weights["OG"] = [("pressures_oe_per_100", .10, -1), ("run_succ_oe", .25, 1), ("run_epa_oe", .15, 1),
                     ("penalty_rate", .20, -1), ("team_unit", .30, 1)]
    weights["C"] = [("pressures_oe_per_100", .10, -1), ("run_succ_oe", .25, 1), ("run_epa_oe", .15, 1),
                    ("penalty_rate", .15, -1), ("team_unit", .35, 1)]
    grades.WEIGHTS_FILE.write_text(json.dumps(weights, indent=1))
    grades.WEIGHTS_FILE.with_name("weights_options.json").write_text(json.dumps(options, indent=1))
    pd.DataFrame(report).to_csv("output/calibration_report.csv", index=False)
    print("wrote", grades.WEIGHTS_FILE)


if __name__ == "__main__":
    main()
