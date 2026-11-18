"""
Free NFL Player Grades — v1
Builds 0-100 grades for every position from free nflverse data.

Run:   pip install nflreadpy pandas pyarrow
       python grades.py --season 2026 --thru-week 6
Out:   output/grades_<POS>.csv  +  output/RESULTS.md (top 10 per position)

Grade scale: 70 = average qualified starter, every 10 points = 1 standard
deviation. 90+ is roughly top 2-3% of qualified players.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

CACHE = Path("cache")
OUT = Path("output")


# ----------------------------------------------------------------- helpers
def cached(name, fn):
    """Load a dataset once, then reuse the local parquet copy."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = fn().to_pandas()
    df.to_parquet(f)
    return df


TRUST_GAMES = 8
OL_TEAM_SHARE = 0.35
COV_K = 25            # coverage stats shrink toward average until ~this many targets (tested 60 and 80: ranks barely moved)
COV_SNAP_K = 120     # same for per-coverage-snap rates
DB_TEAM_SHARE = 1.0    # 1.0 = off: tested 0.5 for corners/safeties, it did not help (2025 check)


def shrink(total, n, k, prior):
    """Regress a rate toward the position average.
    total = summed value, n = sample size, k = how many 'average' plays to blend in."""
    return (total + k * prior) / (n + k)


def z(s, winsor=True, ref=None):
    """z-score against the reference group (regular players at the position).
    Inputs are winsorized (top/bottom 2.5% of the reference group) so one outlier can't dominate."""
    s = s.astype(float)
    r = s if ref is None else s[ref]
    if winsor:
        s = s.clip(r.quantile(0.025), r.quantile(0.975))
        r = s if ref is None else s[ref]
    sd = r.std(ddof=0)
    return ((s - r.mean()) / sd if sd > 0 else s * 0).fillna(0)


def grade(df, spec):
    """spec = [(column, weight, +1 good-high / -1 good-low), ...]
    Everyone who played is graded. The scale (70 = average, 10 = 1 SD) is set by the regulars
    (`qualified`), so part-timers don't distort it; their small samples are already pulled toward
    average by shrink()."""
    ref = df["qualified"].astype(bool) if "qualified" in df else None
    if ref is not None and ref.sum() < 3:
        ref = None
    score = sum(w * sign * z(df[c], ref=ref) for c, w, sign in spec)
    zs_ = z(score, winsor=False, ref=ref)
    if "pt_share" in df:
        # part-timers: pull toward the level of a typical backup (not toward an average starter),
        # in proportion to how much less than a regular they played
        share = df["pt_share"].clip(0, 1)
        semi = (share >= 0.3) & (share < 1)
        backup = float(zs_[semi].mean()) if semi.sum() >= 5 else -0.5
        backup = min(backup, -0.25)
        zs_ = backup + (zs_ - backup) * share
    df["grade"] = (70 + 10 * zs_).clip(30, 99.9).round(1)
    return df.sort_values("grade", ascending=False).reset_index(drop=True)


def passer_rating(cmp, att, yds, td, ints):
    att = np.maximum(att, 1)
    a = np.clip((cmp / att - 0.3) * 5, 0, 2.375)
    b = np.clip((yds / att - 3) * 0.25, 0, 2.375)
    c = np.clip(td / att * 20, 0, 2.375)
    d = np.clip(2.375 - ints / att * 25, 0, 2.375)
    return (a + b + c + d) / 6 * 100


# ----------------------------------------------------------------- weights
# (metric, weight, +1 higher is better / -1 lower is better). weights.json, if present, overrides these;
# calibrate_weights.py writes it from what predicts future performance (see PLAN.md).
DEFAULT_WEIGHTS = {
    "QB": [("epa_play", .35, 1), ("value_over_avg", .25, 1), ("cpoe", .15, 1), ("success", .10, 1),
           ("int_worthy_rate", .10, -1), ("fault_sack_rate", .05, -1)],
    "RB": [("ryoe_att", .25, 1), ("yds_after_contact", .20, 1), ("success", .15, 1), ("epa_per_carry", .10, 1),
           ("broken_tkl_rate", .10, 1), ("rec_yds_per_route", .15, 1), ("fumble_rate", .05, -1)],
    "WR": [("yds_per_route", .25, 1), ("tgt_per_route", .20, 1), ("route_share", .10, 1), ("epa_per_target", .10, 1),
           ("catch_over_exp", .10, 1), ("separation_over_exp", .10, 1), ("yac_over_exp", .05, 1), ("drop_rate", .10, -1)],
    "OT": [("pressures_oe_per_100", .40, -1), ("run_succ_oe", .15, 1), ("run_epa_oe", .10, 1),
           ("penalty_rate", .15, -1), ("team_unit", .20, 1)],
    "OG": [("pressures_oe_per_100", .25, -1), ("run_succ_oe", .25, 1), ("run_epa_oe", .15, 1),
           ("penalty_rate", .15, -1), ("team_unit", .20, 1)],
    "C":  [("pressures_oe_per_100", .20, -1), ("run_succ_oe", .25, 1), ("run_epa_oe", .15, 1),
           ("penalty_rate", .15, -1), ("team_unit", .25, 1)],
    "EDGE": [("pressure_rate", .45, 1), ("sack_rate", .15, 1), ("run_stop_rate", .30, 1), ("missed_tkl_rate", .10, -1)],
    "IDL":  [("pressure_rate", .35, 1), ("sack_rate", .10, 1), ("run_stop_rate", .40, 1), ("missed_tkl_rate", .15, -1)],
    "LB":   [("run_stop_rate", .35, 1), ("yds_per_cov_snap", .25, -1), ("rating_allowed", .10, -1),
             ("plays_per_cov_snap", .10, 1), ("missed_tkl_rate", .10, -1), ("pressure_rate", .10, 1)],
    "CB":   [("tgt_per_cov_snap", .15, -1), ("yds_per_target", .20, -1), ("cmp_over_exp_allowed", .20, -1),
             ("yac_per_cmp_allowed", .10, -1), ("playmaking", .15, 1), ("rating_allowed", .10, -1),
             ("missed_tkl_rate", .05, -1), ("run_stop_rate", .05, 1)],
    "S":    [("tgt_per_cov_snap", .10, -1), ("yds_per_target", .15, -1), ("cmp_over_exp_allowed", .10, -1),
             ("playmaking", .15, 1), ("rating_allowed", .10, -1), ("run_stop_rate", .20, 1), ("missed_tkl_rate", .20, -1)],
}
DEFAULT_WEIGHTS["TE"] = DEFAULT_WEIGHTS["WR"]
WEIGHTS_FILE = Path(__file__).with_name("weights.json")
USE_DEFAULTS = False          # calibration flips this on to compare old vs. new


def W(pos):
    if WEIGHTS_FILE.exists() and not USE_DEFAULTS:
        w = json.loads(WEIGHTS_FILE.read_text()).get(pos)
        if w:
            return [tuple(x) for x in w]
    return DEFAULT_WEIGHTS[pos]


# ----------------------------------------------------------------- data
def load_all(season, thru, playoffs=False):
    """thru = last regular-season week. playoffs=True adds that season's postseason games (weeks 19-22)."""
    d = {}
    last = 22 if playoffs else thru
    reg = lambda df: df[(df["week"] <= last) & (df["week"] >= 1)]
    d["pbp"] = reg(cached(f"pbp_{season}", lambda: nfl.load_pbp([season])))
    if not playoffs:
        d["pbp"] = d["pbp"][d["pbp"]["season_type"] == "REG"]
    d["ftn"] = reg(cached(f"ftn_{season}", lambda: nfl.load_ftn_charting([season])))
    d["snaps"] = reg(cached(f"snaps_{season}", lambda: nfl.load_snap_counts([season])))
    for st in ["def", "pass", "rush", "rec"]:
        d[f"pfr_{st}"] = reg(cached(f"pfr_{st}_{season}",
            lambda st=st: nfl.load_pfr_advstats([season], stat_type=st, summary_level="week")))
    ngs = {}
    for st in ["rushing", "passing", "receiving"]:
        n = cached(f"ngs_{st}_{season}", lambda st=st: nfl.load_nextgen_stats([season], stat_type=st))
        ngs[st] = n[n["week"] == 0]                      # week 0 = season-to-date
    d["ngs"] = ngs
    try:
        d["depth"] = normalize_depth(cached(f"depth_{season}", lambda: nfl.load_depth_charts([season])))
    except Exception:
        d["depth"] = None
    ro = cached(f"rosters_{season}", lambda: nfl.load_rosters([season]))
    ro = ro.sort_values("week").drop_duplicates("gsis_id", keep="last")
    # rosters leave pfr_id blank for some players (all OL in 2026) - fill from the master players table
    ids = cached("players_all", lambda: nfl.load_players())[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    ro["pfr_id"] = ro["pfr_id"].fillna(ro["gsis_id"].map(ids.set_index("gsis_id")["pfr_id"]))
    d["roster"] = ro[["gsis_id", "pfr_id", "full_name", "team", "position", "depth_chart_position", "weight"]]
    # league make-% by distance for kickers, from this season's kicks only
    d["fg_hist"] = d["pbp"][d["pbp"]["field_goal_attempt"] == 1][["kick_distance", "field_goal_result"]].dropna()
    # league completion % by target depth (used to judge coverage: short catches are expected, deep ones aren't)
    pa = d["pbp"][(d["pbp"]["pass_attempt"] == 1) & (d["pbp"]["sack"] == 0) & d["pbp"]["air_yards"].notna()]
    ay = pa["air_yards"].clip(-5, 50).values
    d["cp_w"] = fit_logistic(np.c_[np.ones_like(ay), ay, ay ** 2], pa["complete_pass"].values.astype(float))
    if playoffs:                                          # NGS season rows are regular season only: rebuild from weekly rows
        d = subset_weeks(d, 1, 22, season)
    return d


def fit_logistic(X, y, iters=25):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        w += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(w)), X.T @ (y - p))
    return w


def cp_at_depth(w, adot):
    a = np.clip(adot, -5, 50)
    return 1 / (1 + np.exp(-(w[0] + w[1] * a + w[2] * a ** 2)))


def normalize_depth(dc):
    """Depth charts before 2025 use a different layout; convert to the 2025+ columns the OL code expects."""
    if "pos_abb" in dc.columns or "depth_position" not in dc.columns:
        return dc
    dc = dc.rename(columns={"club_code": "team", "depth_position": "pos_abb"}).copy()
    dc["pos_rank"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    d34 = dc[dc["formation"].eq("Defense") & dc["pos_abb"].isin(["NT", "OLB", "LOLB", "ROLB"])]
    is34 = set(zip(d34["team"], d34["week"]))
    dc["pos_grp"] = np.where(dc["formation"].ne("Defense"), dc["formation"],
                             np.where([(t, w) in is34 for t, w in zip(dc["team"], dc["week"])], "Base 3-4 D", "Base 4-3 D"))
    return dc


def subset_weeks(d, lo, hi, season):
    """Copy of loaded data restricted to weeks lo..hi (NGS rebuilt from weekly rows so nothing leaks)."""
    out = dict(d)
    for k in ["pbp", "ftn", "snaps", "pfr_def", "pfr_pass", "pfr_rush", "pfr_rec"]:
        out[k] = d[k][(d[k]["week"] >= lo) & (d[k]["week"] <= hi)]
    ngs = {}
    for st, idc, wcol, cols in [
        ("rushing", "player_gsis_id", "rush_attempts", ["rush_yards_over_expected_per_att"]),
        ("passing", "team_abbr", "attempts", ["avg_time_to_throw"]),
        ("receiving", "player_gsis_id", "targets",
         ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"])]:
        n = pd.read_parquet(CACHE / f"ngs_{st}_{season}.parquet")
        n = n[(n["week"] >= lo) & (n["week"] <= hi)].dropna(subset=cols)
        agg = {c: (n[c] * n[wcol]).groupby(n[idc]).sum() / n[wcol].groupby(n[idc]).sum() for c in cols}
        agg[wcol] = n[wcol].groupby(n[idc]).sum()
        if st == "receiving":
            agg["receptions"] = n["receptions"].groupby(n[idc]).sum()
        ngs[st] = pd.DataFrame(agg).reset_index()
    out["ngs"] = ngs
    return out


def team_context(d):
    """Per team-game counts used to estimate routes / coverage / pass-rush snaps."""
    p = d["pbp"]
    plays = p[(p["pass"] == 1) | (p["rush"] == 1)]
    off = plays.groupby(["game_id", "posteam"]).agg(
        dropbacks=("pass", "sum"), runs=("rush", "sum")).reset_index()
    off = off.rename(columns={"posteam": "team"})
    off["opp_pass_share"] = off["dropbacks"] / (off["dropbacks"] + off["runs"])
    defn = off.copy()
    # defense faces the opponent's offense in the same game
    games = p[["game_id", "home_team", "away_team"]].drop_duplicates()
    m = off.merge(games, on="game_id")
    m["def_team"] = np.where(m["team"] == m["home_team"], m["away_team"], m["home_team"])
    defn = m[["game_id", "def_team", "opp_pass_share"]].rename(columns={"def_team": "team"})
    team_games = off.groupby("team")["game_id"].nunique()
    return off[["game_id", "team", "dropbacks"]], defn, team_games


# ----------------------------------------------------------------- offense
def grade_qb(d, team_games):
    p, f = d["pbp"], d["ftn"]
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f["play_id"].astype(float)
    q = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["qb_epa"].notna()
          & (p["qb_kneel"] == 0) & (p["qb_spike"] == 0) & p["id"].notna()]
    q = q.merge(f[["game_id", "play_id", "is_interception_worthy", "is_qb_fault_sack"]],
                on=["game_id", "play_id"], how="left")
    qbs = d["roster"].query("position == 'QB'")["gsis_id"]
    q = q[q["id"].isin(qbs)]
    g = q.groupby("id").agg(
        plays=("qb_epa", "size"), epa=("qb_epa", "sum"), succ=("success", "sum"),
        dropbacks=("pass", "sum"), cpoe_sum=("cpoe", "sum"), cpoe_n=("cpoe", "count"),
        int_worthy=("is_interception_worthy", "sum"), fault_sacks=("is_qb_fault_sack", "sum"),
        team=("posteam", "last")).reset_index()
    g["pt_share"] = (g["plays"] / (20 * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    lg_epa, lg_s = g["epa"].sum() / g["plays"].sum(), g["succ"].sum() / g["plays"].sum()
    lg_iw, lg_fs = g["int_worthy"].sum() / g["dropbacks"].sum(), g["fault_sacks"].sum() / g["dropbacks"].sum()
    g["epa_play"] = shrink(g["epa"], g["plays"], 60, lg_epa)
    g["cpoe"] = shrink(g["cpoe_sum"], g["cpoe_n"], 60, 0)
    g["success"] = shrink(g["succ"], g["plays"], 60, lg_s)
    g["int_worthy_rate"] = shrink(g["int_worthy"], g["dropbacks"], 100, lg_iw)
    g["fault_sack_rate"] = shrink(g["fault_sacks"], g["dropbacks"], 100, lg_fs)
    # value over average = how much EPA above an average QB he produced on his actual workload
    g["value_over_avg"] = (g["epa_play"] - lg_epa) * g["plays"]
    g = g.merge(d["roster"][["gsis_id", "full_name"]], left_on="id", right_on="gsis_id")
    g["workload"] = g["plays"] / (20 * g["team"].map(team_games))
    return grade(g, W("QB"))


def est_routes(d, off_ctx):
    """Routes ≈ offensive snap % × team dropbacks, game by game."""
    s = d["snaps"].merge(off_ctx, on=["game_id", "team"], how="left")
    s["routes"] = s["offense_pct"] * s["dropbacks"]
    return s.groupby("pfr_player_id")[["routes", "dropbacks"]].sum()


def receiving_table(d, routes):
    p, f = d["pbp"], d["ftn"]
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f["play_id"].astype(float)
    t = p[(p["pass_attempt"] == 1) & (p["sack"] == 0) & p["receiver_player_id"].notna()]
    t = t.merge(f[["game_id", "play_id", "is_drop", "is_catchable_ball"]], on=["game_id", "play_id"], how="left")
    t["rec_yds"] = np.where(t["complete_pass"] == 1, t["yards_gained"], 0)
    # interceptions are the QB's play — leave them out of the receiver's EPA
    t["epa_rec"] = np.where(t["interception"] == 1, np.nan, t["epa"])
    g = t.groupby("receiver_player_id").agg(
        targets=("epa", "size"), epa=("epa_rec", "sum"), tgt_no_int=("epa_rec", "count"), yds=("rec_yds", "sum"),
        cpoe_sum=("cpoe", "sum"), cpoe_n=("cpoe", "count"),
        drops=("is_drop", "sum"), catchable=("is_catchable_ball", "sum"),
        team=("posteam", "last")).reset_index().rename(columns={"receiver_player_id": "gsis_id"})
    g = g.merge(d["roster"], on="gsis_id", suffixes=("", "_ro"))
    g["routes"] = g["pfr_id"].map(routes["routes"]).fillna(0)
    # share of his team's dropbacks he was on the field for, in games he played (full-time role vs. part-timer)
    g["route_share"] = g["routes"] / g["pfr_id"].map(routes["dropbacks"]).clip(lower=1)
    return g


def ngs_receiving(d):
    """Next Gen Stats tracking: how open a receiver gets, and YAC vs. what the tracking model expected.
    Separation depends on route depth and how far off the DB plays, so grade it versus expectation."""
    n = d["ngs"]["receiving"].copy()
    X = np.c_[np.ones(len(n)), n["avg_intended_air_yards"], n["avg_cushion"]]
    W = np.sqrt(n["targets"].values)
    beta = np.linalg.lstsq(X * W[:, None], n["avg_separation"].values * W, rcond=None)[0]
    n["sep_over_exp"] = n["avg_separation"] - X @ beta
    return n.set_index("player_gsis_id")[["sep_over_exp", "avg_yac_above_expectation", "targets", "receptions"]]


def grade_receivers(g, pos, min_routes_pg, team_games, ngs):
    g = g[g["position"] == pos].copy()
    g = g[g["routes"] > 0].copy()
    g["pt_share"] = (g["routes"] / (min_routes_pg * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    # tracking stats (NGS lists receivers with 20+ targets; anyone below gets league average, i.e. 0)
    for col, ncol, k in [("separation_over_exp", "sep_over_exp", 40),
                         ("yac_over_exp", "avg_yac_above_expectation", 25)]:
        n_obs = g["gsis_id"].map(ngs["targets" if ncol == "sep_over_exp" else "receptions"]).fillna(0)
        val = g["gsis_id"].map(ngs[ncol]).fillna(0)
        g[col] = shrink(val * n_obs, n_obs, k, 0)
    lg_ypr = g["yds"].sum() / g["routes"].sum()
    lg_tpr = g["targets"].sum() / g["routes"].sum()
    lg_ept = g["epa"].sum() / g["tgt_no_int"].sum()
    lg_drop = g["drops"].sum() / g["catchable"].sum()
    # k = how much evidence before we trust a player's rate. Per-target stats are noisy, so they get big k.
    g["yds_per_route"] = shrink(g["yds"], g["routes"], 150, lg_ypr)
    g["tgt_per_route"] = shrink(g["targets"], g["routes"], 80, lg_tpr)
    g["epa_per_target"] = shrink(g["epa"], g["tgt_no_int"], 60, lg_ept)
    g["catch_over_exp"] = shrink(g["cpoe_sum"], g["cpoe_n"], 60, 0)
    g["drop_rate"] = shrink(g["drops"], g["catchable"], 60, lg_drop)
    g["workload"] = g["routes"] / (min_routes_pg * g["team"].map(team_games))
    g["rec_yds_pg"] = g["yds"] / g["team"].map(team_games)
    return grade(g, W(pos))


def grade_rb(d, rec, team_games):
    p = d["pbp"]
    r = p[(p["rush"] == 1) & p["rusher_id"].notna() & p["epa"].notna() & (p["qb_kneel"] == 0)]
    r = r.assign(expl=(r["yards_gained"] >= 10).astype(int))
    g = r.groupby("rusher_id").agg(carries=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"),
                                   expl=("expl", "sum"), fumbles=("fumble_lost", "sum"),
                                   team=("posteam", "last")).reset_index()
    g = g.rename(columns={"rusher_id": "gsis_id"}).merge(d["roster"], on="gsis_id", suffixes=("", "_ro"))
    g = g[g["position"] == "RB"]
    g = g.copy()
    g["pt_share"] = (g["carries"] / (8 * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    pr = d["pfr_rush"].groupby("pfr_player_id")[["carries", "rushing_yards_after_contact",
                                                 "rushing_broken_tackles", "receiving_broken_tackles"]].sum()
    g = g.join(pr.add_prefix("pfr_"), on="pfr_id")
    ngs = d["ngs"]["rushing"].set_index("player_gsis_id")
    g["ryoe_att"] = g["gsis_id"].map(ngs["rush_yards_over_expected_per_att"])
    g["ryoe_att"] = g["ryoe_att"].fillna(g["ryoe_att"].median())
    rec_ = rec.set_index("gsis_id")
    g["rec_yds"] = g["gsis_id"].map(rec_["yds"]).fillna(0)
    g["routes"] = g["gsis_id"].map(rec_["routes"]).fillna(0)
    g["receptions_proxy"] = g["gsis_id"].map(rec_["targets"]).fillna(0)
    lg = lambda a, b: g[a].sum() / g[b].sum()
    g["epa_per_carry"] = shrink(g["epa"], g["carries"], 40, lg("epa", "carries"))
    g["success"] = shrink(g["succ"], g["carries"], 40, lg("succ", "carries"))
    g["yds_after_contact"] = shrink(g["pfr_rushing_yards_after_contact"].fillna(0), g["pfr_carries"].fillna(0), 40,
                                    lg("pfr_rushing_yards_after_contact", "pfr_carries"))
    g["touches"] = g["carries"] + g["receptions_proxy"]
    g["bt"] = g["pfr_rushing_broken_tackles"].fillna(0) + g["pfr_receiving_broken_tackles"].fillna(0)
    g["broken_tkl_rate"] = shrink(g["bt"], g["touches"], 40, lg("bt", "touches"))
    g["rec_yds_per_route"] = shrink(g["rec_yds"], g["routes"], 40, lg("rec_yds", "routes"))
    g["fumble_rate"] = shrink(g["fumbles"], g["touches"], 100, lg("fumbles", "touches"))
    g["explosive_rate"] = shrink(g["expl"], g["carries"], 60, lg("expl", "carries"))
    g["workload"] = g["carries"] / (8 * g["team"].map(team_games))
    g["touches_pg"] = g["touches"] / g["team"].map(team_games)
    return grade(g, W("RB"))


def grade_ol_units(d):
    """Team offensive line units (individual OL needs film charting — see PLAN.md)."""
    p, f = d["pbp"], d["ftn"]
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f["play_id"] = f["play_id"].astype(float)
    db = p[p["pass"] == 1].merge(f[["game_id", "play_id", "is_qb_fault_sack"]], on=["game_id", "play_id"], how="left")
    t = db.groupby("posteam").agg(dropbacks=("pass", "size"), sacks=("sack", "sum"),
                                  qb_fault=("is_qb_fault_sack", "sum")).rename_axis("team")
    t["pressures"] = d["pfr_pass"].groupby("team")["times_pressured"].sum()
    t["pressure_rate"] = t["pressures"] / t["dropbacks"]
    ngs = d["ngs"]["passing"]
    ngs = ngs.assign(w=ngs["avg_time_to_throw"] * ngs["attempts"]).groupby("team_abbr")[["w", "attempts"]].sum()
    t["time_to_throw"] = ngs["w"] / ngs["attempts"]
    t["time_to_throw"] = t["time_to_throw"].fillna(t["time_to_throw"].mean())
    # QBs who hold the ball create pressure; grade the line on pressure ABOVE what that hold time predicts
    b = np.polyfit(t["time_to_throw"], t["pressure_rate"], 1)
    t["pressure_over_exp"] = t["pressure_rate"] - np.polyval(b, t["time_to_throw"])
    t["ol_sack_rate"] = (t["sacks"] - t["qb_fault"]) / t["dropbacks"]
    qbs = set(d["roster"].query("position == 'QB'")["gsis_id"])
    runs = p[(p["rush"] == 1) & ~p["rusher_id"].isin(qbs) & p["epa"].notna()]
    t["stuff_rate"] = runs.assign(st=runs["yards_gained"] <= 0).groupby("posteam")["st"].mean()
    t["run_success"] = runs.groupby("posteam")["success"].mean()
    qb_pfr = set(d["roster"].query("position == 'QB'")["pfr_id"])
    pr = d["pfr_rush"][~d["pfr_rush"]["pfr_player_id"].isin(qb_pfr)]
    pr = pr.groupby("team")[["rushing_yards_before_contact", "carries"]].sum()
    t["ybc_per_carry"] = pr["rushing_yards_before_contact"] / pr["carries"]
    t = t.reset_index().rename(columns={"team": "full_name"})
    t["team"] = t["full_name"]
    return grade(t, [("pressure_over_exp", .35, -1), ("ol_sack_rate", .20, -1), ("ybc_per_carry", .25, 1),
                     ("stuff_rate", .10, -1), ("run_success", .10, 1)])


# ----------------------------------------------------------------- individual OL
OL_SLOTS = ["LT", "LG", "C", "RG", "RT"]
ALL5 = {k: 0.2 for k in OL_SLOTS}
# which blockers own a run, by where it went (nflverse run_location / run_gap)
GAP_CREDIT = {("left", "end"): {"LT": 1.0}, ("left", "tackle"): {"LT": .5, "LG": .5},
              ("left", "guard"): {"LG": .6, "C": .4}, ("middle", "mid"): {"C": .5, "LG": .25, "RG": .25},
              ("right", "guard"): {"RG": .6, "C": .4}, ("right", "tackle"): {"RT": .5, "RG": .5},
              ("right", "end"): {"RT": 1.0}}
# which blockers face a pass rusher, by his depth-chart spot. Defense's RIGHT side faces the offense's LEFT.
_GENERIC = {"DE": {"LT": .5, "RT": .5}, "EDGE": {"LT": .5, "RT": .5}, "RUSH": {"LT": .5, "RT": .5},
            "OLB": {"LT": .5, "RT": .5}, "DT": {"LG": .4, "C": .2, "RG": .4}, "DL": {"LG": .4, "C": .2, "RG": .4}}
RUSH_MATCHUP = {
    "4-3": {**_GENERIC, "RDE": {"LT": 1.0}, "LDE": {"RT": 1.0}, "RDT": {"LG": .7, "C": .3}, "LDT": {"RG": .7, "C": .3},
            "NT": {"C": .6, "LG": .2, "RG": .2}},
    "3-4": {**_GENERIC, "RDE": {"LT": .5, "LG": .5}, "LDE": {"RT": .5, "RG": .5}, "NT": {"C": .6, "LG": .2, "RG": .2},
            "WLB": {"LT": .5, "RT": .5}, "SLB": {"LT": .5, "RT": .5}},
}
OL_PENALTIES = ["Offensive Holding", "False Start", "Illegal Use of Hands", "Chop Block", "Clipping"]


def primary_slot(dc, slots):
    """Each player's usual depth-chart spot among `slots` (his best rank, most often listed)."""
    x = dc[dc["pos_abb"].isin(slots) & dc["gsis_id"].notna()]
    x = x[x["pos_rank"] == x.groupby("gsis_id")["pos_rank"].transform("min")]
    n = x.groupby(["gsis_id", "pos_abb", "pos_grp"]).size().reset_index(name="n")
    n = n.sort_values("n").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    return n["pos_abb"], n["pos_grp"]


def grade_ol_individual(d, off_ctx, def_ctx, team_games, units):
    dc = d["depth"]
    if dc is None or "pos_abb" not in dc.columns:
        return {}
    ol_slot, _ = primary_slot(dc, OL_SLOTS)
    def_slot, def_grp = primary_slot(dc, ["LDE", "RDE", "LDT", "RDT", "NT", "WLB", "SLB", "MLB", "LILB", "RILB",
                                          "LCB", "RCB", "SS", "FS", "NB", "DE", "EDGE", "RUSH", "OLB", "DT", "DL"])
    ro = d["roster"][["pfr_id", "gsis_id", "full_name"]].dropna(subset=["pfr_id"])

    # --- who played where, game by game (snap % split if two players shared a spot)
    sn = d["snaps"].merge(ro, left_on="pfr_player_id", right_on="pfr_id")
    ol = sn[(sn["offense_snaps"] > 0) & sn["gsis_id"].isin(ol_slot.index)].copy()
    ol["slot"] = ol["gsis_id"].map(ol_slot)
    ol["w"] = ol["offense_pct"] / ol.groupby(["game_id", "team", "slot"])["offense_pct"].transform("sum").clip(lower=1)
    ol = ol.merge(off_ctx, on=["game_id", "team"], how="left")
    ol["pass_snaps"] = ol["offense_pct"] * ol["dropbacks"]
    slot_w = ol[["game_id", "team", "slot", "gsis_id", "w"]]

    def credit(events, value_cols):
        """events: game_id, team (offense), slot, share, + value columns -> summed per blocker."""
        m = events.merge(slot_w, on=["game_id", "team", "slot"])
        m["wt"] = m["share"] * m["w"]
        out = {"weight": m.groupby("gsis_id")["wt"].sum()}
        for c in value_cols:
            out[c] = (m[c] * m["wt"]).groupby(m["gsis_id"]).sum()
        return pd.DataFrame(out)

    # --- run blocking: results of runs through his gaps vs. league average for that gap
    p = d["pbp"]
    qbs = set(d["roster"].query("position == 'QB'")["gsis_id"])
    r = p[(p["rush"] == 1) & ~p["rusher_id"].isin(qbs) & p["run_location"].notna() & p["epa"].notna()].copy()
    r["gap"] = np.where(r["run_location"] == "middle", "mid", r["run_gap"])
    r = r[r["gap"].notna()]
    r["succ_oe"] = r["success"] - r.groupby(["run_location", "gap"])["success"].transform("mean")
    r["epa_oe"] = r["epa"] - r.groupby(["run_location", "gap"])["epa"].transform("mean")
    ev = [dict(game_id=g, team=t, slot=sl, share=sh, succ_oe=so, epa_oe=eo)
          for g, t, loc, gp, so, eo in r[["game_id", "posteam", "run_location", "gap", "succ_oe", "epa_oe"]].itertuples(index=False)
          for sl, sh in GAP_CREDIT.get((loc, gp), {}).items()]
    run = credit(pd.DataFrame(ev), ["succ_oe", "epa_oe"])

    # --- pass pro: pressures by the rushers lined up across from him, vs. what those rushers usually get
    dsn = sn[sn["defense_snaps"] > 0].merge(def_ctx, on=["game_id", "team"], how="left")
    dsn["rush_snaps"] = dsn["defense_snaps"] * dsn["opp_pass_share"]
    pr = d["pfr_def"][["game_id", "pfr_player_id", "def_pressures"]]
    dsn = dsn.merge(pr, on=["game_id", "pfr_player_id"], how="left").fillna({"def_pressures": 0})
    tot = dsn.groupby("gsis_id")[["def_pressures", "rush_snaps"]].sum()
    lg_rate = tot["def_pressures"].sum() / tot["rush_snaps"].sum()
    rate = shrink(tot["def_pressures"], tot["rush_snaps"], 100, lg_rate)        # each rusher's usual rate
    dsn["exp_press"] = dsn["gsis_id"].map(rate) * dsn["rush_snaps"]
    dsn["press_oe"] = dsn["def_pressures"] - dsn["exp_press"]
    games = p[["game_id", "home_team", "away_team"]].drop_duplicates().set_index("game_id")
    dsn["offense"] = np.where(dsn["team"].values == dsn["game_id"].map(games["home_team"]).values,
                              dsn["game_id"].map(games["away_team"]).values, dsn["game_id"].map(games["home_team"]).values)
    dsn["scheme"] = np.where(dsn["gsis_id"].map(def_grp).fillna("").str.contains("3-4"), "3-4", "4-3")
    dsn["dslot"] = dsn["gsis_id"].map(def_slot)
    ev = [dict(game_id=g, team=o, slot=sl, share=sh, press_oe=v)
          for g, o, sc, ds, v in dsn[["game_id", "offense", "scheme", "dslot", "press_oe"]].itertuples(index=False)
          if v == v for sl, sh in RUSH_MATCHUP[sc].get(ds, ALL5).items()]
    pas = credit(pd.DataFrame(ev), ["press_oe"])

    # --- penalties charged to him
    pen = p[(p["penalty"] == 1) & p["penalty_type"].isin(OL_PENALTIES) & (p["penalty_team"] == p["posteam"])]
    pens = pen["penalty_player_id"].value_counts()

    g = ol.groupby("gsis_id").agg(snaps=("offense_snaps", "sum"), pass_snaps=("pass_snaps", "sum"),
                                  team=("team", "last"), slot=("slot", "first")).reset_index()
    g["slot"] = g["gsis_id"].map(ol_slot)
    g = g.merge(ro[["gsis_id", "full_name"]].drop_duplicates("gsis_id"), on="gsis_id")
    g = g[g["snaps"] > 0].copy()
    g["pt_share"] = (g["snaps"] / (30 * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    g = g.join(run, on="gsis_id").join(pas.rename(columns={"weight": "pass_weight"}), on="gsis_id")
    g = g.fillna({"weight": 0, "succ_oe": 0, "epa_oe": 0, "pass_weight": 0, "press_oe": 0})
    g["run_succ_oe"] = shrink(g["succ_oe"], g["weight"], 40, 0)
    g["run_epa_oe"] = shrink(g["epa_oe"], g["weight"], 40, 0)
    g["pressures_oe_per_100"] = 100 * shrink(g["press_oe"], g["pass_snaps"], 150, 0)
    g["penalties"] = g["gsis_id"].map(pens).fillna(0)
    lg_pen = g["penalties"].sum() / g["snaps"].sum()
    g["penalty_rate"] = shrink(g["penalties"], g["snaps"], 200, lg_pen)
    g["team_unit"] = g["team"].map(units.set_index("team")["grade"])

    groups = {"OT": ["LT", "RT"], "OG": ["LG", "RG"], "C": ["C"]}
    g["workload"] = g["snaps"] / (30 * g["team"].map(team_games))
    # A lineman's numbers are mostly his line's numbers. Judge each man against his own linemates,
    # and let the line's overall level count for only a slice (TEAM_SHARE) of his individual stats.
    for m in ["pressures_oe_per_100", "run_succ_oe", "run_epa_oe"]:
        if m in g:
            wts = g["snaps"].clip(lower=1)
            team_mean = (g[m] * wts).groupby(g["team"]).sum() / wts.groupby(g["team"]).sum()
            lg = (g[m] * wts).sum() / wts.sum()
            tm = g["team"].map(team_mean)
            g[m] = (g[m] - tm) + OL_TEAM_SHARE * (tm - lg)
    return {pos: grade(g[g["slot"].isin(sl)].copy(), W(pos)) for pos, sl in groups.items()}


# ----------------------------------------------------------------- defense
def defense_table(d, def_ctx):
    s = d["snaps"].merge(def_ctx, on=["game_id", "team"], how="left")
    s["pass_snaps"] = s["defense_snaps"] * s["opp_pass_share"]
    s["run_snaps"] = s["defense_snaps"] - s["pass_snaps"]
    sn = s.groupby("pfr_player_id").agg(def_snaps=("defense_snaps", "sum"), pass_snaps=("pass_snaps", "sum"),
                                         run_snaps=("run_snaps", "sum"), team=("team", "last"))
    cols = ["def_ints", "def_targets", "def_completions_allowed", "def_yards_allowed", "def_receiving_td_allowed",
            "def_pressures", "def_sacks", "def_tackles_combined", "def_missed_tackles"]
    pfd = d["pfr_def"].copy()
    lg_adot = np.average(pfd["def_adot"].fillna(pfd["def_adot"].mean()), weights=pfd["def_targets"].fillna(0) + 1e-9)
    pfd["exp_cmp_allowed"] = pfd["def_targets"].fillna(0) * cp_at_depth(d["cp_w"], pfd["def_adot"].fillna(lg_adot))
    cols += ["exp_cmp_allowed", "def_yards_after_catch"]
    pf = pfd.groupby("pfr_player_id")[cols].sum()
    g = sn.join(pf, how="left").fillna(0).reset_index().rename(columns={"pfr_player_id": "pfr_id"})
    g = g.merge(d["roster"].drop(columns="team"), on="pfr_id")
    # run stops + pass breakups from play-by-play
    p = d["pbp"]
    tk_cols = ["solo_tackle_1_player_id", "solo_tackle_2_player_id", "assist_tackle_1_player_id",
               "assist_tackle_2_player_id", "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id",
               "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id"]
    rs = p[(p["rush"] == 1) & (p["success"] == 0)]
    stops = rs.melt(id_vars=["game_id", "play_id"], value_vars=tk_cols)["value"].dropna()
    stops = rs.melt(id_vars=["game_id", "play_id"], value_vars=tk_cols).dropna().drop_duplicates(
        ["game_id", "play_id", "value"])["value"].value_counts()
    pd_ = pd.concat([p["pass_defense_1_player_id"], p["pass_defense_2_player_id"]]).dropna().value_counts()
    g["run_stops"] = g["gsis_id"].map(stops).fillna(0)
    g["pbus"] = g["gsis_id"].map(pd_).fillna(0)
    g["team_games_key"] = g["team"]
    return g


def def_metrics(g):
    lg = lambda a, b: g[a].sum() / max(g[b].sum(), 1)
    g["pressure_rate"] = shrink(g["def_pressures"], g["pass_snaps"], 100, lg("def_pressures", "pass_snaps"))
    g["sack_rate"] = shrink(g["def_sacks"], g["pass_snaps"], 150, lg("def_sacks", "pass_snaps"))
    g["run_stop_rate"] = shrink(g["run_stops"], g["run_snaps"], 60, lg("run_stops", "run_snaps"))
    g["tkl_att"] = g["def_tackles_combined"] + g["def_missed_tackles"]
    g["missed_tkl_rate"] = shrink(g["def_missed_tackles"], g["tkl_att"], 20, lg("def_missed_tackles", "tkl_att"))
    g["yds_per_target"] = shrink(g["def_yards_allowed"], g["def_targets"], COV_K, lg("def_yards_allowed", "def_targets"))
    g["cmp_over_exp_allowed"] = shrink(g["def_completions_allowed"] - g["exp_cmp_allowed"], g["def_targets"], COV_K, 0)
    g["yac_per_cmp_allowed"] = shrink(g["def_yards_after_catch"], g["def_completions_allowed"], 15,
                                      lg("def_yards_after_catch", "def_completions_allowed"))
    g["tgt_per_cov_snap"] = shrink(g["def_targets"], g["pass_snaps"], COV_SNAP_K, lg("def_targets", "pass_snaps"))
    g["yds_per_cov_snap"] = shrink(g["def_yards_allowed"], g["pass_snaps"], COV_SNAP_K, lg("def_yards_allowed", "pass_snaps"))
    rating = passer_rating(g["def_completions_allowed"], g["def_targets"], g["def_yards_allowed"],
                           g["def_receiving_td_allowed"], g["def_ints"])
    pos_avg = np.average(rating, weights=np.maximum(g["def_targets"], 0.1))
    g["rating_allowed"] = shrink(rating * g["def_targets"], g["def_targets"], COV_K, pos_avg)
    g["plays"] = g["def_ints"] + g["pbus"]
    g["playmaking"] = shrink(g["plays"], g["def_targets"], COV_K, lg("plays", "def_targets"))
    g["plays_per_cov_snap"] = shrink(g["plays"], g["pass_snaps"], COV_SNAP_K, lg("plays", "pass_snaps"))
    return g


def grade_defense(d, def_ctx, team_games):
    g = defense_table(d, def_ctx)
    g = g[g["team"].isin(team_games.index)]
    G = g["team"].map(team_games)
    dc = g["depth_chart_position"]
    # OLBs split by job: EDGE if they generate more pressures than targets, or are built like one (248+ lb)
    rushes = (g["def_pressures"] > g["def_targets"]) | (g["weight"] >= 248)
    heavy = g["weight"] >= 290
    groups = {
        "EDGE": (dc.isin(["DE"]) & ~heavy) | (dc.eq("OLB") & rushes),
        "IDL": dc.isin(["DT", "NT"]) | (dc.isin(["DE"]) & heavy),
        "LB": dc.isin(["ILB", "MLB", "LB"]) | (dc.eq("OLB") & ~rushes),
        "CB": dc.isin(["CB", "DB"]),
        "S": dc.isin(["FS", "SS", "S"]),
    }
    mins = {"EDGE": 25, "IDL": 25, "LB": 30, "CB": 30, "S": 30}
    out = {}
    for pos, mask in groups.items():
        sub = g[mask & (g["def_snaps"] > 0)].copy()
        sub["pt_share"] = (sub["def_snaps"] / (mins[pos] * G[mask & (g["def_snaps"] > 0)].clip(upper=TRUST_GAMES))).fillna(0)
        sub["qualified"] = sub["pt_share"] >= 1
        sub = def_metrics(sub)
        if pos in ("CB", "S"):
            # Coverage numbers are partly the whole defense (pass rush, scheme). Judge each DB mostly
            # against his own secondary; keep only DB_TEAM_SHARE of the unit's overall level.
            for m in ["tgt_per_cov_snap", "yds_per_target", "cmp_over_exp_allowed", "yac_per_cmp_allowed",
                      "rating_allowed", "yds_per_cov_snap"]:
                if m in sub:
                    w_ = sub["def_snaps"].clip(lower=1)
                    tm = ((sub[m] * w_).groupby(sub["team"]).sum() / w_.groupby(sub["team"]).sum())
                    lg = (sub[m] * w_).sum() / w_.sum()
                    t_ = sub["team"].map(tm)
                    sub[m] = (sub[m] - t_) + DB_TEAM_SHARE * (t_ - lg)
        Gs = G[sub.index] if hasattr(G, "loc") else G
        sub["workload"] = sub["def_snaps"] / (mins[pos] * Gs)
        sub["pressures_pg"] = sub["def_pressures"] / Gs
        sub["stops_pg"] = sub["run_stops"] / Gs
        sub["ballplays_pg"] = sub["plays"] / Gs
        sub["tackles_pg"] = sub["def_tackles_combined"] / Gs
        out[pos] = grade(sub, W(pos))
    return out


# ----------------------------------------------------------------- specialists
def grade_k(d, team_games):
    h = d["fg_hist"]
    x, y = h["kick_distance"].values, (h["field_goal_result"] == "made").astype(float).values
    # 2-parameter logistic fit by Newton's method (make% vs distance)
    X = np.c_[np.ones_like(x), x]; w = np.zeros(2)
    for _ in range(25):
        pr = 1 / (1 + np.exp(-X @ w)); W = pr * (1 - pr)
        w += np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - pr))
    p = d["pbp"]
    fg = p[p["field_goal_attempt"] == 1].copy()
    fg["exp"] = 1 / (1 + np.exp(-(w[0] + w[1] * fg["kick_distance"])))
    fg["poe"] = 3 * ((fg["field_goal_result"] == "made") - fg["exp"])
    xp = p[p["extra_point_result"].notna()].copy()
    xp_rate = (xp["extra_point_result"] == "good").mean()
    xp["poe"] = (xp["extra_point_result"] == "good") - xp_rate
    k = pd.concat([fg[["kicker_player_id", "poe", "posteam"]].assign(fga=1),
                   xp[["kicker_player_id", "poe", "posteam"]].assign(fga=0)])
    g = k.groupby("kicker_player_id").agg(points_over_exp=("poe", "sum"), att=("poe", "size"),
                                          fga=("fga", "sum"), team=("posteam", "last")).reset_index()
    g["pt_share"] = (g["fga"] / (1.2 * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    g["poe_per_att"] = shrink(g["points_over_exp"], g["att"], 10, 0)
    g = g.merge(d["roster"][["gsis_id", "full_name"]], left_on="kicker_player_id", right_on="gsis_id")
    return grade(g, [("poe_per_att", 1.0, 1)])


def grade_p(d, team_games):
    p = d["pbp"]
    pu = p[(p["punt_attempt"] == 1) & p["punter_player_id"].notna()].copy()
    pu["net"] = pu["kick_distance"].fillna(0) - pu["return_yards"].fillna(0) - 20 * pu["touchback"].fillna(0)
    g = pu.groupby("punter_player_id").agg(punts=("net", "size"), net_sum=("net", "sum"), epa=("epa", "sum"),
                                           i20=("punt_inside_twenty", "sum"), team=("posteam", "last")).reset_index()
    g["pt_share"] = (g["punts"] / (2.5 * g["team"].map(team_games).clip(upper=TRUST_GAMES))).fillna(0)
    g["qualified"] = g["pt_share"] >= 1
    lg = lambda a: g[a].sum() / g["punts"].sum()
    g["epa_per_punt"] = shrink(g["epa"], g["punts"], 15, lg("epa"))
    g["net_avg"] = shrink(g["net_sum"], g["punts"], 15, lg("net_sum"))
    g["inside20_rate"] = shrink(g["i20"], g["punts"], 15, lg("i20"))
    g = g.merge(d["roster"][["gsis_id", "full_name"]], left_on="punter_player_id", right_on="gsis_id")
    return grade(g, [("epa_per_punt", .5, 1), ("net_avg", .3, 1), ("inside20_rate", .2, 1)])


# ----------------------------------------------------------------- main
def run_all(season, thru):
    return run_on(load_all(season, thru))


def run_on(d):
    off_ctx, def_ctx, team_games = team_context(d)
    routes = est_routes(d, off_ctx)
    rec = receiving_table(d, routes)
    res = {"QB": grade_qb(d, team_games),
           "RB": grade_rb(d, rec, team_games),
           "WR": grade_receivers(rec, "WR", 18, team_games, ngs_receiving(d)),
           "TE": grade_receivers(rec, "TE", 12, team_games, ngs_receiving(d)),
           "OL (team units)": grade_ol_units(d)}
    try:
        res.update(grade_ol_individual(d, off_ctx, def_ctx, team_games, res["OL (team units)"]))
    except Exception as e:                      # e.g. older seasons with a different depth-chart format
        print(f"[warn] individual OL grades skipped: {e}")
    res.update(grade_defense(d, def_ctx, team_games))
    res["K"] = grade_k(d, team_games)
    res["P"] = grade_p(d, team_games)
    return res


# Early-season stabilizer: blend in last season's grade, fading out as this season's sample grows.
# Last season's grade counts like ~2 games of this season's sample:
# ~75% this season by Week 6, ~90% by Week 17. Use --no-prior for pure single-season grades.
PRIOR_SAMPLE = {"QB": ("plays", 70), "RB": ("carries", 30), "WR": ("routes", 60), "TE": ("routes", 40),
                "EDGE": ("def_snaps", 100), "IDL": ("def_snaps", 90), "LB": ("def_snaps", 110),
                "CB": ("def_snaps", 110), "S": ("def_snaps", 110), "K": ("att", 10), "P": ("punts", 8)}


def blend_with_prior(res, prev):
    # last season's grade at any position (players who moved DE->DT etc. keep their history);
    # if graded at two positions last year, use the one with the bigger sample
    allprev = pd.concat([prev[p][["gsis_id", "grade"]] for p in PRIOR_SAMPLE])
    any_pos = allprev.drop_duplicates("gsis_id").set_index("gsis_id")["grade"]
    for pos, (col, k) in PRIOR_SAMPLE.items():
        cur = res[pos]
        cur["grade_this_season"] = cur["grade"]
        same_pos = prev[pos].drop_duplicates("gsis_id").set_index("gsis_id")["grade"]
        cur["grade_last_season"] = cur["gsis_id"].map(same_pos).fillna(cur["gsis_id"].map(any_pos))
        n = cur[col].astype(float)
        blended = (n * cur["grade_this_season"] + k * cur["grade_last_season"]) / (n + k)
        cur["grade"] = blended.fillna(cur["grade_this_season"]).round(1)
        res[pos] = cur.sort_values("grade", ascending=False).reset_index(drop=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--thru-week", type=int, default=7)
    ap.add_argument("--blend", action="store_true",
                    help="optional: blend in last season's grade early in the year (off by default)")
    a = ap.parse_args()
    res = run_all(a.season, a.thru_week)
    if a.blend:
        res = blend_with_prior(res, run_all(a.season - 1, 18))

    out_dir = OUT / str(a.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "blended with last season" if a.blend else f"{a.season} games only"
    wk = "full regular season" if a.thru_week >= 18 else f"Weeks 1–{a.thru_week}"
    lines = [f"# Top 10 by position — {a.season} season, {wk}\n",
             f"70 = average qualified starter · each 10 pts = 1 standard deviation · {mode}\n"]
    for pos, df in res.items():
        df.to_csv(out_dir / f"grades_{pos.split()[0]}.csv", index=False)
        has_split = "grade_this_season" in df.columns
        hdr = "| # | Player | Team | Grade | This season | Last season |\n|---|---|---|---|---|---|" if has_split \
            else "| # | Player | Team | Grade |\n|---|---|---|---|"
        lines.append(f"\n## {pos}  ({len(df)} qualified)\n\n{hdr}")
        for i, r in df.head(10).iterrows():
            row = f"| {i+1} | {r['full_name']} | {r['team']} | {r['grade']:.1f} |"
            if has_split:
                ls = "—" if pd.isna(r["grade_last_season"]) else f"{r['grade_last_season']:.1f}"
                row += f" {r['grade_this_season']:.1f} | {ls} |"
            lines.append(row)
    (OUT / f"RESULTS_{a.season}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
