"""
Stat lines for the site: box-score stats plus the numbers each grade is built from, with our ranks.

Ranks:
  counting stats (tackles, yards, sacks...)  -> #1 = most, among everyone graded at the position
  rates and grade inputs                      -> #1 = best, compared against the position's regulars
Team stats: #1 = best of 32.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl
import grades
import team_grades as tgm

# ---------------------------------------------------------------- box score definitions
# (label, column, kind, direction) kind: n = count, r = rate (ranked vs regulars, direction says which way is good)
OFF_FUM = ("Fumbles lost", "fumbles_lost", "n", -1)
BOX = {
    "QB": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Completions", "completions", "n", 1), ("Attempts", "attempts", "n", 1),
           ("Completion %", "cmp_pct", "r%", 1), ("Passing yards", "passing_yards", "n", 1),
           ("Passing TDs", "passing_tds", "n", 1), ("Interceptions", "passing_interceptions", "n", -1),
           ("Yards per attempt", "ypa", "r1", 1), ("Sacks taken", "sacks_suffered", "n", -1),
           ("Rushing yards", "rushing_yards", "n", 1), ("Rushing TDs", "rushing_tds", "n", 1), OFF_FUM],
    "RB": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Carries", "carries", "n", 1), ("Rushing yards", "rushing_yards", "n", 1),
           ("Yards per carry", "ypc", "r1", 1), ("Rushing TDs", "rushing_tds", "n", 1),
           ("Yards after contact", "rushing_yards_after_contact", "n", 1),
           ("Broken tackles", "broken_tackles", "n", 1), ("Targets", "targets", "n", 1),
           ("Receptions", "receptions", "n", 1), ("Receiving yards", "receiving_yards", "n", 1),
           ("Receiving TDs", "receiving_tds", "n", 1), OFF_FUM],
    "WR": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Targets", "targets", "n", 1), ("Receptions", "receptions", "n", 1),
           ("Receiving yards", "receiving_yards", "n", 1), ("Yards per catch", "ypr", "r1", 1),
           ("Receiving TDs", "receiving_tds", "n", 1), ("Yards after catch", "receiving_yards_after_catch", "n", 1),
           ("First downs", "receiving_first_downs", "n", 1), ("Drops", "receiving_drop", "n", -1),
           ("Broken tackles", "broken_tackles", "n", 1), OFF_FUM],
    "OL": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Offensive snaps", "offense_snaps", "n", 1),
           ("Penalties", "penalties", "n", -1), ("Penalty yards", "penalty_yards", "n", -1)],
    "DL": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Defensive snaps", "defense_snaps", "n", 1),
           ("Tackles", "def_tackles_combined", "n", 1), ("Tackles for loss", "def_tackles_for_loss", "n", 1),
           ("Sacks", "def_sacks", "n", 1), ("QB hits", "def_qb_hits", "n", 1), ("Pressures", "def_pressures", "n", 1),
           ("Pass breakups", "def_pass_defended", "n", 1), ("Forced fumbles", "def_fumbles_forced", "n", 1),
           ("Missed tackles", "def_missed_tackles", "n", -1)],
    "LB": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Defensive snaps", "defense_snaps", "n", 1),
           ("Tackles", "def_tackles_combined", "n", 1), ("Tackles for loss", "def_tackles_for_loss", "n", 1),
           ("Sacks", "def_sacks", "n", 1), ("Pressures", "def_pressures", "n", 1),
           ("Interceptions", "def_interceptions", "n", 1), ("Pass breakups", "def_pass_defended", "n", 1),
           ("Forced fumbles", "def_fumbles_forced", "n", 1), ("Missed tackles", "def_missed_tackles", "n", -1),
           ("Targets allowed", "def_targets", "n", -1), ("Yards allowed", "def_yards_allowed", "n", -1)],
    "DB": [("Games", "games", "n", 0), ("Starts (50%+ of snaps)", "starts", "n", 0), ("Defensive snaps", "defense_snaps", "n", 1),
           ("Tackles", "def_tackles_combined", "n", 1), ("Interceptions", "def_interceptions", "n", 1),
           ("Pass breakups", "def_pass_defended", "n", 1), ("Targets allowed", "def_targets", "n", -1),
           ("Catches allowed", "def_completions_allowed", "n", -1), ("Yards allowed", "def_yards_allowed", "n", -1),
           ("TDs allowed", "def_receiving_td_allowed", "n", -1), ("Passer rating allowed", "rating_allowed_box", "r1", -1),
           ("Tackles for loss", "def_tackles_for_loss", "n", 1), ("Sacks", "def_sacks", "n", 1),
           ("Forced fumbles", "def_fumbles_forced", "n", 1), ("Missed tackles", "def_missed_tackles", "n", -1)],
    "K": [("Games", "games", "n", 0), ("FG made", "fg_made", "n", 1), ("FG attempts", "fg_att", "n", 1),
          ("FG %", "fg_pct_c", "r%", 1), ("Longest FG", "fg_long", "n", 1), ("50+ yard FGs made", "fg_50", "n", 1),
          ("Extra points made", "pat_made", "n", 1), ("Extra point attempts", "pat_att", "n", 1)],
    "P": [("Games", "games", "n", 0), ("Punts", "pt_att", "n", 1), ("Gross average", "pt_gross", "r1", 1),
          ("Net average", "pt_net", "r1", 1), ("Inside the 20", "pt_inside_20", "n", 1),
          ("Touchbacks", "pt_touchback", "n", 1)],
}
GROUP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "WR", "OT": "OL", "OG": "OL", "C": "OL", "EDGE": "DL",
         "IDL": "DL", "LB": "LB", "CB": "DB", "S": "DB", "K": "K", "P": "P"}

# grade inputs: plain label, format
METRIC = {
    "epa_play": ("Points added per play (EPA)", "f3"), "value_over_avg": ("Points added over an average QB", "f1"),
    "cpoe": ("Completion % over expected", "pp"), "success": ("Success rate", "%"),
    "int_worthy_rate": ("Interception-worthy throw rate", "%"), "fault_sack_rate": ("Sacks he caused (rate)", "%"),
    "ryoe_att": ("Rush yards over expected per carry", "f2"), "rec_yds_per_route": ("Receiving yards per route", "f2"),
    "broken_tkl_rate": ("Broken tackles per touch", "%"), "yds_per_route": ("Yards per route run", "f2"),
    "tgt_per_route": ("Targets per route", "%"), "route_share": ("Route share (role size)", "%"),
    "epa_per_target": ("Points added per target", "f2"), "separation_over_exp": ("Separation over expected (yds)", "f2"),
    "catch_over_exp": ("Catch rate over expected", "%"), "yac_over_exp": ("Yards after catch over expected", "f2"),
    "drop_rate": ("Drop rate", "%"), "pressures_oe_per_100": ("Pressures allowed vs expected, per 100 pass snaps", "f1"),
    "run_succ_oe": ("Run success through his gaps vs expected", "%"), "run_epa_oe": ("Run EPA through his gaps vs expected", "f3"),
    "penalty_rate": ("Penalty rate per snap", "%"), "team_unit": ("Team offensive line grade", "f1"),
    "pressure_rate": ("Pressure rate", "%"), "sack_rate": ("Sack rate", "%"), "run_stop_rate": ("Run stop rate", "%"),
    "missed_tkl_rate": ("Missed tackle rate", "%"), "yds_per_cov_snap": ("Yards allowed per coverage snap", "f2"),
    "tgt_per_cov_snap": ("Targets per coverage snap", "%"), "yds_per_target": ("Yards allowed per target", "f1"),
    "cmp_over_exp_allowed": ("Completions allowed over expected", "%"), "yac_per_cmp_allowed": ("YAC allowed per catch", "f1"),
    "playmaking": ("Plays on the ball (INT + PBU) rate", "%"), "rating_allowed": ("Passer rating allowed", "f1"),
    "plays_per_cov_snap": ("Plays on the ball per coverage snap", "%"),
    "poe_per_att": ("Points over expected per kick", "f2"), "epa_per_punt": ("Points added per punt", "f2"),
    "net_avg": ("Net punt average", "f1"), "inside20_rate": ("Inside-20 rate", "%"),
}
KP_SPEC = {"K": [("poe_per_att", 1.0, 1)], "P": [("epa_per_punt", .5, 1), ("net_avg", .3, 1), ("inside20_rate", .2, 1)]}


def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if kind in ("n",):
        return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"
    if kind in ("%", "r%"):
        return f"{100 * v:.1f}%"
    if kind == "pp":
        return f"{v:+.1f}%"
    if kind == "pp100":
        return f"{100 * v:+.1f}%"
    if kind == "f1s":
        return f"{v:+.1f}"
    if kind in ("r1", "f1"):
        return f"{v:.1f}"
    if kind == "f2":
        return f"{v:.2f}"
    if kind == "f3":
        return f"{v:.3f}"
    return str(v)


def _rank_vs(values, ref_values, direction):
    """Rank each value against the regulars: 1 + how many regulars are strictly better."""
    ref = np.sort(np.asarray(ref_values, float)[~np.isnan(np.asarray(ref_values, float))])
    out = []
    for v in values:
        if np.isnan(v):
            out.append(None); continue
        better = (ref > v).sum() if direction > 0 else (ref < v).sum()
        out.append(int(better) + 1)
    return out, len(ref)


def player_stats_week(season, thru, playoffs):
    s = grades.cached(f"pstats_{season}", lambda: nfl.load_player_stats([season], summary_level="week"))
    keep = s["season_type"].eq("REG") & (s["week"] <= thru)
    if playoffs:
        keep = keep | s["season_type"].eq("POST")
    return s[keep]


def box_table(d, season, thru, playoffs):
    w = player_stats_week(season, thru, playoffs)
    num = [c for c in w.columns if c not in ("player_id",) and pd.api.types.is_numeric_dtype(w[c])
           and c not in ("season", "week", "fg_long", "pt_long")]
    b = w.groupby("player_id")[num].sum()
    b["fg_long"] = w.groupby("player_id")["fg_long"].max()
    b["fumbles_lost"] = b[["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]].sum(axis=1)
    b["fg_50"] = b["fg_made_50_59"] + b["fg_made_60_"]
    ro = d["roster"].set_index("gsis_id")["pfr_id"]
    b["pfr_id"] = b.index.map(ro)
    sn = d["snaps"].assign(any_=lambda x: (x["offense_snaps"] + x["defense_snaps"] + x["st_snaps"]) > 0,
                           start_=lambda x: (x["offense_pct"].fillna(0) >= .5) | (x["defense_pct"].fillna(0) >= .5))
    sg = sn.groupby("pfr_player_id").agg(games=("any_", "sum"), starts=("start_", "sum"),
                                         offense_snaps=("offense_snaps", "sum"), defense_snaps=("defense_snaps", "sum"))
    pdf = d["pfr_def"].groupby("pfr_player_id")[["def_tackles_combined", "def_pressures", "def_missed_tackles",
                                                 "def_targets", "def_completions_allowed", "def_yards_allowed",
                                                 "def_receiving_td_allowed", "def_ints"]].sum()
    prs = d["pfr_rush"].groupby("pfr_player_id")[["rushing_yards_after_contact", "rushing_broken_tackles"]].sum()
    prc = d["pfr_rec"].groupby("pfr_player_id")[["receiving_broken_tackles", "receiving_drop"]].sum()
    b = b.join(sg, on="pfr_id").join(pdf, on="pfr_id").join(prs, on="pfr_id").join(prc, on="pfr_id")
    b["broken_tackles"] = b["rushing_broken_tackles"].fillna(0) + b["receiving_broken_tackles"].fillna(0)
    b["games"] = b["games"].fillna(w.groupby("player_id")["game_id"].nunique())
    div = lambda a, c: b[a] / b[c].where(b[c] > 0)
    b["cmp_pct"], b["ypa"] = div("completions", "attempts"), div("passing_yards", "attempts")
    b["ypc"], b["ypr"] = div("rushing_yards", "carries"), div("receiving_yards", "receptions")
    b["fg_pct_c"] = div("fg_made", "fg_att")
    b["pt_gross"], b["pt_net"] = div("pt_yards", "pt_att"), div("pt_net_yards", "pt_att")
    b["rating_allowed_box"] = grades.passer_rating(b["def_completions_allowed"], b["def_targets"].where(b["def_targets"] > 0),
                                                   b["def_yards_allowed"], b["def_receiving_td_allowed"], b["def_ints"])
    return b


def player_rows(res, d, season, thru, playoffs, scope):
    b = box_table(d, season, thru, playoffs)
    rows = []
    for pos, df in res.items():
        if pos not in GROUP:
            continue
        df = df.reset_index(drop=True)
        ids = df["gsis_id"] if "gsis_id" in df else df.get("kicker_player_id", df.get("punter_player_id"))
        reg = df["qualified"].astype(bool) if "qualified" in df else pd.Series(True, index=df.index)
        bx = b.reindex(ids.values)
        bx.index = df.index
        for o, (label, col, kind, dirn) in enumerate(BOX[GROUP[pos]]):
            v = bx[col].astype(float) if col in bx else pd.Series(np.nan, index=df.index)
            if kind == "n":
                v = v.fillna(0)
                # counting stats: #1 = most, and only where more is better (no rank for games, INTs thrown, etc.)
                rk = v.rank(ascending=False, method="min").astype(int).tolist() if dirn == 1 else [None] * len(df)
                of = len(df) if dirn == 1 else None
            else:
                rk, of = _rank_vs(v.values, v[reg].values, dirn)
            for i in df.index:
                rows.append((season, scope, pos, ids[i], "Box score", o, label, fmt(v[i], kind), rk[i], of))
        spec = KP_SPEC.get(pos) or grades.W(pos)
        for o, (m, wt, sign) in enumerate(spec):
            if m not in df or m not in METRIC:
                continue
            label, kind = METRIC[m]
            v = df[m].astype(float)
            rk, of = _rank_vs(v.values, v[reg].values, sign)
            for i in df.index:
                rows.append((season, scope, pos, ids[i], f"What the grade uses", 100 + o,
                             f"{label} ({wt:.0%} of grade)", fmt(v[i], kind), rk[i], of))
    out = pd.DataFrame(rows, columns=["season", "scope", "pos", "gsis_id", "grp", "ord", "stat", "value", "rank", "of"])
    out["rank"] = out["rank"].astype("Int64")
    return out


def team_rows(d, season, scope):
    p = d["pbp"]
    g = tgm.team_games(d)
    ts = tgm.season_grades(d).set_index("team")
    pl = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["posteam"].notna()]
    def side(col):
        x = pl.groupby(col)
        return pd.DataFrame({"plays": x.size(), "yards": x["yards_gained"].sum(), "pass_yds": x["passing_yards"].sum(),
                             "rush_yds": x["rushing_yards"].sum(), "epa": x["epa"].mean(), "succ": x["success"].mean(),
                             "sacks": x["sack"].sum()})
    o, df_ = side("posteam"), side("defteam")
    t3 = p[(p["down"] == 3) & ((p["pass"] == 1) | (p["rush"] == 1))]
    third_o = t3.groupby("posteam")["third_down_converted"].mean()
    third_d = t3.groupby("defteam")["third_down_converted"].mean()
    dr = p[p["drive_inside20"] == 1].groupby(["game_id", "posteam", "fixed_drive"])["fixed_drive_result"].first()
    rz = (dr == "Touchdown").groupby(level=1).mean()
    rz_d = p[p["drive_inside20"] == 1].groupby(["game_id", "defteam", "fixed_drive"])["fixed_drive_result"].first()
    rz_d = (rz_d == "Touchdown").groupby(level=1).mean()
    fg = p[p["field_goal_attempt"] == 1].groupby("posteam")["field_goal_result"].apply(lambda s: (s == "made").mean())
    pu = p[p["punt_attempt"] == 1].assign(net=lambda x: x["kick_distance"].fillna(0) - x["return_yards"].fillna(0) - 20 * x["touchback"].fillna(0))
    net = pu.groupby("posteam")["net"].mean()
    G = g.groupby("team")["game_id"].nunique()
    xp = pl[pl["xpass"].notna()]
    proe = (xp["pass"] - xp["xpass"]).groupby(xp["posteam"]).mean()
    ex = pl.assign(ex=np.where(pl["pass"] == 1, pl["yards_gained"] >= 20, pl["yards_gained"] >= 10))
    expl_o, expl_d = ex.groupby("posteam")["ex"].mean(), ex.groupby("defteam")["ex"].mean()
    f4 = p[(p["down"] == 4) & (p["ydstogo"] <= 2) & p["play_type"].isin(["pass", "run", "punt", "field_goal"])]
    go4 = f4["play_type"].isin(["pass", "run"]).groupby(f4["posteam"]).mean()
    pa_ = p[(p["pass_attempt"] == 1) & (p["sack"] == 0) & p["air_yards"].notna()]
    adot = pa_.groupby("posteam")["air_yards"].mean()
    ed = pl[pl["down"].isin([1, 2])]
    ed_o, ed_d = ed.groupby("posteam")["epa"].mean(), ed.groupby("defteam")["epa"].mean()
    ng = d["ngs"]["passing"]
    ttt = (ng["avg_time_to_throw"] * ng["attempts"]).groupby(ng["team_abbr"]).sum() / ng["attempts"].groupby(ng["team_abbr"]).sum()
    wins = g.assign(w=(g["pf"] > g["pa"]) + 0.5 * (g["pf"] == g["pa"])).groupby("team")["w"].sum()
    pf_, pa_s = g.groupby("team")["pf"].sum(), g.groupby("team")["pa"].sum()
    xw = G * pf_ ** 2.37 / (pf_ ** 2.37 + pa_s ** 2.37)
    gs = g.groupby("team")[["pf", "pa", "turnovers", "d_turnovers"]].sum()
    T = pd.DataFrame(index=ts.index)
    stats = [  # side, label, series, kind, direction (1 = higher is better)
        ("Offense", "Points per game", gs["pf"] / G, "r1", 1), ("Offense", "Yards per play", o["yards"] / o["plays"], "r1", 1),
        ("Offense", "Passing yards per game", o["pass_yds"] / G, "r1", 1), ("Offense", "Rushing yards per game", o["rush_yds"] / G, "r1", 1),
        ("Offense", "Points added per play (EPA)", o["epa"], "f3", 1), ("Offense", "Success rate", o["succ"], "%", 1),
        ("Offense", "3rd-down conversion rate", third_o, "%", 1), ("Offense", "Red-zone TD rate", rz, "%", 1),
        ("Offense", "Turnovers", gs["turnovers"], "n", -1), ("Offense", "Sacks allowed", o["sacks"], "n", -1),
        ("Offense", "Pressure rate allowed", ts["pressure_rate_allowed"], "%", -1),
        ("Defense", "Points allowed per game", gs["pa"] / G, "r1", -1), ("Defense", "Yards per play allowed", df_["yards"] / df_["plays"], "r1", -1),
        ("Defense", "Passing yards allowed per game", df_["pass_yds"] / G, "r1", -1),
        ("Defense", "Rushing yards allowed per game", df_["rush_yds"] / G, "r1", -1),
        ("Defense", "Points added per play allowed (EPA)", df_["epa"], "f3", -1), ("Defense", "Success rate allowed", df_["succ"], "%", -1),
        ("Defense", "3rd-down conversion rate allowed", third_d, "%", -1), ("Defense", "Red-zone TD rate allowed", rz_d, "%", -1),
        ("Defense", "Takeaways", gs["d_turnovers"], "n", 1), ("Defense", "Sacks", df_["sacks"], "n", 1),
        ("Defense", "Pressure rate", ts["pressure_rate_made"], "%", 1),
        ("Offense", "Explosive play rate", expl_o, "%", 1), ("Offense", "Early-down EPA per play", ed_o, "f3", 1),
        ("Offense", "Pass rate over expected", proe, "pp100", 0), ("Offense", "Average depth of target (yds)", adot, "r1", 0),
        ("Offense", "QB time to throw (sec)", ttt, "f2", 0), ("Offense", "Go-for-it rate on 4th and 1-2", go4, "%", 0),
        ("Defense", "Explosive plays allowed rate", expl_d, "%", -1), ("Defense", "Early-down EPA allowed per play", ed_d, "f3", -1),
        ("Team", "Wins above expected (luck)", wins - xw, "f1s", 0),
        ("Special teams", "Field goal %", fg, "%", 1), ("Special teams", "Net punt average", net, "r1", 1),
        ("Special teams", "Special teams points added per game", ts["st_epa_game"], "f2", 1),
    ]
    rows = []
    for o_, (sd, label, s, kind, dirn) in enumerate(stats):
        s = s.reindex(T.index).astype(float)
        rk = s.rank(ascending=dirn < 0, method="min")   # direction 0 = style stat: #1 = highest
        for t in T.index:
            rows.append((season, scope, t, sd, o_, label, fmt(s[t], kind if kind != "n" else "n"),
                         None if np.isnan(rk[t]) else int(rk[t]), len(T)))
    out = pd.DataFrame(rows, columns=["season", "scope", "team", "side", "ord", "stat", "value", "rank", "of"])
    out["rank"] = out["rank"].astype("Int64")
    return out
