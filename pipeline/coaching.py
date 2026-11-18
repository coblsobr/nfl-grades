"""
Coaching, play calling and luck, from free play-by-play (nflverse) and FTN charting.

What it can and can't say: these numbers describe decisions and results. They can't see who
actually called a play or what the coach was thinking, and play-calling results are tangled up
with player talent.
"""
import math
import numpy as np
import pandas as pd
import grades

DIST = [(1, 1.0), (3, .8), (6, .6), (9, .4), (99, .2)]


def surrender_index(r):
    """Our version of the 'Surrender Index' idea (Jon Bois): how timid a punt was.
    Bigger = punting closer to the opponent's end zone, on shorter yardage, while behind, later in the game."""
    fp = 1.1 ** max(0.0, 60 - r["yardline_100"])                     # each yard past your own 40 multiplies by 1.1
    dist = next(m for lim, m in DIST if r["ydstogo"] <= lim)
    sd = r["score_differential"]
    score = 1 if sd > 0 else 2 if sd == 0 else 3 if sd >= -8 else 4
    gsr = r["game_seconds_remaining"]
    clock = 1 + ((1800 - gsr) * 0.001) ** 3 if gsr <= 1800 else 1   # grows through the second half
    return fp * dist * score * clock


def team_coaching(d):
    p = d["pbp"]
    G = p.groupby("posteam")["game_id"].nunique()
    plays = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["posteam"].notna() & p["epa"].notna()]

    # ---- 4th downs: how often a team goes for it vs. what the league does in the same spot
    f4 = p[(p["down"] == 4) & p["play_type"].isin(["pass", "run", "punt", "field_goal"]) & (p["qb_kneel"] != 1)
           & p["wp"].notna()].copy()
    f4["go"] = f4["play_type"].isin(["pass", "run"]).astype(float)
    ytg = f4["ydstogo"].clip(upper=15)
    X = np.c_[np.ones(len(f4)), ytg, np.log(ytg), f4["yardline_100"] / 100, (f4["yardline_100"] / 100) ** 2,
              f4["score_differential"].clip(-21, 21) / 10, f4["game_seconds_remaining"] / 3600, f4["wp"]]
    w = grades.fit_logistic(X, f4["go"].values)
    f4["xgo"] = 1 / (1 + np.exp(-X @ w))
    go = f4.groupby("posteam").agg(go=("go", "sum"), xgo=("xgo", "sum"), n=("go", "size"))
    conv = f4[f4["go"] == 1].groupby("posteam")["fourth_down_converted"].mean()

    # ---- timid punts
    pu = f4[f4["play_type"] == "punt"].copy()
    pu["si"] = pu.apply(surrender_index, axis=1)
    si = pu.groupby("posteam")["si"].agg(["mean", "max", lambda s: (s >= 10).sum()])
    si.columns = ["si_mean", "si_max", "si_timid"]
    worst = pu.loc[pu["si"].idxmax()] if len(pu) else None

    # ---- play calling (neutral situations: early downs, game within reach, not the end of a half)
    nt = plays[plays["down"].isin([1, 2]) & plays["wp"].between(.2, .8) & (plays["half_seconds_remaining"] > 120)
               & plays["xpass"].notna()]
    ed_proe = (nt["pass"] - nt["xpass"]).groupby(nt["posteam"]).mean()
    ed_epa = nt.groupby("posteam")["epa"].mean()
    gl = plays[(plays["goal_to_go"] == 1) & (plays["yardline_100"] <= 3) & plays["down"].isin([1, 2]) & plays["xpass"].notna()]
    gl_pass = gl.groupby("posteam")["pass"].mean()
    gl_proe = (gl["pass"] - gl["xpass"]).groupby(gl["posteam"]).mean()
    gl_n = gl.groupby("posteam").size()
    sg = plays.groupby(["posteam", "shotgun"])["pass"].mean().unstack()
    predict = (sg.get(1, np.nan) - sg.get(0, np.nan))
    ftn = d["ftn"].rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    f = plays.merge(ftn[["game_id", "play_id", "is_play_action", "is_motion", "is_rpo", "is_screen_pass", "is_no_huddle"]],
                    on=["game_id", "play_id"], how="inner")
    db = f[f["pass"] == 1]
    pa_rate = db.groupby("posteam")["is_play_action"].mean()
    pa_gain = db[db["is_play_action"] == 1].groupby("posteam")["epa"].mean() - db[db["is_play_action"] == 0].groupby("posteam")["epa"].mean()
    motion = f.groupby("posteam")["is_motion"].mean()
    screen = db.groupby("posteam")["is_screen_pass"].mean()
    nohud = f.groupby("posteam")["is_no_huddle"].mean()

    # ---- expectations and luck
    gm = p.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"), hs=("home_score", "last"),
                                  as_=("away_score", "last"), spread=("spread_line", "first")).dropna(subset=["spread"])
    rows = []
    for _, g in gm.iterrows():
        ph = 0.5 * (1 + math.erf(g["spread"] / 13.5 / math.sqrt(2)))     # Vegas spread -> chance the home team wins
        res = 1.0 if g["hs"] > g["as_"] else 0.0 if g["hs"] < g["as_"] else 0.5
        m = g["hs"] - g["as_"]
        rows += [dict(team=g["home"], xw=ph, w=res, close=abs(m) <= 8, m=m, ats=m - g["spread"]),
                 dict(team=g["away"], xw=1 - ph, w=1 - res, close=abs(m) <= 8, m=-m, ats=-m + g["spread"])]
    gr = pd.DataFrame(rows)
    vegas = gr.groupby("team")[["w", "xw"]].sum()
    ats = gr.groupby("team")["ats"].mean()                      # points per game better than the Vegas spread
    pen = p[(p["penalty"] == 1) & p["penalty_team"].notna()]
    pen_yds = pen.groupby("penalty_team")["penalty_yards"].sum() / G
    close = gr[gr["close"]].groupby("team")["w"].agg(["sum", "size"])
    fum = p[(p["fumble"] == 1) & p["fumble_recovery_1_team"].notna()]
    fr = {}
    for t in G.index:
        x = fum[(fum["posteam"] == t) | (fum["defteam"] == t)]
        fr[t] = (x["fumble_recovery_1_team"] == t).mean() if len(x) else np.nan
    fr = pd.Series(fr)

    T = pd.DataFrame(index=G.index)
    T["go_rate_oe"] = (go["go"] - go["xgo"]) / go["n"]
    T["go_extra"] = go["go"] - go["xgo"]
    T["fourth_conv"] = conv
    T["si_mean"], T["si_timid"] = si["si_mean"], si["si_timid"]
    T["ed_proe"], T["ed_epa"] = ed_proe, ed_epa
    T["gl_pass"], T["gl_proe"], T["gl_n"] = gl_pass, gl_proe, gl_n
    T["predictability"] = predict
    T["pa_rate"], T["pa_gain"], T["motion"], T["screen"], T["no_huddle"] = pa_rate, pa_gain, motion, screen, nohud
    T["wins"], T["vegas_xw"] = vegas["w"], vegas["xw"]
    T["vs_vegas"] = vegas["w"] - vegas["xw"]
    T["close_w"], T["close_n"] = close["sum"], close["size"]
    T["fumble_rec"] = fr
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    # grades on the usual scale: 70 = average, 10 = one standard deviation
    T["aggressive_grade"] = (70 + 10 * z(0.7 * z(T["go_rate_oe"]) - 0.3 * z(T["si_mean"].fillna(T["si_mean"].mean())))).round(1)
    T["playcall_grade"] = (70 + 10 * z(0.6 * z(T["ed_epa"]) + 0.2 * z(T["pa_gain"].fillna(0)) - 0.2 * z(T["predictability"]))).round(1)
    T["expect_grade"] = (70 + 10 * z(T["vs_vegas"])).round(1)
    T["ats_pg"] = ats
    T["pen_yds_pg"] = pen_yds
    # Overall coaching: beating the spread (results vs. what the market expected from this roster),
    # play calling, and discipline (penalty yards). Aggressiveness is a style stat, not part of it.
    T["coach_grade"] = (70 + 10 * z(0.40 * z(T["ats_pg"]) + 0.35 * z(T["playcall_grade"]) - 0.25 * z(T["pen_yds_pg"]))).round(1)
    return T, worst


def team_rows(d, season, scope):
    T, _ = team_coaching(d)
    pct = lambda v, s=False: "" if pd.isna(v) else (f"{100 * v:+.1f}%" if s else f"{100 * v:.0f}%")
    stats = [  # side, label, series, formatter, direction for rank (1 = higher is #1, -1 = lower is #1)
        ("Coaching", "Aggressiveness grade (4th downs + punts)", T["aggressive_grade"], lambda v: f"{v:.1f}", 1),
        ("Coaching", "Play-calling grade", T["playcall_grade"], lambda v: f"{v:.1f}", 1),
        ("Coaching", "Go-for-it rate vs. league in same spots", T["go_rate_oe"], lambda v: pct(v, True), 1),
        ("Coaching", "Extra 4th-down tries vs. league", T["go_extra"], lambda v: f"{v:+.1f}", 1),
        ("Coaching", "4th-down conversion rate", T["fourth_conv"], pct, 1),
        ("Coaching", "Timid punts (surrender score 10+)", T["si_timid"], lambda v: f"{v:.0f}", -1),
        ("Coaching", "Average surrender score per punt", T["si_mean"], lambda v: f"{v:.1f}", -1),
        ("Play calling", "Early-down pass rate vs. expected (neutral)", T["ed_proe"], lambda v: pct(v, True), 0),
        ("Play calling", "Early-down EPA per play (neutral)", T["ed_epa"], lambda v: f"{v:.3f}", 1),
        ("Play calling", "Pass rate inside the 3 on 1st/2nd down", T["gl_pass"], pct, 0),
        ("Play calling", "Pass rate inside the 3 vs. expected", T["gl_proe"], lambda v: pct(v, True), 0),
        ("Play calling", "Plays inside the 3 on 1st/2nd down", T["gl_n"], lambda v: f"{v:.0f}", 0),
        ("Play calling", "Play-action rate", T["pa_rate"], pct, 0),
        ("Play calling", "Play-action EPA boost", T["pa_gain"], lambda v: f"{v:+.3f}", 1),
        ("Play calling", "Pre-snap motion rate", T["motion"], pct, 0),
        ("Play calling", "Screen rate", T["screen"], pct, 0),
        ("Play calling", "Predictability (shotgun vs under-center pass gap)", T["predictability"], lambda v: pct(v), -1),
        ("Luck & expectations", "Beat-expectations grade", T["expect_grade"], lambda v: f"{v:.1f}", 1),
        ("Luck & expectations", "Wins vs. Vegas expectations", T["vs_vegas"], lambda v: f"{v:+.1f}", 1),
        ("Luck & expectations", "Vegas-expected wins", T["vegas_xw"], lambda v: f"{v:.1f}", 0),
        ("Luck & expectations", "Record in one-score games", T["close_w"] / T["close_n"],
         None, 1),
        ("Luck & expectations", "Fumble recovery rate (all fumbles in its games)", T["fumble_rec"], pct, 0),
    ]
    rows = []
    for o, (side, label, s, f, dirn) in enumerate(stats):
        s = s.reindex(T.index).astype(float)
        rk = s.rank(ascending=dirn < 0, method="min")
        for t in T.index:
            if f is None:
                w_, n_ = T.loc[t, "close_w"], T.loc[t, "close_n"]
                val = "" if pd.isna(n_) else f"{w_:g}-{n_ - w_:g}"
            else:
                val = "" if pd.isna(s[t]) else f(s[t])
            rows.append((season, scope, t, side, 200 + o, label, val, None if np.isnan(rk[t]) else int(rk[t])))
    out = pd.DataFrame(rows, columns=["season", "scope", "team", "side", "ord", "stat", "value", "rank"])
    out["rank"] = out["rank"].astype("Int64")
    return out


def wide(d, season, scope):
    T, _ = team_coaching(d)
    T = T.reset_index().rename(columns={"index": "team", "posteam": "team"})
    T["season"], T["scope"] = season, scope
    return T


def worst_punt_fact(d, season):
    _, w = team_coaching(d)
    if w is None:
        return None
    sd = int(w["score_differential"])
    sit = "tied" if sd == 0 else (f"down {-sd}" if sd < 0 else f"up {sd}")
    q = int(w["qtr"]) if "qtr" in w else None
    return dict(season=season, cat="Teams", icon="🏳️", title="Most timid punt",
                text=f"{w['posteam']} punted on 4th-and-{int(w['ydstogo'])} from "
                     f"{'the opponent' + chr(39) + 's ' + str(int(w['yardline_100'])) if w['yardline_100'] < 50 else ('midfield' if w['yardline_100'] == 50 else 'its own ' + str(int(100 - w['yardline_100'])))}"
                     f" while {sit} in the {'4th quarter' if q == 4 else f'Q{q}' if q else 'game'} (Week {int(w['week'])} vs {w['defteam']})."
                     f" Surrender score {surrender_index(w):.0f}, the most timid punt of the season.", team=w["posteam"])
