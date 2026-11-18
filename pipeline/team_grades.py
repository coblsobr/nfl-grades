"""
Team grades: offense, defense, special teams and overall, by season and by week.

Everything is measured in EPA (expected points added), so units add up:
    overall strength per game = offense EPA - defense EPA allowed + special teams EPA
Season grades are opponent-adjusted (beating up on bad teams counts for less).
Week grades are that game only, unadjusted.
Same scale as players: 70 = average team, each 10 points = 1 standard deviation.
"""
import numpy as np
import pandas as pd

ST_TYPES = ["kickoff", "punt", "field_goal", "extra_point"]


def _z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def _grade(score):
    return (70 + 10 * _z(score)).clip(30, 99.9).round(1)


def team_games(d):
    """One row per team per game with raw offense/defense/special-teams numbers."""
    p = d["pbp"]
    sc = (p.groupby("game_id")[["week", "home_team", "away_team", "home_score", "away_score"]]
          .last().reset_index())
    plays = p[((p["pass"] == 1) | (p["rush"] == 1)) & p["epa"].notna() & p["posteam"].notna()].copy()
    plays["is_pass"] = plays["pass"] == 1
    plays["pass_epa"] = np.where(plays["is_pass"], plays["epa"], np.nan)
    plays["rush_epa"] = np.where(~plays["is_pass"], plays["epa"], np.nan)
    plays["explosive"] = np.where(plays["is_pass"], plays["yards_gained"] >= 20, plays["yards_gained"] >= 10)
    agg = dict(plays=("epa", "size"), epa=("epa", "sum"), success=("success", "sum"),
               dropbacks=("is_pass", "sum"), pass_epa=("pass_epa", "sum"), rush_epa=("rush_epa", "sum"),
               explosive=("explosive", "sum"), sacks=("sack", "sum"), turnovers=("interception", "sum"))
    off = plays.groupby(["game_id", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    fum = p[(p["fumble_lost"] == 1) & p["posteam"].notna()].groupby(["game_id", "posteam"]).size()
    off["turnovers"] = off["turnovers"] + off.set_index(["game_id", "team"]).index.map(fum).fillna(0).values
    off["runs"] = off["plays"] - off["dropbacks"]

    st = p[p["play_type"].isin(ST_TYPES) & p["epa"].notna() & p["posteam"].notna()]
    st_for = st.groupby(["game_id", "posteam"])["epa"].sum()
    st_against = st.groupby(["game_id", "defteam"])["epa"].sum()

    g = off.merge(sc, on="game_id")
    g["opp"] = np.where(g["team"] == g["home_team"], g["away_team"], g["home_team"])
    g["pf"] = np.where(g["team"] == g["home_team"], g["home_score"], g["away_score"])
    g["pa"] = np.where(g["team"] == g["home_team"], g["away_score"], g["home_score"])
    opp = off.add_prefix("d_").rename(columns={"d_game_id": "game_id", "d_team": "opp"})
    g = g.merge(opp, on=["game_id", "opp"])
    key = list(zip(g["game_id"], g["team"]))
    g["st_epa"] = [st_for.get(k, 0) - st_against.get(k, 0) for k in key]

    # pressure (PFR charting): pressures our QBs faced / pressures our defense created
    pf_p = d["pfr_pass"].groupby(["game_id", "team"])["times_pressured"].sum()
    pf_d = d["pfr_def"].groupby(["game_id", "team"])["def_pressures"].sum()
    g["pressures_allowed"] = [pf_p.get(k, np.nan) for k in key]
    g["pressures_made"] = [pf_d.get(k, np.nan) for k in key]
    return g.drop(columns=["home_team", "away_team", "home_score", "away_score"])


def _adjust(g, off_col, def_col, w_off, w_def, iters=10):
    """Opponent adjustment on per-play rates: offense judged vs. the defenses it faced, and vice versa."""
    o = g.groupby("team")[off_col].sum() / g.groupby("team")[w_off].sum()
    dd = g.groupby("team")[def_col].sum() / g.groupby("team")[w_def].sum()
    lo, ld = o.mean(), dd.mean()
    ao, ad = o.copy(), dd.copy()
    for _ in range(iters):
        opp_def = g.assign(x=g["opp"].map(ad) - ld).groupby("team")["x"].mean()
        opp_off = g.assign(x=g["opp"].map(ao) - lo).groupby("team")["x"].mean()
        ao, ad = o - opp_def, dd - opp_off
    return o, dd, ao, ad


def season_grades(d):
    g = team_games(d)
    t = g.groupby("team").agg(games=("game_id", "nunique"), pf=("pf", "sum"), pa=("pa", "sum"),
                              wins=("pf", lambda s: 0), plays=("plays", "sum"), d_plays=("d_plays", "sum"),
                              dropbacks=("dropbacks", "sum"), d_dropbacks=("d_dropbacks", "sum"),
                              runs=("runs", "sum"), d_runs=("d_runs", "sum"),
                              st_epa=("st_epa", "sum"), turnovers=("turnovers", "sum"),
                              takeaways=("d_turnovers", "sum"),
                              pressures_allowed=("pressures_allowed", "sum"), pressures_made=("pressures_made", "sum"))
    res = g.assign(w=(g["pf"] > g["pa"]).astype(int), l=(g["pf"] < g["pa"]).astype(int), t_=(g["pf"] == g["pa"]).astype(int))
    t["wins"], t["losses"], t["ties"] = res.groupby("team")["w"].sum(), res.groupby("team")["l"].sum(), res.groupby("team")["t_"].sum()
    for name, oc, dc, wo, wd in [("epa", "epa", "d_epa", "plays", "d_plays"),
                                 ("pass", "pass_epa", "d_pass_epa", "dropbacks", "d_dropbacks"),
                                 ("rush", "rush_epa", "d_rush_epa", "runs", "d_runs"),
                                 ("succ", "success", "d_success", "plays", "d_plays")]:
        o, dd, ao, ad = _adjust(g, oc, dc, wo, wd)
        t[f"off_{name}_play"], t[f"def_{name}_play"] = o, dd
        t[f"off_{name}_adj"], t[f"def_{name}_adj"] = ao, ad
    t["st_epa_game"] = t["st_epa"] / t["games"]
    t["net_epa_game"] = (t["off_epa_adj"] * t["plays"] - t["def_epa_adj"] * t["d_plays"]) / t["games"] + t["st_epa_game"]
    t["pressure_rate_allowed"] = t["pressures_allowed"] / t["dropbacks"]
    t["pressure_rate_made"] = t["pressures_made"] / t["d_dropbacks"]
    t["turnover_margin"] = t["takeaways"] - t["turnovers"]
    t["point_diff"] = t["pf"] - t["pa"]
    t["off_grade"] = _grade(.7 * _z(t["off_epa_adj"]) + .3 * _z(t["off_succ_adj"]))
    t["pass_off_grade"] = _grade(t["off_pass_adj"])
    t["run_off_grade"] = _grade(t["off_rush_adj"])
    t["pass_pro_grade"] = _grade(-t["pressure_rate_allowed"])
    t["def_grade"] = _grade(-(.7 * _z(t["def_epa_adj"]) + .3 * _z(t["def_succ_adj"])))
    t["pass_def_grade"] = _grade(-t["def_pass_adj"])
    t["run_def_grade"] = _grade(-t["def_rush_adj"])
    t["pass_rush_grade"] = _grade(t["pressure_rate_made"])
    t["st_grade"] = _grade(t["st_epa_game"])
    t["overall_grade"] = _grade(t["net_epa_game"])
    for c in ["off", "def", "st", "overall"]:
        t[f"{c}_rank"] = t[f"{c}_grade"].rank(ascending=False, method="min").astype(int)
    return t.reset_index().sort_values("overall_grade", ascending=False).reset_index(drop=True)


def weekly_grades(d):
    """Each team-game graded against every other team that week (no opponent adjustment)."""
    g = team_games(d)
    g["off_epa_play"] = g["epa"] / g["plays"]
    g["def_epa_play"] = g["d_epa"] / g["d_plays"]
    g["off_succ"] = g["success"] / g["plays"]
    g["def_succ"] = g["d_success"] / g["d_plays"]
    g["net_epa"] = g["epa"] - g["d_epa"] + g["st_epa"]
    out = []
    for wk, x in g.groupby("week"):
        x = x.copy()
        x["off_grade"] = _grade(.7 * _z(x["off_epa_play"]) + .3 * _z(x["off_succ"]))
        x["def_grade"] = _grade(-(.7 * _z(x["def_epa_play"]) + .3 * _z(x["def_succ"])))
        x["st_grade"] = _grade(x["st_epa"])
        x["overall_grade"] = _grade(x["net_epa"])
        out.append(x)
    cols = ["week", "game_id", "team", "opp", "pf", "pa", "off_epa_play", "def_epa_play", "off_succ", "def_succ",
            "st_epa", "net_epa", "off_grade", "def_grade", "st_grade", "overall_grade"]
    return pd.concat(out)[cols].sort_values(["week", "overall_grade"], ascending=[True, False]).reset_index(drop=True)
