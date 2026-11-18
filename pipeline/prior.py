"""Week 0 (preseason) team strength from our own data only:
   half = how good the team was last season (our final power score, playoffs included),
   half = who they're starting this year (Week 1 depth chart), rated by each starter's rank last season.
   No betting lines. It fades as real games come in (worth PRIOR_GAMES games)."""
import numpy as np, pandas as pd

PRIOR_GAMES = 4
SLOTS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OT": 2, "OG": 2, "C": 1, "EDGE": 2, "IDL": 2, "LB": 2, "CB": 3, "S": 2}
POSW = {"QB": 3.0}
GROUP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "LT": "OT", "RT": "OT", "T": "OT", "OT": "OT", "LG": "OG", "RG": "OG", "G": "OG", "OG": "OG",
         "C": "C", "LDE": "EDGE", "RDE": "EDGE", "DE": "EDGE", "EDGE": "EDGE", "OLB": "EDGE", "LOLB": "EDGE", "ROLB": "EDGE", "RUSH": "EDGE",
         "LDT": "IDL", "RDT": "IDL", "DT": "IDL", "NT": "IDL", "DL": "IDL", "MLB": "LB", "WLB": "LB", "SLB": "LB", "LILB": "LB", "RILB": "LB",
         "ILB": "LB", "LB": "LB", "MIKE": "LB", "WILL": "LB", "SAM": "LB", "LCB": "CB", "RCB": "CB", "CB": "CB", "NB": "CB", "NCB": "CB",
         "FS": "S", "SS": "S", "S": "S"}
z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)


def week1_starters(d, sched, season):
    dc = d["depth"].copy()
    if dc is None or not len(dc):
        return pd.DataFrame()
    if "dt" in dc:
        first = str(sched[(sched["season"] == season) & (sched["game_type"] == "REG")]["gameday"].min())
        dc = dc[dc["dt"].astype(str) < first + "T23:59"]
        dc = dc[dc["dt"] == dc.groupby("team")["dt"].transform("max")]
    elif "week" in dc:
        dc = dc[dc["week"] == dc["week"].min()]
    dc["grp"] = dc["pos_abb"].map(GROUP)
    dc = dc[dc["grp"].notna() & dc["gsis_id"].notna()].sort_values(["team", "grp", "pos_rank"])
    dc = dc.drop_duplicates(["team", "gsis_id"])
    return pd.concat([g.head(SLOTS[grp]) for (t, grp), g in dc.groupby(["team", "grp"])])


def roster_prior(d, sched, season, ps_prev):
    st = week1_starters(d, sched, season)
    if not len(st):
        return pd.Series(dtype=float)
    p = ps_prev[ps_prev["scope"] == "regular"].copy()
    p["pct"] = np.where(p["rank"].notna(), 1 - (p["rank"] - 1) / p["pool"], np.nan)
    best = p.groupby("gsis_id")["pct"].max()
    st["pct"] = st["gsis_id"].map(best).fillna(0.30)            # rookies / no 2025 sample: a below-average starter
    st["w"] = st["grp"].map(POSW).fillna(1.0)
    return (st["pct"] * st["w"]).groupby(st["team"]).sum() / st["w"].groupby(st["team"]).sum()


def preseason(season, d, sched, team_power_prev, ps_prev):
    """Score per team (z-scored). None if there is no previous season to build from."""
    if team_power_prev is None or not len(team_power_prev):
        return None
    tp = team_power_prev
    scope = "with_playoffs" if (tp["scope"] == "with_playoffs").any() else "regular"
    last = tp[tp["scope"] == scope].set_index("team")["power_score"]
    ro = roster_prior(d, sched, season, ps_prev)
    teams = sorted(set(last.index) | set(ro.index))
    last = last.reindex(teams).fillna(last.mean()); ro = ro.reindex(teams).fillna(ro.mean())
    return z(.5 * z(last) + .5 * z(ro))
