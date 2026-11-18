"""Power rankings: record, scoring margin, team efficiency and roster strength from our player ranks."""
import numpy as np
import pandas as pd

WEIGHTS = {"strength": .55, "win_pct": .15, "pd_pg": .15, "roster": .15}
H2H_GAP = 0.15   # teams this close in power score: the head-to-head winner goes ahead
STRENGTH = {"off_grade": .45, "def_grade": .45, "st_grade": .10}   # special teams matters, but far less
POS_WEIGHT = {"QB": 3.0, "K": .5, "P": .5}


def roster_strength(roster, ps):
    """Snap-weighted average of where each starter and rotation player ranks at his position (QBs count triple)."""
    p = ps[ps["scope"] == "regular"][["season", "pos", "gsis_id", "rank", "pool"]]
    r = roster.drop(columns=[c for c in ["rank", "grade", "pool"] if c in roster]).merge(p, on=["season", "pos", "gsis_id"], how="left")
    r["pctile"] = np.where(r["rank"].notna(), 1 - (r["rank"] - 1) / r["pool"], 0.25)
    r["share"] = r["snap_pct"].str.rstrip("%").astype(float) / 100
    r["w"] = r["pos"].map(POS_WEIGHT).fillna(1.0) * r["share"].clip(lower=.1)
    return (r["pctile"] * r["w"]).groupby([r["season"], r["team"]]).sum() / r["w"].groupby([r["season"], r["team"]]).sum()


def head_to_head(sched):
    """Wins by each team over each opponent in games already played."""
    sc = sched[(sched["game_type"] == "REG") & sched["home_score"].notna()]
    w = {}
    for _, g in sc.iterrows():
        if g["home_score"] == g["away_score"]:
            continue
        win, lose = (g["home_team"], g["away_team"]) if g["home_score"] > g["away_score"] else (g["away_team"], g["home_team"])
        w[(g["season"], win, lose)] = w.get((g["season"], win, lose), 0) + 1
    return w


def apply_h2h(g, season, h2h):
    """Close calls go to the team that won the head-to-head game(s)."""
    order = list(g.sort_values("power_score", ascending=False)["team"])
    sc = g.set_index("team")["power_score"]
    for _ in range(5):
        moved = False
        for i in range(len(order) - 1):
            a, b = order[i], order[i + 1]
            if sc[a] - sc[b] < H2H_GAP and h2h.get((season, b, a), 0) > h2h.get((season, a, b), 0):
                order[i], order[i + 1] = b, a
                moved = True
        if not moved:
            break
    return {t: i + 1 for i, t in enumerate(order)}


MARKET_GAMES = 4   # the betting market's view counts like this many games of our own numbers


def market_ratings(sched, season, thru_week):
    """Team strength implied by the point spreads (games through thru_week plus the next week's lines).
    Early in a season this is the best free read on who's good before our stats have much to go on."""
    sc = sched[(sched["season"] == season) & (sched["game_type"] == "REG") & (sched["week"] <= thru_week + 1) & sched["spread_line"].notna()]
    teams = sorted(set(sc["home_team"]) | set(sc["away_team"]))
    if not len(sc):
        return pd.Series(0.0, index=teams)
    ix = {t: i for i, t in enumerate(teams)}
    A = np.zeros((len(sc) + len(teams), len(teams) + 1)); y = np.zeros(len(sc) + len(teams))
    for i, (h, a, sp) in enumerate(zip(sc["home_team"], sc["away_team"], sc["spread_line"])):
        A[i, ix[h]], A[i, ix[a]], A[i, -1], y[i] = 1, -1, 1, sp
    for j in range(len(teams)):                       # light ridge so a team with one line stays near average
        A[len(sc) + j, j] = 0.3
    r = np.linalg.lstsq(A, y, rcond=None)[0][:-1]
    return pd.Series(r - r.mean(), index=teams)


def blend_market(g, sched, season, thru_week, games):
    w = games / (games + MARKET_GAMES)
    mk = market_ratings(sched, season, thru_week).reindex(g["team"]).fillna(0).values
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    return w * z(g["power_score"].values) + (1 - w) * z(mk)


def power(ts, roster, ps, sched=None):
    t = ts.copy()
    t["games"] = t["wins"] + t["losses"] + t["ties"]
    t["win_pct"] = (t["wins"] + .5 * t["ties"]) / t["games"]
    t["pd_pg"] = t["point_diff"] / t["games"]
    t["efficiency"] = t["net_epa_game"]
    rs = roster_strength(roster, ps)
    t["roster"] = [rs.get((s, tm), np.nan) for s, tm in zip(t["season"], t["team"])]
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    out = []
    for _, g in t.groupby(["season", "scope"]):
        g = g.copy()
        g["strength"] = sum(w * z(g[k]) for k, w in STRENGTH.items())
        g["power_score"] = sum(w * z(g[k]) for k, w in WEIGHTS.items())
        if sched is not None:
            rk = apply_h2h(g, g["season"].iat[0], head_to_head(sched))
            g["power_rank"] = g["team"].map(rk)
        else:
            g["power_rank"] = g["power_score"].rank(ascending=False, method="min").astype(int)
        g["roster_rank"] = g["roster"].rank(ascending=False, method="min").astype(int)
        out.append(g)
    return pd.concat(out)


def schedule_strength(pw, sched):
    """Strength of schedule from our power scores: games already played and games still to come."""
    sc = sched[sched["game_type"] == "REG"]
    g = pd.concat([sc.assign(team=sc["home_team"], opp=sc["away_team"], ha="Home", pf=sc["home_score"], pa=sc["away_score"]),
                   sc.assign(team=sc["away_team"], opp=sc["home_team"], ha="Away", pf=sc["away_score"], pa=sc["home_score"])])
    g = g[["season", "week", "team", "opp", "ha", "pf", "pa"]].copy()
    p = pw[pw["scope"] == "regular"][["season", "team", "power_score", "power_rank", "wins", "losses", "ties"]]
    g = g.merge(p.rename(columns={"team": "opp", "power_score": "opp_score", "power_rank": "opp_power_rank", "wins": "ow", "losses": "ol", "ties": "ot"}),
                on=["season", "opp"], how="left")
    g["played"] = g["pf"].notna()
    g["result"] = np.where(g["played"], np.where(g["pf"] > g["pa"], "W ", np.where(g["pf"] < g["pa"], "L ", "T "))
                           + g["pf"].fillna(0).astype(int).astype(str) + "-" + g["pa"].fillna(0).astype(int).astype(str), "")
    rows = []
    for (season, team), x in g.groupby(["season", "team"]):
        a, b = x[x["played"]], x[~x["played"]]
        rec = lambda y: f"{int(y['ow'].sum())}-{int(y['ol'].sum())}" + (f"-{int(y['ot'].sum())}" if y["ot"].sum() else "")
        rows.append(dict(season=season, team=team, sos=a["opp_score"].mean(), sos_left=b["opp_score"].mean() if len(b) else np.nan,
                         opp_record=rec(a) if len(a) else "", left_record=rec(b) if len(b) else "", games_left=len(b)))
    s = pd.DataFrame(rows)
    s["sos_rank"] = s.groupby("season")["sos"].rank(ascending=False, method="min").astype(int)
    s["sos_left_rank"] = s.groupby("season")["sos_left"].rank(ascending=False, method="min").astype("Int64")
    return s, g


def weekly_power(d, sched, season, max_week, pre=None):
    """Power ranking as it stood after each week: our team numbers through that week, blended with the
    market's view (which fades as games pile up). Roster strength isn't used here (no week-by-week rosters)."""
    import grades, team_grades as tgm
    z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
    out = []
    for N in range(1, max_week + 1):
        t = tgm.season_grades(grades.subset_weeks(d, 1, N, season))
        gp = t["wins"] + t["losses"] + t["ties"]
        strength = .45 * z(t["off_grade"]) + .45 * z(t["def_grade"]) + .10 * z(t["st_grade"])
        t["power_score"] = .65 * z(strength) + .175 * z((t["wins"] + .5 * t["ties"]) / gp) + .175 * z(t["point_diff"] / gp)
        if pre is not None:
            import prior as _pr
            w = float(gp.median()) / (float(gp.median()) + _pr.PRIOR_GAMES)
            t["power_score"] = w * z(t["power_score"]) + (1 - w) * t["team"].map(pre).fillna(0).values
        t["power_rank_wk"] = t["power_score"].rank(ascending=False, method="min").astype(int)
        t["record_wk"] = t["wins"].astype(str) + "-" + t["losses"].astype(str) + np.where(t["ties"] > 0, "-" + t["ties"].astype(str), "")
        out.append(t[["team", "power_rank_wk", "record_wk", "power_score"]].assign(season=season, week=N))
    if pre is not None:                      # Week 0: the preseason view on its own
        t0 = pd.DataFrame({"team": pre.index, "power_rank_wk": pre.rank(ascending=False, method="min").astype(int).values, "record_wk": "0-0",
                           "power_score": pre.values})
        out.insert(0, t0.assign(season=season, week=0))
    return pd.concat(out)
