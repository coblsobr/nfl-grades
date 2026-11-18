"""Fun facts and 'did you know' cards for the site, one set per season, all computed from the data."""
import numpy as np
import pandas as pd

POSNAME = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "OT": "tackle", "OG": "guard", "C": "center",
           "EDGE": "edge rusher", "IDL": "interior lineman", "LB": "linebacker", "CB": "cornerback", "S": "safety",
           "K": "kicker", "P": "punter"}


def _name(ps, season, pos, gid):
    r = ps[(ps["season"] == season) & (ps["pos"] == pos) & (ps["gsis_id"] == gid)]
    return (r["player"].iat[0], r["team"].iat[0]) if len(r) else (gid, "")


def build(season, d, ps, pw, ts, tst, box):
    """ps/pw/ts/tst: this season's player_season (regular), player_week, team_season (regular), team_stats rows.
    box: stats.box_table for the regular season (indexed by gsis_id)."""
    F = []
    add = lambda cat, icon, title, text, team="": F.append(dict(season=season, cat=cat, icon=icon, title=title,
                                                                    text=text, team=team))
    ps = ps[ps["scope"] == "regular"]
    reg = ps[ps["ranked"] == 1] if "ranked" in ps else ps[ps["reg"] == 1]
    t = ts.set_index("team")
    # --- teams
    gap = (t["off_rank"] - t["def_rank"])
    tm = gap.idxmax()
    add("Teams", "🧱", "Defense carrying the load",
        f"{tm} has the #{t.loc[tm, 'def_rank']:.0f} defense but only the #{t.loc[tm, 'off_rank']:.0f} offense. No team leans harder on its defense.", tm)
    tm = (-gap).idxmax()
    add("Teams", "🎯", "All offense",
        f"{tm} has the #{t.loc[tm, 'off_rank']:.0f} offense but only the #{t.loc[tm, 'def_rank']:.0f} defense, the most offense-first team in the league.", tm)
    st = tst[(tst["scope"] == "regular")]
    def team_stat(label):
        x = st[st["stat"] == label].set_index("team")
        return x
    luck = team_stat("Wins above expected (luck)")
    if len(luck):
        v = luck["value"].str.replace("+", "", regex=False).astype(float)
        add("Teams", "🍀", "Luckiest team", f"{v.idxmax()} has won {v.max():.1f} more games than its point differential predicts. Close games have broken its way.", v.idxmax())
        add("Teams", "😬", "Unluckiest team", f"{v.idxmin()} has won {abs(v.min()):.1f} fewer games than its point differential predicts. It's better than its record.", v.idxmin())
    proe = team_stat("Pass rate over expected")
    if len(proe):
        v = proe["value"].str.replace("%", "").str.replace("+", "", regex=False).astype(float)
        add("Teams", "🛩️", "Air raid", f"{v.idxmax()} throws {v.max():.1f}% more often than the situation calls for, the most pass-happy team in football.", v.idxmax())
        add("Teams", "🚜", "Ground and pound", f"{v.idxmin()} runs it more than any team: it passes {abs(v.min()):.1f}% less often than expected.", v.idxmin())
    go = team_stat("Go-for-it rate on 4th and 1-2")
    if len(go):
        v = go["value"].str.replace("%", "").astype(float)
        add("Teams", "🎲", "Fourth-down gamblers", f"{v.idxmax()} goes for it on {v.max():.0f}% of 4th-and-short tries, the boldest coach in the league.", v.idxmax())
    add("Teams", "⚡", "Best special teams", f"{t['st_grade'].idxmax()} has the #1 special teams, adding {t.loc[t['st_grade'].idxmax(), 'st_epa_game']:.1f} points a game on kicks and returns.", t["st_grade"].idxmax())
    # --- coaching and luck
    try:
        import coaching
        T, _ = coaching.team_coaching(d)
        a = T["go_rate_oe"].idxmax(); c = T["go_rate_oe"].idxmin()
        add("Coaching", "🦁", "Most aggressive on 4th down", f"{a} goes for it {100 * T.loc[a, 'go_rate_oe']:.0f}% more often than the league does in the same spots ({T.loc[a, 'go_extra']:+.0f} extra tries).", a)
        add("Coaching", "🐢", "Most conservative on 4th down", f"{c} goes for it {abs(100 * T.loc[c, 'go_rate_oe']):.0f}% less often than the league does in the same spots.", c)
        v = T["vs_vegas"].idxmax(); u = T["vs_vegas"].idxmin()
        add("Coaching", "📈", "Beating the odds", f"{v} has {T.loc[v, 'wins']:g} wins; Vegas lines said about {T.loc[v, 'vegas_xw']:.1f}. Nobody has outrun expectations more.", v)
        add("Coaching", "📉", "Falling short", f"{u} has {T.loc[u, 'wins']:g} wins; Vegas lines said about {T.loc[u, 'vegas_xw']:.1f}, the biggest shortfall in the league.", u)
        g = T[T["gl_n"] >= 3]["gl_proe"]
        if len(g):
            t = g.idxmax()
            add("Coaching", "🎯", "Throwing at the goal line", f"{t} passes on {100 * T.loc[t, 'gl_pass']:.0f}% of 1st and 2nd downs inside the 3, the most pass-happy goal-line offense ({T.loc[t, 'gl_n']:.0f} plays).", t)
        wp = coaching.worst_punt_fact(d, season)
        if wp:
            add(wp["cat"], wp["icon"], wp["title"], wp["text"], wp["team"])
    except Exception as e:
        print("coaching facts skipped:", e)
    # --- games
    try:
        import team_grades as tgm
        g = tgm.team_games(d)
        hi = g.loc[g["pf"].idxmax()]
        add("Games", "🎆", "Highest-scoring team game", f"{hi['team']} put up {hi['pf']:.0f} points on {hi['opp']} in Week {hi['week']:.0f}, the most by any team this season.", hi["team"])
        g["margin"] = g["pf"] - g["pa"]
        bw = g.loc[g["margin"].idxmax()]
        add("Games", "💨", "Biggest blowout", f"{bw['team']} beat {bw['opp']} {bw['pf']:.0f}-{bw['pa']:.0f} in Week {bw['week']:.0f}, a {bw['margin']:.0f}-point win.", bw["team"])
        close = g[(g["margin"].abs() <= 3) & (g["margin"] > 0)].groupby("team").size()
        if len(close):
            add("Games", "😰", "Cardiac kids", f"{close.idxmax()} has won {close.max()} games by 3 points or fewer, the most nail-biter wins in the league.", close.idxmax())
    except Exception:
        pass
    # --- players
    if len(pw):
        pwr = pw[pw["rank"].notna()]
        hi = pwr[pwr["rank"] <= 3].groupby(["pos", "gsis_id"]).size().sort_values(ascending=False)
        if len(hi):
            (pos, gid), n = hi.index[0], hi.iloc[0]
            nm, tmm = _name(ps, season, pos, gid)
            add("Players", "🔥", "Most dominant games", f"{nm} ({POSNAME[pos]}, {tmm}) has ranked top 3 at his position in {n} games, the most in the league.", tmm)
        weeks = pw["week"].nunique()
        need = max(4, int(0.6 * weeks))
        pwr = pw[pw["rank"].notna()].assign(pct=lambda x: x["rank"] / x["pool"])
        cnt = pwr.groupby(["pos", "gsis_id"])["pct"].agg(["size", "max"])
        cnt = cnt[cnt["size"] >= need]
        if len(cnt):
            (pos, gid) = cnt["max"].idxmin()
            nm, tmm = _name(ps, season, pos, gid)
            row = pwr[(pwr["pos"] == pos) & (pwr["gsis_id"] == gid)]
            add("Players", "📏", "Mr. Consistent", f"{nm} ({POSNAME[pos]}, {tmm}) hasn't had an off week: his worst game still ranked #{row['rank'].max():.0f} of {row['pool'].median():.0f} at his position.", tmm)
        if weeks >= 4:
            half = sorted(pw["week"].unique())[weeks // 2]
            pwr = pw[pw["rank"].notna()].assign(pct=lambda x: 100 * (1 - x["rank"] / x["pool"]))
            a = pwr[pwr["week"] < half].groupby(["pos", "gsis_id"])["pct"].agg(["mean", "size"])
            b = pwr[pwr["week"] >= half].groupby(["pos", "gsis_id"])["pct"].agg(["mean", "size"])
            j = a.join(b, lsuffix="_a", rsuffix="_b").dropna()
            j = j[(j["size_a"] >= 2) & (j["size_b"] >= 2)]
            j = j[j.index.isin(list(zip(reg["pos"], reg["gsis_id"])))]
            if len(j):
                dlt = j["mean_b"] - j["mean_a"]
                (pos, gid) = dlt.idxmax()
                nm, tmm = _name(ps, season, pos, gid)
                add("Players", "📈", "Heating up", f"{nm} ({POSNAME[pos]}, {tmm}) went from beating {j.loc[(pos, gid), 'mean_a']:.0f}% of his position each week early on to beating {j.loc[(pos, gid), 'mean_b']:.0f}% since Week {half}.", tmm)
    bx = box.join(ps.drop_duplicates("gsis_id").set_index("gsis_id")[["player", "team", "pos"]], how="inner")
    def leader(col, icon, title, fmt, unit, pos_filter=None):
        x = bx if pos_filter is None else bx[bx["pos"].isin(pos_filter)]
        x = x[x[col].notna()]
        if not len(x) or x[col].max() <= 0:
            return
        r = x.loc[x[col].idxmax()]
        ties = int((x[col] == x[col].max()).sum()) - 1
        txt = fmt.format(n=r["player"], t=r["team"], v=r[col])
        if ties:
            txt = f"{r['player']} ({r['team']}) and {ties} other player{'s' if ties > 1 else ''} share the league lead with {r[col]:.0f} {unit}."
        add("Players", icon, title, txt, r["team"])
    leader("def_pressures", "💥", "Pressure machine", "{n} ({t}) leads the league with {v:.0f} QB pressures.", "pressures")
    leader("def_interceptions", "🧲", "Ballhawk", "{n} ({t}) has {v:.0f} interceptions, more than anyone.", "interceptions")
    leader("def_tackles_combined", "🛑", "Tackling machine", "{n} ({t}) leads all defenders with {v:.0f} tackles.", "tackles")
    leader("broken_tackles", "🌀", "Hard to bring down", "{n} ({t}) has broken {v:.0f} tackles, the most of any player.", "broken tackles")
    leader("fg_long", "🦵", "Longest field goal", "{n} ({t}) hit the season's longest field goal: {v:.0f} yards.", "yards")
    leader("receiving_yards_after_catch", "🏃", "YAC king", "{n} ({t}) leads the league with {v:.0f} yards after the catch.", "yards after the catch")
    # most / least targeted corners among regulars
    cb = reg[reg["pos"] == "CB"]
    cbx = bx[bx.index.isin(cb["gsis_id"]) & (bx["defense_snaps"] > 0)]
    if len(cbx):
        rate = (cbx["def_targets"] / cbx["defense_snaps"]).dropna()
        if len(rate):
            gid = rate.idxmin()
            add("Players", "🚫", "Nobody throws at him", f"{cbx.loc[gid, 'player']} ({cbx.loc[gid, 'team']}) is targeted once every {1 / rate.min():.0f} snaps, the least of any starting corner.", cbx.loc[gid, "team"])
    df = pd.DataFrame(F)
    df["id"] = [f"{season}-{i}" for i in range(len(df))]
    return df
