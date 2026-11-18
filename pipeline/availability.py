"""Who is actually available right now: IR/reserve lists, the latest injury report, players who have
missed their team's last two games, plus a small hand-entered list for news newer than the data."""
import pandas as pd, numpy as np
import nflreadpy as nfl
import grades

OUT_STATUSES = {"Out", "Doubtful"}


def current(season=2026):
    rw = grades.cached(f"rosters_weekly_{season}", lambda: nfl.load_rosters_weekly([season]))
    inj = grades.cached(f"injuries_{season}_fresh", lambda: nfl.load_injuries([season]))
    sn = grades.cached(f"snaps_{season}_fresh", lambda: nfl.load_snap_counts([season]))
    sn = sn[sn["game_type"] == "REG"]
    wk = int(rw["week"].max())
    out = {}
    r = rw[rw["week"] == wk]
    for _, x in r[r["status"].isin(["RES"])].iterrows():
        out[x["gsis_id"]] = "IR / reserve"
    iw = int(inj["week"].max())
    for _, x in inj[(inj["week"] == iw) & inj["report_status"].isin(OUT_STATUSES)].iterrows():
        out.setdefault(x["gsis_id"], f"{x['report_status']} ({x['report_primary_injury']})")
    # missed the team's last two games after playing earlier
    games = sn.groupby("team")["week"].apply(lambda s: sorted(s.unique()))
    played = sn.assign(any_=(sn["offense_snaps"].fillna(0) + sn["defense_snaps"].fillna(0)) > 0)
    pid = played.groupby(["team", "pfr_player_id"]).apply(lambda g: set(g.loc[g["any_"], "week"]))
    gmap = rw.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    for (team, pfr), weeks in pid.items():
        last2 = games[team][-2:]
        if weeks and not (set(last2) & weeks) and len(games[team]) >= 3:
            g = gmap.get(pfr)
            if g: out.setdefault(g, "Missed last 2 games")
    man = pd.read_csv("injuries_manual.csv")
    names = rw[rw["week"] == wk].set_index(["team", "full_name"])["gsis_id"]
    for _, x in man[man["season"] == season].iterrows():
        g = names.get((x["team"], x["player"]))
        if g is not None:
            out[g] = f"{x['status']} ({x['note']})"
    # recent snap share: last 3 games each team played
    last3 = {t: set(w[-3:]) for t, w in games.items()}
    s3 = sn[[w in last3[t] for t, w in zip(sn["team"], sn["week"])]]
    tot = s3.groupby(["team", "game_id"])[["offense_snaps", "defense_snaps"]].max().groupby("team").sum()
    a = s3.groupby(["team", "pfr_player_id"])[["offense_snaps", "defense_snaps"]].sum().reset_index()
    a["recent"] = np.maximum(a["offense_snaps"] / a["team"].map(tot["offense_snaps"]), a["defense_snaps"] / a["team"].map(tot["defense_snaps"]))
    a["gsis_id"] = a["pfr_player_id"].map(gmap)
    recent = a.dropna(subset=["gsis_id"]).groupby("gsis_id")["recent"].max()
    return out, recent, wk
