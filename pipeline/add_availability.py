"""Mark who is available now (2026), add anyone who recently took over a starting job, and score each team's
current lineup vs. its full-strength lineup (used by the power rankings)."""
import sys; sys.argv = ['x']
import numpy as np, pandas as pd, availability as av
SLOTS = {"QB": 1, "RB": 1, "WR": 3, "TE": 1, "OT": 2, "OG": 2, "C": 1, "EDGE": 2, "IDL": 2, "LB": 2, "CB": 3, "S": 2}
SIDE = {p: ("off" if p in ("QB", "RB", "WR", "TE", "OT", "OG", "C") else "def") for p in SLOTS}
POSW = {"QB": 3.0}
ro = pd.read_csv('output/roster.csv'); ps = pd.read_csv('output/player_season.csv')
out, recent, wk = av.current(2026)
ro["avail"] = 1; ro["status"] = ""; ro["recent"] = ro["snap_pct"].str.rstrip("%").astype(float) / 100
cur = ro["season"] == 2026
ro.loc[cur, "recent"] = ro.loc[cur, "gsis_id"].map(recent).fillna(0)
ro.loc[cur, "status"] = ro.loc[cur, "gsis_id"].map(out).fillna("")
ro.loc[cur & (ro["status"] != ""), "avail"] = 0
# players who recently took over (25%+ of snaps in the last 3 games) but weren't on the season roster
p26 = ps[(ps.season == 2026) & (ps.scope == "regular")].sort_values("ranked", ascending=False).drop_duplicates("gsis_id").set_index("gsis_id")
have = set(ro.loc[cur, "gsis_id"]); add = []
for g, rv in recent.items():
    if rv >= .25 and g not in have and g in p26.index and p26.loc[g, "pos"] in SLOTS:
        x = p26.loc[g]
        add.append(dict(key=f"2026|{x.team}|{g}", season=2026, team=x.team, unit="Offense" if SIDE[x.pos] == "off" else "Defense", pos=x.pos,
                        player=x.player, gsis_id=g, role="Rotation", starts=0, games=0, snap_pct=f"{round(100*rv)}%", grade=x.grade,
                        rank=x["rank"], ord=999, avail=0 if g in out else 1, status=out.get(g, ""), recent=rv))
ro = pd.concat([ro, pd.DataFrame(add)], ignore_index=True)
ro["role0"] = ro["role"]
ro.loc[(ro["avail"] == 0), "role"] = "Out"
ro["cur_ord"] = ro.groupby(["season", "team", "pos"])["recent"].rank(ascending=False, method="first")
ro.drop(columns=['role0']).to_csv('output/roster.csv', index=False)
# lineup scores: full strength (season snaps) vs. now (healthy players, recent snaps)
pool = ps[ps.scope == "regular"].groupby(["season", "pos"])["pool"].max()
def pct(row):
    pl = pool.get((row.season, row.pos))
    return 1 - (row["rank"] - 1) / pl if pd.notna(row["rank"]) and pl else 0.25
ro["pctile"] = ro.apply(pct, axis=1)
rows = []
for (s, t), g in ro[ro.pos.isin(SLOTS)].groupby(["season", "team"]):
    for mode in ["full", "cur"]:
        gg = g if mode == "full" else g[g["avail"] == 1]
        key = "snap_pct" if mode == "full" else "recent"
        gg = gg.assign(k=(gg["snap_pct"].str.rstrip("%").astype(float) / 100 + (gg["role0"] == "Starter")) if mode == "full" else gg["recent"])
        for side in ["off", "def"]:
            sel = pd.concat([gg[gg.pos == p].nlargest(n, "k") for p, n in SLOTS.items() if SIDE[p] == side])
            w = sel["pos"].map(POSW).fillna(1.0)
            rows.append(dict(season=s, team=t, side=side, mode=mode, score=float((sel["pctile"] * w).sum() / w.sum()) if len(sel) else np.nan))
ls = pd.DataFrame(rows).pivot_table(index=["season", "team", "side"], columns="mode", values="score").reset_index()
ls.to_csv('output/lineup_scores.csv', index=False)
x = ls[(ls.season == 2026)].assign(delta=lambda d: d["cur"] - d["full"]).sort_values("delta")
print(x.head(6).round(3).to_string(index=False))
print('added recent starters:', len(add), '| out marked:', int((ro.season.eq(2026) & ro.avail.eq(0)).sum()))
