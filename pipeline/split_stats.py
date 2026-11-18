"""Passing vs. rushing team stats (offense and defense) with ranks, added to output/team_stats.csv (ord 300+)."""
import sys, json; sys.argv = ['x']
import numpy as np, pandas as pd, grades

def side_stats(pl, key, games):
    db = pl[(pl["pass"] == 1)]; ru = pl[(pl["rush"] == 1)]
    g = lambda df: df.groupby(key)
    att = db[db["pass_attempt"] == 1]
    out = {
        "pass": {
            "Yards per dropback": (g(db)["yards_gained"].sum() / g(db).size(), "r1"),
            "EPA per dropback": (g(db)["epa"].mean(), "f3"),
            "Pass success rate": (g(db)["success"].mean(), "%"),
            "Completion %": (g(att)["complete_pass"].mean(), "%"),
            "Passing TDs": (g(db)["pass_touchdown"].sum(), "n"),
            "Interceptions": (g(db)["interception"].sum(), "n"),
            "Big pass plays (20+ yds)": (g(db).apply(lambda x: (x["yards_gained"] >= 20).mean()), "%"),
        },
        "rush": {
            "Yards per carry": (g(ru)["yards_gained"].mean(), "r1"),
            "EPA per run": (g(ru)["epa"].mean(), "f3"),
            "Run success rate": (g(ru)["success"].mean(), "%"),
            "Rushing TDs": (g(ru)["rush_touchdown"].sum(), "n"),
            "Big runs (10+ yds)": (g(ru).apply(lambda x: (x["yards_gained"] >= 10).mean()), "%"),
            "Stuffed (0 or fewer yds)": (g(ru).apply(lambda x: (x["yards_gained"] <= 0).mean()), "%"),
        }}
    return out

def fmt(v, k):
    if pd.isna(v): return ""
    return {"r1": f"{v:.1f}", "f3": f"{v:.3f}", "%": f"{100 * v:.1f}%", "n": f"{v:.0f}"}[k]

# direction: 1 = higher is better for the offense; defense flips it
BETTER_LOW = {"Interceptions", "Stuffed (0 or fewer yds)"}
cur = json.load(open('current.json')); rows = []
for season, thru in {2023: 18, 2024: 18, 2025: 18, cur['season']: cur['thru']}.items():
    for po in ([False, True] if thru >= 18 else [False]):
        d = grades.load_all(season, thru, playoffs=po); sc = 'with_playoffs' if po else 'regular'
        p = d['pbp']; pl = p[((p['pass'] == 1) | (p['rush'] == 1)) & p['posteam'].notna() & p['epa'].notna()]
        games = p.groupby('posteam')['game_id'].nunique()
        ordn = 300
        for unit, key, flip in [("off", "posteam", 1), ("def", "defteam", -1)]:
            st = side_stats(pl, key, games)
            for kind, side in [("pass", "Pass offense" if unit == "off" else "Pass defense"), ("rush", "Run offense" if unit == "off" else "Run defense")]:
                for label, (ser, k) in st[kind].items():
                    lab = label if unit == "off" else {"Interceptions": "Interceptions made", "Passing TDs": "Passing TDs allowed", "Rushing TDs": "Rushing TDs allowed",
                                                         "Completion %": "Completion % allowed", "Stuffed (0 or fewer yds)": "Runs stuffed"}.get(label, label + " allowed")
                    dirn = (-1 if label in BETTER_LOW else 1) * flip
                    ser = ser.reindex(games.index).astype(float)
                    rk = ser.rank(ascending=dirn < 0, method="min")
                    for t in games.index:
                        rows.append((season, sc, t, side, ordn, lab, fmt(ser[t], k), None if np.isnan(rk[t]) else int(rk[t])))
                    ordn += 1
new = pd.DataFrame(rows, columns=["season", "scope", "team", "side", "ord", "stat", "value", "rank"])
new["rank"] = new["rank"].astype("Int64")
ts = pd.read_csv('output/team_stats.csv', dtype={'value': str}); ts = ts[ts['ord'] < 300]
pd.concat([ts, new]).to_csv('output/team_stats.csv', index=False)
x = new[(new.season == 2026) & (new.scope == 'regular') & (new.team == 'PIT')]
print(x[['side', 'stat', 'value', 'rank']].to_string(index=False))
