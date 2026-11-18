"""Per-game stat lines for every player (for the player page's week picker), and our pick record."""
import sys, json; sys.argv = ['x']
import numpy as np, pandas as pd, grades, nflreadpy as nfl
cur = json.load(open('current.json'))
COLS = {"completions": "Comp", "attempts": "Att", "passing_yards": "Pass yds", "passing_tds": "Pass TD", "passing_interceptions": "INT",
        "sacks_suffered": "Sacked", "carries": "Carries", "rushing_yards": "Rush yds", "rushing_tds": "Rush TD", "targets": "Targets",
        "receptions": "Catches", "receiving_yards": "Rec yds", "receiving_tds": "Rec TD", "receiving_yards_after_catch": "YAC",
        "tackles": "Tackles", "def_tackles_for_loss": "TFL", "def_sacks": "Sacks", "def_qb_hits": "QB hits", "def_interceptions": "INTs",
        "def_pass_defended": "Passes broken up", "fg_made": "FG made", "fg_att": "FG att", "pat_made": "XP made", "pt_att": "Punts"}
for season, thru in {2023: 18, 2024: 18, 2025: 18, cur['season']: cur['thru']}.items():
    s = grades.cached(f"pstats_{season}", lambda: nfl.load_player_stats([season], summary_level="week"))
    s = s[(s["season_type"] == "REG") & (s["week"] <= thru)].copy()
    s["tackles"] = s["def_tackles_solo"].fillna(0) + s["def_tackle_assists"].fillna(0)
    keep = s[["week", "player_id", "team", "opponent_team"] + [c for c in COLS if c in s]].rename(columns={"player_id": "gsis_id", "opponent_team": "opp", **COLS})
    num = [v for v in COLS.values() if v in keep]
    keep[num] = keep[num].fillna(0).round(1)
    keep = keep[(keep[num] != 0).any(axis=1)]
    for c in num:
        keep[c] = keep[c].map(lambda v: f"{v:g}")
    keep["key"] = keep["week"].astype(str) + "|" + keep["gsis_id"]
    keep.drop_duplicates("key").to_csv(f"output/player_week_stats_{season}.csv", index=False)
    print(season, len(keep), "stat lines")
# ---- pick record: every game where both teams had a power score the week before ----
sched = grades.cached('schedules_all', lambda: nfl.load_schedules([2023, 2024, 2025, 2026]))
sched = sched[(sched['game_type'] == 'REG') & sched['home_score'].notna()]
wp = pd.read_csv('output/team_week_power.csv')
score = {(r.season, r.week, r.team): r.power_score for r in wp.itertuples()}
rows = []
for g in sched.itertuples():
    if g.season == cur['season'] and g.week > cur['thru']:
        continue
    sh, sa = score.get((g.season, g.week - 1, g.home_team)), score.get((g.season, g.week - 1, g.away_team))
    if sh is None or sa is None or g.home_score == g.away_score:
        continue
    pick_home = 4.1 * (sh - sa) + 2.0 > 0                     # same line as the app: score gap + home field
    rows.append(dict(season=g.season, week=g.week, right=int(pick_home == (g.home_score > g.away_score))))
r = pd.DataFrame(rows)
rec = r.groupby(["season", "week"])["right"].agg(["sum", "size"]).reset_index().rename(columns={"sum": "won", "size": "games"})
rec["lost"] = rec["games"] - rec["won"]
tot = rec.groupby("season")[["won", "lost"]].sum().reset_index().assign(week=0)   # week 0 row = whole season
out = pd.concat([tot, rec[["season", "week", "won", "lost"]]])
out["pct"] = (100 * out["won"] / (out["won"] + out["lost"])).round(1).map(lambda v: f"{v:.1f}%")
out["record"] = out["won"].astype(str) + "-" + out["lost"].astype(str)
out["key"] = out["season"].astype(str) + "|" + out["week"].astype(str)
out[["key", "season", "week", "record", "pct", "won", "lost"]].sort_values(["season", "week"]).to_csv("output/pick_record.csv", index=False)
print(out[out.week == 0][["season", "record", "pct"]].to_string(index=False))
