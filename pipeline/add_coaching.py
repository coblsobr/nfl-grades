"""One-off: add coaching rows to existing team_stats.csv and write coaching.csv without a full rebuild."""
import sys; sys.argv = ['x']
import pandas as pd, grades, coaching
tst = pd.read_csv('output/team_stats.csv', dtype={'value': str})
tst = tst[tst['ord'] < 200]
rows, wide = [tst], []
for season, thru in {2023: 18, 2024: 18, 2025: 18, 2026: 7}.items():
    for po in ([False, True] if thru >= 18 else [False]):
        d = grades.load_all(season, thru, playoffs=po)
        sc = 'with_playoffs' if po else 'regular'
        rows.append(coaching.team_rows(d, season, sc))
        wide.append(coaching.wide(d, season, sc))
out = pd.concat(rows); out.to_csv('output/team_stats.csv', index=False)
pd.concat(wide).to_csv('output/coaching.csv', index=False)
print(len(out), out[out.ord >= 200].groupby('season').size().to_dict())
