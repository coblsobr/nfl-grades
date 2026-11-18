import sys; sys.argv = ['x']
import pandas as pd, grades, roster
ps = pd.read_csv('output/player_season.csv'); out = []
import json as _json
CUR = _json.load(open('current.json'))
for season, thru in {2023: 18, 2024: 18, 2025: 18, CUR['season']: CUR['thru']}.items():
    d = grades.load_all(season, thru)
    out.append(roster.build(season, d, ps[(ps.season == season) & (ps.scope == 'regular')]))
pd.concat(out).to_csv('output/roster.csv', index=False); print('roster rebuilt')
