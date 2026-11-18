import sys, json; sys.argv = ['x']
import pandas as pd, grades, coaching
CUR = json.load(open('current.json')); wide = []
for season, thru in {2023: 18, 2024: 18, 2025: 18, CUR['season']: CUR['thru']}.items():
    for po in ([False, True] if thru >= 18 else [False]):
        wide.append(coaching.wide(grades.load_all(season, thru, playoffs=po), season, 'with_playoffs' if po else 'regular'))
pd.concat(wide).to_csv('output/coaching.csv', index=False); print('coaching wide rebuilt')
