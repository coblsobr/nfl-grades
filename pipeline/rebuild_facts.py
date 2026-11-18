import sys; sys.argv=['x']
import pandas as pd, grades, stats as st, facts as fx
ps=pd.read_csv('output/player_season.csv'); pw=pd.read_csv('output/player_week.csv')
ts=pd.read_csv('output/team_season.csv'); tst=pd.read_csv('output/team_stats.csv', dtype={'value':str})
out=[]
import json as _json
CUR = _json.load(open('current.json'))
for season, thru in {2023: 18, 2024: 18, 2025: 18, CUR['season']: CUR['thru']}.items():
    d=grades.load_all(season, thru)
    out.append(fx.build(season, d, ps[ps.season==season], pw[pw.season==season], ts[(ts.season==season)&(ts.scope=='regular')], tst[tst.season==season], st.box_table(d, season, thru, False)))
f=pd.concat(out); f.to_csv('output/fun_facts.csv', index=False); print(len(f)); print(f[f.text.str.contains('tied')].text.tolist())
