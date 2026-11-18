import sys; sys.argv = ['x']
import pandas as pd, numpy as np, grades, power
ts = pd.read_csv('output/team_season.csv'); ro = pd.read_csv('output/roster.csv'); ps = pd.read_csv('output/player_season.csv')
import nflreadpy as _nfl
sched = grades.cached('schedules_all', lambda: _nfl.load_schedules([2023, 2024, 2025, 2026]))
# Current season: who's actually healthy. Injured players don't count toward roster strength, and losing
# starters lowers that side's strength in proportion to how much worse the replacements are.
ls = pd.read_csv('output/lineup_scores.csv')
if 'avail' in ro:
    cur = ro['season'] == 2026
    ro = ro[~cur | (ro['avail'] == 1)].copy()
    ro.loc[ro['season'] == 2026, 'snap_pct'] = (100 * ro.loc[ro['season'] == 2026, 'recent']).round().astype(int).astype(str) + '%'
for side, col in [('off', 'off_grade'), ('def', 'def_grade')]:
    x = ls[(ls.season == 2026) & (ls.side == side)].set_index('team')
    sd = x['full'].std(ddof=0) or 1
    adj = 10 * 0.6 * (x['cur'] - x['full']) / sd
    m = ts['season'] == 2026
    ts.loc[m, col] = ts.loc[m, col] + ts.loc[m, 'team'].map(adj).fillna(0)
pw = power.power(ts, ro, ps, sched); pw.to_csv('output/team_power.csv', index=False)
sos, games = power.schedule_strength(pw, sched)
sos.to_csv('output/team_sos.csv', index=False); games.to_csv('output/team_schedule.csv', index=False)
x = pw[(pw.season == 2026) & (pw.scope == 'regular')].sort_values('power_rank')
print(x[['power_rank', 'team', 'wins', 'losses', 'point_diff', 'off_rank', 'def_rank', 'st_rank', 'roster_rank']].head(8).to_string(index=False))
y = sos[sos.season == 2026].sort_values('sos_rank')
print(y.head(3)[['team', 'sos_rank', 'opp_record', 'sos_left_rank', 'left_record']].to_string(index=False))
print(y[y.team.isin(['PIT', 'MIN'])][['team', 'sos_rank', 'opp_record', 'sos_left_rank', 'left_record', 'games_left']].to_string(index=False))

# ---- preseason priors (our own data only) and the blend into season + weekly power ----
import prior, json as _json
_cur = _json.load(open('current.json')); wp = []
seasons = {2023: 18, 2024: 18, 2025: 18, _cur['season']: _cur['thru']}
pw = pw.copy(); pw['prior'] = np.nan
for season, thru in seasons.items():
    d_ = grades.load_all(season, thru)
    prev = pw[pw.season == season - 1]
    pre = prior.preseason(season, d_, sched, prev if len(prev) else None, ps[ps.season == season - 1])
    if pre is not None:
        for sc in pw.loc[pw.season == season, 'scope'].unique():
            m = (pw.season == season) & (pw.scope == sc)
            g = pw[m]; gp = float(g['games'].median()); w = gp / (gp + prior.PRIOR_GAMES)
            zz = (g['power_score'] - g['power_score'].mean()) / g['power_score'].std(ddof=0)
            pw.loc[m, 'power_score'] = w * zz + (1 - w) * g['team'].map(pre).fillna(0)
            rk = power.apply_h2h(pw[m], season, power.head_to_head(sched))
            pw.loc[m, 'power_rank'] = pw.loc[m, 'team'].map(rk)
    wp.append(power.weekly_power(d_, sched, season, thru, pre))
pw.to_csv('output/team_power.csv', index=False)
pd.concat(wp).to_csv('output/team_week_power.csv', index=False); print('weekly power done')
sos, games = power.schedule_strength(pw, sched)
sos.to_csv('output/team_sos.csv', index=False); games.to_csv('output/team_schedule.csv', index=False)
