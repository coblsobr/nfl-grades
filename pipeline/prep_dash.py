"""Rebuild every dashboard CSV in /home/claude/dash from output/ (one place, so nothing drifts)."""
import sys; sys.argv = ['x']
import pandas as pd, numpy as np, direction_check as dc
import os
O = os.environ.get('DASH_DIR', '/home/claude/dash/')
ps = pd.read_csv('output/player_season.csv'); ps['grade'] = ps['grade'].round(1)
ps['key'] = ps['season'].astype(str) + '|' + ps['scope'] + '|' + ps['pos'] + '|' + ps['gsis_id']
ps = ps.drop_duplicates('key'); ps['rank'] = ps['rank'].astype('Int64')
ps['order'] = ps.groupby(['season', 'scope', 'pos'])['grade'].rank(ascending=False, method='first').astype(int)
ps[['key', 'season', 'scope', 'pos', 'rank', 'pool', 'gsis_id', 'player', 'team', 'reg', 'ranked', 'order']].to_csv(O + 'player_season.csv', index=False)
pw = pd.read_csv('output/player_week.csv'); pw['grade'] = pw['grade'].round(1); pw['rank'] = pw['rank'].astype('Int64')
for s, x in pw.groupby('season'):
    x[['week', 'pos', 'gsis_id', 'team', 'grade', 'rank', 'pool']].to_csv(O + f'player_week_{s}.csv', index=False)
st = pd.read_csv('output/player_stats.csv', dtype={'value': str}); st['value'] = st['value'].fillna('')
st['rank'] = st['rank'].astype('Int64'); st['of'] = st['of'].astype('Int64')
st['key'] = st['pos'] + '|' + st['gsis_id'] + '|' + st['ord'].astype(str)
for (s, sc), x in st.groupby(['season', 'scope']):
    x = x.drop_duplicates('key').sort_values(['pos', 'gsis_id', 'ord'])
    x[['key', 'pos', 'gsis_id', 'ord', 'stat', 'value', 'rank', 'of']].to_csv(O + f'player_stats_{s}_{"r" if sc == "regular" else "p"}.csv', index=False)
r = pd.read_csv('output/roster.csv'); r['grade'] = r['grade'].round(1); r['rank'] = r['rank'].astype('Int64'); r['ord'] = r['ord'].astype(int)
for c, dflt in [('avail', 1), ('status', ''), ('recent', 0.0), ('cur_ord', 99)]:
    if c not in r: r[c] = dflt
r['status'] = r['status'].fillna(''); r['cur_ord'] = r['cur_ord'].fillna(99).astype(int); r['avail'] = r['avail'].fillna(1).astype(int)
r[['key', 'season', 'team', 'unit', 'pos', 'player', 'gsis_id', 'role', 'starts', 'games', 'snap_pct', 'grade', 'rank', 'ord', 'avail', 'status', 'cur_ord']].drop_duplicates('key').to_csv(O + 'roster.csv', index=False)
# teams
ts = pd.read_csv('output/team_season.csv'); pw_ = pd.read_csv('output/team_power.csv'); sos = pd.read_csv('output/team_sos.csv')
ts['record'] = ts.wins.astype(str) + '-' + ts.losses.astype(str) + np.where(ts.ties > 0, '-' + ts.ties.astype(str), '')
ts['key'] = ts['season'].astype(str) + '|' + ts['scope'] + '|' + ts['team']
for g in ['pass_off', 'run_off', 'pass_def', 'run_def', 'pass_pro', 'pass_rush']:
    ts[g + '_rank'] = ts.groupby(['season', 'scope'])[g + '_grade'].rank(ascending=False, method='min').astype(int)
ts = ts.merge(pw_[['season', 'scope', 'team', 'power_rank', 'roster_rank']], on=['season', 'scope', 'team'])
ts = ts.merge(sos[['season', 'team', 'sos_rank', 'sos_left_rank', 'opp_record', 'left_record']], on=['season', 'team'], how='left')
ts['sos_left_rank'] = ts['sos_left_rank'].astype('Int64'); ts['pd_txt'] = ts['point_diff'].map(lambda v: f"{int(v):+d}")
import os as _os
if _os.path.exists('output/projections.csv'):
    pj = pd.read_csv('output/projections.csv', dtype={'spread': str})[['season', 'team', 'week', 'opp', 'ha', 'pick', 'chance', 'next_txt', 'spread']].rename(columns={'week': 'next_week'})
    ts = ts.merge(pj.assign(scope='regular'), on=['season', 'scope', 'team'], how='left')
for c in ['next_week', 'opp', 'ha', 'pick', 'chance', 'next_txt', 'spread']:
    if c not in ts: ts[c] = ''
ts['next_week'] = pd.to_numeric(ts['next_week'], errors='coerce').astype('Int64')
ts[['key', 'season', 'scope', 'team', 'record', 'overall_rank', 'off_rank', 'def_rank', 'st_rank', 'pass_off_rank', 'run_off_rank', 'pass_def_rank',
    'run_def_rank', 'pass_pro_rank', 'pass_rush_rank', 'point_diff', 'power_rank', 'roster_rank', 'sos_rank', 'sos_left_rank', 'opp_record',
    'left_record', 'pd_txt', 'next_week', 'opp', 'ha', 'pick', 'chance', 'next_txt', 'spread']].to_csv(O + 'team_season.csv', index=False)
g = pd.read_csv('output/team_schedule.csv')
g['key'] = g['season'].astype(str) + '|' + g['team'] + '|' + g['week'].astype(str); g['opp_power_rank'] = g['opp_power_rank'].astype('Int64')
g.sort_values(['season', 'team', 'week'])[['key', 'season', 'week', 'team', 'opp', 'ha', 'result', 'opp_power_rank']].to_csv(O + 'schedule.csv', index=False)
f = pd.read_csv('output/fun_facts.csv'); f[['id', 'season', 'cat', 'icon', 'title', 'text', 'team']].to_csv(O + 'fun_facts.csv', index=False)
s = pd.read_csv('output/direction_check.csv'); s[['pos', 'experts_found', 'hits', 'random_hits', 'agreement', 'status', 'not_qualified']].to_csv(O + 'direction_check.csv', index=False)
p = pd.read_csv('output/direction_check_players.csv')
order = {(pos, n): i + 1 for pos, names in dc.CONSENSUS[2025].items() for i, n in enumerate(names)}
p['expert_order'] = [order.get((a, b)) for a, b in zip(p.pos, p.player)]
p['key'] = p['pos'] + '|' + p['player']; p['our_rank'] = p['our_rank'].astype('Int64'); p['note'] = p['note'].fillna('')
p[['key', 'pos', 'expert_order', 'player', 'our_rank', 'pool', 'note']].to_csv(O + 'expert_picks.csv', index=False)
print('prepared')

# ---- team weekly ranks (overall = offense 45 / defense 45 / special teams 10), league stats, coaching ----
tw = pd.read_csv('output/team_week.csv')
z = lambda x: (x - x.mean()) / x.std(ddof=0)
tw['score'] = tw.groupby(['season', 'week'], group_keys=False).apply(lambda x: .45 * z(x.off_grade) + .45 * z(x.def_grade) + .10 * z(x.st_grade))
tw['overall_rank'] = tw.groupby(['season', 'week'])['score'].rank(ascending=False, method='min').astype(int)
for g, rn in [('off_grade', 'off_rank'), ('def_grade', 'def_rank'), ('st_grade', 'st_rank')]:
    tw[rn] = tw.groupby(['season', 'week'])[g].rank(ascending=False, method='min').astype(int)
tw['teams'] = tw.groupby(['season', 'week'])['team'].transform('size')
tw['result'] = [('W' if a > b else 'L' if a < b else 'T') + f" {a}-{b}" for a, b in zip(tw.pf, tw.pa)]
tw['key'] = tw['season'].astype(str) + '|' + tw['week'].astype(str) + '|' + tw['team']
wpw = pd.read_csv('output/team_week_power.csv')
tw = tw.merge(wpw, on=['season', 'week', 'team'], how='left')
w0 = wpw[wpw['week'] == 0].assign(key=lambda x: x['season'].astype(str) + '|0|' + x['team'], result='', opp='')
tw = pd.concat([w0, tw], ignore_index=True)
tw['power_rank_wk'] = tw['power_rank_wk'].astype('Int64')
for c in ['pf', 'pa', 'overall_rank', 'off_rank', 'def_rank', 'st_rank', 'teams']:
    tw[c] = tw[c].astype('Int64')
tw[['season', 'week', 'team', 'opp', 'pf', 'pa', 'overall_rank', 'off_rank', 'def_rank', 'st_rank', 'teams', 'result', 'key', 'power_rank_wk', 'record_wk']].to_csv(O + 'team_week.csv', index=False)
tst = pd.read_csv('output/team_stats.csv', dtype={'value': str}); tst['value'] = tst['value'].fillna('')
gr = tst['stat'].str.contains(' grade', regex=False)
tst.loc[gr, 'value'] = ''; tst.loc[gr, 'stat'] = tst.loc[gr, 'stat'].str.replace(' grade', ' rank', regex=False)
tst['rank'] = tst['rank'].astype('Int64')
tst['key'] = tst['season'].astype(str) + '|' + tst['scope'] + '|' + tst['team'] + '|' + tst['ord'].astype(str)
tst[['key', 'season', 'scope', 'team', 'side', 'ord', 'stat', 'value', 'rank']].to_csv(O + 'team_stats.csv', index=False)
c = pd.read_csv('output/coaching.csv')
c['key'] = c['season'].astype(str) + '|' + c['scope'] + '|' + c['team']
gg = c.groupby(['season', 'scope'])
c['coach_rank'] = gg['coach_grade'].rank(ascending=False, method='min').astype(int)
c['playcall_rank'] = gg['playcall_grade'].rank(ascending=False, method='min').astype(int)
c['ats_rank'] = gg['ats_pg'].rank(ascending=False, method='min').astype(int)
c['discipline_rank'] = gg['pen_yds_pg'].rank(ascending=True, method='min').astype(int)
c['aggressive_rank'] = gg['aggressive_grade'].rank(ascending=False, method='min').astype(int)
c['ats_txt'] = c['ats_pg'].map(lambda v: f"{v:+.1f} pts/game"); c['pen_txt'] = c['pen_yds_pg'].map(lambda v: f"{v:.0f} penalty yds/game")
c['vs_vegas_txt'] = c['vs_vegas'].map(lambda v: '' if pd.isna(v) else f"{v:+.1f}")
c['close_txt'] = [('' if pd.isna(n) else f"{w:g}-{n - w:g}") for w, n in zip(c.close_w.fillna(0), c.close_n)]
c['go_txt'] = c['go_rate_oe'].map(lambda v: '' if pd.isna(v) else f"{100 * v:+.1f}%"); c['timid_punts'] = c['si_timid'].fillna(0).astype(int)
c[['key', 'season', 'scope', 'team', 'coach_rank', 'playcall_rank', 'ats_rank', 'discipline_rank', 'aggressive_rank', 'ats_txt', 'pen_txt', 'vs_vegas_txt', 'close_txt', 'go_txt', 'timid_punts']].to_csv(O + 'coaching.csv', index=False)
print('prepared team_week, team_stats, coaching')

import shutil, glob as _g
for f in _g.glob('output/player_week_stats_*.csv'):
    shutil.copy(f, O + f.split('/')[-1])
if os.path.exists('output/pick_record.csv'):
    shutil.copy('output/pick_record.csv', O + 'pick_record.csv')
print('copied week stats and pick record')
