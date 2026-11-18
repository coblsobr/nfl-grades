"""Next-game projections from our power scores.
Win chance = logistic(a * (our score - their score) + b * home), fit on every 2023-2026 game using each
team's power score from the week before the game (so the model never sees the result it predicts)."""
import sys, json; sys.argv = ['x']
import numpy as np, pandas as pd, grades, nflreadpy as nfl

cur = json.load(open('current.json'))
sched = grades.cached('schedules_all', lambda: nfl.load_schedules([2023, 2024, 2025, 2026]))
sched = sched[sched['game_type'] == 'REG']
wp = pd.read_csv('output/team_week_power.csv')
score = {(r.season, r.week, r.team): r.power_score for r in wp.itertuples()}
X, y = [], []
for g in sched[sched['home_score'].notna()].itertuples():
    sh, sa = score.get((g.season, g.week - 1, g.home_team)), score.get((g.season, g.week - 1, g.away_team))
    if sh is None or sa is None or g.home_score == g.away_score:
        continue
    X.append([sh - sa, 1.0]); y.append(1.0 if g.home_score > g.away_score else 0.0)
X, y = np.array(X), np.array(y)
w = np.zeros(2)
for _ in range(50):                                   # logistic regression (Newton steps)
    p = 1 / (1 + np.exp(-X @ w)); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
    w += np.linalg.solve(H, X.T @ (y - p))
M = []
for g in sched[sched['home_score'].notna()].itertuples():
    sh, sa = score.get((g.season, g.week - 1, g.home_team)), score.get((g.season, g.week - 1, g.away_team))
    if sh is None or sa is None:
        continue
    M.append([sh - sa, 1.0, g.home_score - g.away_score])
M = np.array(M); mcoef = np.linalg.lstsq(M[:, :2], M[:, 2], rcond=None)[0]   # margin = slope * score gap + home edge
resid = M[:, 2] - M[:, :2] @ mcoef
print(f"margin model: {mcoef[0]:.1f} pts per score unit, home edge {mcoef[1]:.1f}, typical miss {np.abs(resid).mean():.1f} pts")
p = 1 / (1 + np.exp(-X @ w))
print(f"fit on {len(y)} games: slope {w[0]:.2f}, home edge {w[1]:.2f}; picks right {((p > .5) == (y == 1)).mean():.1%} of the time")
# upcoming week for the current season, using today's power scores
tp = pd.read_csv('output/team_power.csv'); tp = tp[(tp.season == cur['season']) & (tp.scope == 'regular')].set_index('team')['power_score']
nxt = cur['thru'] + 1
games = sched[(sched.season == cur['season']) & (sched.week == nxt)]
rows = []
for g in games.itertuples():
    ph = float(1 / (1 + np.exp(-(w[0] * (tp[g.home_team] - tp[g.away_team]) + w[1]))))
    mh = float(mcoef[0] * (tp[g.home_team] - tp[g.away_team]) + mcoef[1])        # predicted home margin
    rows += [dict(team=g.home_team, opp=g.away_team, ha='vs', prob=ph, margin=mh), dict(team=g.away_team, opp=g.home_team, ha='@', prob=1 - ph, margin=-mh)]
pr = pd.DataFrame(rows)
playing = set(pr.team) if len(pr) else set()
for t in tp.index:
    if t not in playing:
        rows.append(dict(team=t, opp='', ha='', prob=np.nan))
pr = pd.DataFrame(rows)
pr['pick'] = np.where(pr['prob'].isna(), 'Bye', np.where(pr['prob'] >= .5, 'Win', 'Loss'))
pr['chance'] = pr['prob'].map(lambda v: '' if pd.isna(v) else f"{round(100 * v)}%")
def _spread(m):
    if pd.isna(m): return ''
    if m == 0: return '+0'                       # exactly even
    v = -round(m * 2) / 2                       # spread convention: favorite negative, underdog positive
    if v == 0: v = -0.5 if m > 0 else 0.5       # close but not even: lean the half point toward the favorite
    return f"{v:+g}"
pr['spread'] = pr['margin'].map(_spread) if 'margin' in pr else ''
# one consistent pick: whoever the spread favors
pr['pick'] = np.where(pr['prob'].isna(), 'Bye', np.where(pr['margin'] > 0, 'Win', 'Loss'))
pr['next_txt'] = np.where(pr['pick'] == 'Bye', 'Bye', pr['ha'] + ' ' + pr['opp'] + ' ' + pr['spread'])
pr.assign(season=cur['season'], week=nxt).to_csv('output/projections.csv', index=False)
print(f"Week {nxt}:", ', '.join(f"{r.team} {r.next_txt} ({r.chance})" for r in pr[pr.team.isin(['PIT', 'BAL', 'SEA', 'JAX'])].itertuples()))
