"""
Validate the individual OL method on 2025, the last season with free per-play participation data.

Three checks:
  1. Split-half: do Weeks 1-9 grades predict Weeks 10-18 grades? (is there real signal, or noise?)
  2. Independent method: a ridge regression on 2025 per-play data (who was on the field + was the QB
     pressured) estimates each lineman's pressure impact without any matchup assumptions.
     Does it agree with our matchup-based pass-pro numbers?
  3. Output 2025 full-season OT/OG/C grades for comparison with published rankings.
"""
import sys
import numpy as np
import pandas as pd
import nflreadpy as nfl
from scipy import sparse
from sklearn.linear_model import Ridge

sys.argv = [sys.argv[0]]
import grades

SEASON = 2025


def subset(d, lo, hi):
    """Copy of the loaded data restricted to weeks lo..hi."""
    out = dict(d)
    for k in ["pbp", "ftn", "snaps", "pfr_def", "pfr_pass", "pfr_rush", "pfr_rec"]:
        out[k] = d[k][(d[k]["week"] >= lo) & (d[k]["week"] <= hi)]
    return out


def ol_grades(d):
    off, de, tg = grades.team_context(d)
    units = grades.grade_ol_units(d)
    res = grades.grade_ol_individual(d, off, de, tg, units)
    return pd.concat([v.assign(pos=k) for k, v in res.items()])


def within_team(df, col):
    return df[col] - df.groupby("team")[col].transform("mean")


def main():
    d = grades.load_all(SEASON, 18)

    # ---------------- 1. split-half reliability
    a = ol_grades(subset(d, 1, 9)).set_index("gsis_id")
    b = ol_grades(subset(d, 10, 18)).set_index("gsis_id")
    both = a[["grade", "team", "pressures_oe_per_100", "run_succ_oe", "team_unit"]].join(
        b[["grade", "pressures_oe_per_100", "run_succ_oe", "team_unit"]], rsuffix="_2", how="inner")
    both = both.reset_index()
    print(f"\n[1] split-half, {len(both)} linemen graded in both halves")
    print(f"    grade            r = {both['grade'].corr(both['grade_2']):.2f}")
    print(f"    pass pro (matchup) r = {both['pressures_oe_per_100'].corr(both['pressures_oe_per_100_2']):.2f}")
    print(f"    run block (gaps)   r = {both['run_succ_oe'].corr(both['run_succ_oe_2']):.2f}")
    print(f"    team unit          r = {both['team_unit'].corr(both['team_unit_2']):.2f}")
    # individual part only: grade relative to his own line's average (same team both halves)
    same = both[both["team"].notna()]
    wa, wb = within_team(same, "grade"), within_team(same.assign(team=same["team"]), "grade_2")
    print(f"    within-line grade  r = {wa.corr(wb):.2f}   (does he beat his own linemates consistently?)")

    # ---------------- 2. independent per-play ridge model (2025 participation)
    part = nfl.load_participation([SEASON]).to_pandas()
    part = part.rename(columns={"nflverse_game_id": "game_id"})
    p = d["pbp"]
    db = p[(p["pass"] == 1) & (p["qb_spike"] == 0) & p["id"].notna()][["game_id", "play_id", "id", "defteam", "posteam"]]
    m = db.merge(part[["game_id", "play_id", "offense_players", "was_pressure"]], on=["game_id", "play_id"])
    m = m[m["was_pressure"].notna() & m["offense_players"].notna()].reset_index(drop=True)
    players = grades.cached("players_all", lambda: nfl.load_players())
    ol_ids = set(players[players["position"].isin(["T", "G", "C", "OL", "OT", "OG"])]["gsis_id"])
    on = m["offense_players"].str.split(";")
    rows, cols, names = [], [], {}
    for i, ids in enumerate(on):
        for pid in ids:
            if pid in ol_ids:
                rows.append(i); cols.append(names.setdefault(pid, len(names)))
    X_ol = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(m), len(names)))
    X_qb = sparse.csr_matrix(pd.get_dummies(m["id"]).values.astype(float))
    X_def = sparse.csr_matrix(pd.get_dummies(m["defteam"]).values.astype(float))
    X = sparse.hstack([X_ol, X_qb, X_def]).tocsr()
    y = m["was_pressure"].astype(float).values
    model = Ridge(alpha=300).fit(X, y)
    coef = pd.Series(model.coef_[:len(names)] * 100, index=list(names))       # pressures per 100 dropbacks
    snaps = pd.Series(np.asarray(X_ol.sum(axis=0)).ravel(), index=list(names))
    print(f"\n[2] ridge model: {len(m):,} dropbacks, league pressure rate {y.mean():.1%}")

    full = ol_grades(d)
    full["ridge_press_per_100"] = full["gsis_id"].map(coef)
    full["ridge_snaps"] = full["gsis_id"].map(snaps)
    f = full[full["ridge_press_per_100"].notna()]
    print(f"    linemen matched: {len(f)}")
    print(f"    matchup pass-pro vs ridge, all linemen   r = {f['pressures_oe_per_100'].corr(f['ridge_press_per_100']):.2f}")
    print(f"    matchup pass-pro vs ridge, within a line r = "
          f"{within_team(f, 'pressures_oe_per_100').corr(within_team(f, 'ridge_press_per_100')):.2f}")
    for pos in ["OT", "OG", "C"]:
        q = f[f["pos"] == pos]
        print(f"      {pos}: r = {q['pressures_oe_per_100'].corr(q['ridge_press_per_100']):.2f}  (n={len(q)})")

    full.to_csv("output/validation_2025_ol.csv", index=False)
    both.to_csv("output/validation_2025_splithalf.csv", index=False)
    for pos in ["OT", "OG", "C"]:
        print(f"\n2025 {pos} top 15:", list(full[full["pos"] == pos].sort_values("grade", ascending=False)["full_name"].head(15)))


if __name__ == "__main__":
    main()
