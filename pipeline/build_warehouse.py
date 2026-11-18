"""
Build everything for the site in one go:
  player grades by season and by week, team grades (offense / defense / special teams / overall)
  by season and by week, and the expert direction check.

  python build_warehouse.py                 # 2023-2026 (2026 through the latest complete week)
Writes output/nfl_grades.db (SQLite) plus CSVs and RESULTS_<season>.md files.
"""
import sqlite3
import sys
import pandas as pd

sys.argv = [sys.argv[0]]
import grades
import team_grades as tgm
import direction_check as dcheck
import stats as st
import facts as fx
import roster as rost
import coaching as coach

import json as _json
CUR = _json.load(open("current.json"))
SEASONS = {2023: 18, 2024: 18, 2025: 18, CUR["season"]: CUR["thru"]}
SAMPLE = {"QB": "plays", "RB": "carries", "WR": "routes", "TE": "routes", "OT": "snaps", "OG": "snaps", "C": "snaps",
          "EDGE": "def_snaps", "IDL": "def_snaps", "LB": "def_snaps", "CB": "def_snaps", "S": "def_snaps",
          "K": "att", "P": "punts"}
OUT = grades.OUT


RANK_SHARE = 0.5   # need about half of a regular's workload (over ~8 games, or this week) to get a rank


def ranked_mask(df):
    return (df["pt_share"] >= RANK_SHARE) if "pt_share" in df else pd.Series(True, index=df.index)


def flatten(res, season, week=None):
    rows = []
    for pos, df in res.items():
        if pos not in SAMPLE:
            continue
        x = df.reset_index(drop=True)
        rk = ranked_mask(x)
        rank = rk.cumsum().where(rk)
        rows.append(pd.DataFrame({
            "season": season, "week": week, "pos": pos, "rank": rank.astype("Int64"), "pool": int(rk.sum()),
            "gsis_id": x.get("gsis_id"), "player": x["full_name"], "team": x["team"], "grade": x["grade"],
            "sample": x[SAMPLE[pos]].round(1) if SAMPLE[pos] in x else None,
            "reg": x["qualified"].astype(int) if "qualified" in x else 1,
            "ranked": rk.astype(int)}))
    return pd.concat(rows)


def results_md(season, thru, res, teams, summ):
    wk = "full regular season" if thru >= 18 else f"Weeks 1–{thru}"
    L = [f"# {season} season, {wk} — {season} games only\n",
         "Scale: 70 = average · each 10 points = 1 standard deviation\n",
         "## Teams (opponent-adjusted)\n",
         "| # | Team | Record | Overall | Offense | Defense | Special teams | Pass off | Run off | Pass def | Run def |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in teams.iterrows():
        rec = f"{r.wins}-{r.losses}" + (f"-{r.ties}" if r.ties else "")
        L.append(f"| {i+1} | {r.team} | {rec} | {r.overall_grade:.1f} | {r.off_grade:.1f} | {r.def_grade:.1f} | "
                 f"{r.st_grade:.1f} | {r.pass_off_grade:.1f} | {r.run_off_grade:.1f} | {r.pass_def_grade:.1f} | {r.run_def_grade:.1f} |")
    if summ is not None:
        L += ["\n## Direction check vs. expert consensus (PFF published lists + AP All-Pro)\n",
              "Agreement: 0 = no better than random, 1 = every expert pick near our top.\n",
              "| Pos | Expert picks graded | Near our top | Random would get | Agreement | Status |",
              "|---|---|---|---|---|---|"]
        for p, r in summ.iterrows():
            L.append(f"| {p} | {r.experts_found} | {r.hits} of {r.experts_found} (top {r.near_top_cutoff}) | "
                     f"{r.random_hits} | {r.agreement:.2f} | {r.status} |")
    for pos, df in res.items():
        L.append(f"\n## {pos}  ({len(df)} qualified)\n\n| # | Player | Team | Grade |\n|---|---|---|---|")
        for i, r in df[ranked_mask(df)].head(10).reset_index(drop=True).iterrows():
            L.append(f"| {i+1} | {r['full_name']} | {r['team']} | {r['grade']:.1f} |")
    (OUT / f"RESULTS_{season}.md").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    ps, pw, ts, tw, dsum, ddet, pst, tst, ff, ro = [], [], [], [], [], [], [], [], [], []
    for season, thru in SEASONS.items():
        d = grades.load_all(season, thru)
        res = grades.run_on(d)
        ps.append(flatten(res, season).assign(scope="regular"))
        pst.append(st.player_rows(res, d, season, thru, False, "regular"))
        tst.append(pd.concat([st.team_rows(d, season, "regular"), coach.team_rows(d, season, "regular")]))
        (OUT / str(season)).mkdir(exist_ok=True)
        for pos, df in res.items():
            df.to_csv(OUT / str(season) / f"grades_{pos.split()[0]}.csv", index=False)
        teams = tgm.season_grades(d).assign(season=season)
        ts.append(teams.assign(scope="regular"))
        if thru >= 18:                       # completed season: also a regular season + playoffs view (how PFF counts year-end)
            dp = grades.load_all(season, 18, playoffs=True)
            rp = grades.run_on(dp)
            ps.append(flatten(rp, season).assign(scope="with_playoffs"))
            pst.append(st.player_rows(rp, dp, season, 22, True, "with_playoffs"))
            tst.append(pd.concat([st.team_rows(dp, season, "with_playoffs"), coach.team_rows(dp, season, "with_playoffs")]))
            ts.append(tgm.season_grades(dp).assign(season=season, scope="with_playoffs"))
        tw.append(tgm.weekly_grades(d).assign(season=season))
        for wk in range(1, thru + 1):
            sub = grades.subset_weeks(d, wk, wk, season)
            if sub["pbp"].empty:
                continue
            try:
                pw.append(flatten(grades.run_on(sub), season, wk))
            except Exception as e:
                print(f"  week {season}-{wk} skipped: {e}")
        try:
            ff.append(fx.build(season, d, ps[-1] if season == 2026 else pd.concat([x for x in ps if (x["season"] == season).all()]),
                               pd.concat([x for x in pw if (x["season"] == season).all()]), teams,
                               pd.concat([x for x in tst if (x["season"] == season).all()]),
                               st.box_table(d, season, thru, False)))
        except Exception as e:
            import traceback; traceback.print_exc()
        ro.append(rost.build(season, d, ps[-1] if season == 2026 else [x for x in ps if (x["season"] == season).all() and (x["scope"] == "regular").all()][0]))
        summ, det = dcheck.check(res, season)
        if summ is not None:
            dsum.append(summ.reset_index().assign(season=season))
            ddet.append(det.assign(season=season))
        results_md(season, thru, res, teams, summ)
        print(f"{season}: done ({thru} weeks)")

    tables = {"player_season": pd.concat(ps), "player_week": pd.concat(pw), "team_season": pd.concat(ts),
              "team_week": pd.concat(tw), "player_stats": pd.concat(pst), "team_stats": pd.concat(tst),
              "fun_facts": pd.concat(ff), "roster": pd.concat(ro), "direction_check": pd.concat(dsum), "direction_check_players": pd.concat(ddet)}
    db = OUT / "nfl_grades.db"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False)
            df.to_csv(OUT / f"{name}.csv", index=False)
    for name, df in tables.items():
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
