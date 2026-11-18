"""
Weekly auto-update. Run it any time (or on a schedule): it only does work when a new week is fully in.

  python update.py            # check; rebuild only if a new week is complete in every source
  python update.py --force    # rebuild the current week anyway

A week counts as ready when (1) every game that week has a final score in the nflverse schedule and
(2) every one of those games is in the nflverse play-by-play. Snap counts, PFR advanced stats, FTN
charting and Next Gen Stats usually land the same night or the next morning; the check waits for
snap counts too, since rosters and injuries lean on them.
"""
import json, sys, subprocess, glob, os
import nflreadpy as nfl

STATE = "current.json"


def check(season):
    """Latest finished week and whether every source has it. A week can also go ahead 'partial' when the only
    gap is one game's PFR advanced stats two or more days after the week ended (usually Monday night's game);
    the next run rebuilds it again once that last game is posted."""
    import datetime
    s = nfl.load_schedules([season]).to_pandas(); s = s[s["game_type"] == "REG"]
    final = s.groupby("week").apply(lambda g: g["home_score"].notna().all())
    done = [int(w) for w, v in final.items() if v]
    if not done:
        return dict(week=0, full=True, partial_ok=False, missing=[], games_through="")
    wk = max(done)
    need = set(s.loc[s["week"] == wk, "game_id"])
    last_day = str(s.loc[s["week"] == wk, "gameday"].max())
    pbp = nfl.load_pbp([season]).to_pandas()
    snaps = nfl.load_snap_counts([season]).to_pandas()
    plays = pbp[(pbp["week"] == wk) & ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
    ftn = nfl.load_ftn_charting([season]).to_pandas()
    pfr = nfl.load_pfr_advstats([season], stat_type="def", summary_level="week").to_pandas()
    ngs = nfl.load_nextgen_stats([season], stat_type="receiving").to_pandas()
    pfr_games = pfr.loc[pfr["week"] == wk, "game_id"].nunique()
    checks = {
        "play-by-play": need <= set(pbp.loc[pbp["week"] == wk, "game_id"]),
        "snap counts": snaps["week"].max() >= wk,
        "FTN charting": (ftn["week"] == wk).sum() >= 0.9 * len(plays),
        "PFR advanced stats": pfr_games >= len(need),
        "Next Gen Stats": ngs.loc[ngs["week"] == wk, "team_abbr"].nunique() >= 0.9 * 2 * len(need),
    }
    missing = [k for k, ok in checks.items() if not ok]
    days = (datetime.date.today() - datetime.date.fromisoformat(last_day)).days
    partial_ok = missing == ["PFR advanced stats"] and pfr_games >= len(need) - 1 and days >= 2
    return dict(week=wk, full=not missing, partial_ok=partial_ok, missing=missing, games_through=last_day)


def ready_week(season):                      # kept for older scripts
    c = check(season)
    return c["week"] if (c["full"] or c["partial_ok"]) else c["week"] - 1


def run(step):
    print(">>", step, flush=True)
    subprocess.run([sys.executable] + step.split(), check=True)


def main():
    import datetime
    cur = json.load(open(STATE))
    out = os.environ.get("DASH_DIR", "/home/claude/dash/")
    c = check(cur["season"])
    target = c["week"] if (c["full"] or c["partial_ok"]) else c["week"] - 1
    if "--week" in sys.argv:
        target = int(sys.argv[sys.argv.index("--week") + 1])
    rebuild = target > cur["thru"] or (target == cur["thru"] and cur.get("partial") and c["full"]) or "--force" in sys.argv
    # what the app shows about the next update, written every run
    json.dump({"checked": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "latest_played": c["week"],
               "waiting_on": [] if c["full"] else c["missing"]}, open(os.path.join(out, "status.json"), "w"))
    if not rebuild:
        print(f"Nothing new: built through Week {cur['thru']}; Week {c['week']} waiting on: {', '.join(c['missing']) or 'nothing'}.")
        return
    for f in glob.glob(f"cache/*{cur['season']}*.parquet") + glob.glob("cache/schedules_all.parquet"):
        os.remove(f)
    cur["thru"] = target; cur["partial"] = bool(target == c["week"] and not c["full"])
    json.dump(cur, open(STATE, "w"))
    for step in ["build_warehouse.py", "build_coaching_wide.py", "split_stats.py", "add_availability.py", "add_power.py", "project.py",
                 "week_stats.py", "rebuild_facts.py", "prep_dash.py"]:
        run(step)
    sch = nfl.load_schedules([cur["season"]]).to_pandas()
    thru_day = str(sch.loc[(sch["game_type"] == "REG") & (sch["week"] == cur["thru"]), "gameday"].max())
    json.dump({"season": cur["season"], "thru": cur["thru"], "games_through": thru_day, "partial": cur["partial"],
               "updated": datetime.date.today().isoformat()}, open(os.path.join(out, "version.json"), "w"))
    print(f"Updated through Week {cur['thru']}{' (one game still missing PFR stats)' if cur['partial'] else ''}. App data is in {out}")


if __name__ == "__main__":
    main()
