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


def ready_week(season):
    s = nfl.load_schedules([season]).to_pandas(); s = s[s["game_type"] == "REG"]
    final = s.groupby("week").apply(lambda g: g["home_score"].notna().all())
    done = [int(w) for w, v in final.items() if v]
    if not done:
        return 0
    wk = max(done)
    pbp = nfl.load_pbp([season]).to_pandas()
    snaps = nfl.load_snap_counts([season]).to_pandas()
    need = set(s.loc[s["week"] == wk, "game_id"])
    plays = pbp[(pbp["week"] == wk) & ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
    ftn = nfl.load_ftn_charting([season]).to_pandas()
    pfr = nfl.load_pfr_advstats([season], stat_type="def", summary_level="week").to_pandas()
    ngs = nfl.load_nextgen_stats([season], stat_type="receiving").to_pandas()
    checks = {
        "play-by-play": need <= set(pbp.loc[pbp["week"] == wk, "game_id"]),
        "snap counts": snaps["week"].max() >= wk,
        "FTN charting": (ftn["week"] == wk).sum() >= 0.9 * len(plays),
        "PFR advanced stats": pfr.loc[pfr["week"] == wk, "game_id"].nunique() >= len(need),
        "Next Gen Stats": ngs.loc[ngs["week"] == wk, "team_abbr"].nunique() >= 0.9 * 2 * len(need),
    }
    missing = [k for k, ok in checks.items() if not ok]
    if missing:
        print(f"Week {wk} is played but not fully posted yet: waiting on {', '.join(missing)}.")
        return wk - 1          # try again later
    return wk


def run(step):
    print(">>", step, flush=True)
    subprocess.run([sys.executable] + step.split(), check=True)


def main():
    cur = json.load(open(STATE))
    wk = ready_week(cur["season"])
    if "--week" in sys.argv:                      # build a specific week even if a source is still missing a game
        wk = int(sys.argv[sys.argv.index("--week") + 1])
    if wk <= cur["thru"] and "--force" not in sys.argv:
        print(f"Nothing new: data through Week {cur['thru']} already built (latest ready: Week {wk}).")
        return
    for f in glob.glob(f"cache/*{cur['season']}*.parquet") + glob.glob("cache/schedules_all.parquet"):
        os.remove(f)                                   # pull fresh copies of this season's data
    cur["thru"] = max(wk, cur["thru"]); json.dump(cur, open(STATE, "w"))
    for step in ["build_warehouse.py", "build_coaching_wide.py", "split_stats.py", "add_availability.py", "add_power.py", "project.py", "week_stats.py", "rebuild_facts.py", "prep_dash.py"]:
        run(step)
    out = os.environ.get("DASH_DIR", "/home/claude/dash/")
    import datetime
    json.dump({"season": cur["season"], "thru": cur["thru"], "updated": datetime.date.today().isoformat()}, open(os.path.join(out, "version.json"), "w"))
    print(f"Updated through Week {cur['thru']}. App data is in {out}")


if __name__ == "__main__":
    main()
