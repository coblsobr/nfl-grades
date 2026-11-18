# NFL Grades — your own app

A free app that ranks every NFL player and team from free public data (nflverse, FTN Data via nflverse,
NFL Next Gen Stats, Pro Football Reference). It updates itself each week.

```
 GitHub (free)                                   Your phone
 ┌──────────────────────────────┐                ┌─────────────────────┐
 │ every morning: is a new week │   new data     │ NFL Grades app      │
 │ fully posted?  ── no ──> stop │ ────────────>  │ checks for new data │
 │      │ yes                    │  (docs/data)   │ every time you open │
 │ rebuild grades, ranks, picks  │                │ it                  │
 └──────────────────────────────┘                └─────────────────────┘
```

## One-time setup (about 15 minutes)

1. **Make a free GitHub account** at github.com.
2. **New repository** (the + at the top right → New repository). Name it `nfl-grades`, choose **Public**
   (free GitHub Pages needs public), and create it.
3. **Upload the files.** On the new repo page click "uploading an existing file", then drag in the
   `docs` and `pipeline` folders and this README. Click **Commit changes**.
4. **Add the update schedule.** Folders starting with a dot often don't upload by drag-and-drop, so make
   this one by hand: **Add file → Create new file**, name it `.github/workflows/update.yml`, paste in the
   contents of `.github/workflows/update.yml` from this package, and **Commit**.
5. **Let the schedule save data:** Settings → Actions → General → *Workflow permissions* →
   **Read and write permissions** → Save.
6. **Turn on the website:** Settings → Pages → *Build and deployment* → Source **Deploy from a branch**,
   Branch **main**, folder **/docs** → Save. After a minute your app is live at
   `https://YOUR-USERNAME.github.io/nfl-grades/`
7. **Put it on your phone:** open that link.
   - iPhone (Safari): Share button → **Add to Home Screen**.
   - Android (Chrome): ⋮ menu → **Install app** / Add to Home screen.
8. **Test the updater once:** Actions tab → *weekly-update* → **Run workflow**. With nothing new it
   finishes in a few minutes and says "No new week yet."

## How updates work
- Every morning GitHub checks whether the latest week is fully posted in every source (scores,
  play-by-play, snaps, FTN charting, PFR advanced stats, Next Gen Stats). If yes, it rebuilds everything
  (20–30 minutes) and saves the new data. If not, it stops.
- Every time you open the app it checks for new data. If there is some, it loads it and shows
  "New data: through Week N". The line under the top bar always says what week the data runs through.

## Things you might edit (right on github.com, pencil icon)
- `pipeline/current.json` — at the start of a new season change `"season"` to the new year and `"thru"` to 0.
- `pipeline/injuries_manual.csv` — breaking injury news before the official report (one line per player).
- `pipeline/weights.json` — the formula weights for each position.

`pipeline/PLAN.md` has the full write-up of how every grade and rank works.
