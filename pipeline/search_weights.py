"""
Search each position's weights so that (in order of priority):
  1. every expert top-10 player (2025) lands inside our top 25          -> "outliers" = 0
  2. the wider expert consensus (PFF lists + All-Pros) sits high in our list
  3. the formula still predicts each player's own future play (2023-25, weeks 1-9 -> 10-18)
Only the mix of existing, explainable stats changes; no player is adjusted by hand.
"""
import sys, json
sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd
import grades, direction_check as dc, calibrate_weights as cw
from outliers import TOP10

SIGN = dict(epa_play=1, value_over_avg=1, cpoe=1, success=1, int_worthy_rate=-1, fault_sack_rate=-1, ryoe_att=1,
            rec_yds_per_route=1, broken_tkl_rate=1, yds_after_contact=1, explosive_rate=1, epa_per_carry=1,
            fumble_rate=-1, yds_per_route=1, tgt_per_route=1, route_share=1, epa_per_target=1,
            separation_over_exp=1, catch_over_exp=1, yac_over_exp=1, drop_rate=-1, pressures_oe_per_100=-1,
            run_succ_oe=1, run_epa_oe=1, penalty_rate=-1, team_unit=1, pressure_rate=1, sack_rate=1,
            workload=1, rec_yds_pg=1, touches_pg=1, pressures_pg=1, stops_pg=1, ballplays_pg=1, tackles_pg=1,
            run_stop_rate=1, missed_tkl_rate=-1, yds_per_cov_snap=-1, tgt_per_cov_snap=-1, yds_per_target=-1,
            cmp_over_exp_allowed=-1, yac_per_cmp_allowed=-1, playmaking=1, rating_allowed=-1, plays_per_cov_snap=1)
# keep every formula football-sensible: no single stat runs a grade, and side jobs stay side jobs
CAP = {("OT", "team_unit"): .15, ("OG", "team_unit"): .15, ("C", "team_unit"): .15,
       ("OT", "workload"): .15, ("OG", "workload"): .15, ("C", "workload"): .15,("CB", "plays_per_cov_snap"): .20, ("CB", "playmaking"): .20, ("CB", "tackles_pg"): .05, ("CB", "stops_pg"): .05,
       ("CB", "ballplays_pg"): .10, ("CB", "pressure_rate"): .05, ("CB", "workload"): .10,
       "workload": .20, "rec_yds_pg": .30, "touches_pg": .25, "pressures_pg": .30, "stops_pg": .30, "tackles_pg": .20,
       "ballplays_pg": .25, ("RB", "rec_yds_per_route"): .30, ("RB", "explosive_rate"): .25, "penalty_rate": .30, "sack_rate": .25, ("CB", "pressure_rate"): .10, ("S", "pressure_rate"): .15,
       ("LB", "pressure_rate"): .20, ("CB", "run_stop_rate"): .15, ("CB", "missed_tkl_rate"): .15,
       ("EDGE", "yds_per_cov_snap"): .10, "team_unit": .35, "route_share": .30}
COVER = {"tgt_per_cov_snap", "yds_per_target", "cmp_over_exp_allowed", "yac_per_cmp_allowed", "rating_allowed", "yds_per_cov_snap"}
POOL = {
    "QB": ["epa_play", "value_over_avg", "cpoe", "success", "int_worthy_rate", "fault_sack_rate", "workload"],
    "RB": ["ryoe_att", "success", "rec_yds_per_route", "broken_tkl_rate", "yds_after_contact", "explosive_rate", "epa_per_carry", "fumble_rate", "workload", "touches_pg"],
    "WR": ["yds_per_route", "tgt_per_route", "route_share", "epa_per_target", "separation_over_exp", "catch_over_exp", "yac_over_exp", "drop_rate", "workload", "rec_yds_pg"],
    "TE": ["yds_per_route", "tgt_per_route", "route_share", "epa_per_target", "separation_over_exp", "catch_over_exp", "yac_over_exp", "drop_rate", "workload", "rec_yds_pg"],
    "OT": ["pressures_oe_per_100", "run_succ_oe", "run_epa_oe", "penalty_rate", "team_unit", "workload"],
    "OG": ["pressures_oe_per_100", "run_succ_oe", "run_epa_oe", "penalty_rate", "team_unit", "workload"],
    "C": ["pressures_oe_per_100", "run_succ_oe", "run_epa_oe", "penalty_rate", "team_unit", "workload"],
    "EDGE": ["pressure_rate", "sack_rate", "run_stop_rate", "missed_tkl_rate", "yds_per_cov_snap", "workload", "pressures_pg", "stops_pg", "tackles_pg"],
    "IDL": ["pressure_rate", "sack_rate", "run_stop_rate", "missed_tkl_rate", "workload", "pressures_pg", "stops_pg", "tackles_pg"],
    "LB": ["run_stop_rate", "yds_per_cov_snap", "pressure_rate", "missed_tkl_rate", "playmaking", "tgt_per_cov_snap", "rating_allowed", "sack_rate", "workload", "stops_pg", "tackles_pg", "ballplays_pg", "pressures_pg"],
    "CB": ["tgt_per_cov_snap", "yds_per_target", "cmp_over_exp_allowed", "yac_per_cmp_allowed", "playmaking", "rating_allowed",
           "run_stop_rate", "missed_tkl_rate", "pressure_rate", "yds_per_cov_snap", "plays_per_cov_snap", "workload", "ballplays_pg", "tackles_pg", "stops_pg"],
    "S": ["tgt_per_cov_snap", "yds_per_target", "cmp_over_exp_allowed", "yac_per_cmp_allowed", "playmaking", "rating_allowed",
          "run_stop_rate", "missed_tkl_rate", "pressure_rate", "yds_per_cov_snap", "plays_per_cov_snap", "workload", "ballplays_pg", "tackles_pg", "stops_pg"],
}


def zmat(df, mets):
    ref = df["qualified"].astype(bool).values if "qualified" in df else np.ones(len(df), bool)
    cols = []
    for m in mets:
        x = df[m].astype(float).values
        r = x[ref]
        lo, hi = np.nanquantile(r, .025), np.nanquantile(r, .975)
        x = np.clip(x, lo, hi); r = x[ref]
        sd = np.nanstd(r)
        cols.append(np.nan_to_num((x - np.nanmean(r)) / sd if sd > 0 else x * 0) * SIGN[m])
    return np.column_stack(cols), ref


def ranks(Z, ref, share, w):
    sc = Z @ w
    zs = (sc - sc[ref].mean()) / (sc[ref].std() or 1)
    semi = (share >= .3) & (share < 1)
    backup = min(zs[semi].mean() if semi.sum() >= 5 else -.5, -.25)
    zs = backup + (zs - backup) * share
    elig = share >= 0.5
    order = np.argsort(-np.where(elig, zs, -1e9))
    rk = np.full(len(zs), 999); pos_ = np.arange(1, len(zs) + 1)
    rk[order] = pos_
    rk[~elig] = 999
    return rk


def main(only=None, n=4000, seed=7):
    cur = json.loads(grades.WEIGHTS_FILE.read_text())
    seasons = {}
    for po in [True, False]:
        seasons[po] = grades.run_on(grades.load_all(2025, 18, playoffs=po))
    halves = []
    for s in [2023, 2024, 2025]:
        d = grades.load_all(s, 18)
        halves.append((cw.half_tables(cw.subset(d, 1, 9, s)), cw.half_tables(cw.subset(d, 10, 18, s))))
    rng = np.random.default_rng(seed)
    out, final = [], dict(cur)
    for pos, mets in POOL.items():
        if only and pos not in only:
            continue
        mets = [m for m in mets if m in seasons[True][pos]]
        prep = {}
        for po, res in seasons.items():
            df = res[pos].reset_index(drop=True)
            Z, ref = zmat(df, mets)
            keys = df["full_name"].map(dc.key).values
            prep[po] = (Z, ref, df["pt_share"].clip(0, 1).values if "pt_share" in df else np.ones(len(df)), keys)
        top = [dc.key(n) for n in TOP10.get(pos, [])]
        cons = list({dc.key(n) for n in dc.CONSENSUS[2025].get(pos, []) + TOP10.get(pos, [])})
        pred = []
        if pos not in ("OT", "OG", "C"):
            for a, b in halves:
                A = a[pos][a[pos]["qualified"]].drop_duplicates("gsis_id").set_index("gsis_id")
                B = b[pos].drop_duplicates("gsis_id").set_index("gsis_id")
                y = cw.outcome(pos, B.reset_index(), b["_rb_epa_touch"]).set_axis(B.index).dropna()
                c = A.index.intersection(y.index)
                Za, _ = zmat(A.loc[c].assign(qualified=True), mets)
                pred.append((Za, cw.zs(y.loc[c]).values))

        ordered = [dc.key(n) for n in (dc.CONSENSUS[2025][pos][:20] if pos in ("QB", "RB", "OT", "OG", "C", "CB") else TOP10.get(pos, []))]

        teams_of = {po: res_[pos].reset_index(drop=True)["team"].values for po, res_ in seasons.items()}

        def evaluate(w):
            o, lr, rho = 0.0, 0.0, 0.0
            for po, (Z, ref, share, keys) in prep.items():
                rk = ranks(Z, ref, share, w)
                if pos in ("OT", "OG", "C"):
                    top = teams_of[po][rk <= 10]
                    o += 0.5 * (len(top) - len(set(top)))          # each repeat team in the top 10 counts against
                    if (rk <= 2).sum() == 2 and len(set(teams_of[po][rk <= 2])) == 1:
                        o += 1.0                                    # same team at #1 and #2
                top = [dc.key(n) for n in TOP10.get(pos, [])]
                pos_of = dict(zip(keys, rk))
                o += (1.0 if po else 0.5) * sum((pos_of.get(k, 999) > 25) for k in top)
                if po:
                    lr = float(np.mean([np.log(min(pos_of.get(k, 200), 200)) for k in cons])) if cons else 0
                    got = [(i, pos_of[k]) for i, k in enumerate(ordered) if pos_of.get(k, 999) < 999]
                    if len(got) >= 4:
                        a = np.argsort(np.argsort([g[0] for g in got])); b = np.argsort(np.argsort([g[1] for g in got]))
                        rho = float(np.corrcoef(a, b)[0, 1])
            r = float(np.mean([np.corrcoef(Za @ w, y)[0, 1] for Za, y in pred])) if pred else 0.0
            return o, lr, r, rho

        def total(o, lr, r, rho):
            return 10 * o + lr + 3 * (1 - rho) - 1.5 * r

        cands = []
        base = {m: 0.0 for m in mets}
        for m, v, sg in cur[pos]:
            if m in base: base[m] = v
        cw0 = np.array([base[m] for m in mets])
        cap0 = np.array([CAP.get((pos, m), CAP.get(m, .45)) for m in mets])
        if not (cw0 > cap0 + 1e-9).any():
            cands.append(("current", cw0))
        else:
            cands.append(("current_capped_out", cw0))   # evaluated for the before-numbers, never picked
        cap = np.array([CAP.get((pos, m), CAP.get(m, .45)) for m in mets])
        tries = 0
        while len(cands) < n and tries < 30 * n:
            tries += 1
            w = rng.dirichlet(np.full(len(mets), .6))
            w = np.round(w / .05) * .05
            if w.sum() == 0: continue
            w = w / w.sum()
            if (w > cap + 1e-9).any() or (w > 0).sum() < min(3, len(mets)): continue
            if pos == "CB" and sum(w[i] for i, m in enumerate(mets) if m in COVER) < .5: continue
            cands.append(("search", w))
        rows = []
        for tag, w in cands:
            o, lr, r, rho = evaluate(w)
            rows.append((tag, o, lr, r, w, rho, total(o, lr, r, rho)))
        cur_o, cur_lr, cur_r = rows[0][1:4]
        rows = [x for x in rows if x[0] != "current_capped_out"]
        # local refinement: nudge the 40 best candidates thousands of times
        best = sorted(rows, key=lambda x: x[6])[:40]
        for _ in range(n // 2):
            base_w = best[rng.integers(len(best))][4]
            w = np.clip(base_w + rng.normal(0, .06, len(base_w)) * (rng.random(len(base_w)) < .5), 0, None)
            w = np.round(w / .05) * .05
            if w.sum() == 0: continue
            w = w / w.sum()
            if (w > cap + 1e-9).any() or (w > 0).sum() < min(3, len(mets)): continue
            if pos == "CB" and sum(w[i] for i, m in enumerate(mets) if m in COVER) < .5: continue
            o, lr, r, rho = evaluate(w)
            rows.append(("refine", o, lr, r, w, rho, total(o, lr, r, rho)))
        pick = min(rows, key=lambda x: x[6])
        w = pick[4]
        final[pos] = [(m, round(float(v), 3), SIGN[m]) for m, v in zip(mets, w) if v > 0.001]
        out.append(dict(pos=pos, outliers_before=cur_o, outliers_after=pick[1], consensus_logrank_before=round(cur_lr, 2),
                        consensus_logrank_after=round(pick[2], 2), predict_r_before=round(cur_r, 3),
                        predict_r_after=round(pick[3], 3), order_corr=round(pick[5], 2), tried=len(rows),
                        weights=", ".join(f"{'-' if SIGN[m] < 0 else ''}{m} {v:.0%}" for m, v in zip(mets, w) if v > 0.001)))
        print(out[-1], flush=True)
    grades.WEIGHTS_FILE.write_text(json.dumps(final, indent=1))
    pd.DataFrame(out).to_csv("output/weight_search.csv" if not only else "output/weight_search_partial.csv", index=False)


if __name__ == "__main__":
    main()
