"""Pick each position's weights using BOTH checks:
  1. Guardrail (free data): a candidate may not predict future play worse than our original hand-set formula
     (out-of-season r >= original r - 0.02).
  2. Direction (experts as a guide): among the candidates that pass, take the one whose 2025 grades best agree
     with the expert consensus (PFF's published lists + AP All-Pro).
Candidates per position: original hand-set, fitted-to-predict, and a 50/50 mix."""
import sys, json, pandas as pd
sys.argv = [sys.argv[0]]
import grades, direction_check as dc

opts = json.loads(grades.WEIGHTS_FILE.with_name("weights_options.json").read_text())
cur = json.loads(grades.WEIGHTS_FILE.read_text())
for p in ["OT", "OG", "C"]:   # OL: original vs. current (no prediction target in free data)
    opts[p] = {"old": {"r": None, "w": grades.DEFAULT_WEIGHTS[p]}, "new": {"r": None, "w": cur[p]}}
d25 = grades.load_all(2025, 18)
agree = {}
for variant in ["old", "new", "blend"]:
    w = dict(cur); w.update({p: o[variant]["w"] for p, o in opts.items() if variant in o})
    grades.WEIGHTS_FILE.write_text(json.dumps(w))
    summ, _ = dc.check(grades.run_on(d25), 2025)
    for p in summ.index:
        agree[(p, variant)] = float(summ.loc[p, "agreement"])
final, rows = dict(cur), []
for p, o in opts.items():
    r_old = o["old"]["r"]
    ok = [k for k in o if r_old is None or o[k]["r"] >= r_old - 0.02]
    pick = max(ok, key=lambda k: (agree.get((p, k), -9), o[k]["r"] or 0))
    final[p] = o[pick]["w"]
    rows.append(dict(pos=p, **{f"predict_r_{k}": (round(o[k]["r"], 3) if o[k]["r"] else None) for k in o},
                     **{f"expert_agree_{k}": agree.get((p, k)) for k in o}, chosen=pick))
grades.WEIGHTS_FILE.write_text(json.dumps(final, indent=1))
r = pd.DataFrame(rows); r.to_csv("output/weight_selection.csv", index=False)
print(r.to_string(index=False))
