"""Pick final weights per position.
Rule: keep only candidate weight sets (old / fitted / 50-50 blend) whose out-of-season prediction r is within
0.03 of the best. Among those, take the one whose 2025 grades best match the AP All-Pro vote (lower median rank).
Prediction is the main test; expert consensus only breaks near-ties."""
import sys, json, numpy as np, pandas as pd
sys.argv = [sys.argv[0]]
import grades
from evaluate_2025 import ALL_PRO_2025, ranks

opts = json.loads(grades.WEIGHTS_FILE.with_name("weights_options.json").read_text())
base = json.loads(grades.WEIGHTS_FILE.read_text())
med = {}
for variant in ["old", "new", "blend"]:
    w = dict(base); w.update({pos: o[variant]["w"] for pos, o in opts.items()})
    grades.WEIGHTS_FILE.write_text(json.dumps(w))
    rk = ranks(grades.run_all(2025, 18))
    for pos in opts:
        med[(pos, variant)] = float(np.median([r for (p, _), (r, _, _) in rk.items() if p == pos and r is not None]))
final, rows = dict(base), []
for pos, o in opts.items():
    best_r = max(v["r"] for v in o.values())
    ok = [k for k, v in o.items() if v["r"] >= best_r - 0.03]
    pick = min(ok, key=lambda k: (med[(pos, k)], -o[k]["r"]))
    final[pos] = o[pick]["w"]
    rows.append(dict(pos=pos, **{f"r_{k}": round(o[k]["r"], 3) for k in o}, **{f"allpro_med_{k}": med[(pos, k)] for k in o},
                     chosen=pick, weights=", ".join(f"{m} {w:.0%}" for m, w, _ in o[pick]["w"])))
grades.WEIGHTS_FILE.write_text(json.dumps(final, indent=1))
r = pd.DataFrame(rows); r.to_csv("output/weight_selection.csv", index=False)
print(r.drop(columns="weights").to_string(index=False))
for x in rows: print(x["pos"], "->", x["weights"])
