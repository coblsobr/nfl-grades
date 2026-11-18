"""Outlier check: every player in an expert top 10 (2025) should be inside our top 25 at his position."""
import sys, re, json
sys.argv = [sys.argv[0]]
import numpy as np, pandas as pd
import grades, direction_check as dc

C = dc.CONSENSUS[2025]
TOP10 = {p: C[p][:10] for p in ["QB", "RB", "OT", "OG", "C"]}
TOP10.update({
    "WR": ["Puka Nacua", "Jaxon Smith-Njigba", "Amon-Ra St. Brown", "Ja'Marr Chase", "Drake London", "George Pickens", "Davante Adams"],
    "TE": ["George Kittle", "Dalton Kincaid", "Colston Loveland", "Trey McBride", "Brock Bowers", "Kyle Pitts"],
    "EDGE": ["Will Anderson Jr.", "Myles Garrett", "Micah Parsons", "Danielle Hunter", "Aidan Hutchinson", "Greg Rousseau", "Laiatu Latu"],
    "IDL": ["Cameron Heyward", "Quinnen Williams", "Jeffery Simmons", "Poona Ford", "Kobie Turner", "Leonard Williams"],
    "LB": ["Fred Warner", "Jack Campbell", "Devin Lloyd", "Devin Bush", "Demario Davis"],
    "CB": ['Devon Witherspoon', 'James Pierre', 'Mike Jackson', "Ja'Quan McMillian", 'Jamel Dean', 'Quinyon Mitchell', 'Kamari Lassiter', 'Cooper DeJean', 'Charvarius Ward', 'Sauce Gardner'],
    "S": ["Antonio Johnson", "Kyle Hamilton", "Jalen Pitre", "Minkah Fitzpatrick", "Derwin James", "Xavier McKinney"],
})


def find(res):
    rows = []
    for pos, names in TOP10.items():
        df = res[pos]
        df = df[df["pt_share"] >= 0.5].reset_index(drop=True) if "pt_share" in df else df.reset_index(drop=True)
        idx = {dc.key(n): i + 1 for i, n in enumerate(df["full_name"])}
        for i, nm in enumerate(names):
            r = idx.get(dc.key(nm))
            where = None
            if r is None:
                for p2, d2 in res.items():
                    if "full_name" in d2:
                        k2 = {dc.key(n): j + 1 for j, n in enumerate(d2["full_name"].reset_index(drop=True))}
                        if dc.key(nm) in k2:
                            where = f"{p2} #{k2[dc.key(nm)]}"
            rows.append(dict(pos=pos, expert=i + 1, player=nm, ours=r, pool=len(df), where=where))
    t = pd.DataFrame(rows)
    t["outlier"] = t["ours"].isna() | (t["ours"] > 25)
    return t


if __name__ == "__main__":
    for po in [True, False]:
        res = grades.run_on(grades.load_all(2025, 18, playoffs=po))
        t = find(res)
        o = t[t.outlier]
        print(("WITH PLAYOFFS" if po else "REGULAR SEASON"), f"outliers: {len(o)} of {len(t)}")
        print(o.to_string(index=False))
