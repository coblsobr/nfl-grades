"""External sanity check (not used for fitting): where do 2025 AP All-Pros land in our 2025 full-season grades,
with the old hand-set weights vs. the calibrated weights?"""
import sys, re, pandas as pd, numpy as np
sys.argv = [sys.argv[0]]
import grades

ALL_PRO_2025 = {   # AP first + second team (Jan 2026)
 "QB": ["Matthew Stafford", "Drake Maye"], "RB": ["Bijan Robinson", "James Cook"],
 "WR": ["Puka Nacua", "Jaxon Smith-Njigba", "Ja'Marr Chase", "George Pickens", "Amon-Ra St. Brown", "Chris Olave"],
 "TE": ["Trey McBride", "Kyle Pitts"],
 "OT": ["Garett Bolles", "Penei Sewell", "Trent Williams", "Darnell Wright"],
 "OG": ["Joe Thuney", "Quinn Meinerz", "Quenton Nelson", "Chris Lindstrom"], "C": ["Creed Humphrey", "Aaron Brewer"],
 "EDGE": ["Myles Garrett", "Will Anderson Jr.", "Micah Parsons", "Brian Burns", "Danielle Hunter", "Aidan Hutchinson"],
 "IDL": ["Jeffery Simmons", "Zach Allen", "Leonard Williams", "Cameron Heyward"],
 "LB": ["Jack Campbell", "Jordyn Brooks", "Devin Lloyd", "Ernest Jones"],
 "CB": ["Derek Stingley Jr.", "Quinyon Mitchell", "Cooper DeJean", "Patrick Surtain II", "Devon Witherspoon", "Derwin James"],
 "S": ["Kyle Hamilton", "Kevin Byard", "Jessie Bates III", "Talanoa Hufanga", "Xavier McKinney"],
}
key = lambda n: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n.lower().replace("patrick surtain", "pat surtain")))


def ranks(res):
    out = {}
    for pos, names in ALL_PRO_2025.items():
        df = res[pos].reset_index(drop=True)
        allk = {key(n): (p, i + 1, len(res[p])) for p in ALL_PRO_2025 for i, n in enumerate(res[p]["full_name"])}
        for nm in names:
            k = key(nm)
            hit = df.index[df["full_name"].map(key) == k]
            if len(hit):
                out[(pos, nm)] = (int(hit[0]) + 1, len(df), pos)
            elif k in allk:
                p, r, n = allk[k]; out[(pos, nm)] = (r, n, p)
            else:
                out[(pos, nm)] = (None, len(df), "-")
    return out


if __name__ == "__main__":
    grades.USE_DEFAULTS = True
    old = ranks(grades.run_all(2025, 18))
    grades.USE_DEFAULTS = False
    new = ranks(grades.run_all(2025, 18))
    rows = []
    for (pos, nm), (r_old, n, p) in old.items():
        r_new = new[(pos, nm)][0]
        rows.append(dict(pos=pos, player=nm, graded_as=p, pool=n, rank_old=r_old, rank_new=r_new))
    df = pd.DataFrame(rows)
    df.to_csv("output/allpro_check_2025.csv", index=False)
    summ = df.groupby("pos").agg(n=("player", "size"), pool=("pool", "first"),
                                 median_old=("rank_old", "median"), median_new=("rank_new", "median"),
                                 top10_old=("rank_old", lambda s: (s <= 10).sum()),
                                 top10_new=("rank_new", lambda s: (s <= 10).sum()))
    print(summ.to_string())
    print("\nALL positions: median rank old %.0f -> new %.0f | All-Pros in our top 10: old %d -> new %d of %d"
          % (df.rank_old.median(), df.rank_new.median(), (df.rank_old <= 10).sum(), (df.rank_new <= 10).sum(), len(df)))
    print(df.to_string(index=False))
