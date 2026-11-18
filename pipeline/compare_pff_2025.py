"""Compare our 2025 OT/OG/C grades with PFF's published 2025 top lists (free articles, Feb 2026)."""
import pandas as pd, numpy as np, re

PFF = {  # rank order as published (ties keep listed order); PFF qualifier = 700+ snaps
 "OT": ["Penei Sewell","Tristan Wirfs","Trent Williams","Garett Bolles","Andrew Thomas","Jordan Mailata","Laremy Tunsil",
        "Colton McKivitz","Alaric Jackson","Bernhard Raimann","Brian O'Neill","Taylor Moton","Abraham Lucas","Darnell Wright",
        "Warren McClendon","Mike McGlinchey","Charles Cross","Spencer Brown","Paris Johnson","Morgan Moses"],
 "OG": ["Quinn Meinerz","Chris Lindstrom","Kevin Dotson","Quenton Nelson","Peter Skoronski","Joe Thuney","Damien Lewis",
        "Mike Onwenu","Steve Avila","Kevin Zeitler","Ed Ingram","Tyler Smith","Tyler Steen","Isaac Seumalo","Mason McCormick",
        "Jonah Jackson","Tyler Booker","Matthew Bergeron","Joel Bitonio","Tate Ratledge","David Edwards"],
 "C":  ["Creed Humphrey","Aaron Brewer","Tanor Bortolini","Ryan Kelly","Tyler Linderbaum","Ryan Neuzil","Zach Frazier",
        "Drew Dalman","Jake Brendel","Connor McGovern","Coleman Shelton"],
}
key = lambda n: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n.lower().replace("michael", "mike")))

ours = pd.read_csv("output/validation_2025_ol.csv")
allpos = ours.copy(); allpos["k"] = allpos["full_name"].map(key)
rows = []
for pos, names in PFF.items():
    g = ours[ours["pos"] == pos].sort_values("grade", ascending=False).reset_index(drop=True)
    g["our_rank"] = g.index + 1
    g["team_rank"] = g["team_unit"].rank(ascending=False, method="min").astype(int)   # baseline: team line only
    g["k"] = g["full_name"].map(key)
    n = len(g); top = len(names)
    for i, nm in enumerate(names):
        r = g[g["k"] == key(nm)]
        other = allpos[(allpos["k"] == key(nm)) & (allpos["pos"] != pos)]
        rows.append(dict(pos=pos, pff_rank=i + 1, player=nm,
                         our_rank=int(r["our_rank"].iat[0]) if len(r) else None,
                         team_only_rank=int(r["team_rank"].iat[0]) if len(r) else None,
                         pool=n, note="" if len(r) else ("graded at " + other["pos"].iat[0] if len(other) else "not qualified / missed time")))
    sub = pd.DataFrame([x for x in rows if x["pos"] == pos])
    found = sub[sub["our_rank"].notna()]
    hit = (found["our_rank"] <= top).sum(); hit_team = (found["team_only_rank"] <= top).sum()
    print(f"{pos}: PFF top {top} -> {len(found)} in our pool of {n}; in our top {top}: {hit} "
          f"(team-only baseline {hit_team}, random ~{top*top/n:.1f}); median our rank {found['our_rank'].median():.0f}")
    print("   our top 10:", list(g["full_name"].head(10)))
out = pd.DataFrame(rows); out.to_csv("output/compare_pff_2025.csv", index=False)
print(out.to_string(index=False))
