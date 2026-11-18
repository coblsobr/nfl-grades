"""
Direction check: are our grades in the same ballpark as outside experts?

NOT used to set weights. It answers one question per position: do the players experts call elite
land near the top of our list? If almost none do, something in our formula is off.

2025 "consensus elite" = PFF's published top lists (free articles, Feb-Mar 2026; full top 20 where we
have it, top 3 otherwise) plus the AP All-Pro first and second teams. Names only.
"""
import re
import numpy as np
import pandas as pd

CONSENSUS = {2025: {
    "QB": ["Matthew Stafford", "Joe Burrow", "Josh Allen", "Jordan Love", "Dak Prescott", "Brock Purdy",
           "Trevor Lawrence", "Drake Maye", "Sam Darnold", "Jalen Hurts", "Justin Herbert", "Jared Goff", "Bo Nix",
           "Caleb Williams", "Mac Jones", "Patrick Mahomes", "Kirk Cousins", "Jayden Daniels", "Tyler Shough",
           "Lamar Jackson"],
    "RB": ["Kenneth Walker III", "De'Von Achane", "Bijan Robinson", "Zach Charbonnet", "Josh Jacobs", "Jahmyr Gibbs",
           "Kyren Williams", "Breece Hall", "D'Andre Swift", "James Cook", "Jaylen Warren", "Rachaad White",
           "Cameron Skattebo", "Derrick Henry", "Jonathan Taylor", "Blake Corum", "Omarion Hampton",
           "Rhamondre Stevenson", "Christian McCaffrey", "J.K. Dobbins"],
    "WR": ["Puka Nacua", "Jaxon Smith-Njigba", "Amon-Ra St. Brown", "Ja'Marr Chase", "George Pickens", "Chris Olave"],
    "TE": ["George Kittle", "Dalton Kincaid", "Colston Loveland", "Trey McBride", "Kyle Pitts"],
    "OT": ["Penei Sewell", "Tristan Wirfs", "Trent Williams", "Garett Bolles", "Andrew Thomas", "Jordan Mailata",
           "Laremy Tunsil", "Colton McKivitz", "Alaric Jackson", "Bernhard Raimann", "Brian O'Neill", "Taylor Moton",
           "Abraham Lucas", "Darnell Wright", "Warren McClendon", "Mike McGlinchey", "Charles Cross", "Spencer Brown",
           "Paris Johnson", "Morgan Moses"],
    "OG": ["Quinn Meinerz", "Chris Lindstrom", "Kevin Dotson", "Quenton Nelson", "Peter Skoronski", "Joe Thuney",
           "Damien Lewis", "Mike Onwenu", "Steve Avila", "Kevin Zeitler", "Ed Ingram", "Tyler Smith", "Tyler Steen",
           "Isaac Seumalo", "Mason McCormick", "Jonah Jackson", "Tyler Booker", "Matthew Bergeron", "Joel Bitonio",
           "Tate Ratledge", "David Edwards"],
    "C": ["Creed Humphrey", "Aaron Brewer", "Tanor Bortolini", "Ryan Kelly", "Tyler Linderbaum", "Ryan Neuzil",
          "Zach Frazier", "Drew Dalman", "Jake Brendel", "Connor McGovern", "Coleman Shelton"],
    "EDGE": ["Will Anderson Jr.", "Myles Garrett", "Micah Parsons", "Brian Burns", "Danielle Hunter", "Aidan Hutchinson"],
    "IDL": ["Cameron Heyward", "Quinnen Williams", "Jeffery Simmons", "Zach Allen", "Leonard Williams"],
    "LB": ["Fred Warner", "Jack Campbell", "Devin Lloyd", "Jordyn Brooks", "Ernest Jones"],
    "CB": ['Devon Witherspoon', 'James Pierre', 'Mike Jackson', "Ja'Quan McMillian", 'Jamel Dean', 'Quinyon Mitchell', 'Kamari Lassiter', 'Cooper DeJean', 'Charvarius Ward', 'Sauce Gardner', 'Benjamin St-Juste', 'Trent McDuffie', 'Christian Gonzalez', 'Jarrian Jones', 'Chidobe Awuzie', 'Nohl Williams', 'Jaylen Watson', 'Carlton Davis III', 'Isaiah Rodgers', 'Eric Stokes', 'Derek Stingley Jr.', 'Patrick Surtain II'],
    "S": ["Antonio Johnson", "Kyle Hamilton", "Jalen Pitre", "Kevin Byard", "Jessie Bates III", "Talanoa Hufanga",
          "Xavier McKinney", "Derwin James"],
}}


def key(n):
    n = n.lower().replace("michael", "mike").replace("patrick surtain", "pat surtain")
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n))


def check(res, season):
    """res = {position: graded DataFrame} for one season. Returns (summary, detail)."""
    if season not in CONSENSUS:
        return None, None
    res = {p: (df[df["pt_share"] >= 0.5].reset_index(drop=True) if "pt_share" in df else df) for p, df in res.items()}
    rows = []
    where = {key(n): (p, i + 1, len(df)) for p, df in res.items() if "full_name" in df
             for i, n in enumerate(df["full_name"])}
    for pos, names in CONSENSUS[season].items():
        df = res[pos]
        idx = {key(n): i + 1 for i, n in enumerate(df["full_name"])}
        for nm in names:
            k = key(nm)
            if k in idx:
                r, n, p = idx[k], len(df), pos
            elif k in where:
                p, r, n = where[k]
            else:
                rows.append(dict(pos=pos, player=nm, our_pos=None, our_rank=None, pool=len(df), pct=None,
                                 note="did not qualify (injury / low snaps)"))
                continue
            rows.append(dict(pos=pos, player=nm, our_pos=p, our_rank=r, pool=n, pct=round(r / n, 3),
                             note="" if p == pos else f"graded at {p}"))
    det = pd.DataFrame(rows)
    out = []
    for pos, q in det[det["pct"].notna()].groupby("pos"):
        n, pool = len(q), int(q["pool"].iat[0])
        cut = max(n, int(np.ceil(.25 * pool)))           # "near the top" = top N or top quarter, whichever is bigger
        hits = int((q["our_rank"] <= cut).sum())
        rand = n * cut / pool                             # what random rankings would get
        best = min(n, cut)
        score = (hits - rand) / (best - rand) if best > rand else 0
        # rank agreement for ordered expert lists (Spearman on players both lists contain)
        order = {key(nm): i for i, nm in enumerate(CONSENSUS[season][pos])}
        rho = q.assign(er=q["player"].map(key).map(order))[["er", "our_rank"]].corr(method="spearman").iat[0, 1]
        out.append(dict(pos=pos, experts_found=n, pool=pool, near_top_cutoff=cut, hits=hits,
                        random_hits=round(rand, 1), agreement=round(score, 2),
                        rank_corr=round(rho, 2) if n >= 8 else None,
                        in_our_bottom_half=int((q["pct"] > .5).sum())))
    summ = pd.DataFrame(out).set_index("pos")
    summ["status"] = np.select([summ["agreement"] >= .5, summ["agreement"] >= .2], ["on track", "partly"], "off")
    summ["not_qualified"] = det[det["pct"].isna()].groupby("pos").size().reindex(summ.index).fillna(0).astype(int)
    order = list(CONSENSUS[season])
    return summ.reindex([p for p in order if p in summ.index]), det
