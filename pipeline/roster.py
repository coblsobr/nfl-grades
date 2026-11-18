"""Team rosters for the site: every starter plus backups who get real playing time, with our grades."""
import numpy as np
import pandas as pd

ORDER = ["QB", "RB", "FB", "WR", "TE", "OT", "OG", "C", "EDGE", "IDL", "LB", "CB", "S", "K", "P"]
UNIT = {**{p: "Offense" for p in ["QB", "RB", "FB", "WR", "TE", "OT", "OG", "C"]},
        **{p: "Defense" for p in ["EDGE", "IDL", "LB", "CB", "S"]}, "K": "Special teams", "P": "Special teams"}
ROTATION_SHARE = 0.10     # backups need at least 10% of their unit's snaps...
DEPTH = {"QB": 2, "RB": 3, "FB": 1, "WR": 5, "TE": 3, "OT": 3, "OG": 3, "C": 2, "EDGE": 4, "IDL": 4, "LB": 4, "CB": 4,
         "S": 3, "K": 1, "P": 1}  # ...and to be among the top this-many at their position on the team


def build(season, d, ps):
    """ps = this season's regular-season player_season rows."""
    sn = d["snaps"].copy()
    for c in ["offense_snaps", "defense_snaps", "st_snaps", "offense_pct", "defense_pct", "st_pct"]:
        sn[c] = sn[c].fillna(0)
    tot = sn.groupby(["game_id", "team"])[["offense_snaps", "defense_snaps", "st_snaps"]].max().groupby("team").sum()
    G = sn.groupby("team")["game_id"].nunique()
    sn["played"] = (sn["offense_snaps"] + sn["defense_snaps"] + sn["st_snaps"]) > 0
    a = sn.groupby(["team", "pfr_player_id"]).agg(
        player=("player", "last"), position=("position", "last"), games=("played", "sum"),
        off=("offense_snaps", "sum"), de=("defense_snaps", "sum"), st=("st_snaps", "sum"),
        s_off=("offense_pct", lambda x: (x >= .5).sum()), s_def=("defense_pct", lambda x: (x >= .5).sum()),
        s_st=("st_pct", lambda x: (x > 0).sum())).reset_index()
    gs = d["roster"].dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    a["gsis_id"] = a["pfr_player_id"].map(gs)
    best = (ps.sort_values(["ranked", "grade"], ascending=False).drop_duplicates("gsis_id")
              .set_index("gsis_id")[["pos", "grade", "rank", "player"]])
    a = a.join(best, on="gsis_id", rsuffix="_ps")
    kp = a["position"].isin(["K", "P"])
    a["pos"] = np.where(a["pos"].isna(), np.where(kp, a["position"], np.where(a["position"].eq("FB"), "FB", None)), a["pos"])
    a = a[a["pos"].notna()].copy()
    a["unit"] = a["pos"].map(UNIT)
    side = np.select([a["unit"].eq("Offense"), a["unit"].eq("Defense")], ["offense_snaps", "defense_snaps"], "st_snaps")
    a["snaps"] = np.select([a["unit"].eq("Offense"), a["unit"].eq("Defense")], [a["off"], a["de"]], a["st"])
    a["team_snaps"] = [tot.loc[t, c] if t in tot.index else np.nan for t, c in zip(a["team"], side)]
    a["share"] = a["snaps"] / a["team_snaps"]
    a["starts"] = np.select([a["unit"].eq("Offense"), a["unit"].eq("Defense")], [a["s_off"], a["s_def"]], a["s_st"]).astype(int)
    a["G"] = a["team"].map(G)
    a["depth"] = a.groupby(["team", "pos"])["share"].rank(ascending=False, method="first")
    backup = (a["share"] >= ROTATION_SHARE) & (a["depth"] <= a["pos"].map(DEPTH))
    a["role"] = np.where(a["unit"].eq("Special teams"), np.where(a["games"] >= .5 * a["G"], "Starter", ""),
                         np.where(a["starts"] >= .5 * a["G"], "Starter", np.where(backup, "Rotation", "")))
    a = a[a["role"] != ""].copy()
    a["player"] = a["player_ps"].fillna(a["player"])
    a["ord"] = a["pos"].map({p: i for i, p in enumerate(ORDER)}) * 1000 + (a["role"] != "Starter") * 500 - (a["share"] * 100).round()
    a["snap_pct"] = (100 * a["share"]).round().astype("Int64").astype(str) + "%"
    a["season"] = season
    a["gsis_id"] = a["gsis_id"].fillna("pfr:" + a["pfr_player_id"])
    a["key"] = a["season"].astype(str) + "|" + a["team"] + "|" + a["gsis_id"]
    a["rank"] = a["rank"].astype("Int64")
    return a.sort_values(["team", "ord"])[["key", "season", "team", "unit", "pos", "player", "gsis_id", "role", "starts",
                                          "games", "snap_pct", "grade", "rank", "ord"]]
