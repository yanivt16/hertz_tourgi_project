"""Two crowds of non-experts on hard questions, with gold that does not come from the crowd.

* Deepfakes (Groh et al. 2022, PNAS; experiment 1, Prolific sample): 54 videos x 304 participants,
  each rated ~20 videos on a slider from "100% confidence this is NOT a DeepFake" (0) to
  "100% confidence this is a DeepFake" (100). Gold: the video is a deepfake (known by construction).
  We keep the first guess per (participant, video) (`guess_round == 1`; in round 2 participants could
  revise) and drop the attention-check video.
  Source: github.com/mattgroh/cognitive-science-detecting-deepfakes, replication/e1_prolific.csv.
* Fact-checking (Allen et al. 2021, Science Advances, "Scaling up fact-checking using the wisdom of
  crowds"): 207 news articles x 1,128 MTurk laypeople (~19 articles each, ~100 per article). Each
  rated 7 items on 1-7 (happened, unbiased, objective, accurate, reliable, trustworthy, true); we use
  their mean. Gold: the modal verdict of the 3 professional fact-checkers (`pro == 1` in the same
  file), positive = False or Misleading (survey codes 1 True, 2 Misleading, 3 False, 4 Not sure;
  checked against FB_News_Evaluation_Study_MTurk.docx on OSF). Articles without a majority verdict,
  or whose majority is Not sure, are dropped.
  Source: osf.io/hts3w, Data/RepAB.csv.

`load(name)` returns a long DataFrame (item, rater, score, plus rater demographics) where a higher
score means "more likely positive", and a bool Series of gold labels indexed by item.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "hard_crowds"
URLS = {
    "deepfake": ("https://raw.githubusercontent.com/mattgroh/cognitive-science-detecting-deepfakes/main/replication/e1_prolific.csv", "e1_prolific.csv"),
    "factcheck": ("https://osf.io/download/f9z6s/", "RepAB.csv"),
}
NAMES = list(URLS)


def _fetch(name: str) -> Path:
    url, fn = URLS[name]
    path = CACHE / fn
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _deepfake():
    d = pd.read_csv(_fetch("deepfake"), low_memory=False)
    d = d[(d.guess_round == 1) & (d.video != "v2attentioncheck.mp4")]
    d = d.drop_duplicates(["prolific_pid", "video"])
    age = pd.cut(pd.to_numeric(d.age, errors="coerce"), [0, 29, 44, 200], labels=["<30", "30-44", "45+"])
    out = pd.DataFrame({"item": d.video.to_numpy(), "rater": d.prolific_pid.to_numpy(),
                        "score": d.guess_int.astype(float).to_numpy(),
                        "sex": d.Sex.astype(str).to_numpy(), "age": age.astype(str).to_numpy(),
                        "ethnicity": d["Ethnicity (Simplified)"].astype(str).to_numpy()})
    gold = d.groupby("video").fake.first().astype(bool)
    return out, gold


def _factcheck():
    d = pd.read_csv(_fetch("factcheck"), low_memory=False)
    items = ["mru", "mro", "mra", "mrc", "mrr", "mrw", "mrh"]
    pro = d[d.pro == 1]

    def verdict(v):
        c = v.value_counts()
        return c.index[0] if c.iloc[0] >= 2 and (len(c) == 1 or c.iloc[0] > c.iloc[1]) else np.nan

    v = pro.groupby("msg_id").mis.agg(verdict)
    v = v[v.isin([1, 2, 3])]
    gold = v.isin([2, 3]).rename("gold")
    lay = d[(d.pro == 0) & d.msg_id.isin(gold.index)].copy()
    lay["score"] = 8 - lay[items].mean(1)  # higher = rated less accurate = more likely false/misleading
    lay = lay[lay.score.notna()].drop_duplicates(["w_id", "msg_id"])
    party = pd.cut(lay.polpref, [0, 3, 6], labels=["Dem", "Rep"]).astype(str)
    out = pd.DataFrame({"item": lay.msg_id.to_numpy(), "rater": lay.w_id.astype(str).to_numpy(),
                        "score": lay.score.to_numpy(), "party": party.to_numpy(),
                        "polpref": lay.polpref.to_numpy(), "sex": lay.gender.astype(str).to_numpy(),
                        "age": pd.cut(lay.age, [0, 29, 44, 200], labels=["<30", "30-44", "45+"]).astype(str).to_numpy()})
    return out, gold


def load(name: str):
    return {"deepfake": _deepfake, "factcheck": _factcheck}[name]()


def matrix(long: pd.DataFrame, gold: pd.Series, by: str = "rater"):
    """Items x raters (or x groups, mean score per group) matrix with NaN for missing, plus y."""
    M = long.pivot_table(index="item", columns=by, values="score", aggfunc="mean")
    M = M.reindex(gold.index)
    return M.to_numpy(float), gold.to_numpy(bool), list(M.columns)
