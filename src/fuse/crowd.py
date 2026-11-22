"""Binary crowdsourcing benchmarks with per-worker labels and gold (Stage 1 datasets).

* RTE (Snow et al. 2008): 800 textual-entailment pairs, 164 MTurk workers, 10 labels each.
  From CrowdTruth/CrowdTruth-core `tutorial/data/rte.standardized.csv`.
* D_PosSent, D_Product, Bluebird ("Duck Identification") from the Zheng et al. (2017) truth
  inference benchmark (zhydhkcws.github.io/crowd_truth_inference/datasets.zip).

`load(name)` returns (V, gold): V is items x workers with +1 / -1 for the worker's label and
NaN where the worker did not label the item; gold is a bool array.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "crowd"
RTE_URL = "https://raw.githubusercontent.com/CrowdTruth/CrowdTruth-core/master/tutorial/data/rte.standardized.csv"
ZHENG_URL = "https://zhydhkcws.github.io/crowd_truth_inference/datasets.zip"
ZHENG = {"D_PosSent": "d_sentiment", "D_Product": "d_jn-product", "Bluebird": "d_Duck Identification"}
NAMES = ["RTE", *ZHENG]


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _matrix(items, workers, labels, gold_by_item):
    iu, ii = np.unique(items, return_inverse=True)
    wu, jj = np.unique(workers, return_inverse=True)
    V = np.full((len(iu), len(wu)), np.nan)
    V[ii, jj] = np.where(np.asarray(labels).astype(int) == 1, 1.0, -1.0)
    gold = np.array([gold_by_item[i] for i in iu]).astype(int) == 1
    return V, gold


def load(name: str, cache_dir: Path = CACHE):
    import pandas as pd

    if name == "RTE":
        df = pd.read_csv(_fetch(RTE_URL, Path(cache_dir) / "rte.standardized.csv"))
        df = df.drop_duplicates(["orig_id", "!amt_worker_ids"])
        gold = df.groupby("orig_id")["gold"].first().to_dict()
        return _matrix(df.orig_id.to_numpy(), df["!amt_worker_ids"].to_numpy(), df.response.to_numpy(), gold)
    z = zipfile.ZipFile(_fetch(ZHENG_URL, Path(cache_dir) / "zheng_datasets.zip"))
    d = ZHENG[name]
    ans = pd.read_csv(io.BytesIO(z.read(f"datasets/{d}/answer.csv")))
    tru = pd.read_csv(io.BytesIO(z.read(f"datasets/{d}/truth.csv")))
    ans.columns, tru.columns = ["i", "w", "a"], ["i", "t"]
    ans = ans.drop_duplicates(["i", "w"])
    ans = ans[ans.i.isin(tru.i)]
    return _matrix(ans.i.to_numpy(), ans.w.astype(str).to_numpy(), ans.a.to_numpy(), tru.set_index("i").t.to_dict())
