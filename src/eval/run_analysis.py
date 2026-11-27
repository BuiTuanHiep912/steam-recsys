"""Chạy toàn bộ phân tích của C. Chạy: python -m src.eval.run_analysis --data data_fake"""
import argparse
import json

import numpy as np
import pandas as pd

from src.eval.analyze import (make_all_figures, metrics_by_breadth, popularity_bias,
                              profile_entropy, rq3_table, sample_recommendations)
from src.eval.evaluate import build_truth, rank_all
from src.eval.run_main import get_build_confidence
from src.models.knn import ItemKNN
from src.models.popular import MostPopular
from src.models.wmf import WMF

p = argparse.ArgumentParser()
p.add_argument("--data", default="data_fake")
p.add_argument("--scheme", default="log")       # đổi thành cấu hình thắng RQ1 sau ngày 14
p.add_argument("--alpha", type=float, default=40.0)
p.add_argument("--out", default="reports/figures")
a = p.parse_args()

maps = json.load(open(f"{a.data}/interim/maps.json"))
train = pd.read_parquet(f"{a.data}/splits/train.parquet")
test = pd.read_parquet(f"{a.data}/splits/test.parquet")
items = pd.read_parquet(f"{a.data}/interim/items.parquet")
nu, ni = maps["n_users"], maps["n_items"]
item_pop = np.bincount(train.i_idx.to_numpy(), minlength=ni)
users = test.u_idx.unique()

pop = MostPopular(nu, ni); pop.fit(train)
knn = ItemKNN(nu, ni, 50); knn.fit(train)
wmf = WMF(nu, ni); wmf.fit(train, confidence=get_build_confidence()(train, a.scheme, a.alpha))
models = {"MostPopular": pop, "ItemKNN": knn, "WMF": wmf}

print("\n## 1. Thiên lệch phổ biến (top-10)")
for name, m in models.items():
    print(f"\n{name}\n", popularity_bias(rank_all(m, users, 10), item_pop).round(3).to_string(index=False))

print("\n## 2. Đọc thử gợi ý")
rng = np.random.default_rng(42)
print(sample_recommendations(wmf, train, items, rng.choice(users, 5, replace=False)))

print("\n## 3. RQ3")
rq3 = rq3_table({"WMF": wmf, "ItemKNN": knn}, train, test, items, maps["cutoff_train"])
print(rq3.round(4))

print("\n## 4. Độ rộng sở thích (WMF)")
ent = profile_entropy(train, items)
breadth = metrics_by_breadth(rank_all(wmf, users, 20), build_truth(test, users), users,
                             ent, item_pop, train.u_idx.nunique())
print(breadth.round(4).to_string(index=False))

runs = pd.read_csv("results/runs.csv")
make_all_figures(runs, item_pop, rq3, breadth, a.out)
print(f"\nĐã xuất hình vào {a.out}/")