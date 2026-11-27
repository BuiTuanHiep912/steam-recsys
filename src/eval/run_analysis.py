"""Chạy toàn bộ phân tích của C. NGƯỜI C — ngày 11-21.

Quyết định của nhóm (xem fit_and_target trong run_main.py):
    --split valid  (mặc định) -> học trên train,       RQ3 dùng cutoff_train
    --split test   (ngày 21)  -> học trên train+valid, RQ3 dùng cutoff_valid

Chạy:
    python -m src.eval.run_analysis --data data_fake --scheme linear --alpha 40
    python -m src.eval.run_analysis --data data --split test --scheme <thắng> --alpha <thắng>
"""
import argparse

import numpy as np
import pandas as pd

from src.eval.analyze import (make_all_figures, metrics_by_breadth, popularity_bias,
                              profile_entropy, rq3_table, sample_recommendations)
from src.eval.evaluate import build_truth, rank_all
from src.eval.run_main import fit_and_target, get_build_confidence, load
from src.models.knn import ItemKNN
from src.models.popular import MostPopular
from src.models.wmf import WMF

p = argparse.ArgumentParser()
p.add_argument("--data", default="data_fake")
p.add_argument("--split", default="valid", choices=["valid", "test"])
p.add_argument("--knn-k", type=int, default=50)
p.add_argument("--scheme", default="log")       # đổi thành cấu hình thắng RQ1 sau ngày 14
p.add_argument("--alpha", type=float, default=40.0)
p.add_argument("--out", default="reports/figures")
a = p.parse_args()

# ---- Dữ liệu: tập học và mốc cắt đi theo tập được chấm ----------------------
maps, sp = load(a.data)
fit_df, target, train_set, cutoff_ts = fit_and_target(sp, maps, a.split)
items = pd.read_parquet(f"{a.data}/interim/items.parquet")
nu, ni = maps["n_users"], maps["n_items"]
item_pop = np.bincount(fit_df.i_idx.to_numpy(), minlength=ni)   # đếm từ TẬP HỌC
users = target.u_idx.unique()
print(f"học trên {train_set} ({len(fit_df):,} dòng), chấm {a.split} ({len(target):,} dòng)")

pop = MostPopular(nu, ni); pop.fit(fit_df)
knn = ItemKNN(nu, ni, a.knn_k); knn.fit(fit_df)
wmf = WMF(nu, ni)
wmf.fit(fit_df, confidence=get_build_confidence()(fit_df, scheme=a.scheme, alpha=a.alpha))
models = {"MostPopular": pop, "ItemKNN": knn, "WMF": wmf}

print("\n## 1. Thiên lệch phổ biến (top-10)")
for name, m in models.items():
    print(f"\n{name}\n", popularity_bias(rank_all(m, users, 10), item_pop).round(3).to_string(index=False))

print("\n## 2. Đọc thử gợi ý")
rng = np.random.default_rng(42)
print(sample_recommendations(wmf, fit_df, items, rng.choice(users, 5, replace=False)))

print(f"\n## 3. RQ3 (game mới = phát hành từ mốc cắt của {train_set})")
rq3 = rq3_table({"WMF": wmf, "ItemKNN": knn}, fit_df, target, items, cutoff_ts)
print(rq3.round(4))

print("\n## 4. Độ rộng sở thích (WMF)")
ent = profile_entropy(fit_df, items)
breadth = metrics_by_breadth(rank_all(wmf, users, 20), build_truth(target, users), users,
                             ent, item_pop, fit_df.u_idx.nunique())
print(breadth.round(4).to_string(index=False))

# ---- Hình: chỉ lấy dòng của ĐÚNG bộ dữ liệu này, bỏ lần chạy trùng -----------
runs = pd.read_csv("results/runs.csv")
runs = runs[runs["data"] == a.data].drop_duplicates(
    ["model", "split", "scheme", "alpha"], keep="last")
out = f"{a.out}/{a.split}"                      # valid và test không ghi đè nhau
make_all_figures(runs, item_pop, rq3, breadth, out)
print(f"\nĐã xuất hình vào {out}/")
