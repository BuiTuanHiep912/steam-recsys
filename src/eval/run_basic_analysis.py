"""Hai phân tích cơ bản. NGƯỜI C — ngày 11-14.

Chạy: python -m src.eval.run_basic_analysis --data data_fake
Ghi ra: reports/phan_tich_11-14/  (bảng .csv, hình .png, gợi ý mẫu .txt)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.analyze import popularity_bias, sample_recommendations
from src.eval.evaluate import rank_all
from src.eval.run_main import get_build_confidence
from src.models.knn import ItemKNN
from src.models.popular import MostPopular
from src.models.wmf import WMF

p = argparse.ArgumentParser()
p.add_argument("--data", default="data_fake")
p.add_argument("--split", default="valid")      # KHÔNG dùng test trước ngày 21
p.add_argument("--knn-k", type=int, default=50)
p.add_argument("--scheme", default="log")
p.add_argument("--alpha", type=float, default=40.0)
p.add_argument("--n-sample", type=int, default=5)
a = p.parse_args()

out = Path("reports/phan_tich_11-14")
out.mkdir(parents=True, exist_ok=True)

# ---- 1. Dữ liệu và mô hình -------------------------------------------------
maps = json.load(open(f"{a.data}/interim/maps.json"))
train = pd.read_parquet(f"{a.data}/splits/train.parquet")
target = pd.read_parquet(f"{a.data}/splits/{a.split}.parquet")
items = pd.read_parquet(f"{a.data}/interim/items.parquet")
nu, ni = maps["n_users"], maps["n_items"]
item_pop = np.bincount(train["i_idx"].to_numpy(), minlength=ni)
users = target["u_idx"].unique()

models = [MostPopular(nu, ni), ItemKNN(nu, ni, a.knn_k), WMF(nu, ni)]
conf = get_build_confidence()(train, scheme=a.scheme, alpha=a.alpha)
for m in models:
    m.fit(train, confidence=conf)

# ---- 2. Phân tích 1: thiên lệch phổ biến -----------------------------------
tables = []
for m in models:
    t = popularity_bias(rank_all(m, users, k=10), item_pop)
    t.insert(0, "model", m.name)
    tables.append(t)
bias = pd.concat(tables, ignore_index=True)
bias.round(4).to_csv(out / "popularity_bias.csv", index=False)
print(bias.round(3).to_string(index=False))

# Hình: mỗi nhóm một cụm cột; cột xám là tỷ lệ thật trong train (mốc so sánh)
wide = bias.pivot(index="nhóm", columns="model", values="% gợi ý")
wide = wide.loc[bias["nhóm"].unique(), [m.name for m in models]]
wide.insert(0, "train (mốc)", bias.drop_duplicates("nhóm").set_index("nhóm")["% tương tác train"])
fig, ax = plt.subplots(figsize=(7, 4))
wide.plot.bar(ax=ax, rot=0, color=["#BBBBBB", "#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("tỷ lệ gợi ý top-10"); ax.set_xlabel("")
ax.set_title("Gợi ý rơi vào nhóm game nào?")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(out / "popularity_bias.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- 3. Phân tích 2: đọc thử gợi ý -----------------------------------------
# Chọn user có lịch sử dài vừa phải (5-30 game) để dễ đọc, seed cố định để tái lập
n_hist = train.groupby("u_idx").size()
pool = [u for u in users if 5 <= n_hist.get(u, 0) <= 30]
rng = np.random.default_rng(42)
chosen = rng.choice(pool, size=min(a.n_sample, len(pool)), replace=False)

text = []
for m in models[1:]:                          # MostPopular giống nhau cho mọi người, bỏ
    text.append(f"\n################ {m.name} ################")
    text.append(sample_recommendations(m, train, items, chosen))
(out / "sample_recommendations.txt").write_text("\n".join(text), encoding="utf-8")
print(f"\nĐã ghi {out}/popularity_bias.csv, popularity_bias.png, sample_recommendations.txt")