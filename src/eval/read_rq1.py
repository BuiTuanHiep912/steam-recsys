"""Đọc bảng RQ1 từ results/runs.csv. NGƯỜI C — ngày 14.
Chạy: python -m src.eval.read_rq1 --data data_fake"""
import argparse

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--data", default="data_fake")
p.add_argument("--runs", default="results/runs.csv")
a = p.parse_args()

r = pd.read_csv(a.runs)
r = r[(r.model == "WMF") & (r.split == "valid") & (r.data == a.data)]
print(f"Số dòng WMF/valid: {len(r)}")

# Cùng (scheme, alpha) chạy nhiều lần -> giữ lần MỚI NHẤT (file ghi theo thời gian)
r = r.drop_duplicates(subset=["scheme", "alpha"], keep="last")
print(f"Sau khi bỏ trùng: {len(r)}  (phải là 16)")

table = r.pivot(index="scheme", columns="alpha", values="ndcg@10")
table = table.reindex(["binary", "log", "linear", "capped"])
print("\nnDCG@10 trên valid:")
print(table.round(4).to_string())

best = r.loc[r["ndcg@10"].idxmax()]
print(f"\nCấu hình thắng: scheme={best.scheme}, alpha={best.alpha:g}, "
      f"nDCG@10={best['ndcg@10']:.4f}, seed={best.seed}")
b = r[r.scheme == "binary"].sort_values("ndcg@10").iloc[-1]
print(f"Binary tốt nhất: alpha={b.alpha:g}, nDCG@10={b['ndcg@10']:.4f}")
