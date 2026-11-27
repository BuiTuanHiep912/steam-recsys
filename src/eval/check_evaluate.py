"""Tiêu chí Xong khi của ngày 4-5: run_eval chạy với MostPopular VÀ ItemKNN mà
không sửa gì. Chạy: python -m src.eval.check_evaluate --data data_fake"""
import argparse
import json

import pandas as pd

from src.eval.evaluate import run_eval
from src.models.knn import ItemKNN
from src.models.popular import MostPopular

p = argparse.ArgumentParser()
p.add_argument("--data", default="data_fake")
a = p.parse_args()

maps = json.load(open(f"{a.data}/interim/maps.json"))
train = pd.read_parquet(f"{a.data}/splits/train.parquet")
test = pd.read_parquet(f"{a.data}/splits/test.parquet")
n_users, n_items = maps["n_users"], maps["n_items"]

rows = []
for model in [MostPopular(n_users, n_items), ItemKNN(n_users, n_items, k_neighbors=50)]:
    model.fit(train)                          # cùng MỘT cách gọi cho mọi mô hình
    rows.append({"model": model.name, **run_eval(model, train, test)})

print(pd.DataFrame(rows).round(4).to_string(index=False))