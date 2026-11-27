import numpy as np
import pandas as pd

from src.eval.metrics import evaluate

d = "data_fake/splits/"
train = pd.read_parquet(d + "train.parquet")
test = pd.read_parquet(d + "test.parquet")
n_items = 300

# (1) Độ phổ biến đếm từ TRAIN, xếp giảm dần
pop = np.bincount(train.i_idx, minlength=n_items)
order = np.argsort(-pop, kind="stable")

# (2) Lịch sử train của từng user, để loại game đã xem
seen = train.groupby("u_idx").i_idx.apply(set).to_dict()

# (3) Top-20 cho mỗi user có trong test
users = test.u_idx.unique()
preds = np.array([[i for i in order if i not in seen.get(u, set())][:20] for u in users])

# (4) truth: khóa là CHỈ SỐ HÀNG, không phải u_idx
pos = {u: r for r, u in enumerate(users)}
truth = {}
for u, i in zip(test.u_idx, test.i_idx):
    truth.setdefault(pos[u], set()).add(int(i))

# (5) Chấm
out = evaluate(preds, truth, (10, 20), pop, n_items, train.u_idx.nunique())
for k, v in out.items():
    print(f"{k:12s} {v:.4f}")