"""
Sinh bộ dữ liệu giả đúng lược đồ trong contracts.py.

Chạy trong vài giây, để B và C phát triển song song từ ngày 2 mà không phải
chờ A xử lý xong 41 triệu dòng thật.

    python make_fake.py --out data_fake

Cấu trúc được CÀI SẴN, không ngẫu nhiên hoàn toàn:

  * 500 người dùng chia thành 5 nhóm sở thích.
  * 300 game chia thành 5 nhóm tương ứng, mỗi nhóm có bộ thẻ riêng.
  * Mỗi người tương tác chủ yếu trong nhóm của mình, lẫn khoảng 10% nhiễu.
  * Trong mỗi nhóm, độ hút khách của game lệch theo Zipf.

NGƯỠNG KIỂM THỬ. Với seed 42, một ItemKNN cosine viết đúng cho nDCG@10 ≈ 0,25
còn MostPopular chỉ khoảng 0,04 — chênh hơn sáu lần. Mô hình của bạn chạy trên
bộ này mà không bỏ xa MostPopular thì lỗi nằm ở code, không phải ở dữ liệu.
Đây là cách bắt lỗi rẻ nhất trong cả đồ án.

Dữ liệu cũng chứa sẵn một nhóm game phát hành sau mốc cắt huấn luyện, để phát
triển phần RQ3 (cold-start item) trước khi có dữ liệu thật.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 500
N_ITEMS = 300
N_GROUPS = 5
ITEMS_PER_GROUP = N_ITEMS // N_GROUPS

# Mốc thời gian: cửa sổ 300 ngày kể từ 01/01/2022.
DAY = 86_400
T0 = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp())
WINDOW_DAYS = 300
CUT_TRAIN_DAY = 240   # train:  ts <  T0 + 240 ngày
CUT_VALID_DAY = 270   # valid:  240 <= ngày < 270   | test: >= 270

TAGS_BY_GROUP = [
    ["Roguelike", "Pixel Graphics", "Difficult", "Indie", "Action"],
    ["Grand Strategy", "Historical", "Turn-Based", "Simulation", "Complex"],
    ["FPS", "Multiplayer", "Competitive", "Shooter", "Team-Based"],
    ["Story Rich", "RPG", "Open World", "Fantasy", "Singleplayer"],
    ["Cozy", "Farming Sim", "Relaxing", "Crafting", "Cute"],
]
GROUP_NAMES = ["Roguelite", "Strategy", "Shooter", "RPG", "Cozy"]


def build_items(rng: np.random.Generator) -> pd.DataFrame:
    i_idx = np.arange(N_ITEMS, dtype=np.int32)
    group = i_idx // ITEMS_PER_GROUP

    # Phần lớn game phát hành trước cửa sổ; khoảng 12% ra sau mốc cắt train
    # để luôn có nhóm cold-start item khi phát triển RQ3.
    release_day = rng.integers(-400, CUT_TRAIN_DAY, size=N_ITEMS)
    late = rng.random(N_ITEMS) < 0.12
    release_day[late] = rng.integers(CUT_TRAIN_DAY, WINDOW_DAYS, size=int(late.sum()))

    tags, descriptions, titles = [], [], []
    for i in range(N_ITEMS):
        g = int(group[i])
        pool = TAGS_BY_GROUP[g]
        picked = list(rng.choice(pool, size=3, replace=False))
        # một thẻ lạc sang nhóm khác, để TF-IDF không tách nhóm quá dễ
        if rng.random() < 0.3:
            other = int(rng.integers(0, N_GROUPS))
            picked.append(str(rng.choice(TAGS_BY_GROUP[other])))
        tags.append("|".join(picked))
        descriptions.append(
            f"A {picked[0].lower()} game with {picked[1].lower()} elements. "
            f"Part of the {GROUP_NAMES[g].lower()} catalogue."
        )
        titles.append(f"{GROUP_NAMES[g]} Game {i:03d}")

    return pd.DataFrame({
        "i_idx": i_idx,
        "app_id": (100_000 + i_idx * 7).astype(np.int64),
        "title": pd.array(titles, dtype="string"),
        "date_release": (T0 + release_day * DAY).astype(np.int64),
        "price_final": np.round(rng.uniform(0, 60, N_ITEMS), 2).astype(np.float32),
        "win": np.ones(N_ITEMS, dtype=bool),
        "mac": rng.random(N_ITEMS) < 0.45,
        "linux": rng.random(N_ITEMS) < 0.30,
        "tags": pd.array(tags, dtype="string"),
        "description": pd.array(descriptions, dtype="string"),
    })


def build_interactions(rng: np.random.Generator, items: pd.DataFrame) -> pd.DataFrame:
    release_day = ((items["date_release"].to_numpy() - T0) // DAY).astype(int)

    # Trong mỗi nhóm, độ hút khách của game lệch theo Zipf: vài game được hầu
    # hết thành viên của nhóm chơi, phần còn lại thưa dần. Không có bước này
    # thì mọi game trong nhóm đều như nhau, mô hình chỉ đoán được nhóm chứ
    # không xếp hạng được bên trong nhóm — và dữ liệu giả mất tác dụng kiểm thử.
    appeal = {}
    for g in range(N_GROUPS):
        w = 1.0 / np.power(np.arange(1, ITEMS_PER_GROUP + 1), 1.1)
        appeal[g] = rng.permutation(w / w.sum())

    rows = []

    for u in range(N_USERS):
        g = u % N_GROUPS
        own = np.arange(g * ITEMS_PER_GROUP, (g + 1) * ITEMS_PER_GROUP)
        n = int(rng.integers(5, 26))

        n_noise = max(1, int(round(n * 0.1)))
        picked = list(rng.choice(own, size=min(n, len(own)),
                                 replace=False, p=appeal[g]))
        picked += list(rng.choice(N_ITEMS, size=n_noise, replace=False))

        for i in dict.fromkeys(picked):          # bỏ trùng, giữ thứ tự
            lo = max(0, int(release_day[i]))
            if lo >= WINDOW_DAYS:
                continue                          # chưa phát hành trong cửa sổ
            day = int(rng.integers(lo, WINDOW_DAYS))
            # giờ chơi lệch nặng: trung vị khoảng 12h, đuôi dài tới vài nghìn
            hours = float(np.clip(rng.lognormal(2.5, 1.3), 0.1, 5000.0))
            rows.append((u, int(i), hours, T0 + day * DAY))

    df = pd.DataFrame(rows, columns=["u_idx", "i_idx", "hours", "ts"])
    df = df.drop_duplicates(subset=["u_idx", "i_idx"], keep="last")
    return df.astype({"u_idx": "int32", "i_idx": "int32",
                      "hours": "float32", "ts": "int64"})


def main(out_dir: str) -> None:
    rng = np.random.default_rng(SEED)
    out = Path(out_dir)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "interim").mkdir(parents=True, exist_ok=True)

    items = build_items(rng)
    inter = build_interactions(rng, items)

    cut_train = T0 + CUT_TRAIN_DAY * DAY
    cut_valid = T0 + CUT_VALID_DAY * DAY

    train = inter[inter["ts"] < cut_train]
    valid = inter[(inter["ts"] >= cut_train) & (inter["ts"] < cut_valid)]
    test = inter[inter["ts"] >= cut_valid]

    train.to_parquet(out / "splits" / "train.parquet", index=False)
    valid.to_parquet(out / "splits" / "valid.parquet", index=False)
    test.to_parquet(out / "splits" / "test.parquet", index=False)
    items.to_parquet(out / "interim" / "items.parquet", index=False)

    (out / "interim" / "maps.json").write_text(json.dumps({
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "cutoff_train": int(cut_train),
        "cutoff_valid": int(cut_valid),
        "seed": SEED,
    }, indent=2), encoding="utf-8")

    cold = (items["date_release"].to_numpy() >= cut_train).sum()
    test_users = test["u_idx"].nunique()
    print(f"train {len(train):>6}  valid {len(valid):>5}  test {len(test):>5}")
    print(f"người dùng có mặt trong test: {test_users}")
    print(f"game phát hành sau mốc train (cold-start): {cold}")
    print(f"giờ chơi — trung vị {inter['hours'].median():.1f}h, "
          f"lớn nhất {inter['hours'].max():.0f}h")
    print(f"đã ghi vào {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data_fake")
    main(p.parse_args().out)
