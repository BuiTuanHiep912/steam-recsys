"""Khung đánh giá. NGƯỜI C — ngày 4-5.

Nhận một Recommender bất kỳ và chấm điểm. A và B gọi run_eval khi chạy thí nghiệm.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.eval.metrics import evaluate


def build_truth(test_df: pd.DataFrame, user_ids: np.ndarray) -> dict[int, set[int]]:
    """{chỉ số hàng trong user_ids -> tập i_idx đúng}.

    Chú ý: khóa là CHỈ SỐ HÀNG, không phải u_idx. Đây là chỗ dễ nhầm nhất.

    đưa ra tập game mỗi người thực sự thích, dạng dict[user_id] = {tập game dungd}
    """
    pos = {int(u): r for r, u in enumerate(user_ids)}
    truth: dict[int, set[int]] = {}
    for u, i in zip(test_df["u_idx"].to_numpy(), test_df["i_idx"].to_numpy()):
        r = pos.get(int(u))
        if r is not None:                      # user không nằm trong user_ids -> bỏ
            truth.setdefault(r, set()).add(int(i))
    return truth

def _check_block(block: np.ndarray, n_rows: int, k: int, n_items: int | None,
                 name: str) -> None:
    """Kiểm tra đầu ra của recommend đúng hợp đồng. KHÔNG sửa, chỉ báo lỗi."""
    assert block.shape == (n_rows, k), f"{name}.recommend trả về {block.shape}, cần {(n_rows, k)}"
    assert np.issubdtype(block.dtype, np.integer), f"{name}.recommend trả về kiểu {block.dtype}, cần int"
    assert block.min() >= 0, f"{name}.recommend trả về chỉ số âm (-1?)"
    if n_items is not None:
        assert block.max() < n_items, f"{name}.recommend trả về i_idx >= n_items"

def rank_all(
    model,
    user_ids: np.ndarray,
    k: int = 20,
    item_subset: np.ndarray | None = None,
    seen: dict[int, set[int]] | None = None,
    batch_size: int = 2048,
) -> np.ndarray:
    """Sinh dự đoán cho một danh sách người dùng.

    item_subset giới hạn catalog được xếp hạng — dùng cho RQ3 khi chỉ chấm riêng
    nhóm game mới phát hành. None nghĩa là toàn bộ catalog.
    seen        lịch sử train {u_idx -> tập i_idx}; CHỈ cần khi có item_subset.

    * Chia lô người dùng để không nổ bộ nhớ.
    * KHÔNG lấy mẫu 100 negative: xếp hạng TOÀN catalog.
    * Không có item_subset: recommend của B đã loại game đã xem, ở đây chỉ kiểm tra.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    name = getattr(model, "name", type(model).__name__)
    n_items = getattr(model, "n_items", None)

    keep = None
    if item_subset is not None:
        if n_items is None:
            raise ValueError("item_subset cần model.n_items để xếp hạng toàn catalog")
        keep = np.zeros(n_items, dtype=bool)
        keep[np.asarray(item_subset, dtype=np.int64)] = True
        seen = seen or {}

    blocks = []
    for s in range(0, len(user_ids), batch_size):
        batch = user_ids[s:s + batch_size]
        if keep is None:
            block = np.asarray(model.recommend(batch, k=k, exclude_seen=True))
            _check_block(block, len(batch), k, n_items, name)
        else:
            # Xin xếp hạng TOÀN catalog (exclude_seen=False để luôn đủ n_items cột),
            # rồi tự bỏ game đã xem và game ngoài subset, GIỮ NGUYÊN thứ tự.
            full = np.asarray(model.recommend(batch, k=n_items, exclude_seen=False))
            _check_block(full, len(batch), n_items, n_items, name)
            block = np.empty((len(batch), k), dtype=np.int32)
            for r, (u, row) in enumerate(zip(batch, full)):
                ok = keep[row]
                hist = seen.get(int(u))
                if hist:
                    ok &= ~np.isin(row, list(hist))
                cand = row[ok]
                if len(cand) < k:
                    raise ValueError(f"user {u}: subset chỉ còn {len(cand)} game chưa xem < k={k}")
                block[r] = cand[:k]
        blocks.append(block.astype(np.int32))
    if not blocks:
        return np.empty((0, k), dtype=np.int32)
    return np.vstack(blocks)


def run_eval(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: Iterable[int] = (10, 20),
    item_subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Tiện ích gộp rank_all + evaluate.

    Chỉ chấm trên những người dùng có ít nhất một tương tác trong test_df
    (khi có item_subset: ít nhất một tương tác VỚI GAME TRONG SUBSET).
    """
    k_list = list(k_list)
    n_items = getattr(model, "n_items", None)
    if n_items is None:
        n_items = int(max(train_df["i_idx"].max(), test_df["i_idx"].max()) + 1)

    if item_subset is not None:
        item_subset = np.unique(np.asarray(item_subset, dtype=np.int64))
        test_df = test_df[test_df["i_idx"].isin(item_subset)]

    user_ids = test_df["u_idx"].unique()
    seen = None
    if item_subset is not None:
        seen = train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
    preds = rank_all(model, user_ids, k=max(k_list), item_subset=item_subset, seen=seen)
    truth = build_truth(test_df, user_ids)

    item_pop = np.bincount(train_df["i_idx"].to_numpy(), minlength=n_items)
    n_users = int(train_df["u_idx"].nunique())          # số user TRONG TRAIN
    n_catalog = n_items if item_subset is None else len(item_subset)
    return evaluate(preds, truth, k_list, item_pop, n_catalog, n_users)


def log_run(results: dict, config: dict, path: str = "results/runs.csv") -> None:
    """Ghi một dòng vào results/runs.csv kèm siêu tham số và seed.

    Mỗi con số trong báo cáo phải truy được về một dòng ở đây.
    Bắt buộc có cột train_set ∈ {"train", "train+valid"}.
    """
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **config, **results}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():
        new = pd.concat([pd.read_csv(p), new], ignore_index=True)
    new.to_csv(p, index=False)
