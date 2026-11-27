"""Test khung đánh giá bằng một mô hình giả biết trước đáp án.
Chạy: python -m src.eval.test_evaluate"""
import numpy as np
import pandas as pd

from src.eval.evaluate import build_truth, rank_all, run_eval


class FixedModel:
    """Mô hình giả: mọi user nhận cùng thứ tự 0, 1, 2, ... (bỏ game đã xem)."""
    name = "Fixed"

    def __init__(self, n_items, seen):
        self.n_items, self.seen = n_items, seen

    def recommend(self, user_ids, k=10, exclude_seen=True):
        rows = []
        for u in user_ids:
            hist = self.seen.get(int(u), set()) if exclude_seen else set()
            order = [i for i in range(self.n_items) if i not in hist]
            rows.append(order[:k])
        return np.array(rows, dtype=np.int32)


train = pd.DataFrame({"u_idx": [0, 0, 1], "i_idx": [0, 1, 2]})
test = pd.DataFrame({"u_idx": [1, 0, 1], "i_idx": [0, 2, 4]})   # user 1 xuất hiện TRƯỚC user 0
seen = {0: {0, 1}, 1: {2}}


# Mục đích: khóa của truth là CHỈ SỐ HÀNG trong user_ids, không phải u_idx
def test_build_truth():
    users = test.u_idx.unique()                       # [1, 0]
    assert list(users) == [1, 0]
    assert build_truth(test, users) == {0: {0, 4}, 1: {2}}


# Mục đích: chia lô không làm đổi kết quả, đầu ra đúng kích thước
def test_rank_all_batch():
    m = FixedModel(6, seen)
    a = rank_all(m, np.array([1, 0]), k=3, batch_size=1)
    b = rank_all(m, np.array([1, 0]), k=3, batch_size=100)
    assert (a == b).all() and a.shape == (2, 3) and a.dtype == np.int32
    assert a.tolist() == [[0, 1, 3], [2, 3, 4]]


# Mục đích: item_subset chỉ giữ game trong subset, GIỮ NGUYÊN thứ tự
def test_rank_all_subset():
    m = FixedModel(6, seen)
    p = rank_all(m, np.array([1, 0]), k=2, item_subset=np.array([5, 4, 3, 2]), seen=seen)
    assert p.tolist() == [[3, 4], [2, 3]]           # user 1 đã xem game 2 -> bị loại


# Mục đích: run_eval khớp con số tính tay
# user 1: gợi ý [0,1,3], đúng {0,4} -> trúng hạng 1 -> nDCG@3 = 1/(1+1/log2 3) = 0,6131
# user 0: gợi ý [2,3,4], đúng {2}   -> trúng hạng 1 -> nDCG@3 = 1
def test_run_eval():
    out = run_eval(FixedModel(6, seen), train, test, k_list=(3,))
    assert np.isclose(out["ndcg@3"], (0.6131 + 1.0) / 2, atol=1e-4)
    assert np.isclose(out["mrr"], 1.0)
    assert np.isclose(out["coverage@10"], 5 / 6)        # {0,1,3,2,4}


# Mục đích: mô hình trả về -1 thì phải BÁO LỖI, không âm thầm chấm sai
def test_bat_loi():
    class Bad(FixedModel):
        def recommend(self, user_ids, k=10, exclude_seen=True):
            return -np.ones((len(user_ids), k), dtype=np.int32)
    try:
        rank_all(Bad(6, seen), np.array([0]), k=3)
    except AssertionError:
        return
    raise RuntimeError("rank_all không bắt được -1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK  ", name)