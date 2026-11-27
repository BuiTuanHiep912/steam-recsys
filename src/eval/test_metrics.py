"""Test bộ độ đo bằng ví dụ tính tay. Chạy: python -m src.eval.test_metrics"""
import numpy as np

from src.eval.metrics import (coverage_at_k, evaluate, ndcg_at_k, novelty_at_k,
                              recall_at_k, reciprocal_rank)


def close(a, b):
    return np.isclose(a, b, atol=1e-4)


# Mục đích: ba độ đo ĐỘ ĐÚNG khớp con số trên giấy (ví dụ 1)
def test_vi_du_1():
    p, t = np.array([7, 2, 9]), {2, 5}
    assert close(ndcg_at_k(p, t, 3), 0.3869)
    assert close(recall_at_k(p, t, 3), 0.5)
    assert close(reciprocal_rank(p, t), 0.5)


# Mục đích: hai độ đo ĐỘ ĐA DẠNG khớp con số trên giấy (ví dụ 2)
def test_vi_du_2():
    P = np.array([[0, 1], [0, 3]])
    assert close(coverage_at_k(P, 5), 0.6)
    assert close(novelty_at_k(P, np.array([3, 1, 0, 2, 1]), 3), 0.3538)


# Mục đích: các trường hợp BIÊN — nơi lỗi hay trốn
def test_bien():
    assert close(ndcg_at_k(np.array([5, 1, 2]), {5}, 3), 1.0)       # trúng vị trí 1 → đúng 1
    assert ndcg_at_k(np.array([1, 2, 3]), {9}, 3) == 0.0            # không trúng → 0
    assert reciprocal_rank(np.array([1, 2, 3]), {9}) == 0.0
    assert ndcg_at_k(np.array([1, 2]), set(), 2) == 0.0             # T rỗng → không chia 0
    assert recall_at_k(np.array([1, 2]), set(), 2) == 0.0
    # quyết định 3: 2 trúng / min(5, 2) = 1.0
    assert close(recall_at_k(np.array([1, 2]), {1, 2, 3, 4, 5}, 2), 1.0)
    # quyết định 2: pop = 0 vẫn hữu hạn
    assert np.isfinite(novelty_at_k(np.array([[0]]), np.array([0]), 10))


# Mục đích: hàm gộp đúng HỢP ĐỒNG — đúng khóa, đúng kiểu, thiếu tham số không lỗi
def test_evaluate():
    preds = np.array([[7, 2, 9], [5, 1, 2]])
    out = evaluate(preds, {0: {2, 5}, 1: {5}}, k_list=(3,),
                   item_pop=np.ones(10), n_items=10, n_users=4)
    assert set(out) == {"ndcg@3", "recall@3", "mrr", "coverage@10", "novelty@10"}
    assert close(out["ndcg@3"], (0.3869 + 1.0) / 2)                 # trung bình theo user
    assert all(type(v) is float for v in out.values())              # float thường
    assert "novelty@10" not in evaluate(preds, {0: {2}}, k_list=(3,))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK  ", name)
