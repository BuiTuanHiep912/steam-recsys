"""Bộ độ đo. NGƯỜI C — ngày 1-3. LÀM TRƯỚC MỌI THỨ KHÁC.

evaluate là HÀM THUẦN: không đọc file, không gọi model. Nhờ vậy kiểm thử được bằng
một ví dụ nhỏ tính tay trên giấy — đó là tiêu chí Xong khi của việc này.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def dcg_at_k(hits: np.ndarray, k: int) -> float:
    """DCG = sum_{r=1..k} rel_r / log2(r + 1)

    hits: mảng 0/1 độ dài >= k, phần tử r là 1 nếu vị trí r gợi ý đúng.
    cộng điểm cho mỗi lần trúng tức là một game trong danh sách gợi ý và người chơi thực sự có tương tác với game này
    hits[i] = 0 tức game ở vị trí i + 1 không trúng
    hits[i] = 1 tức game ở vị trí i + 1 trúng
    """
    hits = np.asarray(hits, dtype=np.float64)[:k]
    if hits.size == 0:
        return 0.0
    return float(np.sum(hits / np.log2(np.arange(2, hits.size + 2)))) # công thức tính nếu trúng là 1 / log2(hạng + 1)


def ndcg_at_k(pred_row: np.ndarray, truth_set: set[int], k: int) -> float:
    """nDCG = DCG / IDCG.

    IDCG là DCG của trường hợp lý tưởng: min(len(truth_set), k) item đúng xếp đầu.

    nDCG đo cùng lúc 2 thứ: model gợi ý trúng bao nhiêu game và các game trúng nằm ở đâu trong danh sách
    "Trúng" ở vị trí cao là dựa trên số điểm tính được ở hàm dcg_at_k
    chuẩn hoá về 0 - 1
    nDCG = DCG / IDCG.
    IDCG là số điểm DCG cao nhất mà mỗi user đạt được
    """
    if not truth_set:
        return 0.0
    hits = np.isin(np.asarray(pred_row)[:k], list(truth_set))
    idcg = dcg_at_k(np.ones(min(len(truth_set), k)), k)
    return dcg_at_k(hits, k) / idcg


def recall_at_k(pred_row: np.ndarray, truth_set: set[int], k: int) -> float:
    """|top-k ∩ T| / min(|T|, k). Nhất quán với IDCG của nDCG.

    Chia cho min(|T|, k) để người có nhiều game trong test vẫn đạt được 1.
    T rỗng -> 0.

    Đo tìm được bao nhiêu game đúng (game mà thực sự được recommend)
    """
    if not truth_set:
        return 0.0
    n_hit = int(np.isin(np.asarray(pred_row)[:k], list(truth_set)).sum())
    return n_hit / min(len(truth_set), k)


def reciprocal_rank(pred_row: np.ndarray, truth_set: set[int]) -> float:
    """1 / thứ hạng của item đúng ĐẦU TIÊN, xét TOÀN BỘ pred_row (k=20 khi gọi
    từ run_eval). Không có item đúng nào thì bằng 0.

    Người dùng phải kéo xuống bao xa mới gặp được game mình thích
    pred_row là danh sách gợi ý của một model
    """
    for r, i in enumerate(pred_row, start=1):
        if int(i) in truth_set:
            return 1.0 / r
    return 0.0


def coverage_at_k(preds: np.ndarray, n_items: int) -> float:
    """Tỷ lệ catalog từng xuất hiện trong gợi ý của ít nhất một người dùng."""
    return len(np.unique(preds)) / n_items


def novelty_at_k(preds: np.ndarray, item_pop: np.ndarray, n_users: int) -> float:
    """Độ phổ biến nghịch đảo trung bình, có làm trơn +1.

        novelty = trung bình trên mọi ô của preds của  -log2( (pop(i)+1) / (n_users+1) )

    item_pop và n_users đếm TỪ TẬP HUẤN LUYỆN. Làm trơn +1 để game chưa ai chơi
    (game phát hành sau mốc cắt, dùng cho RQ3) có novelty cao nhất nhưng hữu hạn.

    đo mức độ phổ biến của game được rcm
    độ phổ biến tính dựa trên số lượng user dã recommend
    Giới hạn của báo cáo: "độ phổ biến được ước lượng qua số review tích cực trong train, vì dữ liệu không có số người chơi thực tế".
    """
    pop = np.asarray(item_pop, dtype=np.float64)[np.asarray(preds)]
    return float(np.mean(-np.log2((pop + 1) / (n_users + 1))))


def evaluate(
    preds: np.ndarray,
    truth: Mapping[int, set[int]],
    k_list: Iterable[int] = (10, 20),
    item_pop: np.ndarray | None = None,
    n_items: int | None = None,
    n_users: int | None = None,
) -> dict[str, float]:
    """Gộp mọi độ đo thành một dict.

    preds : (n_users, k) các i_idx đã xếp hạng, hàng r ứng với người dùng thứ r
    truth : {chỉ số hàng -> tập i_idx đúng trong tập kiểm thử}

    Khóa trả về: ndcg@10, ndcg@20, recall@10, recall@20, mrr,
                 coverage@10, novelty@10

    Giá trị là float thường, không phải np.float32 — để ghi thẳng ra CSV.

    ndcg@10: trúng bao nhiêu và có xếp game đúng lên cao không trong top 10 giá trị [0, 1]
    ndcg@20: trúng bao nhiêu và có xếp game đúng lên cao không trong top 20 giá trị [0, 1]
    recall@10: tìm được bao nhiêu phần game đúng trong top 10 giá trị [0, 1]
    recall@20: tìm được bao nhiêu phần game đúng trong top 20 giá trị [0, 1]
    mrr: trung bình người dùng phải kéo xuống bao xa mới gặp game đúng đầu tiên top 20 giá trị [0, 1]
    coverage@10: model dùng được bao nhiêu phần catalog (catalog là danh sách game mà hệ thống có thể gợi ý) -> số game khác nhau xuất hiện trong top-10 của mọi user chia cho n_items
    novelty@10: game được gợi ý ít phổ biến đến mức nào -log2((pop + 1)/(n_users + 1))
    """
    preds = np.asarray(preds)
    k_list = list(k_list)
    rows = list(truth.keys())
    out: dict[str, float] = {}

    for k in k_list:
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(preds[r], truth[r], k) for r in rows]))
        out[f"recall@{k}"] = float(np.mean([recall_at_k(preds[r], truth[r], k) for r in rows]))
    out["mrr"] = float(np.mean([reciprocal_rank(preds[r], truth[r]) for r in rows]))

    top10 = preds[rows, :10]
    if n_items is not None:
        out["coverage@10"] = float(coverage_at_k(top10, n_items))
    if item_pop is not None and n_users is not None:
        out["novelty@10"] = float(novelty_at_k(top10, item_pop, n_users))
    return out


# ---------------------------------------------------------------------------
# Ví dụ tính tay — đây là tiêu chí Xong khi của metrics.py
# ---------------------------------------------------------------------------
# Một người dùng, k = 3, gợi ý [7, 2, 9], item đúng = {2, 5}.
#
#   vị trí 1: item 7  -> sai  -> 0
#   vị trí 2: item 2  -> ĐÚNG -> 1/log2(3) = 0,6309
#   vị trí 3: item 9  -> sai  -> 0
#   DCG  = 0,6309
#   IDCG = 1/log2(2) + 1/log2(3) = 1 + 0,6309 = 1,6309
#   nDCG = 0,6309 / 1,6309 = 0,3868
#   recall@3 = 1/2 = 0,5
#   MRR = 1/2 = 0,5
#
# Viết test khẳng định đúng các con số này trước khi chạy bất cứ mô hình nào.
# Ví dụ 2 — coverage và novelty
# n_items = 5, n_users = 3, item_pop = [3, 1, 0, 2, 1], k = 2
# preds: user 0 -> [0, 1], user 1 -> [0, 3]
#
#   coverage@2 = |{0, 1, 3}| / 5 = 0,6
#   novelty: item 0 -> -log2(4/4) = 0
#            item 1 -> -log2(2/4) = 1
#            item 3 -> -log2(3/4) = 0,4150
#   novelty@2 = (0 + 1 + 0 + 0,4150) / 4 = 0,3538
